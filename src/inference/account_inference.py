from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from openai import OpenAI

from src.preprocessing.baf_preprocessor import BAFPreprocessor
from src.retriever.enrichment import build_retriever_features_for_records, retrieve_retriever_evidence


@dataclass(frozen=True)
class DecisionPolicy:
    approve_threshold: float = 0.30
    escalate_threshold: float = 0.65


def recommend_action(score: float, policy: DecisionPolicy) -> str:
    if score < policy.approve_threshold:
        return "approve"
    if score < policy.escalate_threshold:
        return "escalate"
    return "reject"


def _top_shap_features(explainer: shap.Explainer | None, model: Any, X: pd.DataFrame, top_n: int = 5) -> list[dict[str, float]]:
    if explainer is not None:
        values = explainer(X)
        shap_values = values.values[0]
    else:
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        contrib = model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_values = contrib[0, :-1]
    ranked = np.argsort(np.abs(shap_values))[::-1][:top_n]
    return [{"feature": X.columns[i], "shap_value": float(shap_values[i])} for i in ranked]


def _generate_llm_report(payload: dict[str, Any]) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        return "LLM report unavailable: DEEPSEEK_API_KEY not set."

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = (
        "Write a concise fraud analyst report using only provided facts. "
        "Include score interpretation, top drivers, retriever evidence, and final recommendation."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a fraud risk analyst. Do not invent facts."},
            {"role": "user", "content": prompt + "\n\nFacts:\n" + json.dumps(payload, indent=2)},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


class AccountFraudInferenceService:
    def __init__(
        self,
        model_path: str | Path,
        preprocessor_path: str | Path,
        calibrator_path: str | Path | None = None,
        risk_calibrator_path: str | Path | None = None,
        threshold_bands_path: str | Path | None = None,
        enriched: bool = True,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        self.preprocessor: BAFPreprocessor = BAFPreprocessor.load(preprocessor_path)
        self.calibrator = joblib.load(calibrator_path) if calibrator_path else None
        if risk_calibrator_path is None:
            candidate = self.model_path.parent / "risk_calibrator.pkl"
            risk_calibrator_path = candidate if candidate.exists() else None
        self.risk_calibrator = joblib.load(risk_calibrator_path) if risk_calibrator_path else None
        self.risk_calibrator_meta = None
        meta_path = self.model_path.parent / "risk_calibrator_metadata.json"
        if meta_path.exists():
            self.risk_calibrator_meta = json.loads(meta_path.read_text())

        if threshold_bands_path is None:
            candidate = self.model_path.parent / "threshold_bands.json"
            threshold_bands_path = candidate if candidate.exists() else None
        self.threshold_bands = json.loads(Path(threshold_bands_path).read_text()) if threshold_bands_path else None
        self.enriched = enriched
        self.policy = policy or DecisionPolicy()
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = None

    def score(self, record: dict[str, Any]) -> dict[str, Any]:
        raw_df = pd.DataFrame([record])
        X = self.preprocessor.transform_records(raw_df)
        retr = build_retriever_features_for_records(raw_df)
        retriever_features = retr.iloc[0].to_dict()
        retriever_evidence = retrieve_retriever_evidence(record, top_k=5)
        if self.enriched:
            X = pd.concat([X, retr], axis=1)

        p_model = float(self.model.predict_proba(X)[:, 1][0])
        p_model_cal = p_model
        if self.calibrator is not None:
            p_model_cal = float(self.calibrator.predict_proba(np.array([[p_model]], dtype=float))[:, 1][0])

        p_final = p_model_cal
        if self.risk_calibrator is not None and retriever_features:
            eps = 1e-6
            p_clip = float(min(max(p_model_cal, eps), 1.0 - eps))
            logit_p = float(np.log(p_clip / (1.0 - p_clip)))
            feature_cols = None
            if isinstance(self.risk_calibrator_meta, dict):
                feature_cols = self.risk_calibrator_meta.get("feature_cols")
            if isinstance(feature_cols, list) and feature_cols:
                row = {}
                row["logit_p_model_calibrated"] = logit_p
                row.update({k: float(v) for k, v in retriever_features.items()})
                x_vec = np.array([[row.get(col, 0.0) for col in feature_cols]], dtype=float)
            else:
                x_vec = np.array([[logit_p] + [float(v) for v in retriever_features.values()]], dtype=float)
            p_final = float(self.risk_calibrator.predict_proba(x_vec)[:, 1][0])

        t_med = None
        t_high = None
        if isinstance(self.threshold_bands, dict):
            selected = self.threshold_bands.get("selected", {})
            t_med = selected.get("t_med")
            t_high = selected.get("t_high")
        confidence_band = "unknown"
        if isinstance(t_med, (float, int)) and isinstance(t_high, (float, int)):
            confidence_band = "high" if p_final >= float(t_high) else ("medium" if p_final >= float(t_med) else "low")
        elif isinstance(t_med, (float, int)):
            confidence_band = "medium" if p_final >= float(t_med) else "low"

        predicted_label = int(p_final >= float(t_med)) if isinstance(t_med, (float, int)) else int(p_final >= 0.5)

        action = recommend_action(p_final, self.policy)
        top_drivers = _top_shap_features(self.explainer, self.model, X, top_n=5)
        grounded = {
            "p_model": p_model,
            "p_model_calibrated": p_model_cal,
            "p_final": p_final,
            "predicted_label": predicted_label,
            "confidence_band": confidence_band,
            "thresholds": {"t_med": t_med, "t_high": t_high},
            "decision_policy": asdict(self.policy),
            "recommended_action": action,
            "top_shap_drivers": top_drivers,
            "retriever_features": retriever_features,
            "retriever_cases": retriever_evidence.get("top_cases") if isinstance(retriever_evidence, dict) else [],
        }
        report_text = _generate_llm_report(grounded)
        return {
            "report": grounded,
            "llm_report_text": report_text,
        }

    @classmethod
    def from_champion_manifest(
        cls,
        manifest_path: str | Path,
        variant_name: str,
        enriched: bool = True,
        policy: DecisionPolicy | None = None,
    ) -> "AccountFraudInferenceService":
        payload = json.loads(Path(manifest_path).read_text())
        champion = payload["overall_champion"]
        model_path = payload["artifact_template"].format(variant=variant_name, champion=champion)
        calibrator_path = payload["calibrator_template"].format(variant=variant_name, champion=champion)
        preprocessor_path = payload["preprocessor_template"].format(variant=variant_name, champion=champion)
        risk_calibrator_template = payload.get("risk_calibrator_template")
        threshold_bands_template = payload.get("threshold_bands_template")
        risk_calibrator_path = (
            risk_calibrator_template.format(variant=variant_name, champion=champion) if isinstance(risk_calibrator_template, str) else None
        )
        threshold_bands_path = (
            threshold_bands_template.format(variant=variant_name, champion=champion) if isinstance(threshold_bands_template, str) else None
        )
        return cls(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            calibrator_path=calibrator_path,
            risk_calibrator_path=risk_calibrator_path,
            threshold_bands_path=threshold_bands_path,
            enriched=enriched,
            policy=policy,
        )
