"""
Optional SHAP + calibrated risk scores via the champion XGBoost bundle (same as training/inference).

Set CHAMPION_MANIFEST_PATH (e.g. results/variants/champion_model.json) and optionally
CHAMPION_VARIANT_NAME (default: variant_base). Requires process CWD = repo root
so relative artifact paths resolve.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def attach_champion_explanation(app_data: dict[str, Any]) -> dict[str, Any] | None:
    """
    Run AccountFraudInferenceService.score() and return a JSON-serializable bundle
    for retriever / writer / UI. Returns None if not configured.
    """
    manifest = os.getenv("CHAMPION_MANIFEST_PATH")
    if not manifest:
        return None
    mp = Path(manifest)
    if not mp.is_file():
        return {"shap_available": False, "shap_note": f"CHAMPION_MANIFEST_PATH not found: {manifest}"}

    variant = os.getenv("CHAMPION_VARIANT_NAME", "variant_base")
    try:
        from src.inference.account_inference import AccountFraudInferenceService

        svc = AccountFraudInferenceService.from_champion_manifest(str(mp), variant, enriched=True)
        out = svc.score(app_data)
        r = out["report"]
        return {
            "shap_available": True,
            "top_shap_drivers": r.get("top_shap_drivers"),
            "risk_scores": {
                "p_model": r.get("p_model"),
                "p_model_calibrated": r.get("p_model_calibrated"),
                "p_final": r.get("p_final"),
                "confidence_band": r.get("confidence_band"),
                "recommended_action": r.get("recommended_action"),
            },
            "retriever_features_model": r.get("retriever_features"),
        }
    except Exception as e:
        return {"shap_available": False, "shap_error": str(e)}
