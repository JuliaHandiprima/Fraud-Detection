from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    brier_raw_valid: float
    brier_cal_valid: float
    improvement: float
    model: LogisticRegression


def calibrate_platt(y_true_valid: np.ndarray, y_score_valid: np.ndarray) -> CalibrationResult:
    y_score_valid = np.asarray(y_score_valid, dtype=float).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(y_score_valid, y_true_valid)
    y_cal = model.predict_proba(y_score_valid)[:, 1]
    brier_raw = float(brier_score_loss(y_true_valid, y_score_valid.ravel()))
    brier_cal = float(brier_score_loss(y_true_valid, y_cal))
    return CalibrationResult(
        method="platt_sigmoid",
        brier_raw_valid=brier_raw,
        brier_cal_valid=brier_cal,
        improvement=brier_raw - brier_cal,
        model=model,
    )


def _subgroup_parity_difference(y_pred: np.ndarray, groups: pd.DataFrame) -> float | None:
    if groups is None or groups.empty:
        return None
    diffs = []
    for col in groups.columns:
        rates = groups.assign(pred=y_pred).groupby(col)["pred"].mean()
        if len(rates) > 1:
            diffs.append(float(rates.max() - rates.min()))
    return float(max(diffs)) if diffs else None


def _equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.DataFrame) -> float | None:
    if groups is None or groups.empty:
        return None
    diffs = []
    for col in groups.columns:
        tprs = []
        for _, idx in groups.groupby(col).groups.items():
            yt = y_true[list(idx)]
            yp = y_pred[list(idx)]
            pos = (yt == 1)
            if pos.sum() == 0:
                continue
            tprs.append(float((yp[pos] == 1).mean()))
        if len(tprs) > 1:
            diffs.append(float(max(tprs) - min(tprs)))
    return float(max(diffs)) if diffs else None


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_score_calibrated: np.ndarray,
    threshold: float,
    groups: pd.DataFrame | None = None,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    y_pred_cal = (y_score_calibrated >= threshold).astype(int)
    subgroup_parity_diff = _subgroup_parity_difference(y_pred, groups) if groups is not None else None
    eq_opp_diff = _equal_opportunity_difference(y_true, y_pred, groups) if groups is not None else None
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier_raw": float(brier_score_loss(y_true, y_score)),
        "brier_calibrated": float(brier_score_loss(y_true, y_score_calibrated)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_calibrated": float(precision_score(y_true, y_pred_cal, zero_division=0)),
        "recall_calibrated": float(recall_score(y_true, y_pred_cal, zero_division=0)),
        "f1_calibrated": float(f1_score(y_true, y_pred_cal, zero_division=0)),
        "threshold": float(threshold),
        "alert_yield_pct": float(y_pred.mean() * 100.0),
        "fairness": {
            "subgroup_parity_difference": subgroup_parity_diff,
            "equal_opportunity_difference": eq_opp_diff,
            "status": "available" if groups is not None and not groups.empty else "skipped_no_groups",
        },
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 19)
    f1_vals = [f1_score(y_true, (y_score >= thr).astype(int), zero_division=0) for thr in candidates]
    return float(candidates[int(np.argmax(f1_vals))])
