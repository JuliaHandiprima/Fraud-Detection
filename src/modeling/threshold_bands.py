from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve


def _pick_threshold_for_precision(
    *,
    precisions: np.ndarray,
    thresholds: np.ndarray,
    target_precision: float,
) -> float | None:
    """
    Given arrays from precision_recall_curve, choose the smallest threshold
    (highest recall) that achieves precision >= target_precision.
    """
    # precision_recall_curve returns precisions/recalls with len = len(thresholds) + 1
    if thresholds.size == 0:
        return None
    p = precisions[1:]
    ok = np.where(p >= target_precision)[0]
    if ok.size == 0:
        return None
    # thresholds are ascending; choose the smallest threshold that still satisfies precision
    return float(thresholds[int(ok[0])])


def _metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, t: float) -> dict[str, float]:
    y_pred = (y_score >= t).astype(int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    flagged_rate = float(y_pred.mean())
    return {"threshold": float(t), "precision": float(precision), "recall": float(recall), "f1": float(f1), "flagged_rate": flagged_rate}


def compute_threshold_bands(
    *,
    y_valid: np.ndarray,
    p_valid: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    target_precision_high: float = 0.80,
    target_precision_med: float = 0.60,
) -> dict:
    precisions, recalls, thresholds = precision_recall_curve(y_valid, p_valid)
    t_high = _pick_threshold_for_precision(precisions=precisions, thresholds=thresholds, target_precision=target_precision_high)
    t_med = _pick_threshold_for_precision(precisions=precisions, thresholds=thresholds, target_precision=target_precision_med)

    report: dict = {
        "targets": {"precision_high": float(target_precision_high), "precision_med": float(target_precision_med)},
        "selected": {"t_high": t_high, "t_med": t_med},
        "valid": {},
        "test": {},
    }
    if t_high is not None:
        report["valid"]["high"] = _metrics_at_threshold(y_valid, p_valid, t_high)
        report["test"]["high"] = _metrics_at_threshold(y_test, p_test, t_high)
    if t_med is not None:
        report["valid"]["med"] = _metrics_at_threshold(y_valid, p_valid, t_med)
        report["test"]["med"] = _metrics_at_threshold(y_test, p_test, t_med)
    return report


def save_threshold_bands(report: dict, output_dir: str | Path, filename: str = "threshold_bands.json") -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(report, indent=2))
    return str(path)

