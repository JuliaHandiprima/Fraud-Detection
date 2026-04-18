from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import shap


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, values.shape[0])
    return np.argsort(np.abs(values))[::-1][:k]


def save_shap_artifacts(
    *,
    model: Any,
    X: pd.DataFrame,
    record_index: Iterable[int] | np.ndarray,
    output_dir: str | Path,
    prefix: str,
    top_k: int = 10,
) -> dict[str, str]:
    """
    Persist SHAP evidence artifacts for a given matrix.

    - `{prefix}_global_mean_abs.json`: mean(|SHAP|) per feature
    - `{prefix}_top_drivers.jsonl`: per-record top-k drivers for interpretability
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_mat = X.to_numpy()
    feature_names = list(X.columns)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_mat)
    if isinstance(shap_values, list):
        # Some SHAP versions return [class0, class1] for classifiers
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values, dtype=float)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    global_path = out / f"{prefix}_global_mean_abs.json"
    global_path.write_text(json.dumps({name: float(val) for name, val in zip(feature_names, mean_abs)}, indent=2))

    drivers_path = out / f"{prefix}_top_drivers.jsonl"
    with drivers_path.open("w", encoding="utf-8") as f:
        for idx, row_idx in enumerate(record_index):
            sv = shap_values[idx]
            top_idx = _top_k_indices(sv, top_k)
            drivers = []
            for j in top_idx:
                drivers.append(
                    {
                        "feature": feature_names[int(j)],
                        "value": float(X_mat[idx, int(j)]),
                        "shap_value": float(sv[int(j)]),
                    }
                )
            f.write(json.dumps({"record_index": int(row_idx), "top_drivers": drivers}) + "\n")

    return {"global_mean_abs_json": str(global_path), "top_drivers_jsonl": str(drivers_path)}

