from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def bootstrap_delta_pr_auc(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        pr_a = average_precision_score(yt, score_a[idx])
        pr_b = average_precision_score(yt, score_b[idx])
        deltas.append(pr_b - pr_a)
    if not deltas:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    deltas = np.array(deltas, dtype=float)
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }


def compare_and_select(
    vanilla_dir: str | Path = "results/vanilla",
    enriched_dir: str | Path = "results/enriched",
    output_path: str | Path = "results/model_selection.json",
) -> dict:
    vanilla_dir = Path(vanilla_dir)
    enriched_dir = Path(enriched_dir)
    vanilla_pred = pd.read_csv(vanilla_dir / "test_predictions.csv")
    enriched_pred = pd.read_csv(enriched_dir / "test_predictions.csv")

    y_true = vanilla_pred["label"].to_numpy(dtype=int)
    if not np.array_equal(y_true, enriched_pred["label"].to_numpy(dtype=int)):
        raise ValueError("Mismatched labels between vanilla and enriched predictions.")

    stats = bootstrap_delta_pr_auc(
        y_true=y_true,
        score_a=vanilla_pred["score"].to_numpy(dtype=float),
        score_b=enriched_pred["score"].to_numpy(dtype=float),
    )

    vanilla_metrics = json.loads((vanilla_dir / "metrics.json").read_text())["metrics"]
    enriched_metrics = json.loads((enriched_dir / "metrics.json").read_text())["metrics"]
    if enriched_metrics["pr_auc"] > vanilla_metrics["pr_auc"]:
        preferred = "enriched"
    elif enriched_metrics["pr_auc"] < vanilla_metrics["pr_auc"]:
        preferred = "vanilla"
    else:
        preferred = "enriched" if enriched_metrics.get("brier_calibrated", 1.0) <= vanilla_metrics.get("brier_calibrated", 1.0) else "vanilla"
    report = {
        "preferred_model": preferred,
        "vanilla_metrics": vanilla_metrics,
        "enriched_metrics": enriched_metrics,
        "delta_pr_auc_bootstrap": stats,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    result = compare_and_select()
    print(json.dumps(result, indent=2))
