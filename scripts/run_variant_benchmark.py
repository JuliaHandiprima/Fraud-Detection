from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modeling.compare_models import compare_and_select
from src.modeling.train_enriched import train_enriched
from src.modeling.train_vanilla import train_vanilla


def _mean_std(records: list[float]) -> dict[str, float]:
    arr = np.array(records, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def run_variant(
    variant_name: str,
    data_path: str,
    out_root: Path,
    cpu_only: bool,
    disable_yeojohnson: bool,
    disable_smote: bool,
    smote_sampling_strategy: float,
    smote_random_state: int,
    fairness_group_cols: list[str] | None,
) -> dict:
    variant_dir = out_root / variant_name
    vanilla_dir = variant_dir / "vanilla"
    enriched_dir = variant_dir / "enriched"
    vanilla = train_vanilla(
        data_path=data_path,
        output_dir=vanilla_dir,
        prefer_gpu=not cpu_only,
        use_yeo_johnson=not disable_yeojohnson,
        use_smote=not disable_smote,
        smote_sampling_strategy=smote_sampling_strategy,
        smote_random_state=smote_random_state,
        fairness_group_cols=fairness_group_cols,
    )
    enriched = train_enriched(
        data_path=data_path,
        output_dir=enriched_dir,
        prefer_gpu=not cpu_only,
        use_yeo_johnson=not disable_yeojohnson,
        use_smote=not disable_smote,
        smote_sampling_strategy=smote_sampling_strategy,
        smote_random_state=smote_random_state,
        fairness_group_cols=fairness_group_cols,
    )
    selection = compare_and_select(
        vanilla_dir=vanilla_dir,
        enriched_dir=enriched_dir,
        output_path=variant_dir / "model_selection.json",
    )
    result = {
        "variant": variant_name,
        "data_path": data_path,
        "vanilla": vanilla["metrics"],
        "enriched": enriched["metrics"],
        "selection": selection,
    }
    (variant_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def aggregate(results: list[dict]) -> dict:
    vanilla_pr = [r["vanilla"]["pr_auc"] for r in results]
    enriched_pr = [r["enriched"]["pr_auc"] for r in results]
    vanilla_brier = [r["vanilla"]["brier_calibrated"] for r in results]
    enriched_brier = [r["enriched"]["brier_calibrated"] for r in results]
    winner_counts = {
        "vanilla": sum(1 for r in results if r["selection"]["preferred_model"] == "vanilla"),
        "enriched": sum(1 for r in results if r["selection"]["preferred_model"] == "enriched"),
    }
    overall_champion = "enriched" if winner_counts["enriched"] >= winner_counts["vanilla"] else "vanilla"
    return {
        "variant_count": len(results),
        "winner_counts": winner_counts,
        "overall_champion": overall_champion,
        "metrics_aggregate": {
            "vanilla_pr_auc": _mean_std(vanilla_pr),
            "enriched_pr_auc": _mean_std(enriched_pr),
            "vanilla_brier_calibrated": _mean_std(vanilla_brier),
            "enriched_brier_calibrated": _mean_std(enriched_brier),
        },
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full protocol across 6 BAF variants")
    parser.add_argument("--variants-json", required=True, help="JSON string: [{\"name\":\"v1\",\"path\":\"...\"}, ...]")
    parser.add_argument("--results-dir", default="results/variants")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--disable-yeojohnson", action="store_true")
    parser.add_argument("--disable-smote", action="store_true")
    parser.add_argument("--smote-sampling-strategy", type=float, default=0.5)
    parser.add_argument("--smote-random-state", type=int, default=42)
    parser.add_argument("--fairness-group-cols", nargs="*", default=None)
    args = parser.parse_args()

    variants = json.loads(args.variants_json)
    if len(variants) != 6:
        raise ValueError("Expected exactly 6 variants for this benchmark.")

    out_root = Path(args.results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    per_variant = []
    for item in variants:
        per_variant.append(
            run_variant(
                variant_name=item["name"],
                data_path=item["path"],
                out_root=out_root,
                cpu_only=args.cpu_only,
                disable_yeojohnson=args.disable_yeojohnson,
                disable_smote=args.disable_smote,
                smote_sampling_strategy=args.smote_sampling_strategy,
                smote_random_state=args.smote_random_state,
                fairness_group_cols=args.fairness_group_cols,
            )
        )

    summary = aggregate(per_variant)
    (out_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "champion_model.json").write_text(
        json.dumps(
            {
                "overall_champion": summary["overall_champion"],
                "artifact_template": "results/variants/{variant}/{champion}/model.pkl",
                "calibrator_template": "results/variants/{variant}/{champion}/platt_calibrator.pkl",
                "preprocessor_template": "results/variants/{variant}/{champion}/baf_preprocessor.pkl",
                "risk_calibrator_template": "results/variants/{variant}/{champion}/risk_calibrator.pkl",
                "threshold_bands_template": "results/variants/{variant}/{champion}/threshold_bands.json",
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
