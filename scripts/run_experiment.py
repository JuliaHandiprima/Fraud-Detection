from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modeling.compare_models import compare_and_select
from src.modeling.train_enriched import train_enriched
from src.modeling.train_vanilla import train_vanilla


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BAF vanilla vs enriched experiment")
    parser.add_argument("--data", required=True, help="Path to BAF csv")
    parser.add_argument("--results-dir", default="results", help="Output results directory")
    parser.add_argument("--cpu-only", action="store_true", help="Disable CUDA and force CPU training")
    parser.add_argument("--disable-yeojohnson", action="store_true", help="Disable Yeo-Johnson transform")
    parser.add_argument("--disable-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--smote-sampling-strategy", type=float, default=0.5)
    parser.add_argument("--smote-random-state", type=int, default=42)
    parser.add_argument("--fairness-group-cols", nargs="*", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    vanilla_dir = results_dir / "vanilla"
    enriched_dir = results_dir / "enriched"

    vanilla = train_vanilla(
        args.data,
        vanilla_dir,
        prefer_gpu=not args.cpu_only,
        use_yeo_johnson=not args.disable_yeojohnson,
        use_smote=not args.disable_smote,
        smote_sampling_strategy=args.smote_sampling_strategy,
        smote_random_state=args.smote_random_state,
        fairness_group_cols=args.fairness_group_cols,
    )
    enriched = train_enriched(
        args.data,
        enriched_dir,
        prefer_gpu=not args.cpu_only,
        use_yeo_johnson=not args.disable_yeojohnson,
        use_smote=not args.disable_smote,
        smote_sampling_strategy=args.smote_sampling_strategy,
        smote_random_state=args.smote_random_state,
        fairness_group_cols=args.fairness_group_cols,
    )
    selected = compare_and_select(vanilla_dir=vanilla_dir, enriched_dir=enriched_dir, output_path=results_dir / "model_selection.json")

    summary = {
        "vanilla": vanilla["metrics"],
        "enriched": enriched["metrics"],
        "selection": selected,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
