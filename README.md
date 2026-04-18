# BAF Agentic Fraud Detection

End-to-end account-level fraud workflow:
- BAF preprocessing with strict time split,
- vanilla and retriever-enriched XGBoost training (SMOTE optional),
- statistical model comparison,
- real-time scoring with SHAP,
- DeepSeek-backed fraud report and recommendation (`approve | escalate | reject`).

## 1) Prerequisites (new machine)

- OS with NVIDIA GPU support (tested target: GeForce RTX 4090).
- NVIDIA driver `560+` with CUDA runtime compatibility (`12.5` target).
- Python `3.11` recommended.
- Git and Docker installed.

## 2) Clone and create environment

```bash
git clone https://github.com/JuliaHandiprima/Fraud-Detection.git
cd Fraud-Detection
python -m venv .venv
```

Activate environment:

- PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

- bash/zsh:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3) Required data and artifacts

### BAF datasets

For the gap-matrix benchmark, prepare 6 BAF variant CSV files (each must include `month` and `fraud_bool`) for example:

```text
data/variant_1.csv
data/variant_2.csv
data/variant_3.csv
data/variant_4.csv
data/variant_5.csv
data/variant_6.csv
```

### Retriever backend (required for enriched model + inference evidence)

The retriever module expects pgvector/Postgres and preprocessing artifacts (`scaler.pkl`, `encoder.pkl`, `medians.pkl`, `feature_cols.pkl`) compatible with the BAF schema.

If you are setting up from scratch, generate/load those artifacts using the retriever notebooks in:
- `notebooks/experiments/retriever_preprocess_load_data.ipynb`
- `notebooks/experiments/retriever_run_dataset.ipynb`

and run the retriever DB/API stack per the retriever workflow before enriched training.

## 4) Time split protocol (fixed)

- Train: months `0-5`
- Validation: month `6`
- Test: month `7`

This split is enforced before fitting preprocessing to avoid leakage.

## 5) Run full experiment

GPU-first (auto-detect CUDA, fallback to CPU):

```bash
python scripts/run_experiment.py --data data/base.csv --results-dir results
```

## 5b) Run full 6-variant benchmark (recommended)

```bash
python scripts/run_variant_benchmark.py \
  --variants-json "[{\"name\":\"variant_1\",\"path\":\"data/variant_1.csv\"},{\"name\":\"variant_2\",\"path\":\"data/variant_2.csv\"},{\"name\":\"variant_3\",\"path\":\"data/variant_3.csv\"},{\"name\":\"variant_4\",\"path\":\"data/variant_4.csv\"},{\"name\":\"variant_5\",\"path\":\"data/variant_5.csv\"},{\"name\":\"variant_6\",\"path\":\"data/variant_6.csv\"}]" \
  --results-dir results/variants
```

Optional fairness columns:

```bash
python scripts/run_variant_benchmark.py ... --fairness-group-cols source device_os
```

Force CPU:

```bash
python scripts/run_experiment.py --data data/base.csv --results-dir results --cpu-only
```

Produced outputs:
- `results/vanilla/metrics.json`
- `results/enriched/metrics.json`
- `results/model_selection.json`
- `results/summary.json`

For the 6-variant run:
- `results/variants/{variant}/vanilla/*`
- `results/variants/{variant}/enriched/*`
- `results/variants/{variant}/model_selection.json`
- `results/variants/aggregate_summary.json`
- `results/variants/champion_model.json`

## 6) Prepare real-time inference + report generation

Set DeepSeek credentials:

```bash
export DEEPSEEK_API_KEY=your_key_here
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
```

PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your_key_here"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

## 7) Start account-level inference API

```bash
python -m src.inference.app
```

To serve benchmark-selected champion artifacts, set:

```bash
export CHAMPION_MANIFEST_PATH=results/variants/champion_model.json
export CHAMPION_VARIANT_NAME=variant_1
python -m src.inference.app
```

Endpoint: `POST /agent/account_fraud_report`

Example request:

```json
{
  "id": "request-001",
  "input": {
    "query": {
      "month": 6,
      "income": 0.8,
      "source": "INTERNET",
      "device_os": "windows"
    }
  }
}
```

Response includes:
- `fraud_score`
- `recommended_action`
- SHAP top drivers
- retriever evidence summary
- LLM narrative report

## 8) Evaluation metrics included

- Performance: PR-AUC, ROC-AUC
- Calibration: Brier raw and calibrated (Platt scaling)
- Threshold metrics: precision, recall, F1 (raw + calibrated)
- Alert yield: `% flagged` at selected threshold
- Fairness (when subgroup columns provided):
  - subgroup parity difference
  - equal opportunity difference

## 9) Repository structure

- `src/preprocessing/baf_preprocessor.py`: deterministic BAF preprocessing.
- `src/modeling/train_vanilla.py`: vanilla XGBoost pipeline.
- `src/modeling/train_enriched.py`: retriever-enriched XGBoost pipeline.
- `src/modeling/compare_models.py`: bootstrap delta PR-AUC and winner selection.
- `scripts/run_variant_benchmark.py`: six-variant orchestration and aggregate summary.
- `src/inference/account_inference.py`: scoring + SHAP + LLM reporting logic.
- `src/inference/app.py`: Flask API service.
- `src/retriever/`: migrated retriever components.
- `configs/baf_experiment.yaml`: experiment defaults.

## 10) Notes

- Scope is account-level only for this stage.
- Training is GPU-aware and uses `xgboost` with `tree_method=hist`, `device=cuda` when available.
- Champion preprocessing uses median imputation + Yeo-Johnson + standardization.
- Imbalance handling is SMOTE on train split only.
- Generated artifacts (`results/`, `*.pkl`, caches) are intentionally gitignored for clean reproducibility.
