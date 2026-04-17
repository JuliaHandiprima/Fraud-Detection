from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.modeling.metrics import CalibrationResult, best_f1_threshold, calibrate_platt, evaluate_binary_classifier
from src.modeling.shap_artifacts import save_shap_artifacts
from src.modeling.threshold_bands import compute_threshold_bands, save_threshold_bands
from src.modeling.xgb_runtime import resolve_xgb_compute
from src.preprocessing.baf_preprocessor import BAFPreprocessor, TimeSplit
from src.retriever.enrichment import build_retriever_features_for_records


def train_vanilla(
    data_path: str | Path,
    output_dir: str | Path = "results/vanilla",
    target_col: str = "fraud_bool",
    month_col: str = "month",
    prefer_gpu: bool = True,
    use_yeo_johnson: bool = True,
    use_smote: bool = True,
    smote_sampling_strategy: float = 0.5,
    smote_random_state: int = 42,
    fairness_group_cols: list[str] | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    split = TimeSplit()
    preprocessor = BAFPreprocessor(
        target_col=target_col,
        month_col=month_col,
        use_yeo_johnson=use_yeo_johnson,
    )
    train_df, valid_df, test_df = preprocessor.split_by_month(df, split)
    preprocessor.fit(train_df)
    preprocessor.save(output)

    X_train, y_train = preprocessor.transform_with_target(train_df)
    X_valid, y_valid = preprocessor.transform_with_target(valid_df)
    X_test, y_test = preprocessor.transform_with_target(test_df)

    retr_valid = build_retriever_features_for_records(valid_df.drop(columns=[target_col], errors="ignore"))
    retr_test = build_retriever_features_for_records(test_df.drop(columns=[target_col], errors="ignore"))
    X_train_fit, y_train_fit = X_train, y_train
    if use_smote:
        smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=smote_random_state)
        X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)

    compute = resolve_xgb_compute(prefer_gpu=prefer_gpu)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        tree_method=compute["tree_method"],
        device=compute["device"],
    )
    model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    calibrator: CalibrationResult = calibrate_platt(y_valid.to_numpy(), valid_scores)
    valid_scores_cal = calibrator.model.predict_proba(valid_scores.reshape(-1, 1))[:, 1]
    test_scores_cal = calibrator.model.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    threshold = best_f1_threshold(y_valid.to_numpy(), valid_scores)
    fairness_groups = None
    if fairness_group_cols:
        present_cols = [c for c in fairness_group_cols if c in test_df.columns]
        fairness_groups = test_df[present_cols] if present_cols else None
    metrics = evaluate_binary_classifier(
        y_true=y_test.to_numpy(),
        y_score=test_scores,
        y_score_calibrated=test_scores_cal,
        threshold=threshold,
        groups=fairness_groups,
    )

    joblib.dump(model, output / "model.pkl")
    joblib.dump(calibrator.model, output / "platt_calibrator.pkl")

    def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    calibrator_feature_cols = ["logit_p_model_calibrated"] + retr_valid.columns.tolist()
    X_cal_valid = pd.concat(
        [
            pd.Series(_logit(valid_scores_cal), index=valid_df.index, name="logit_p_model_calibrated"),
            retr_valid,
        ],
        axis=1,
    )
    risk_calibrator = LogisticRegression(max_iter=2000, solver="lbfgs")
    risk_calibrator.fit(X_cal_valid.to_numpy(), y_valid.to_numpy())
    joblib.dump(risk_calibrator, output / "risk_calibrator.pkl")
    (output / "risk_calibrator_metadata.json").write_text(
        json.dumps(
            {
                "type": "logistic_regression",
                "feature_cols": calibrator_feature_cols,
                "p_base_col": "p_model_calibrated",
                "logit_base_col": "logit_p_model_calibrated",
            },
            indent=2,
        )
    )

    X_cal_test = pd.concat(
        [
            pd.Series(_logit(test_scores_cal), index=test_df.index, name="logit_p_model_calibrated"),
            retr_test,
        ],
        axis=1,
    )
    p_final_valid = risk_calibrator.predict_proba(X_cal_valid.to_numpy())[:, 1]
    p_final_test = risk_calibrator.predict_proba(X_cal_test.to_numpy())[:, 1]

    save_shap_artifacts(
        model=model,
        X=X_valid,
        record_index=valid_df.index.to_numpy(),
        output_dir=output,
        prefix="shap_month6",
        top_k=10,
    )
    save_shap_artifacts(
        model=model,
        X=X_test,
        record_index=test_df.index.to_numpy(),
        output_dir=output,
        prefix="shap_month7",
        top_k=10,
    )

    bands = compute_threshold_bands(
        y_valid=y_valid.to_numpy(),
        p_valid=p_final_valid,
        y_test=y_test.to_numpy(),
        p_test=p_final_test,
        target_precision_high=0.80,
        target_precision_med=0.60,
    )
    save_threshold_bands(bands, output_dir=output, filename="threshold_bands.json")

    def _pick_id_column(frame: pd.DataFrame) -> str | None:
        for candidate in ("account_id", "application_id", "id"):
            if candidate in frame.columns:
                return candidate
        return None

    id_col = _pick_id_column(df)
    valid_pred = pd.DataFrame(
        {
            "record_index": valid_df.index.to_numpy(),
            "month": valid_df[month_col].to_numpy() if month_col in valid_df.columns else 6,
            "label": y_valid.to_numpy(),
            "p_model": valid_scores,
            "p_model_calibrated": valid_scores_cal,
            "p_final": p_final_valid,
        },
        index=valid_df.index,
    )
    test_pred = pd.DataFrame(
        {
            "record_index": test_df.index.to_numpy(),
            "month": test_df[month_col].to_numpy() if month_col in test_df.columns else 7,
            "label": y_test.to_numpy(),
            "p_model": test_scores,
            "p_model_calibrated": test_scores_cal,
            "p_final": p_final_test,
        },
        index=test_df.index,
    )
    if id_col is not None:
        valid_pred[id_col] = valid_df[id_col].to_numpy()
        test_pred[id_col] = test_df[id_col].to_numpy()
    valid_pred = pd.concat([valid_pred, retr_valid], axis=1)
    test_pred = pd.concat([test_pred, retr_test], axis=1)
    valid_pred.to_csv(output / "predictions_month6.csv", index=False)
    test_pred.to_csv(output / "predictions_month7.csv", index=False)

    pd.DataFrame({"score": test_scores, "label": y_test.to_numpy()}).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame({"score_calibrated": test_scores_cal, "label": y_test.to_numpy()}).to_csv(output / "test_predictions_calibrated.csv", index=False)
    calibration_report = {
        "method": calibrator.method,
        "brier_raw_valid": float(calibrator.brier_raw_valid),
        "brier_cal_valid": float(calibrator.brier_cal_valid),
        "improvement": float(calibrator.improvement),
    }
    report = {
        "split": {"train_months": [0, 1, 2, 3, 4, 5], "valid_months": [6], "test_months": [7]},
        "counts": {"train": int(len(train_df)), "valid": int(len(valid_df)), "test": int(len(test_df))},
        "metrics": metrics,
        "compute": compute,
        "preprocessing": {"use_yeo_johnson": use_yeo_johnson},
        "imbalance": {
            "method": "smote" if use_smote else "none",
            "sampling_strategy": smote_sampling_strategy if use_smote else None,
            "train_rows_before": int(len(X_train)),
            "train_rows_after": int(len(X_train_fit)),
        },
        "calibration": calibration_report,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate vanilla BAF XGBoost")
    parser.add_argument("--data", required=True, help="Path to BAF csv")
    parser.add_argument("--output", default="results/vanilla")
    parser.add_argument("--cpu-only", action="store_true", help="Disable CUDA and force CPU training")
    parser.add_argument("--disable-yeojohnson", action="store_true", help="Disable Yeo-Johnson transform")
    parser.add_argument("--disable-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--smote-sampling-strategy", type=float, default=0.5)
    parser.add_argument("--smote-random-state", type=int, default=42)
    parser.add_argument("--fairness-group-cols", nargs="*", default=None)
    args = parser.parse_args()
    result = train_vanilla(
        args.data,
        args.output,
        prefer_gpu=not args.cpu_only,
        use_yeo_johnson=not args.disable_yeojohnson,
        use_smote=not args.disable_smote,
        smote_sampling_strategy=args.smote_sampling_strategy,
        smote_random_state=args.smote_random_state,
        fairness_group_cols=args.fairness_group_cols,
    )
    print(json.dumps(result, indent=2))
