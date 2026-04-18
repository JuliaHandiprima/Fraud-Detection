from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler


@dataclass(frozen=True)
class TimeSplit:
    train_months: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    valid_months: tuple[int, ...] = (6,)
    test_months: tuple[int, ...] = (7,)


class BAFPreprocessor:
    """
    Leakage-safe BAF preprocessor.

    Fit only on train split; transform val/test/inference with frozen schema.
    """

    def __init__(
        self,
        target_col: str = "fraud_bool",
        month_col: str = "month",
        treat_minus_one_as_missing: bool = True,
        use_yeo_johnson: bool = True,
    ) -> None:
        self.target_col = target_col
        self.month_col = month_col
        self.treat_minus_one_as_missing = treat_minus_one_as_missing
        self.use_yeo_johnson = use_yeo_johnson
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.feature_cols_: list[str] = []
        self.pipeline_: Pipeline | None = None

    @staticmethod
    def _flatten_records(data: list[dict[str, Any]] | dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            return pd.json_normalize(data)
        return pd.json_normalize(data)

    def split_by_month(self, df: pd.DataFrame, split: TimeSplit) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = df[df[self.month_col].isin(split.train_months)].copy()
        valid_df = df[df[self.month_col].isin(split.valid_months)].copy()
        test_df = df[df[self.month_col].isin(split.test_months)].copy()
        return train_df, valid_df, test_df

    def _infer_feature_types(self, X: pd.DataFrame) -> None:
        numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]
        self.numeric_cols = [col for col in numeric_cols if col != self.target_col]
        self.categorical_cols = [col for col in categorical_cols if col != self.target_col]

    def _replace_missing_markers(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.treat_minus_one_as_missing:
            return X
        X = X.copy()
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = X[col].replace(-1, np.nan)
        return X

    def fit(self, train_df: pd.DataFrame) -> "BAFPreprocessor":
        if self.target_col not in train_df.columns:
            raise ValueError(f"Missing target column '{self.target_col}' in training data.")

        X_train = train_df.drop(columns=[self.target_col])
        self._infer_feature_types(X_train)
        X_train = self._replace_missing_markers(X_train)

        numeric_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        if self.use_yeo_johnson:
            numeric_pipe.steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
        numeric_pipe.steps.append(("scaler", StandardScaler()))
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_cols),
                ("cat", categorical_pipe, self.categorical_cols),
            ],
            remainder="drop",
        )
        self.pipeline_ = Pipeline(steps=[("preprocessor", preprocessor)])
        self.pipeline_.fit(X_train)
        self.feature_cols_ = self.get_feature_names()
        return self

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        X = df.drop(columns=[self.target_col], errors="ignore")
        X = self._replace_missing_markers(X)
        transformed = self.pipeline_.transform(X)
        return pd.DataFrame(transformed, columns=self.feature_cols_, index=df.index)

    def transform_with_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = self.transform_features(df)
        if self.target_col not in df.columns:
            raise ValueError(f"Missing target column '{self.target_col}' in transform input.")
        y = df[self.target_col].astype(int)
        return X, y

    def transform_records(self, records: list[dict[str, Any]] | dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        raw = self._flatten_records(records)
        for col in self.numeric_cols + self.categorical_cols:
            if col not in raw.columns:
                raw[col] = np.nan
        ordered = raw[self.numeric_cols + self.categorical_cols]
        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        transformed = self.pipeline_.transform(self._replace_missing_markers(ordered))
        return pd.DataFrame(transformed, columns=self.feature_cols_, index=raw.index)

    def get_feature_names(self) -> list[str]:
        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        preprocessor: ColumnTransformer = self.pipeline_.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out().tolist()
        return [name.replace("num__", "").replace("cat__", "") for name in names]

    def save(self, out_dir: Path | str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path / "baf_preprocessor.pkl")

    @classmethod
    def load(cls, path: Path | str) -> "BAFPreprocessor":
        return joblib.load(path)
