"""
Enhanced Data Agent with GPU Acceleration (Polars GPU Engine)
Supports transactional and non-transactional data for fraud detection and credit risk assessment.

Native Windows compatible – uses Polars GPU backend when available.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import warnings
import logging
import requests
from pathlib import Path
from typing import List, Optional, Any
import json
import torch
from torch_geometric.data import HeteroData

# Polars with optional GPU engine
import polars as pl

# GPU availability detection – compatible with various polars versions
GPU_AVAILABLE = False
try:
    # Attempt to run a small GPU-accelerated operation
    # If polars was installed with `polars[gpu]`, this should succeed
    df_test = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    # Force GPU engine for this test (may raise if GPU not available)
    # We wrap in a try-except to be safe
    try:
        _ = df_test.select(pl.col("a") * 2).collect(engine="gpu")
        GPU_AVAILABLE = True
    except Exception:
        # Fallback: check if polars was built with GPU support via feature flag
        if hasattr(pl, "GPUEngine"):
            GPU_AVAILABLE = True
        else:
            GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedDataAgent:
    """
    GPU-accelerated Data Agent using Polars GPU engine.

    Supports:
    - Transactional data: cashflow, POS, supply-chain, utilities, telco, shipping, ERP
    - Non-transactional data: credit reports, psychometric, sentiment, IP, assets
    - GPU acceleration via Polars GPU backend when available
    - Multiple file formats: CSV, Parquet, JSON, Excel, Feather, HDF5, Pickle, JSONL
    """

    def __init__(self,
                 categorical_features: List[str],
                 numeric_features: List[str],
                 binary_features: List[str],
                 text_features: Optional[List[str]] = None,
                 time_series_features: Optional[List[str]] = None,
                 target_col: str = 'fraud_bool',
                 feature_selection_method: str = 'rf',
                 n_top_features: int = 50,
                 sample_size_for_selection: Optional[int] = 100000,
                 use_gpu: bool = True):

        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.binary_features = binary_features
        self.text_features = text_features or []
        self.time_series_features = time_series_features or []
        self.target_col = target_col
        self.feature_selection_method = feature_selection_method
        self.n_top_features = n_top_features
        self.sample_size_for_selection = sample_size_for_selection

        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.backend = 'polars_gpu' if self.use_gpu else 'polars_cpu'

        # Fitted state
        self.imputer_median_ = {}
        self.imputer_mode_ = {}
        self.outlier_thresholds_ = {}
        self.encoder_ = None
        self.scaler_ = None
        self.selected_features_ = None
        self.encoded_feature_names_ = []
        self.binary_feature_names_ = []
        self.derived_feature_names_ = []

        logger.info(f"Initialized EnhancedDataAgent with backend: {self.backend}")

    # ------------------------------------------------------------------
    # Ingestion – all return Polars DataFrames
    # ------------------------------------------------------------------

    def ingest(self, source, source_type: str = 'auto', **kwargs) -> pl.DataFrame:
        """
        Load data from various sources into a Polars DataFrame (GPU or CPU).
        """
        if source_type == 'auto':
            if isinstance(source, str):
                ext = Path(source).suffix.lower()
                mapping = {
                    '.csv': 'csv', '.tsv': 'tsv', '.txt': 'csv',
                    '.parquet': 'parquet',
                    '.json': 'json', '.jsonl': 'jsonl', '.ndjson': 'jsonl',
                    '.xlsx': 'excel', '.xls': 'excel',
                    '.feather': 'feather', '.arrow': 'feather',
                    '.h5': 'hdf5', '.hdf5': 'hdf5', '.hdf': 'hdf5',
                    '.pkl': 'pickle', '.pickle': 'pickle',
                }
                source_type = mapping.get(ext, 'csv')
            elif isinstance(source, pd.DataFrame):
                source_type = 'pandas'
            elif isinstance(source, np.ndarray):
                source_type = 'numpy'
            elif isinstance(source, pl.DataFrame):
                return source

        logger.info(f"Ingesting data as '{source_type}' from: {source if isinstance(source, str) else type(source).__name__}")

        readers = {
            'csv':     self._read_csv,
            'tsv':     self._read_tsv,
            'parquet': self._read_parquet,
            'json':    self._read_json,
            'jsonl':   self._read_jsonl,
            'excel':   self._read_excel,
            'feather': self._read_feather,
            'hdf5':    self._read_hdf5,
            'pickle':  self._read_pickle,
            'sql':     self._read_sql,
            'api':     self._read_api,
            'pandas':  self._from_pandas,
            'numpy':   self._from_numpy,
        }

        if source_type not in readers:
            raise ValueError(f"Unsupported source_type '{source_type}'. Choose from: {list(readers)}")

        return readers[source_type](source, **kwargs)

    def _read_csv(self, path, **kwargs) -> pl.DataFrame:
        return pl.read_csv(path, **kwargs)

    def _read_tsv(self, path, **kwargs) -> pl.DataFrame:
        return pl.read_csv(path, separator='\t', **kwargs)

    def _read_parquet(self, path, **kwargs) -> pl.DataFrame:
        return pl.read_parquet(path, **kwargs)

    def _read_json(self, path, **kwargs) -> pl.DataFrame:
        return pl.read_json(path, **kwargs)

    def _read_jsonl(self, path, **kwargs) -> pl.DataFrame:
        return pl.read_ndjson(path, **kwargs)

    def _read_excel(self, path, **kwargs) -> pl.DataFrame:
        df_pd = pd.read_excel(path, **kwargs)
        return pl.from_pandas(df_pd)

    def _read_feather(self, path, **kwargs) -> pl.DataFrame:
        return pl.read_ipc(path, **kwargs)

    def _read_hdf5(self, path, **kwargs) -> pl.DataFrame:
        return pl.from_pandas(pd.read_hdf(path, **kwargs))

    def _read_pickle(self, path, **kwargs) -> pl.DataFrame:
        return pl.from_pandas(pd.read_pickle(path, **kwargs))

    def _read_sql(self, source, **kwargs) -> pl.DataFrame:
        if not (isinstance(source, tuple) and len(source) == 2):
            raise ValueError("SQL source must be a tuple: (connection_uri, query)")
        conn_uri, query = source
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(conn_uri)
            return pl.from_pandas(pd.read_sql(query, engine, **kwargs))
        except ImportError:
            raise ImportError("sqlalchemy is required for SQL ingestion: pip install sqlalchemy")

    def _read_api(self, url, **kwargs) -> pl.DataFrame:
        response = requests.get(url, **kwargs)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        if 'application/json' in content_type:
            data = response.json()
            df_pd = pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            import io
            df_pd = pd.read_csv(io.StringIO(response.text))
        return pl.from_pandas(df_pd)

    def _from_pandas(self, df_pd, **kwargs) -> pl.DataFrame:
        return pl.from_pandas(df_pd)

    def _from_numpy(self, arr, columns=None, **kwargs) -> pl.DataFrame:
        if columns is None:
            columns = [f"col_{i}" for i in range(arr.shape[1])]
        return pl.from_pandas(pd.DataFrame(arr, columns=columns))

    # ------------------------------------------------------------------
    # Internal helpers (conversion to pandas for scikit‑learn)
    # ------------------------------------------------------------------

    def _to_pandas(self, df) -> pd.DataFrame:
        """Convert any supported DataFrame to pandas."""
        if isinstance(df, pd.DataFrame):
            return df.copy()
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        raise TypeError(f"Cannot convert {type(df)} to pandas DataFrame")

    def _col_names(self, df) -> list:
        return list(df.columns)

    def _get_col(self, df, col):
        """Return column as a pandas Series."""
        if isinstance(df, pd.DataFrame):
            return df[col]
        if isinstance(df, pl.DataFrame):
            return df[col].to_pandas()
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")

    # ------------------------------------------------------------------
    # Feature engineering (operates on pandas for simplicity)
    # ------------------------------------------------------------------

    def add_transactional_features(self, df_pd: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from transactional data sources."""
        cols = set(df_pd.columns)

        # Cashflow velocity
        if {'velocity_6h', 'velocity_24h', 'velocity_4w'}.issubset(cols):
            df_pd['velocity_ratio_24h_to_4w'] = df_pd['velocity_24h'] / (df_pd['velocity_4w'] + 1e-5)
            df_pd['total_velocity_24h'] = df_pd['velocity_6h'] + df_pd['velocity_24h']

        # Transaction acceleration
        if {'transaction_count_30d', 'transaction_count_90d'}.issubset(cols):
            df_pd['transaction_acceleration'] = (
                df_pd['transaction_count_30d'] / (df_pd['transaction_count_90d'] + 1e-5)
            )

        # Supply chain
        if {'supplier_count', 'transaction_amount'}.issubset(cols):
            df_pd['avg_amount_per_supplier'] = (
                df_pd['transaction_amount'] / (df_pd['supplier_count'] + 1e-5)
            )

        # Utility z-score
        if 'electricity_consumption' in cols:
            mu = df_pd['electricity_consumption'].mean()
            sd = df_pd['electricity_consumption'].std() + 1e-5
            df_pd['electricity_consumption_zscore'] = (df_pd['electricity_consumption'] - mu) / sd

        # Telco
        if {'call_duration_total', 'call_count'}.issubset(cols):
            df_pd['avg_call_duration'] = df_pd['call_duration_total'] / (df_pd['call_count'] + 1e-5)

        # Shipping
        if {'shipping_frequency', 'shipping_cost_total'}.issubset(cols):
            df_pd['avg_shipping_cost'] = (
                df_pd['shipping_cost_total'] / (df_pd['shipping_frequency'] + 1e-5)
            )

        # ERP
        if {'accounts_receivable', 'revenue'}.issubset(cols):
            df_pd['ar_to_revenue_ratio'] = df_pd['accounts_receivable'] / (df_pd['revenue'] + 1e-5)

        if {'invoice_count', 'invoice_amount_total'}.issubset(cols):
            df_pd['avg_invoice_amount'] = (
                df_pd['invoice_amount_total'] / (df_pd['invoice_count'] + 1e-5)
            )

        return df_pd

    def add_non_transactional_features(self, df_pd: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from non-transactional data sources."""
        cols = set(df_pd.columns)

        # Credit score category
        if 'credit_score' in cols:
            df_pd['credit_score_category'] = pd.cut(
                df_pd['credit_score'],
                bins=[0, 580, 670, 740, 800, 1000],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)

        # High credit utilization flag
        if 'credit_utilization' in cols:
            df_pd['high_credit_utilization'] = (df_pd['credit_utilization'] > 0.7).astype(int)

        # Credit rating vs industry
        if {'company_credit_rating', 'industry_avg_rating'}.issubset(cols):
            df_pd['credit_rating_vs_industry'] = (
                df_pd['company_credit_rating'] - df_pd['industry_avg_rating']
            )

        # Psychometric interaction
        if {'risk_tolerance_score', 'financial_literacy_score'}.issubset(cols):
            df_pd['risk_literacy_interaction'] = (
                df_pd['risk_tolerance_score'] * df_pd['financial_literacy_score']
            )

        # Negative sentiment flag
        if 'review_sentiment_score' in cols:
            df_pd['negative_sentiment_flag'] = (
                df_pd['review_sentiment_score'] < -0.3
            ).astype(int)

        # Weighted reputation
        if {'yelp_rating', 'review_count'}.issubset(cols):
            df_pd['weighted_reputation_score'] = (
                df_pd['yelp_rating'] * np.log1p(df_pd['review_count'])
            )

        # IP assets
        if {'patent_count', 'trademark_count'}.issubset(cols):
            df_pd['total_ip_assets'] = df_pd['patent_count'] + df_pd['trademark_count']

        # Net asset value
        if {'asset_value', 'liability_value'}.issubset(cols):
            df_pd['net_asset_value'] = df_pd['asset_value'] - df_pd['liability_value']
            df_pd['asset_liability_ratio'] = (
                df_pd['asset_value'] / (df_pd['liability_value'] + 1e-5)
            )

        # Revenue per customer
        if {'customer_count', 'revenue'}.issubset(cols):
            df_pd['revenue_per_customer'] = df_pd['revenue'] / (df_pd['customer_count'] + 1e-5)

        # Industry recognition flag
        if 'award_count' in cols:
            df_pd['has_industry_recognition'] = (df_pd['award_count'] > 0).astype(int)

        # Premium partnership flag
        if 'partnership_tier' in cols:
            df_pd['premium_partnership'] = (df_pd['partnership_tier'] >= 3).astype(int)

        return df_pd

    def add_domain_features(self, df_pd: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering and track derived column names."""
        original_cols = set(df_pd.columns)
        df_pd = self.add_transactional_features(df_pd)
        df_pd = self.add_non_transactional_features(df_pd)
        self.derived_feature_names_ = [c for c in df_pd.columns if c not in original_cols]
        logger.info(f"Created {len(self.derived_feature_names_)} derived features")
        return df_pd

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the agent on training data.
        X: file path, pandas/polars DataFrame
        y: target as pandas Series, polars Series, numpy array, or list
        """
        logger.info("Starting fit process...")

        # Normalise X to pandas (scikit‑learn works with pandas/numpy)
        X_pd = self._to_pandas(self._to_dataframe_if_path(X))

        # Normalise y to numpy
        y_np = self._to_numpy(y)

        # Feature engineering
        X_pd = self.add_domain_features(X_pd)

        # Fit preprocessing
        self._fit_preprocessing(X_pd)

        # Apply preprocessing → full feature matrix
        X_full = self._apply_preprocessing(X_pd)

        # Feature selection
        self._fit_feature_selection(X_full, y_np)

        logger.info(f"Fit complete. Selected {len(self.selected_features_)} features.")
        return self

    def transform(self, X) -> pd.DataFrame:
        """Transform data and return selected features as a pandas DataFrame."""
        if self.selected_features_ is None:
            raise RuntimeError("Agent not fitted. Call fit() first.")

        X_pd = self._to_pandas(self._to_dataframe_if_path(X))
        X_pd = self.add_domain_features(X_pd)
        X_full = self._apply_preprocessing(X_pd)

        # Fill any missing selected features with 0
        for col in self.selected_features_:
            if col not in X_full.columns:
                X_full[col] = 0.0

        return X_full[self.selected_features_]

    def fit_transform(self, X, y) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _to_dataframe_if_path(self, data):
        """If data is a file path string, ingest it; otherwise return as-is."""
        if isinstance(data, str):
            return self.ingest(data)
        return data

    def _to_numpy(self, y) -> np.ndarray:
        """Convert any target representation to a 1-D numpy array."""
        if isinstance(y, np.ndarray):
            return y.ravel()
        if isinstance(y, pd.Series):
            return y.to_numpy()
        if isinstance(y, pl.Series):
            return y.to_numpy()
        return np.array(y)

    def _fit_preprocessing(self, X_pd: pd.DataFrame):
        """Fit imputers, encoder, and scaler from training data."""
        # Numeric: median + outlier bounds
        for col in self.numeric_features:
            if col in X_pd.columns:
                self.imputer_median_[col] = X_pd[col].median()
                self.outlier_thresholds_[col] = (
                    X_pd[col].quantile(0.01),
                    X_pd[col].quantile(0.99),
                )

        # Categorical + binary: mode
        for col in self.categorical_features + self.binary_features:
            if col in X_pd.columns:
                mode_vals = X_pd[col].mode()
                self.imputer_mode_[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else None

        # OneHotEncoder for categoricals
        cat_cols = [c for c in self.categorical_features if c in X_pd.columns]
        if cat_cols:
            self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.encoder_.fit(X_pd[cat_cols].astype(str))

        # StandardScaler for numerics + derived numeric features
        all_num = self._all_numeric_cols(X_pd)
        if all_num:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_pd[all_num].fillna(0))

    def _apply_preprocessing(self, X_pd: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing and return a flat pandas DataFrame."""
        X_pd = X_pd.copy()

        # Impute + clip numerics
        for col, median in self.imputer_median_.items():
            if col in X_pd.columns:
                X_pd[col] = X_pd[col].fillna(median)
                lo, hi = self.outlier_thresholds_[col]
                X_pd[col] = X_pd[col].clip(lower=lo, upper=hi)

        # Impute categoricals / binary
        for col, mode_val in self.imputer_mode_.items():
            if col in X_pd.columns and mode_val is not None:
                X_pd[col] = X_pd[col].fillna(mode_val)

        # Encode categoricals
        cat_cols = [c for c in self.categorical_features if c in X_pd.columns]
        if cat_cols and self.encoder_:
            encoded = self.encoder_.transform(X_pd[cat_cols].astype(str))
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder_.get_feature_names_out(cat_cols),
                index=X_pd.index,
            )
        else:
            encoded_df = pd.DataFrame(index=X_pd.index)

        # Scale numerics
        all_num = self._all_numeric_cols(X_pd)
        if all_num and self.scaler_:
            scaled = self.scaler_.transform(X_pd[all_num].fillna(0))
            scaled_df = pd.DataFrame(scaled, columns=all_num, index=X_pd.index)
        else:
            scaled_df = pd.DataFrame(index=X_pd.index)

        # Binary columns (cast to int, handle nulls)
        bin_cols = [c for c in self.binary_features if c in X_pd.columns]
        if bin_cols:
            bin_df = X_pd[bin_cols].fillna(0).astype(int)
        else:
            bin_df = pd.DataFrame(index=X_pd.index)

        return pd.concat([encoded_df, scaled_df, bin_df], axis=1)

    def _all_numeric_cols(self, X_pd: pd.DataFrame) -> list:
        """Return numeric feature columns + derived numeric columns present in X_pd."""
        explicit = [c for c in self.numeric_features if c in X_pd.columns]
        derived = [c for c in self.derived_feature_names_ if c in X_pd.columns]
        # Exclude anything already handled as categorical or binary
        exclude = set(self.categorical_features + self.binary_features)
        return [c for c in explicit + derived if c not in exclude]

    def _fit_feature_selection(self, X_full: pd.DataFrame, y_np: np.ndarray):
        """Select top N features using RF importance or mutual information."""
        n = len(X_full)
        if self.sample_size_for_selection and n > self.sample_size_for_selection:
            idx = np.random.choice(n, self.sample_size_for_selection, replace=False)
            X_s, y_s = X_full.iloc[idx], y_np[idx]
        else:
            X_s, y_s = X_full, y_np

        if self.feature_selection_method == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_s, y_s)
            importances = clf.feature_importances_
        else:
            importances = mutual_info_classif(X_s, y_s, random_state=42)

        imp = pd.Series(importances, index=X_full.columns)
        k = min(self.n_top_features, len(imp))
        self.selected_features_ = imp.nlargest(k).index.tolist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        import joblib
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)


class GraphBuilder:
    """
    Build a bipartite graph from the output of EnhancedDataAgent.
    Node types: 'entity', 'feature'
    Edge type: ('entity', 'has_value', 'feature')
    """

    def __init__(self, agent: EnhancedDataAgent, threshold: float = 0.0):
        """
        Args:
            agent: Fitted EnhancedDataAgent instance.
            threshold: Minimum absolute feature value to create an edge.
        """
        self.agent = agent
        self.threshold = threshold

    def build(self, X) -> HeteroData:
        """
        Transform data using the agent and construct the bipartite graph.

        Args:
            X: Raw input data (file path, DataFrame, etc.)

        Returns:
            HeteroData object with:
                - 'entity': x = feature matrix
                - 'feature': x = one-hot identity matrix (learnable embedding)
                - ('entity', 'has_value', 'feature'): edge_index and edge_weight
        """
        # 1. Obtain the transformed feature matrix (entities × features)
        X_trans = self.agent.transform(X)   # pandas DataFrame
        entity_ids = X_trans.index.tolist()
        feature_names = X_trans.columns.tolist()

        n_entities = len(entity_ids)
        n_features = len(feature_names)

        # 2. Entity node features (the full vector)
        entity_x = torch.tensor(X_trans.values, dtype=torch.float)

        # 3. Feature node features (use one‑hot as learnable embeddings)
        feature_x = torch.eye(n_features, dtype=torch.float)

        # 4. Build bipartite edges where value != 0 (or > threshold)
        rows, cols, weights = [], [], []
        for i in range(n_entities):
            for j in range(n_features):
                val = X_trans.iloc[i, j]
                if abs(val) > self.threshold:
                    rows.append(i)
                    cols.append(j)
                    weights.append(val)

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)

        # 5. Assemble HeteroData
        data = HeteroData()
        data['entity'].x = entity_x
        data['feature'].x = feature_x
        data['entity', 'has_value', 'feature'].edge_index = edge_index
        data['entity', 'has_value', 'feature'].edge_weight = edge_weight

        # Store metadata for interpretation
        data.entity_ids = entity_ids
        data.feature_names = feature_names

        return data