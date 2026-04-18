import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
import warnings
import logging
import requests
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAgent:
    """
    Scalable Data Agent for fraud detection pipelines.
    Supports large-scale data via Polars, feature engineering, and time-series aware processing.
    """
    
    def __init__(self,
                 categorical_features,
                 numeric_features,
                 binary_features,
                 target_col='fraud_bool',
                 feature_selection_method='rf',
                 n_top_features=20,
                 sample_size_for_selection=100000,  # sample rows for feature selection
                 use_polars=True):
        """
        Parameters:
        - categorical_features: list of column names (low cardinality)
        - numeric_features: list of numeric columns
        - binary_features: list of binary columns (0/1)
        - target_col: name of target column
        - feature_selection_method: 'rf' or 'mutual_info'
        - n_top_features: number of top features to retain
        - sample_size_for_selection: rows to sample for feature selection (None = use all)
        - use_polars: if True, use Polars for preprocessing (recommended)
        """
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.binary_features = binary_features
        self.target_col = target_col
        self.feature_selection_method = feature_selection_method
        self.n_top_features = n_top_features
        self.sample_size_for_selection = sample_size_for_selection
        self.use_polars = use_polars
        
        # Fitted parameters
        self.imputer_median_ = {}      # for numeric columns
        self.imputer_mode_ = {}        # for categorical/binary
        self.outlier_thresholds_ = {}  # (lower, upper) per numeric col
        self.encoder_ = None
        self.scaler_ = None
        self.selected_features_ = None
        
        # Track feature names after encoding
        self.encoded_feature_names_ = None
        self.binary_feature_names_ = None
        self.derived_feature_names_ = None  # for domain features
        self._readers = {
            'csv': self._read_csv,
            'json': self._read_json,
            'parquet': self._read_parquet,
            'feather': self._read_feather,
            'excel': self._read_excel,
            'sql': self._read_sql,
            'api': self._read_api,
            'pandas': self._read_pandas,
            'numpy': self._read_numpy,
            'stream': self._read_stream,
        }

    def _read_csv(self, source, **kwargs):
        return pl.read_csv(source, **kwargs)

    def _read_json(self, source, **kwargs):
        return pl.read_json(source, **kwargs)

    def _read_parquet(self, source, **kwargs):
        return pl.read_parquet(source, **kwargs)

    def _read_feather(self, source, **kwargs):
        return pl.read_feather(source, **kwargs)

    def _read_excel(self, source, **kwargs):
        # Polars doesn't support Excel, use pandas
        df_pd = pd.read_excel(source, **kwargs)
        return pl.from_pandas(df_pd)

    def _read_sql(self, source, **kwargs):
        # source can be a tuple (connection_uri, query) or (engine, query)
        if isinstance(source, tuple) and len(source) == 2:
            conn_uri, query = source
            # Use polars read_database if available, else pandas
            try:
                return pl.read_database(query, conn_uri, **kwargs)
            except Exception:
                # Fallback to pandas + convert
                import sqlalchemy
                engine = sqlalchemy.create_engine(conn_uri)
                df_pd = pd.read_sql(query, engine, **kwargs)
                return pl.from_pandas(df_pd)
        else:
            raise ValueError("SQL source must be (connection_uri, query)")

    def _read_api(self, source, **kwargs):
        # source is a URL, kwargs can include 'params', 'headers', etc.
        response = requests.get(source, **kwargs)
        response.raise_for_status()
        # Assume JSON response, but could be CSV; we can handle different content types
        content_type = response.headers.get('content-type', '')
        if 'application/json' in content_type:
            data = response.json()
            # Try to convert to Polars DataFrame
            return pl.DataFrame(data)
        else:
            # If CSV text, use StringIO
            import io
            return pl.read_csv(io.StringIO(response.text))
        # Could add more formats as needed

    def _read_pandas(self, source, **kwargs):
        return pl.from_pandas(source, **kwargs)

    def _read_numpy(self, source, **kwargs):
        # source is a numpy array, column names from kwargs or default
        col_names = kwargs.pop('columns', None)
        if col_names is None:
            col_names = [f"col_{i}" for i in range(source.shape[1])]
        return pl.DataFrame(source, schema=col_names, **kwargs)

    def _read_stream(self, source, **kwargs):
        # source is a generator that yields chunks (list of dicts, pandas, etc.)
        # We'll collect all chunks into a list of Polars DataFrames and concatenate
        chunks = []
        for chunk in source:
            if isinstance(chunk, pl.DataFrame):
                chunks.append(chunk)
            elif isinstance(chunk, pd.DataFrame):
                chunks.append(pl.from_pandas(chunk))
            elif isinstance(chunk, (list, dict)):
                # Assume list of dicts or dict of lists
                chunks.append(pl.DataFrame(chunk))
            else:
                raise ValueError(f"Unsupported chunk type: {type(chunk)}")
        return pl.concat(chunks) if chunks else pl.DataFrame()

    def ingest(self, source, source_type='auto', **kwargs):
        """
        Ingest data from various sources.

        Parameters:
        - source: data source (file path, URL, DataFrame, generator, etc.)
        - source_type: one of 'auto', 'csv', 'json', 'parquet', 'feather', 'excel', 'sql', 'api', 'pandas', 'numpy', 'stream'
        - **kwargs: additional arguments passed to the reader

        Returns:
        - pl.DataFrame: ingested data
        """
        if source_type == 'auto' and isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()
            mapping = {'.csv': 'csv', '.json': 'json', '.parquet': 'parquet', 
                       '.feather': 'feather', '.xlsx': 'excel', '.xls': 'excel'}
            source_type = mapping.get(ext, 'csv')  # default to CSV if unknown

        if source_type not in self._readers:
            raise ValueError(f"Unsupported source_type: {source_type}. Available: {list(self._readers.keys())}")

        reader = self._readers[source_type]
        try:
            df = reader(source, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to ingest data from {source_type} source: {e}")

        # Ensure we return a Polars DataFrame
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else pl.DataFrame(df)

        return df
    
    def _to_polars(self, data):
        """Convert input (DataFrame, path, or Polars) to Polars DataFrame."""
        if isinstance(data, str) and data.endswith('.csv'):
            return pl.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        elif isinstance(data, pl.DataFrame):
            return data
        else:
            raise ValueError("Input must be CSV path, pandas DataFrame, or Polars DataFrame.")
    
    def add_domain_features(self, df):
        """
        Create domain-specific features.
        Input is a Polars DataFrame (or pandas, converted internally).
        Returns Polars DataFrame with added columns.
        """
        # Convert to Polars if not already
        if not isinstance(df, pl.DataFrame):
            df = self._to_polars(df)
        
        # Ratio: velocity_24h / velocity_4w (avoid division by zero)
        df = df.with_columns(
            (pl.col('velocity_24h') / (pl.col('velocity_4w') + 1e-5)).alias('velocity_ratio_24h_to_4w')
        )
        
        # Total velocity in last 24h
        df = df.with_columns(
            (pl.col('velocity_6h') + pl.col('velocity_24h')).alias('total_velocity_24h')
        )
        
        # Interaction: name_email_similarity * velocity_24h (high similarity + high velocity suspicious)
        df = df.with_columns(
            (pl.col('name_email_similarity') * pl.col('velocity_24h')).alias('similarity_velocity_interaction')
        )
        
        # Age group bins (optional, but we keep numeric)
        # Could also create one-hot encoded age groups, but we'll keep numeric for now.
        
        # Device fraud count normalized by distinct emails (risk per email)
        df = df.with_columns(
            (pl.col('device_fraud_count') / (pl.col('device_distinct_emails_8w') + 1)).alias('device_fraud_per_email')
        )
        
        # If income is present, create income_to_credit_limit ratio
        if 'income' in df.columns and 'proposed_credit_limit' in df.columns:
            df = df.with_columns(
                (pl.col('income') / (pl.col('proposed_credit_limit') + 1)).alias('income_to_limit_ratio')
            )
        
        # Flag: if request is foreign and high velocity in 6h
        if 'foreign_request' in df.columns:
            df = df.with_columns(
                ((pl.col('foreign_request') == 1) & (pl.col('velocity_6h') > df['velocity_6h'].quantile(0.9))).alias('foreign_high_velocity')
            ).with_columns(
                pl.col('foreign_high_velocity').cast(pl.Int64)
            )
        
        # Store derived feature names for later reference
        self.derived_feature_names_ = [col for col in df.columns if col not in self.categorical_features + self.numeric_features + self.binary_features + [self.target_col]]
        return df
    
    def _fit_imputers_outliers(self, X_df, y=None):
        """
        Compute median for numeric, mode for categorical/binary, and outlier bounds.
        X_df is a Polars DataFrame.
        """
        # Numeric columns
        for col in self.numeric_features:
            if col in X_df.columns:
                # Compute median
                median = X_df[col].median()
                self.imputer_median_[col] = median
                # Compute outlier thresholds (1st and 99th percentiles)
                q1 = X_df[col].quantile(0.01)
                q99 = X_df[col].quantile(0.99)
                self.outlier_thresholds_[col] = (q1, q99)
        
        # Categorical and binary: mode
        for col in self.categorical_features + self.binary_features:
            if col in X_df.columns:
                # Mode: most frequent value
                mode = X_df[col].mode().to_numpy()[0] if len(X_df[col].mode()) > 0 else None
                self.imputer_mode_[col] = mode
    
    def _apply_imputers_outliers(self, X_df):
        """
        Apply imputation and outlier capping using fitted parameters.
        Returns Polars DataFrame.
        """
        # Impute numeric with median
        for col, median in self.imputer_median_.items():
            X_df = X_df.with_columns(pl.col(col).fill_null(median))
        
        # Impute categorical/binary with mode
        for col, mode in self.imputer_mode_.items():
            X_df = X_df.with_columns(pl.col(col).fill_null(mode))
        
        # Cap outliers
        for col, (low, high) in self.outlier_thresholds_.items():
            X_df = X_df.with_columns(
                pl.col(col).clip(lower_bound=low, upper_bound=high)
            )
        
        return X_df
    
    def _prepare_for_sklearn(self, X_df):
        """
        Convert to pandas, separate features by type, and ensure correct dtypes.
        """
        # Convert to pandas
        X_pd = X_df.to_pandas()
        
        # Ensure categorical columns are categorical for OneHotEncoder
        for col in self.categorical_features:
            if col in X_pd.columns:
                X_pd[col] = X_pd[col].astype('category')
        
        # Ensure binary columns are int (handle nulls and non-integer values)
        for col in self.binary_features:
            if col in X_pd.columns:
                # Fill nulls with mode if available, otherwise 0
                if X_pd[col].isnull().any():
                    if col in self.imputer_mode_ and self.imputer_mode_[col] is not None:
                        X_pd[col] = X_pd[col].fillna(self.imputer_mode_[col])
                    else:
                        X_pd[col] = X_pd[col].fillna(0)
                # Convert to int
                try:
                    X_pd[col] = X_pd[col].astype(int)
                except (ValueError, TypeError):
                    # If conversion fails, try to convert to numeric first
                    X_pd[col] = pd.to_numeric(X_pd[col], errors='coerce').fillna(0).astype(int)
        
        return X_pd
    
    def fit(self, X, y):
        """
        Fit the Data Agent on training data.
        X can be a path, pandas DataFrame, or Polars DataFrame.
        y is the target array (pandas Series or Polars Series).
        """
        # Convert X to Polars
        X_pl = self._to_polars(X)
        if isinstance(y, pl.Series):
            y_pl = y
        else:
            y_pl = pl.Series(y) if y is not None else None
        
        # 1. Add domain features
        X_pl = self.add_domain_features(X_pl)
        
        # 2. Fit imputers and outlier thresholds
        self._fit_imputers_outliers(X_pl, y_pl)
        
        # 3. Apply imputation and capping (to get clean data for encoding/scaling)
        X_clean = self._apply_imputers_outliers(X_pl)
        
        # 4. Convert to pandas for sklearn
        X_pd = self._prepare_for_sklearn(X_clean)
        
        # 5. Separate features by type
        cat_cols = [c for c in self.categorical_features if c in X_pd.columns]
        num_cols = [c for c in self.numeric_features if c in X_pd.columns]
        bin_cols = [c for c in self.binary_features if c in X_pd.columns]
        # Derived features are automatically included; they are numeric
        derived_cols = [c for c in X_pd.columns if c not in cat_cols + num_cols + bin_cols + [self.target_col]]
        
        # 6. Fit OneHotEncoder for categorical
        if cat_cols:
            self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.encoder_.fit(X_pd[cat_cols])
            encoded_names = self.encoder_.get_feature_names_out(cat_cols)
        else:
            self.encoder_ = None
            encoded_names = []
        
        # 7. Fit StandardScaler for numeric (including derived features)
        all_num_cols = num_cols + derived_cols
        if all_num_cols:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_pd[all_num_cols])
        else:
            self.scaler_ = None
        
        # 8. Store column names for transform
        self.encoded_feature_names_ = list(encoded_names)
        self.binary_feature_names_ = bin_cols
        self.derived_feature_names_ = derived_cols
        
        # 9. Feature selection (using cleaned data, but need to combine all features)
        # Build the full feature matrix for selection
        X_full = self._build_feature_matrix(X_pd, cat_cols, num_cols, bin_cols, derived_cols)
        
        # Sample for feature selection if needed
        if self.sample_size_for_selection and len(X_full) > self.sample_size_for_selection:
            indices = np.random.choice(len(X_full), self.sample_size_for_selection, replace=False)
            X_sample = X_full.iloc[indices]
            y_sample = y_pl.to_pandas().iloc[indices] if y_pl is not None else None
        else:
            X_sample = X_full
            y_sample = y_pl.to_pandas() if y_pl is not None else None
        
        # Perform feature selection
        if y_sample is not None:
            if self.feature_selection_method == 'rf':
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_sample, y_sample)
                importances = rf.feature_importances_
            elif self.feature_selection_method == 'mutual_info':
                mi = mutual_info_classif(X_sample, y_sample, random_state=42)
                importances = mi
            else:
                raise ValueError("Invalid feature selection method.")
            
            imp_series = pd.Series(importances, index=X_full.columns)
            top_features = imp_series.nlargest(self.n_top_features).index.tolist()
            self.selected_features_ = top_features
        else:
            # If no target, keep all features
            self.selected_features_ = X_full.columns.tolist()
        
        logger.info(f"Feature selection complete. Selected {len(self.selected_features_)} features.")
        return self
    
    def _build_feature_matrix(self, X_pd, cat_cols, num_cols, bin_cols, derived_cols):
        """
        Build the full feature matrix from cleaned pandas DataFrame.
        Returns pandas DataFrame with all features.
        """
        # Encode categorical
        if cat_cols and self.encoder_:
            cat_encoded = pd.DataFrame(self.encoder_.transform(X_pd[cat_cols]),
                                       columns=self.encoder_.get_feature_names_out(cat_cols),
                                       index=X_pd.index)
        else:
            cat_encoded = pd.DataFrame(index=X_pd.index)
        
        # Scale numeric (including derived)
        all_num_cols = num_cols + derived_cols
        if all_num_cols and self.scaler_:
            num_scaled = pd.DataFrame(self.scaler_.transform(X_pd[all_num_cols]),
                                      columns=all_num_cols,
                                      index=X_pd.index)
        else:
            num_scaled = pd.DataFrame(index=X_pd.index)
        
        # Binary columns unchanged
        if bin_cols:
            bin_df = X_pd[bin_cols].astype(int)
        else:
            bin_df = pd.DataFrame(index=X_pd.index)
        
        # Combine
        X_full = pd.concat([cat_encoded, num_scaled, bin_df], axis=1)
        return X_full
    
    def transform(self, X):
        """
        Apply all transformations to new data (for inference).
        X can be path, pandas, or Polars.
        Returns pandas DataFrame with selected features.
        """
        if self.encoder_ is None or self.scaler_ is None:
            raise RuntimeError("DataAgent not fitted. Call fit() first.")
        
        # Convert to Polars
        X_pl = self._to_polars(X)
        
        # Add domain features
        X_pl = self.add_domain_features(X_pl)
        
        # Apply imputation and capping
        X_clean = self._apply_imputers_outliers(X_pl)
        
        # Convert to pandas
        X_pd = self._prepare_for_sklearn(X_clean)
        
        # Identify columns
        cat_cols = [c for c in self.categorical_features if c in X_pd.columns]
        num_cols = [c for c in self.numeric_features if c in X_pd.columns]
        bin_cols = [c for c in self.binary_features if c in X_pd.columns]
        derived_cols = [c for c in X_pd.columns if c not in cat_cols + num_cols + bin_cols + [self.target_col]]
        
        # Build full feature matrix
        X_full = self._build_feature_matrix(X_pd, cat_cols, num_cols, bin_cols, derived_cols)
        
        # Select only the top features (if selection was performed)
        if self.selected_features_:
            # Ensure all selected features exist
            missing = set(self.selected_features_) - set(X_full.columns)
            if missing:
                logger.warning(f"Missing features in transform: {missing}. These will be filled with zeros.")
                for m in missing:
                    X_full[m] = 0
            X_selected = X_full[self.selected_features_]
        else:
            X_selected = X_full
        
        return X_selected
    
    def fit_transform(self, X, y):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
    
    def time_series_split(self, X, month_col='month', train_months=None, test_months=None):
        """
        Split data by month to avoid leakage.
        X is a Polars or pandas DataFrame containing the 'month' column.
        train_months: list of months for training (e.g., [1,2,3,4,5])
        test_months: list of months for testing (e.g., [6])
        Returns (train_idx, test_idx) as Polars expressions or pandas indices.
        """
        # Convert to Polars if needed
        if not isinstance(X, pl.DataFrame):
            X = self._to_polars(X)
        
        if train_months is None:
            # Default: use first 80% of months for training, last 20% for testing
            unique_months = X.select(pl.col(month_col).unique()).sort(month_col).to_series().to_list()
            split = int(0.8 * len(unique_months))
            train_months = unique_months[:split]
            test_months = unique_months[split:]
        
        train_mask = X[month_col].is_in(train_months)
        test_mask = X[month_col].is_in(test_months)
        
        train_idx = X.filter(train_mask).select(pl.all()).to_pandas().index
        test_idx = X.filter(test_mask).select(pl.all()).to_pandas().index
        
        return train_idx, test_idx
    
    def save(self, path):
        """Save fitted agent using joblib."""
        import joblib
        joblib.dump(self, path)
        logger.info(f"Agent saved to {path}")
    
    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)