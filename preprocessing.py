"""
preprocessing.py
Handles dataset loading, cleaning, feature engineering, encoding,
and train/test splitting for the Sales Forecaster application.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import FEATURE_COLUMNS, TARGET_COLUMN, CATEGORICAL_COLUMNS


class DataPreprocessor:
    """
    Encapsulates the full preprocessing pipeline:
        load -> clean -> engineer -> encode -> scale -> split
    Keeps encoders/scaler so the same transformations can be applied
    later to single-row prediction inputs.
    """

    def __init__(self):
        self.raw_df = None
        self.processed_df = None
        self.label_encoders = {}
        self.scaler = None
        self.summary = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_csv(self, path):
        """Load a CSV file into the raw dataframe."""
        self.raw_df = pd.read_csv(path)
        return self.raw_df

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean(self, df=None):
        """
        Handles missing values and duplicate rows.
        Returns the cleaned dataframe.
        """
        if df is None:
            df = self.raw_df.copy()
        else:
            df = df.copy()

        n_before = len(df)
        n_duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates()

        n_missing_before = int(df.isnull().sum().sum())

        # Numeric columns: fill missing with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Categorical / object columns: fill missing with mode
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)

        self.summary["rows_before"] = n_before
        self.summary["rows_after_dedup"] = len(df)
        self.summary["duplicates_removed"] = n_duplicates
        self.summary["missing_values_filled"] = n_missing_before

        return df

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df):
        """
        Derives Date-based features (Day_of_Week, Month, Year) if a
        'Date' column is present and those columns are missing.
        Also fills in any expected columns absent from a user's CSV
        with reasonable defaults so the pipeline stays robust.
        """
        df = df.copy()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if "Day_of_Week" not in df.columns:
                df["Day_of_Week"] = df["Date"].dt.dayofweek
            if "Month" not in df.columns:
                df["Month"] = df["Date"].dt.month
            if "Year" not in df.columns:
                df["Year"] = df["Date"].dt.year

        # Ensure Units_Sold * Selling_Price -> Sales if Sales missing
        if TARGET_COLUMN not in df.columns and {"Units_Sold", "Selling_Price"}.issubset(df.columns):
            discount = df["Discount"] if "Discount" in df.columns else 0
            df[TARGET_COLUMN] = df["Units_Sold"] * df["Selling_Price"] * (1 - discount / 100)

        # Fill any missing expected feature columns with defaults
        defaults = {
            "Store_ID": 1,
            "Category": "General",
            "Selling_Price": df["Selling_Price"].mean() if "Selling_Price" in df.columns else 0,
            "Discount": 0,
            "Promotion": "No",
            "Holiday_Indicator": "No",
            "Day_of_Week": 0,
            "Month": 1,
            "Year": 2024,
            "Previous_Sales": 0,
        }
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val

        return df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, df):
        """
        Label-encodes categorical columns. Encoders are stored so the
        same mapping can be reused for single-row predictions.
        """
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        return df

    def encode_single_value(self, column, value):
        """
        Encode a single categorical value using a previously fitted
        LabelEncoder. Falls back to 0 for unseen categories.
        """
        le = self.label_encoders.get(column)
        if le is None:
            return 0
        value = str(value)
        if value in le.classes_:
            return int(le.transform([value])[0])
        return 0

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    def run_pipeline(self, path):
        """
        Runs the complete preprocessing pipeline end-to-end and
        stores the resulting processed dataframe.
        """
        self.load_csv(path)
        df = self.clean(self.raw_df)
        df = self.engineer_features(df)
        df = self.encode(df)

        missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset missing required columns: {missing_features}")
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Dataset must contain a '{TARGET_COLUMN}' column, or 'Units_Sold' "
                f"and 'Selling_Price' columns so it can be derived."
            )

        self.processed_df = df
        return df

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def split(self, df=None, test_size=0.2, random_state=42):
        """
        Splits processed data into train/test X and y sets using the
        canonical FEATURE_COLUMNS / TARGET_COLUMN.
        """
        if df is None:
            df = self.processed_df
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
