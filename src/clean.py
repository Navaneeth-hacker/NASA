"""
src/clean.py
------------
Universal Data Cleaner:
1. Auto-detects / enforces datetime index on any CSV.
2. Replaces arbitrary missing sentinels (-999, -9999, NA, etc.).
3. Standardizes column names to clean snake_case and mapped aliases.
4. Intelligent time-series interpolation (works on any continuous data).
5. Dynamic calendar feature generation (year, month, day, day_of_week, day_of_year, etc.).
6. Memory downcasting (64-bit to 32-bit floats for fast ML).
"""

import logging
import re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import DatasetConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def to_snake_case(name: str) -> str:
    """Column names clean snake_case aakkunnu."""
    s = re.sub(r"[^\w\s]", "", str(name))
    s = re.sub(r"\s+", "_", s.strip())
    return s.lower()


class UniversalDataCleaner:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Datetime index auto-detect cheyyunnu or constructs synthetic one."""
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = "date"
            return df.sort_index()

        # Check for 'date' column
        cols_lower = {str(c).lower(): c for c in df.columns}
        if "date" in cols_lower:
            date_col = cols_lower["date"]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df.set_index(date_col, inplace=True)
            df.index.name = "date"
            return df.sort_index()

        # Check for candidates
        candidate_cols = [c for c in df.columns if any(term in str(c).lower() for term in ["date", "time", "timestamp", "datetime", "dt", "day"])]
        found_date = False
        for c in candidate_cols:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > 0.4 * len(df):
                df[c] = parsed
                df = df.dropna(subset=[c])
                df.set_index(c, inplace=True)
                df.index.name = "date"
                found_date = True
                break

        if not found_date:
            logger.warning("No datetime column detected. Generating synthetic daily date index.")
            df.index = pd.date_range("2020-01-01", periods=len(df), freq="D")
            df.index.name = "date"

        return df.sort_index()

    def _handle_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """NASA -999.0 poleyulla sentinels replace cheyyunnu."""
        df = df.copy()
        sentinels = set(self.config.sentinel_values) | {-999, -999.0, -9999, -9999.0, 9999, 9999.0}

        for col in df.columns:
            # Handle string columns containing numeric values with sentinels
            if df[col].dtype == object:
                cleaned = df[col].astype(str).str.strip().str.replace(",", "")
                cleaned = cleaned.replace({str(s): np.nan for s in sentinels})
                cleaned = cleaned.replace({
                    "NA": np.nan, "null": np.nan, "nan": np.nan,
                    "None": np.nan, "?": np.nan, "-": np.nan, "": np.nan
                })
                try_num = pd.to_numeric(cleaned, errors="coerce")
                if try_num.notna().sum() > 0.3 * len(df):
                    df[col] = try_num

            if pd.api.types.is_numeric_dtype(df[col]):
                for sentinel in sentinels:
                    df[col] = df[col].replace(sentinel, np.nan)

        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adaptive time-series interpolation."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if any(term in str(col).lower() for term in ["rain", "precip", "prcp"]):
                df[col] = df[col].fillna(0.0)
            else:
                if isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df[col] = df[col].interpolate(method="time", limit=5)
                    except Exception:
                        df[col] = df[col].interpolate(method="linear", limit=5)
                else:
                    df[col] = df[col].interpolate(method="linear", limit=5)

        df[numeric_cols] = df[numeric_cols].bfill().ffill().fillna(0.0)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Config mapping apply cheyyunnu + rest columns snake_case aakkunnu."""
        df = df.copy()
        if self.config.column_mapping:
            mapping_lower = {k.lower(): v for k, v in self.config.column_mapping.items()}
            renames = {}
            for c in df.columns:
                cl = str(c).lower()
                if cl in mapping_lower:
                    renames[c] = mapping_lower[cl]
            df = df.rename(columns=renames)

        df.columns = [to_snake_case(str(c)) for c in df.columns]
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calendar features generate cheyyunnu."""
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df["year"] = df.index.year
            df["month"] = df.index.month
            df["day"] = df.index.day
            df["day_of_week"] = df.index.dayofweek
            df["day_of_year"] = df.index.dayofyear
            df["quarter"] = df.index.quarter
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        return df

    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Float64 -> Float32 downcast cheyyunnu."""
        df = df.copy()
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        return df

    def clean(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning execution."""
        logger.info(f"Cleaning dataset: '{self.config.name}' (Initial shape: {df_raw.shape})")
        df = self._ensure_datetime_index(df_raw)
        df = self._handle_sentinels(df)
        df = self._standardize_columns(df)
        df = self._impute_missing(df)
        df = self._add_calendar_features(df)
        df = self._optimize_memory(df)
        logger.info(f"Cleaning complete. Output shape: {df.shape}")
        return df


def clean_and_save_dataset(
    df_raw: pd.DataFrame,
    config: DatasetConfig,
    clean_dir: str = "data/clean"
) -> Path:
    """Clean cheythu Parquet aayi save cheyyunnu."""
    cleaner = UniversalDataCleaner(config)
    df_clean = cleaner.clean(df_raw)

    clean_path = Path(clean_dir) / f"{config.name}.parquet"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(clean_path, engine="pyarrow", compression="snappy")

    # Quick inspection sample CSV
    csv_path = Path(clean_dir) / f"{config.name}_sample.csv"
    df_clean.head(100).to_csv(csv_path)

    logger.info(f"Saved clean Parquet to: {clean_path}")
    return clean_path


if __name__ == "__main__":
    from src.load import load_raw_dataset
    cfg = DatasetConfig()
    df_raw = load_raw_dataset(cfg)
    clean_and_save_dataset(df_raw, cfg)
