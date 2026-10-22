"""
src/clean.py
------------
Universal Data Cleaner:
1. Auto-detects / enforces datetime index.
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
    s = re.sub(r"[^\w\s]", "", name)
    s = re.sub(r"\s+", "_", s.strip())
    return s.lower()


class UniversalDataCleaner:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Datetime index auto-detect cheyyunnu."""
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = "date"
            return df.sort_index()

        candidate_cols = [c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()]
        if candidate_cols:
            date_col = candidate_cols[0]
            logger.info(f"Auto-detected date column: '{date_col}'")
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df.set_index(date_col, inplace=True)
            df.index.name = "date"
        else:
            logger.warning("No datetime column detected. Treating index as integer sequence.")

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        return df

    def _handle_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """NASA -999.0 poleyulla sentinels replace cheyyunnu."""
        df = df.copy()
        for sentinel in self.config.sentinel_values:
            df = df.replace(sentinel, np.nan)
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adaptive time-series interpolation."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Rainfall null aayaal 0.0 vekkunnu
            if any(term in col.lower() for term in ["rain", "precip", "prcp"]):
                df[col] = df[col].fillna(0.0)
            else:
                if isinstance(df.index, pd.DatetimeIndex):
                    df[col] = df[col].interpolate(method="time", limit=5)
                else:
                    df[col] = df[col].interpolate(method="linear", limit=5)

        df[numeric_cols] = df[numeric_cols].bfill().ffill()
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Config mapping apply cheyyunnu + rest columns snake_case aakkunnu."""
        df = df.copy()
        if self.config.column_mapping:
            df = df.rename(columns=self.config.column_mapping)
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
