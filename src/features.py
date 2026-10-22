"""
src/features.py
---------------
Universal Time-Series Feature Engineering Engine.
Generates:
- Lag features (t-1, t-2, t-7, t-14, t-30)
- Rolling aggregations (Mean, Std, Min, Max over 7, 14, 30 day windows)
- Day-over-day and Week-over-week Momentum (Differences)
- Cyclical calendar encodings (Sine/Cosine for seasonal continuity)
- Strict leakage-safe window shifts
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Feature generation configuration."""
    target_column: str = "temp_mean_c"
    lag_days: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14, 30])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    diff_lags: List[int] = field(default_factory=lambda: [1, 7])
    cyclical_encoding: bool = True
    drop_initial_nans: bool = True


class UniversalFeatureEngineer:
    """Dataset-agnostic feature engineering pipeline."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "day_of_year" in df.columns or isinstance(df.index, pd.DatetimeIndex):
            doy = df["day_of_year"] if "day_of_year" in df.columns else df.index.dayofyear
            df["sin_day_of_year"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
            df["cos_day_of_year"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

        if "month" in df.columns or isinstance(df.index, pd.DatetimeIndex):
            month = df["month"] if "month" in df.columns else df.index.month
            df["sin_month"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
            df["cos_month"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)

        return df

    def _add_lags_and_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        base_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if not c.startswith(("sin_", "cos_", "year", "day_of", "is_weekend", "quarter"))
        ]

        logger.info(f"Generating features for base columns: {base_cols}")
        new_features = {}

        for col in base_cols:
            for lag in self.config.lag_days:
                new_features[f"{col}_lag_{lag}"] = df[col].shift(lag)

            shifted = df[col].shift(1)
            for window in self.config.rolling_windows:
                rolling_obj = shifted.rolling(window=window, min_periods=max(2, window // 2))
                if "mean" in self.config.rolling_stats:
                    new_features[f"{col}_roll_mean_{window}"] = rolling_obj.mean()
                if "std" in self.config.rolling_stats:
                    new_features[f"{col}_roll_std_{window}"] = rolling_obj.std()
                if "min" in self.config.rolling_stats:
                    new_features[f"{col}_roll_min_{window}"] = rolling_obj.min()
                if "max" in self.config.rolling_stats:
                    new_features[f"{col}_roll_max_{window}"] = rolling_obj.max()

            for diff in self.config.diff_lags:
                new_features[f"{col}_diff_{diff}"] = df[col].shift(1) - df[col].shift(diff + 1)

        features_df = pd.DataFrame(new_features, index=df.index, dtype=np.float32)
        df = pd.concat([df, features_df], axis=1)
        return df

    def transform(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering. Initial shape: {df_clean.shape}")
        df = df_clean.copy()

        if self.config.cyclical_encoding:
            df = self._add_cyclical_features(df)

        df = self._add_lags_and_rolling(df)

        if self.config.drop_initial_nans:
            max_window = max(max(self.config.lag_days), max(self.config.rolling_windows))
            initial_len = len(df)
            df = df.iloc[max_window:].copy()
            logger.info(f"Dropped first {max_window} warmup rows. Rows: {initial_len} -> {len(df)}")

        df = df.bfill().ffill()
        logger.info(f"Feature engineering complete. Final feature matrix shape: {df.shape}")
        return df


def engineer_and_save_features(
    clean_parquet_path: str = "data/clean/kollam_climate.parquet",
    output_parquet_path: str = "data/clean/kollam_climate_features.parquet",
    config: Optional[FeatureConfig] = None
) -> pd.DataFrame:
    df_clean = pd.read_parquet(clean_parquet_path)
    fe = UniversalFeatureEngineer(config)
    df_features = fe.transform(df_clean)

    out_path = Path(output_parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out_path, engine="pyarrow", compression="snappy")
    logger.info(f"Feature matrix saved successfully to: {out_path}")
    return df_features


if __name__ == "__main__":
    engineer_and_save_features()
