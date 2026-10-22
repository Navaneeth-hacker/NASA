"""
src/load.py
-----------
Universal Data Loader.
Supports:
1. NASA POWER API (any lat/lon point)
2. Local CSV files (regular CSV or NASA POWER header format with YEAR+DOY)
3. NetCDF / xarray files (gridded satellite/reanalysis data)
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd
import requests

from src.config import DatasetConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Raw data load cheythu DataFrame aayi return cheyyunnu."""
        pass


class NasaPowerLoader(BaseDataLoader):
    """NASA POWER Daily Point API Loader."""

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def load(self) -> pd.DataFrame:
        logger.info(
            f"Fetching NASA POWER API: Lat={self.config.latitude}, "
            f"Lon={self.config.longitude}, Range={self.config.start_date} to {self.config.end_date}"
        )

        params = {
            "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PS",
            "community": "AG",
            "longitude": self.config.longitude,
            "latitude": self.config.latitude,
            "start": self.config.start_date,
            "end": self.config.end_date,
            "format": "JSON",
        }

        response = requests.get(self.BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        raw_json = response.json()

        # Cache raw JSON
        raw_json_path = Path("data/raw") / f"{self.config.name}_raw.json"
        raw_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_json, f, indent=2)

        data_dict = raw_json["properties"]["parameter"]
        df = pd.DataFrame(data_dict)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "date"
        return df


class CSVDataLoader(BaseDataLoader):
    """
    Universal CSV loader.
    Handles standard CSVs and NASA POWER CSVs with metadata headers & YEAR+DOY.
    """

    def load(self) -> pd.DataFrame:
        if not self.config.file_path:
            raise ValueError("CSV loader requires 'file_path' in config.")

        file_path = Path(self.config.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading CSV data from: {file_path}")

        # Check for NASA POWER header banner
        skip_rows = 0
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                if "-END HEADER-" in line:
                    skip_rows = idx + 1
                    break

        df = pd.read_csv(file_path, skiprows=skip_rows)

        # Handle NASA POWER YEAR + DOY format if present
        if "YEAR" in df.columns and "DOY" in df.columns:
            df["date"] = pd.to_datetime(
                df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
                format="%Y%j",
            )
            df = df.drop(columns=["YEAR", "DOY"])
            df.set_index("date", inplace=True)
        elif self.config.date_column and self.config.date_column in df.columns:
            df["date"] = pd.to_datetime(df[self.config.date_column])
            df.set_index("date", inplace=True)
        
        return df


class NetCDFDataLoader(BaseDataLoader):
    """Satellite / Gridded NetCDF files loader (via xarray)."""

    def load(self) -> pd.DataFrame:
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("NetCDF load cheyyan 'xarray' package install cheyyuka.")

        logger.info(f"Loading NetCDF data from: {self.config.file_path}")
        ds = xr.open_dataset(self.config.file_path)
        return ds.to_dataframe()


def get_data_loader(config: DatasetConfig) -> BaseDataLoader:
    """Factory function: source_type anussarichu loader select cheyyunnu."""
    loaders = {
        "nasa_power": NasaPowerLoader,
        "csv": CSVDataLoader,
        "netcdf": NetCDFDataLoader,
    }
    loader_cls = loaders.get(config.source_type.lower())
    if not loader_cls:
        raise ValueError(f"Unsupported source_type: {config.source_type}. Options: {list(loaders.keys())}")
    return loader_cls(config)


def load_raw_dataset(config: DatasetConfig, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Universal fetch & raw save function."""
    loader = get_data_loader(config)
    df_raw = loader.load()

    raw_path = Path(raw_dir) / f"{config.name}_raw.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(raw_path)
    logger.info(f"Raw data cached to: {raw_path} | Shape: {df_raw.shape}")
    return df_raw


if __name__ == "__main__":
    cfg = DatasetConfig()
    df = load_raw_dataset(cfg)
    print(df.head())
