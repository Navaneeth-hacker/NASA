"""
src/load.py
-----------
Universal Data Loader.
Supports:
1. NASA POWER API (any lat/lon point)
2. Universal CSV files (any header, delimiter, date format, or column naming)
3. NetCDF / xarray files (gridded satellite/reanalysis data)
"""

import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO
import numpy as np
import pandas as pd
import requests

from src.config import DatasetConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_universal_csv(
    source: Union[str, Path, BinaryIO, TextIO, bytes, Any],
    date_column: Optional[str] = None,
    sentinel_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Ultra-resilient CSV reader.
    Handles:
    - NASA POWER headers (-BEGIN HEADER- ... -END HEADER-)
    - Metadata and comment preambles (#, //, ;, etc.)
    - Arbitrary delimiters (comma, semicolon, tab, pipe)
    - Various encodings (utf-8, utf-8-sig, cp1252, latin1)
    - Flexible date formats (YEAR+DOY, YEAR+MO+DY, ISO, timestamps, or synthetic timeline)
    - Missing value sentinels (-999, -999.0, -9999, NA, null, etc.)
    - File paths, in-memory buffers, Streamlit UploadedFile, or raw strings
    """
    if sentinel_values is None:
        sentinel_values = [-999.0, -9999.0, 9999.0, -999.0]

    text = ""
    # 1. Extract text content safely from any source
    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        raw_bytes = Path(str(source)).read_bytes()
        for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            try:
                text = raw_bytes.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
    elif hasattr(source, "read"):
        pos = getattr(source, "tell", lambda: None)()
        raw = source.read()
        if hasattr(source, "seek") and pos is not None:
            try:
                source.seek(pos)
            except Exception:
                pass
        if isinstance(raw, bytes):
            for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
                try:
                    text = raw.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
        else:
            text = str(raw)
    elif isinstance(source, bytes):
        for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            try:
                text = source.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
    elif isinstance(source, str):
        # Might be a file path that doesn't exist, or raw CSV text
        if "\n" not in source and (source.endswith(".csv") or source.endswith(".txt")):
            p = Path(source)
            if p.exists():
                text = p.read_text(encoding="utf-8", errors="ignore")
            else:
                text = source
        else:
            text = source
    else:
        raise ValueError(f"Unsupported source type for CSV reader: {type(source)}")

    if not text.strip():
        logger.warning("Empty CSV source provided.")
        return pd.DataFrame()

    lines = text.splitlines()

    # 2. Skip preamble / header comments
    start_idx = 0
    # Check for NASA POWER header end
    for idx, line in enumerate(lines):
        if "-END HEADER-" in line or "END_OF_HEADER" in line or "*END*" in line:
            start_idx = idx + 1
            break
    else:
        # If no explicit NASA end header, skip comment and empty lines
        while start_idx < len(lines):
            stripped = lines[start_idx].strip()
            if not stripped or stripped.startswith(("#", "//", "/*", ";", "%")):
                start_idx += 1
            else:
                break

    if start_idx >= len(lines):
        logger.warning("No data rows found in CSV after skipping preamble.")
        return pd.DataFrame()

    # 3. Delimiter Detection
    header_line = lines[start_idx]
    delimiter_counts = {
        ",": header_line.count(","),
        ";": header_line.count(";"),
        "\t": header_line.count("\t"),
        "|": header_line.count("|"),
    }
    detected_sep = max(delimiter_counts, key=delimiter_counts.get)
    if delimiter_counts[detected_sep] == 0:
        detected_sep = ","

    csv_data = "\n".join(lines[start_idx:])

    # 4. Parse DataFrame
    try:
        df = pd.read_csv(
            io.StringIO(csv_data),
            sep=detected_sep,
            engine="python",
            on_bad_lines="skip",
        )
    except Exception as e:
        logger.warning(f"Python engine parsing failed with delimiter '{detected_sep}': {e}. Retrying with default.")
        df = pd.read_csv(io.StringIO(csv_data), engine="python")

    # Clean column names
    df.columns = [str(c).strip().strip("'\"") for c in df.columns]

    # 5. Universal Date Parsing
    cols_lower = {str(c).lower(): c for c in df.columns}
    parsed_date = False

    # Check for NASA POWER YEAR + DOY format
    if "year" in cols_lower and "doy" in cols_lower:
        try:
            y_col = cols_lower["year"]
            d_col = cols_lower["doy"]
            y_str = pd.to_numeric(df[y_col], errors="coerce").fillna(2000).astype(int).astype(str)
            d_str = pd.to_numeric(df[d_col], errors="coerce").fillna(1).astype(int).astype(str).str.zfill(3)
            df["date"] = pd.to_datetime(y_str + d_str, format="%Y%j", errors="coerce")
            if df["date"].notna().sum() > 0.5 * len(df):
                parsed_date = True
        except Exception as e:
            logger.debug(f"YEAR+DOY parsing attempt failed: {e}")

    # Check for Year + Month + Day triplet
    if not parsed_date:
        y_name = next((cols_lower[k] for k in ["year", "yr", "yyyy"] if k in cols_lower), None)
        m_name = next((cols_lower[k] for k in ["month", "mo", "mm", "mon"] if k in cols_lower), None)
        d_name = next((cols_lower[k] for k in ["day", "dy", "dd"] if k in cols_lower), None)
        if y_name and m_name and d_name:
            try:
                date_parts = pd.DataFrame({
                    "year": pd.to_numeric(df[y_name], errors="coerce").fillna(2000).astype(int),
                    "month": pd.to_numeric(df[m_name], errors="coerce").fillna(1).astype(int),
                    "day": pd.to_numeric(df[d_name], errors="coerce").fillna(1).astype(int),
                })
                df["date"] = pd.to_datetime(date_parts, errors="coerce")
                if df["date"].notna().sum() > 0.5 * len(df):
                    parsed_date = True
            except Exception as e:
                logger.debug(f"Year/Month/Day triplet parsing failed: {e}")

    # Check explicit date column or candidate name
    if not parsed_date:
        candidates = []
        if date_column and date_column in df.columns:
            candidates.append(date_column)

        for c in df.columns:
            cl = str(c).lower()
            if any(term in cl for term in ["date", "time", "timestamp", "datetime", "dt", "obs_date", "period"]):
                if c not in candidates:
                    candidates.append(c)

        for cand in candidates:
            try:
                parsed = pd.to_datetime(df[cand], errors="coerce")
                if parsed.notna().sum() > 0.4 * len(df):
                    df["date"] = parsed
                    parsed_date = True
                    break
            except Exception:
                continue

    # Search object columns if date still not found
    if not parsed_date:
        for c in df.columns:
            if df[c].dtype == object:
                sample = df[c].dropna().head(30)
                if len(sample) > 0:
                    try:
                        parsed_sample = pd.to_datetime(sample, errors="coerce")
                        if parsed_sample.notna().sum() >= 0.7 * len(sample):
                            df["date"] = pd.to_datetime(df[c], errors="coerce")
                            parsed_date = True
                            break
                    except Exception:
                        continue

    # Fallback: create synthetic daily dates if none found or all dates NaT
    if not parsed_date or df["date"].isna().all():
        df["date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
        df.attrs["date_inferred"] = True
    else:
        df["date"] = df["date"].ffill().bfill()
        df.attrs["date_inferred"] = False

    # Ensure integer YEAR column exists
    df["YEAR"] = df["date"].dt.year

    # 6. Clean numeric columns & Sentinels
    all_sentinels = set(sentinel_values) | {-999, -999.0, -9999, -9999.0, 9999, 9999.0}
    for col in df.columns:
        if col in ["date"]:
            continue
        # Convert string numeric columns to float
        if df[col].dtype == object:
            cleaned_str = df[col].astype(str).str.strip().str.replace(",", "")
            cleaned_str = cleaned_str.replace({str(s): np.nan for s in all_sentinels})
            cleaned_str = cleaned_str.replace({
                "NA": np.nan, "null": np.nan, "nan": np.nan,
                "None": np.nan, "?": np.nan, "-": np.nan, "": np.nan
            })
            num_series = pd.to_numeric(cleaned_str, errors="coerce")
            if num_series.notna().sum() > 0.3 * len(df):
                df[col] = num_series

        if pd.api.types.is_numeric_dtype(df[col]):
            for s in all_sentinels:
                df[col] = df[col].replace(s, np.nan)

    return df


def detect_weather_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently maps columns in any DataFrame to meteorological concepts.
    Returns:
        dict: {'temperature': col or None, 'rainfall': col or None, 'humidity': col or None, ...}
    """
    cols = list(df.columns)
    cols_lower = {str(c).lower().strip(): c for c in cols}

    synonyms = {
        "temperature": [
            "t2m", "temp", "temperature", "temp_mean", "temp_mean_c", "tavg",
            "t_avg", "air_temp", "temperature_c", "temp_c", "t2m_max", "t2m_min", "tmp"
        ],
        "rainfall": [
            "prectotcorr", "rainfall", "rain", "precip", "precipitation",
            "prcp", "rainfall_mm", "rain_mm", "precip_mm", "precip_tot", "rr", "total_precipitation"
        ],
        "humidity": [
            "rh2m", "humidity", "rh", "relative_humidity", "humidity_pct",
            "rel_hum", "rh_pct", "relative_humidity_2m"
        ],
        "wind": [
            "ws2m", "wind", "wind_speed", "wind_speed_ms", "wspd", "wind_spd"
        ],
        "solar": [
            "allsky_sfc_sw_dwn", "solar_radiation", "solar", "radiation", "srad"
        ],
        "pressure": [
            "ps", "pressure", "surface_pressure", "surface_pressure_kpa", "pres"
        ],
    }

    result: Dict[str, Optional[str]] = {k: None for k in synonyms}

    for concept, terms in synonyms.items():
        # 1. Exact match
        for term in terms:
            if term in cols_lower:
                result[concept] = cols_lower[term]
                break
        # 2. Substring match if not found
        if not result[concept]:
            for col_lower, orig_col in cols_lower.items():
                if orig_col in ["date", "year", "month", "day", "is_weekend"]:
                    continue
                if any(term in col_lower for term in terms):
                    result[concept] = orig_col
                    break

    return result


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
    Handles ANY CSV file thrown at it: standard CSVs, NASA POWER CSVs with metadata headers,
    different delimiters, date representations, and sentinels.
    """

    def load(self) -> pd.DataFrame:
        source = self.config.file_source or self.config.file_path
        if source is None:
            raise ValueError("CSV loader requires 'file_path' or 'file_source' in config.")

        logger.info(f"Loading CSV data from source: {source}")
        df = read_universal_csv(
            source=source,
            date_column=self.config.date_column,
            sentinel_values=self.config.sentinel_values,
        )

        # Set datetime index
        if "date" in df.columns:
            # If YEAR and DOY are present as redundant columns, keep or drop cleanly
            df.set_index("date", inplace=True)
            df.index.name = "date"

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
    print("Loaded shape:", df.shape)
    print(df.head())
