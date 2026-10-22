"""
src/config.py
-------------
Universal dataset configuration.
Allows switching datasets (NASA API, local CSV, NetCDF) without code changes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetConfig:
    name: str = "kollam_climate"
    source_type: str = "csv"  # 'nasa_power', 'csv', 'netcdf'
    
    # Coordinates / Geographic info (for NASA POWER API)
    latitude: float = 8.89
    longitude: float = 76.61
    start_date: str = "20200101"     # YYYYMMDD
    end_date: str = "20251231"
    
    # File paths & sources (if CSV / NetCDF based)
    file_path: Optional[str] = "power.csv"
    file_source: Optional[Any] = None  # Buffer, UploadedFile, Bytes, or StringIO
    date_column: Optional[str] = None  # Auto-detected if None
    
    # Missing value sentinels used by NASA POWER and meteorological datasets
    sentinel_values: List[float] = field(default_factory=lambda: [-999.0, -9999.0, 9999.0, -999])
    
    # Canonical column rename map
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        "T2M": "temp_mean_c",
        "T2M_MAX": "temp_max_c",
        "T2M_MIN": "temp_min_c",
        "PRECTOTCORR": "rainfall_mm",
        "RH2M": "humidity_pct",
        "WS2M": "wind_speed_ms",
        "ALLSKY_SFC_SW_DWN": "solar_radiation_mj",
        "PS": "surface_pressure_kpa",
    })
