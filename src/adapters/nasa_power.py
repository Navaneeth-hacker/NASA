"""
src/adapters/nasa_power.py
--------------------------
NASA POWER API Client and Adapter.
Implements the formal adapter contract: provider, discover, fetch, validate,
normalize, metadata, and structured error handling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class NasaAdapterError:
    """Structured error representation for API failures."""
    code: str
    message: str
    status_code: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class NasaPowerAdapter:
    """
    Client adapter for the NASA Prediction Of Worldwide Energy Resources (POWER) API.
    Provides verified daily meteorological and solar irradiance observations globally.
    """

    PROVIDER = "nasa_power"
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    # Supported meteorological parameters and their scientific designations
    SUPPORTED_PARAMETERS = {
        "T2M": "Temperature at 2 Meters (°C)",
        "T2M_MAX": "Maximum Temperature at 2 Meters (°C)",
        "T2M_MIN": "Minimum Temperature at 2 Meters (°C)",
        "PRECTOTCORR": "Precipitation Corrected (mm/day)",
        "RH2M": "Relative Humidity at 2 Meters (%)",
        "WS2M": "Wind Speed at 2 Meters (m/s)",
        "ALLSKY_SFC_SW_DWN": "All Sky Surface Shortwave Downward Irradiance (MJ/m²/day)",
        "PS": "Surface Pressure (kPa)",
    }

    COMMUNITIES = ["AG", "RE", "SB"]

    def __init__(self, timeout_sec: int = 45, max_retries: int = 3):
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def discover(self) -> Dict[str, Any]:
        """Returns catalog of supported parameters, capabilities, and API specifications."""
        return {
            "provider": self.PROVIDER,
            "description": "NASA POWER Daily Point API (MERRA-2 reanalysis and satellite observations).",
            "endpoint": self.BASE_URL,
            "supported_parameters": self.SUPPORTED_PARAMETERS,
            "supported_communities": self.COMMUNITIES,
            "min_date": "1981-01-01",
            "max_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "resolution": "Daily point, native 0.5° x 0.625°",
        }

    def fetch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,  # 'YYYYMMDD' or 'YYYY-MM-DD'
        end_date: str,    # 'YYYYMMDD' or 'YYYY-MM-DD'
        parameters: Optional[List[str]] = None,
        community: str = "AG",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[NasaAdapterError]]:
        """
        Executes bounded HTTP request with exponential backoff and timeout handling.
        """
        # Coordinate bounds validation
        if not (-90.0 <= latitude <= 90.0):
            return None, NasaAdapterError(
                code="INVALID_COORDINATES",
                message=f"Latitude {latitude} out of valid range [-90.0, 90.0]."
            )
        if not (-180.0 <= longitude <= 180.0):
            return None, NasaAdapterError(
                code="INVALID_COORDINATES",
                message=f"Longitude {longitude} out of valid range [-180.0, 180.0]."
            )

        # Normalize date strings to YYYYMMDD
        start_clean = str(start_date).replace("-", "")
        end_clean = str(end_date).replace("-", "")

        param_list = parameters or list(self.SUPPORTED_PARAMETERS.keys())
        params_str = ",".join(param_list)

        query_params = {
            "parameters": params_str,
            "community": community if community in self.COMMUNITIES else "AG",
            "longitude": f"{longitude:.4f}",
            "latitude": f"{latitude:.4f}",
            "start": start_clean,
            "end": end_clean,
            "format": "JSON",
        }

        logger.info(f"Querying NASA POWER API: Lat={latitude}, Lon={longitude}, Dates={start_clean}-{end_clean}")

        attempt = 0
        backoff_sec = 2.0

        while attempt < self.max_retries:
            attempt += 1
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=query_params,
                    timeout=self.timeout_sec,
                    headers={"User-Agent": "NASA-Climate-Dashboard/2.0"},
                )

                if response.status_code == 200:
                    payload = response.json()
                    val_err = self.validate(payload)
                    if val_err:
                        return None, val_err
                    return payload, None

                elif response.status_code == 429:
                    logger.warning(f"NASA POWER rate limit encountered. Retrying in {backoff_sec}s...")
                    time.sleep(backoff_sec)
                    backoff_sec *= 2.0
                    continue

                elif response.status_code in [500, 502, 503, 504]:
                    logger.warning(f"NASA POWER server error ({response.status_code}). Retrying in {backoff_sec}s...")
                    time.sleep(backoff_sec)
                    backoff_sec *= 2.0
                    continue

                else:
                    return None, NasaAdapterError(
                        code="HTTP_ERROR",
                        message=f"NASA API returned status {response.status_code}: {response.text[:200]}",
                        status_code=response.status_code,
                    )

            except requests.exceptions.Timeout:
                logger.warning(f"NASA API request timed out (Attempt {attempt}/{self.max_retries}).")
                if attempt >= self.max_retries:
                    return None, NasaAdapterError(
                        code="TIMEOUT",
                        message=f"Request to NASA POWER API timed out after {self.timeout_sec} seconds."
                    )
                time.sleep(backoff_sec)
                backoff_sec *= 2.0

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error querying NASA API: {e}")
                return None, NasaAdapterError(
                    code="NETWORK_ERROR",
                    message=f"Unable to connect to NASA POWER endpoint: {str(e)}"
                )

        return None, NasaAdapterError(code="MAX_RETRIES_EXCEEDED", message="Failed to obtain response after max retries.")

    def validate(self, payload: Dict[str, Any]) -> Optional[NasaAdapterError]:
        """Validates JSON response structure according to official schema."""
        if not isinstance(payload, dict):
            return NasaAdapterError(code="INVALID_PAYLOAD", message="Response is not a valid JSON object.")

        properties = payload.get("properties")
        if properties is None or not isinstance(properties, dict):
            return NasaAdapterError(code="SCHEMA_MISMATCH", message="Missing 'properties' root object in NASA response.")

        parameter_data = properties.get("parameter")
        if not parameter_data or not isinstance(parameter_data, dict):
            return NasaAdapterError(code="EMPTY_DATA", message="No 'parameter' timeseries data in NASA response.")

        return None

    def normalize(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts NASA POWER JSON structure to standard internal tabular DataFrame
        with datetime index and proper sentinels (-999 replaced with NaN).
        """
        parameter_data = payload["properties"]["parameter"]
        df = pd.DataFrame(parameter_data)

        # Index is formatted as %Y%m%d (e.g. "20240101")
        df["date"] = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
        df = df.reset_index(drop=True)

        # Clean sentinels
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce").replace([-999.0, -999, -9999.0], float("nan"))

        df["YEAR"] = df["date"].dt.year
        return df.sort_values("date").reset_index(drop=True)

    def metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts spatial, elevation, and provenance metadata from response."""
        geom = payload.get("geometry", {})
        coords = geom.get("coordinates", [None, None, None])
        header = payload.get("header", {})
        params = payload.get("parameters", {})

        return {
            "provider": self.PROVIDER,
            "latitude": coords[1] if len(coords) > 1 else None,
            "longitude": coords[0] if len(coords) > 0 else None,
            "elevation": coords[2] if len(coords) > 2 else None,
            "api_version": header.get("api_version"),
            "title": header.get("title"),
            "parameter_descriptions": params,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
