"""
src/registry.py
---------------
Persistent Multi-Dataset Registry Architecture.
Maintains dataset identity, metadata, schema, provenance, and storage isolation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Standardized metadata record for any registered dataset."""
    dataset_id: str
    name: str
    description: str = ""
    source: str = ""
    source_type: str = "local_file"  # 'file_upload', 'nasa_api', 'local_file', 'combined'
    format: str = "csv"              # 'csv', 'parquet', 'json', 'jsonl', 'txt'
    schema: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    imported_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "ready"            # 'pending', 'importing', 'ready', 'processing', 'failed', 'archived'
    record_count: int = 0
    size_bytes: int = 0
    checksum: str = ""
    version: int = 1
    errors: List[str] = field(default_factory=list)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    raw_path: Optional[str] = None
    clean_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_combined: bool = False
    parent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetMetadata:
        # Filter to known fields to handle forward compatibility
        known_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


class DatasetRegistry:
    """
    Manages persistent multi-dataset catalog with atomic writes,
    deduplication, search, and lifecycle operations.
    """

    def __init__(self, base_dir: Union[str, Path] = "data"):
        self.base_dir = Path(base_dir).resolve()
        self.registry_dir = self.base_dir / "registry"
        self.raw_dir = self.base_dir / "raw"
        self.clean_dir = self.base_dir / "clean"
        self.combined_dir = self.base_dir / "combined"
        self.registry_file = self.registry_dir / "registry.json"

        # Ensure directory structures exist
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.combined_dir.mkdir(parents=True, exist_ok=True)

        self._datasets: Dict[str, DatasetMetadata] = {}
        self.load()

    def load(self) -> None:
        """Loads registry from disk atomically."""
        if not self.registry_file.exists():
            self._datasets = {}
            logger.info(f"Initialized new empty dataset registry at {self.registry_file}")
            return

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self._datasets = {
                d["dataset_id"]: DatasetMetadata.from_dict(d)
                for d in raw_data.get("datasets", [])
            }
            logger.info(f"Loaded {len(self._datasets)} datasets from registry.")
        except Exception as e:
            logger.error(f"Failed to load registry from {self.registry_file}: {e}")
            self._datasets = {}

    def save(self) -> None:
        """Persists registry using atomic write to avoid corruption."""
        temp_file = self.registry_dir / f"registry_{uuid.uuid4().hex[:8]}.tmp"
        payload = {
            "version": 1,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "datasets": [meta.to_dict() for meta in self._datasets.values()],
        }

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace
            os.replace(temp_file, self.registry_file)
            logger.debug(f"Atomically saved registry with {len(self._datasets)} records.")
        except Exception as e:
            logger.error(f"Failed to atomically persist registry: {e}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise

    @staticmethod
    def calculate_checksum(data: Union[bytes, str, Path]) -> str:
        """Computes SHA-256 fingerprint for deduplication."""
        hasher = hashlib.sha256()
        if isinstance(data, (str, Path)) and Path(str(data)).exists():
            with open(str(data), "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        elif isinstance(data, bytes):
            hasher.update(data)
        elif isinstance(data, str):
            hasher.update(data.encode("utf-8"))
        return hasher.hexdigest()

    def find_by_checksum(self, checksum: str) -> Optional[DatasetMetadata]:
        """Finds existing dataset with identical content fingerprint."""
        if not checksum:
            return None
        for meta in self._datasets.values():
            if meta.checksum == checksum and meta.status != "archived":
                return meta
        return None

    def register_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        raw_bytes: Optional[bytes] = None,
        raw_filename: Optional[str] = None,
        source: str = "upload",
        source_type: str = "file_upload",
        description: str = "",
        weather_mapping: Optional[Dict[str, Optional[str]]] = None,
        allow_duplicate: bool = False,
        parent_ids: Optional[List[str]] = None,
        is_combined: bool = False,
    ) -> Tuple[DatasetMetadata, bool]:
        """
        Registers a new dataset with storage isolation and metadata cataloging.
        Returns (DatasetMetadata, is_new_import).
        """
        dataset_id = str(uuid.uuid4())
        checksum = self.calculate_checksum(raw_bytes if raw_bytes else df.to_csv().encode("utf-8"))

        # Deduplication check
        if not allow_duplicate:
            existing = self.find_by_checksum(checksum)
            if existing is not None:
                logger.info(f"Duplicate content detected for dataset '{name}'. Existing ID: {existing.dataset_id}")
                return existing, False

        # Prepare isolated storage directories
        dataset_raw_dir = self.raw_dir / dataset_id
        dataset_raw_dir.mkdir(parents=True, exist_ok=True)

        raw_stored_path: Optional[str] = None
        if raw_bytes is not None:
            safe_fname = Path(raw_filename).name if raw_filename else "source_data.bin"
            raw_target = dataset_raw_dir / safe_fname
            with open(raw_target, "wb") as f:
                f.write(raw_bytes)
            raw_stored_path = str(raw_target.relative_to(self.base_dir.parent))
            stored_size = len(raw_bytes)
        else:
            raw_csv_target = dataset_raw_dir / "data.csv"
            df.to_csv(raw_csv_target, index=True)
            raw_stored_path = str(raw_csv_target.relative_to(self.base_dir.parent))
            stored_size = raw_csv_target.stat().st_size

        # Save clean Parquet artifact
        clean_target = self.clean_dir / f"{dataset_id}.parquet"
        df.to_parquet(clean_target, engine="pyarrow", compression="snappy")
        clean_stored_path = str(clean_target.relative_to(self.base_dir.parent))

        # Build schema summary
        columns_info = {
            col: str(df[col].dtype) for col in df.columns
        }
        date_start = str(df["date"].min().date()) if "date" in df.columns else (str(df.index.min().date()) if isinstance(df.index, pd.DatetimeIndex) else None)
        date_end = str(df["date"].max().date()) if "date" in df.columns else (str(df.index.max().date()) if isinstance(df.index, pd.DatetimeIndex) else None)

        schema_meta = {
            "columns": columns_info,
            "numeric_columns": [c for c in df.select_dtypes(include=["number"]).columns],
            "date_range": [date_start, date_end] if date_start and date_end else None,
            "weather_mapping": weather_mapping or {},
            "shape": list(df.shape),
        }

        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            description=description,
            source=source,
            source_type=source_type,
            format="csv" if (raw_filename and raw_filename.endswith(".csv")) else "parquet",
            schema=schema_meta,
            created_at=datetime.now(timezone.utc).isoformat(),
            imported_at=datetime.now(timezone.utc).isoformat(),
            status="ready",
            record_count=len(df),
            size_bytes=stored_size,
            checksum=checksum,
            version=1,
            errors=[],
            processing_history=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "import",
                "details": f"Imported {len(df)} rows with {len(df.columns)} columns."
            }],
            raw_path=raw_stored_path,
            clean_path=clean_stored_path,
            is_combined=is_combined,
            parent_ids=parent_ids or [],
        )

        self._datasets[dataset_id] = metadata
        self.save()
        logger.info(f"Registered new dataset '{name}' (ID: {dataset_id}). Total in registry: {len(self._datasets)}")
        return metadata, True

    def get(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Retrieves metadata by dataset_id."""
        return self._datasets.get(dataset_id)

    def load_clean_dataframe(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Loads the cleaned parquet DataFrame for a dataset ID."""
        meta = self.get(dataset_id)
        if not meta or not meta.clean_path:
            return None

        full_path = self.base_dir.parent / meta.clean_path
        if not full_path.exists():
            # Fallback check relative to clean_dir
            full_path = self.clean_dir / f"{dataset_id}.parquet"

        if not full_path.exists():
            logger.error(f"Clean parquet not found for dataset {dataset_id} at {full_path}")
            return None

        try:
            return pd.read_parquet(full_path)
        except Exception as e:
            logger.error(f"Error reading parquet for dataset {dataset_id}: {e}")
            return None

    def list_datasets(
        self,
        include_archived: bool = False,
        source_type: Optional[str] = None,
        search_query: Optional[str] = None,
        sort_by: str = "imported_at",
        ascending: bool = False,
    ) -> List[DatasetMetadata]:
        """Filters, searches, and sorts registered datasets."""
        results = list(self._datasets.values())

        if not include_archived:
            results = [d for d in results if d.status != "archived"]

        if source_type:
            results = [d for d in results if d.source_type == source_type]

        if search_query:
            q = search_query.lower()
            results = [
                d for d in results
                if q in d.name.lower() or q in d.description.lower() or q in d.dataset_id.lower()
            ]

        # Sorting
        def sort_key(meta: DatasetMetadata) -> Any:
            val = getattr(meta, sort_by, None)
            return val if val is not None else ""

        results.sort(key=sort_key, reverse=not ascending)
        return results

    def delete_dataset(self, dataset_id: str, purge_files: bool = True) -> bool:
        """Removes a dataset from registry and optionally purges storage files."""
        meta = self.get(dataset_id)
        if not meta:
            logger.warning(f"Attempted to delete non-existent dataset: {dataset_id}")
            return False

        if purge_files:
            # Purge raw dir
            raw_dir = self.raw_dir / dataset_id
            if raw_dir.exists():
                shutil.rmtree(raw_dir, ignore_errors=True)

            # Purge clean parquet
            clean_file = self.clean_dir / f"{dataset_id}.parquet"
            if clean_file.exists():
                clean_file.unlink(missing_ok=True)

        del self._datasets[dataset_id]
        self.save()
        logger.info(f"Deleted dataset {dataset_id} (Purged files: {purge_files})")
        return True

    def update_metadata(self, dataset_id: str, updates: Dict[str, Any]) -> Optional[DatasetMetadata]:
        """Surgically updates metadata fields for an existing dataset."""
        meta = self.get(dataset_id)
        if not meta:
            return None

        for k, v in updates.items():
            if hasattr(meta, k) and k not in ["dataset_id", "created_at"]:
                setattr(meta, k, v)

        meta.version += 1
        meta.processing_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "metadata_update",
            "details": f"Updated fields: {list(updates.keys())}",
        })

        self.save()
        return meta

    def seed_initial_datasets(self, root_dir: Union[str, Path] = ".") -> int:
        """
        Auto-seeds existing default datasets (power.csv, sample clean csv)
        if not already present in registry. Returns count of seeded items.
        """
        from src.load import detect_weather_columns, read_universal_csv

        root = Path(root_dir).resolve()
        seeded_count = 0

        # Seed 1: power.csv (Kollam Historical Climate)
        power_csv = root / "power.csv"
        if power_csv.exists():
            checksum = self.calculate_checksum(power_csv)
            if not self.find_by_checksum(checksum):
                try:
                    df = read_universal_csv(power_csv)
                    mapping = detect_weather_columns(df)
                    raw_b = power_csv.read_bytes()
                    self.register_dataset(
                        name="Kollam Climate Baseline (2005–2025)",
                        df=df,
                        raw_bytes=raw_b,
                        raw_filename="power.csv",
                        source="NASA POWER Daily Point API (MERRA-2)",
                        source_type="local_file",
                        description="Daily native resolution climate data for latitude 8.89, longitude 76.61 (Kollam, Kerala). Contains T2M, PRECTOTCORR, and RH2M.",
                        weather_mapping=mapping,
                    )
                    seeded_count += 1
                except Exception as e:
                    logger.error(f"Failed to seed power.csv: {e}")

        # Seed 2: data/clean/kollam_climate_sample.csv if exists
        sample_csv = root / "data" / "clean" / "kollam_climate_sample.csv"
        if sample_csv.exists():
            checksum = self.calculate_checksum(sample_csv)
            if not self.find_by_checksum(checksum):
                try:
                    df = read_universal_csv(sample_csv)
                    mapping = detect_weather_columns(df)
                    raw_b = sample_csv.read_bytes()
                    self.register_dataset(
                        name="Kollam Climate Sample (Q1 2005)",
                        df=df,
                        raw_bytes=raw_b,
                        raw_filename="kollam_climate_sample.csv",
                        source="Curated Clean Sample",
                        source_type="local_file",
                        description="100-day verified sample slice of Kollam daily climate observations with normalized fields.",
                        weather_mapping=mapping,
                    )
                    seeded_count += 1
                except Exception as e:
                    logger.error(f"Failed to seed sample csv: {e}")

        return seeded_count


# Global singleton helper
_REGISTRY_INSTANCE: Optional[DatasetRegistry] = None

def get_registry(base_dir: str = "data") -> DatasetRegistry:
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = DatasetRegistry(base_dir=base_dir)
        _REGISTRY_INSTANCE.seed_initial_datasets()
    return _REGISTRY_INSTANCE
