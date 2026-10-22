"""
src/validation.py
-----------------
Data Ingestion and Validation Engine.
Provides format sniffing, schema integrity verification, failure isolation,
and comprehensive validation reporting.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.load import detect_weather_columns, read_universal_csv
from src.registry import DatasetMetadata, DatasetRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Max allowed file size for memory safety (100 MB)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


@dataclass
class ValidationReport:
    """Detailed structural and domain validation report."""
    is_valid: bool
    format_detected: str
    record_count: int = 0
    column_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_summary: Dict[str, str] = field(default_factory=dict)
    weather_mappings: Dict[str, Optional[str]] = field(default_factory=dict)


def sniff_content_format(raw_bytes: bytes, filename: Optional[str] = None) -> str:
    """
    Inspects byte header and structure to identify data format
    without relying solely on file extensions.
    """
    if not raw_bytes:
        return "empty"

    # 1. Parquet Magic Number: PAR1
    if raw_bytes.startswith(b"PAR1") or raw_bytes.endswith(b"PAR1"):
        return "parquet"

    # Try decoding text snippet
    sample_text = ""
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            sample_text = raw_bytes[:4096].decode(enc).strip()
            break
        except Exception:
            continue

    if not sample_text:
        return "binary_unknown"

    # 2. JSON / JSONL
    if sample_text.startswith(("{", "[")):
        try:
            # Full or partial json check
            json.loads(raw_bytes.decode("utf-8", errors="ignore"))
            return "json"
        except Exception:
            # Check JSONL
            lines = sample_text.splitlines()
            if len(lines) > 1:
                try:
                    json.loads(lines[0])
                    json.loads(lines[1])
                    return "jsonl"
                except Exception:
                    pass

    # 3. NASA POWER Header
    if "-BEGIN HEADER-" in sample_text or "-END HEADER-" in sample_text:
        return "csv_nasa_power"

    # 4. Tab-Separated
    if sample_text.count("\t") > sample_text.count(",") and sample_text.count("\t") > 3:
        return "tsv"

    # 5. Semicolon-Separated
    if sample_text.count(";") > sample_text.count(",") and sample_text.count(";") > 3:
        return "csv_semicolon"

    # 6. Standard CSV or Text
    if "," in sample_text:
        return "csv"

    return "txt"


class DataValidator:
    """Validates ingested datasets against structural, scientific, and safety constraints."""

    @classmethod
    def validate_raw(cls, raw_bytes: bytes, filename: Optional[str] = None) -> ValidationReport:
        """Validates raw binary payload prior to full parsing."""
        errors: List[str] = []
        warnings: List[str] = []

        if len(raw_bytes) == 0:
            return ValidationReport(
                is_valid=False,
                format_detected="empty",
                errors=["File is completely empty (0 bytes)."]
            )

        if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
            errors.append(
                f"File size ({len(raw_bytes) / (1024*1024):.1f} MB) exceeds maximum allowed threshold of "
                f"{MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB."
            )

        detected_format = sniff_content_format(raw_bytes, filename)
        if detected_format == "binary_unknown":
            errors.append("Unrecognized binary format. Expected CSV, TSV, JSON, JSONL, Parquet, or TXT.")

        return ValidationReport(
            is_valid=(len(errors) == 0),
            format_detected=detected_format,
            errors=errors,
            warnings=warnings,
            metrics={"size_bytes": len(raw_bytes)},
        )

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, format_name: str = "csv") -> ValidationReport:
        """Validates parsed DataFrame for integrity, schema, and scientific metrics."""
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        if df is None or df.empty:
            return ValidationReport(
                is_valid=False,
                format_detected=format_name,
                errors=["Dataset contains 0 rows after parsing."],
            )

        row_count, col_count = df.shape
        metrics["row_count"] = row_count
        metrics["column_count"] = col_count

        if row_count < 2:
            warnings.append(f"Very small dataset ({row_count} row). Minimum recommended for time-series is 7 rows.")

        # Check date dimension
        has_date = "date" in df.columns or isinstance(df.index, pd.DatetimeIndex)
        metrics["has_date_dimension"] = has_date
        if not has_date:
            warnings.append("No explicit datetime column found; synthetic chronological timeline was inferred.")

        # Check numeric density
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        metrics["numeric_columns"] = numeric_cols
        if len(numeric_cols) == 0:
            errors.append("Dataset does not contain any numeric measurement columns.")

        # Check missingness ratio
        missing_ratios = df.isna().mean()
        high_missing_cols = [c for c, r in missing_ratios.items() if r > 0.5]
        if high_missing_cols:
            warnings.append(f"High missingness (>50% nulls) detected in columns: {high_missing_cols}")

        metrics["overall_null_pct"] = round(float(df.isna().mean().mean() * 100), 2)

        # Detect weather mapping
        weather_map = detect_weather_columns(df)
        has_weather = bool(weather_map.get("temperature") and weather_map.get("rainfall"))
        metrics["is_weather_compatible"] = has_weather
        if not has_weather:
            warnings.append("Dataset does not automatically match both Temperature and Rainfall concepts.")

        schema_summary = {str(c): str(df[c].dtype) for c in df.columns}

        return ValidationReport(
            is_valid=(len(errors) == 0),
            format_detected=format_name,
            record_count=row_count,
            column_count=col_count,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            schema_summary=schema_summary,
            weather_mappings=weather_map,
        )


@dataclass
class SingleImportResult:
    """Outcome report for a single dataset ingestion attempt."""
    filename: str
    dataset_id: Optional[str]
    name: str
    success: bool
    is_duplicate: bool = False
    validation_report: Optional[ValidationReport] = None
    error_message: Optional[str] = None
    metadata: Optional[DatasetMetadata] = None


class IngestionService:
    """
    Orchestrates validated ingestion of single and batch datasets
    with total failure isolation and atomic registry persistence.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None):
        self.registry = registry or DatasetRegistry()

    def ingest_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        display_name: Optional[str] = None,
        description: str = "",
        source_type: str = "file_upload",
        source_url: str = "",
    ) -> SingleImportResult:
        """Ingests a single byte payload with validation and failure isolation."""
        name = display_name or Path(filename).stem.replace("_", " ").title()

        # Step 1: Pre-parse validation
        raw_val = DataValidator.validate_raw(raw_bytes, filename)
        if not raw_val.is_valid:
            logger.warning(f"Raw validation failed for {filename}: {raw_val.errors}")
            return SingleImportResult(
                filename=filename,
                dataset_id=None,
                name=name,
                success=False,
                validation_report=raw_val,
                error_message="; ".join(raw_val.errors),
            )

        # Step 2: Content Parsing
        try:
            detected_fmt = raw_val.format_detected
            if detected_fmt == "parquet":
                df = pd.read_parquet(io.BytesIO(raw_bytes))
            elif detected_fmt in ["json", "jsonl"]:
                try:
                    df = pd.read_json(io.BytesIO(raw_bytes), lines=(detected_fmt == "jsonl"))
                except Exception:
                    df = pd.read_json(io.BytesIO(raw_bytes))
            else:
                # Universal CSV/TSV/NASA reader
                df = read_universal_csv(raw_bytes)
        except Exception as e:
            logger.error(f"Failed to parse content for {filename}: {e}")
            return SingleImportResult(
                filename=filename,
                dataset_id=None,
                name=name,
                success=False,
                error_message=f"File parsing error: {str(e)}",
            )

        # Step 3: DataFrame Validation
        df_val = DataValidator.validate_dataframe(df, format_name=raw_val.format_detected)
        if not df_val.is_valid:
            return SingleImportResult(
                filename=filename,
                dataset_id=None,
                name=name,
                success=False,
                validation_report=df_val,
                error_message="; ".join(df_val.errors),
            )

        # Step 4: Register into storage and metadata catalog
        try:
            meta, is_new = self.registry.register_dataset(
                name=name,
                df=df,
                raw_bytes=raw_bytes,
                raw_filename=filename,
                source=source_url or filename,
                source_type=source_type,
                description=description,
                weather_mapping=df_val.weather_mappings,
            )

            return SingleImportResult(
                filename=filename,
                dataset_id=meta.dataset_id,
                name=meta.name,
                success=True,
                is_duplicate=(not is_new),
                validation_report=df_val,
                metadata=meta,
            )
        except Exception as e:
            logger.error(f"Error persisting dataset {name}: {e}")
            return SingleImportResult(
                filename=filename,
                dataset_id=None,
                name=name,
                success=False,
                error_message=f"Storage error: {str(e)}",
            )

    def ingest_batch(
        self,
        files: List[Tuple[bytes, str]],
    ) -> List[SingleImportResult]:
        """
        Ingests multiple datasets sequentially with isolated failure boundaries.
        A failure in one dataset does not abort or corrupt siblings.
        """
        results: List[SingleImportResult] = []
        for file_bytes, filename in files:
            res = self.ingest_bytes(raw_bytes=file_bytes, filename=filename)
            results.append(res)
        return results
