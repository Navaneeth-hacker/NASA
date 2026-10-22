"""
src/combiner.py
---------------
Dataset Selection and Combination Engine.
Supports validated time-series concatenation, feature joining, and comparative analysis
while enforcing strict compatibility checks and provenance preservation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.registry import DatasetMetadata, DatasetRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CompatibilityReport:
    """Report detailing whether two or more datasets can be safely combined."""
    is_compatible: bool
    operation: str  # 'concatenate', 'join', 'compare'
    shared_columns: List[str] = field(default_factory=list)
    unique_columns: Dict[str, List[str]] = field(default_factory=dict)
    date_overlap: Optional[Tuple[str, str]] = None
    common_date_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimated_output_rows: int = 0


class DatasetCombiner:
    """
    Validates and executes combination operations between datasets
    without mutating the original source datasets.
    """

    def __init__(self, registry: DatasetRegistry):
        self.registry = registry

    def check_concatenation_compatibility(
        self,
        meta1: DatasetMetadata,
        df1: pd.DataFrame,
        meta2: DatasetMetadata,
        df2: pd.DataFrame,
    ) -> CompatibilityReport:
        """
        Validates if two datasets can be vertically concatenated (time-series union).
        Requires compatible schemas and identifies temporal overlaps.
        """
        errors: List[str] = []
        warnings: List[str] = []

        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        shared = sorted(list(cols1.intersection(cols2)))
        only1 = sorted(list(cols1 - cols2))
        only2 = sorted(list(cols2 - cols1))

        if len(shared) == 0:
            errors.append("No shared columns found between the two datasets.")

        # Check numeric type compatibility for shared columns
        for col in shared:
            t1 = df1[col].dtype
            t2 = df2[col].dtype
            if np.issubdtype(t1, np.number) != np.issubdtype(t2, np.number):
                errors.append(f"Incompatible data types for column '{col}': {t1} vs {t2}.")

        # Check date overlap
        dates1 = set(pd.to_datetime(df1["date"]).dropna().dt.date) if "date" in df1.columns else set()
        dates2 = set(pd.to_datetime(df2["date"]).dropna().dt.date) if "date" in df2.columns else set()
        common_dates = dates1.intersection(dates2)

        overlap_range = None
        if common_dates:
            sorted_common = sorted(list(common_dates))
            overlap_range = (str(sorted_common[0]), str(sorted_common[-1]))
            warnings.append(
                f"Datasets share {len(common_dates)} overlapping dates ({overlap_range[0]} to {overlap_range[1]}). "
                f"Duplicates will be resolved by keeping the most complete observation."
            )

        if only1 or only2:
            warnings.append(
                f"Columns unique to '{meta1.name}': {only1}. "
                f"Columns unique to '{meta2.name}': {only2}. Missing values will be filled with NaN."
            )

        est_rows = len(df1) + len(df2) - len(common_dates)

        return CompatibilityReport(
            is_compatible=(len(errors) == 0),
            operation="concatenate",
            shared_columns=shared,
            unique_columns={meta1.name: only1, meta2.name: only2},
            date_overlap=overlap_range,
            common_date_count=len(common_dates),
            errors=errors,
            warnings=warnings,
            estimated_output_rows=est_rows,
        )

    def concatenate_datasets(
        self,
        dataset_ids: List[str],
        combined_name: str,
        description: str = "",
    ) -> Tuple[DatasetMetadata, pd.DataFrame]:
        """
        Vertically combines two or more datasets along chronological axis.
        Preserves original datasets and records full lineage.
        """
        if len(dataset_ids) < 2:
            raise ValueError("Concatenation requires at least two dataset IDs.")

        dfs: List[pd.DataFrame] = []
        metas: List[DatasetMetadata] = []

        for did in dataset_ids:
            meta = self.registry.get(did)
            if not meta:
                raise ValueError(f"Dataset not found: {did}")
            df = self.registry.load_clean_dataframe(did)
            if df is None or df.empty:
                raise ValueError(f"Unable to load data for dataset: {did}")
            dfs.append(df)
            metas.append(meta)

        # Pairwise compatibility check
        for i in range(len(metas) - 1):
            check = self.check_concatenation_compatibility(metas[i], dfs[i], metas[i + 1], dfs[i + 1])
            if not check.is_compatible:
                raise ValueError(f"Incompatible datasets '{metas[i].name}' and '{metas[i+1].name}': {'; '.join(check.errors)}")

        # Perform concatenation
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)

        # Deduplicate on date if present
        if "date" in combined_df.columns:
            combined_df["date"] = pd.to_datetime(combined_df["date"])
            # Keep row with fewest NaNs on duplicate dates
            combined_df["_nan_count"] = combined_df.isna().sum(axis=1)
            combined_df = combined_df.sort_values(by=["date", "_nan_count"])
            combined_df = combined_df.drop_duplicates(subset=["date"], keep="first")
            combined_df = combined_df.drop(columns=["_nan_count"]).sort_values("date").reset_index(drop=True)
            combined_df["YEAR"] = combined_df["date"].dt.year

        # Register the new combined dataset
        meta_out, _ = self.registry.register_dataset(
            name=combined_name,
            df=combined_df,
            source=f"Concatenation of: {', '.join(m.name for m in metas)}",
            source_type="combined",
            description=description or f"Combined time-series formed by concatenating {len(dataset_ids)} datasets.",
            allow_duplicate=True,
            is_combined=True,
            parent_ids=dataset_ids,
        )

        return meta_out, combined_df

    def check_join_compatibility(
        self,
        meta1: DatasetMetadata,
        df1: pd.DataFrame,
        meta2: DatasetMetadata,
        df2: pd.DataFrame,
        join_key: str = "date",
    ) -> CompatibilityReport:
        """
        Validates if two datasets can be horizontally merged (feature union)
        on a shared key (e.g., date or timestamp).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if join_key not in df1.columns:
            errors.append(f"Join key '{join_key}' not found in dataset '{meta1.name}'.")
        if join_key not in df2.columns:
            errors.append(f"Join key '{join_key}' not found in dataset '{meta2.name}'.")

        if errors:
            return CompatibilityReport(
                is_compatible=False,
                operation="join",
                errors=errors,
            )

        keys1 = set(pd.to_datetime(df1[join_key]).dt.date) if join_key == "date" else set(df1[join_key])
        keys2 = set(pd.to_datetime(df2[join_key]).dt.date) if join_key == "date" else set(df2[join_key])
        common_keys = keys1.intersection(keys2)

        if len(common_keys) == 0:
            errors.append(f"No shared {join_key} values found between '{meta1.name}' and '{meta2.name}'.")

        # Check overlapping feature column names (excluding join key)
        overlap_features = [c for c in df1.columns if c in df2.columns and c != join_key]
        if overlap_features:
            warnings.append(
                f"Features overlapping across datasets: {overlap_features}. "
                f"Suffixes will be applied to distinguish source variables."
            )

        overlap_range = None
        if common_keys and join_key == "date":
            sorted_dates = sorted(list(common_keys))
            overlap_range = (str(sorted_dates[0]), str(sorted_dates[-1]))

        return CompatibilityReport(
            is_compatible=(len(errors) == 0),
            operation="join",
            shared_columns=[join_key] + overlap_features,
            date_overlap=overlap_range,
            common_date_count=len(common_keys),
            errors=errors,
            warnings=warnings,
            estimated_output_rows=len(common_keys),
        )

    def join_datasets(
        self,
        dataset_id1: str,
        dataset_id2: str,
        combined_name: str,
        join_key: str = "date",
        how: str = "inner",
        description: str = "",
    ) -> Tuple[DatasetMetadata, pd.DataFrame]:
        """
        Horizontally merges two datasets on a shared key (e.g. date).
        Preserves original datasets and records full lineage.
        """
        meta1 = self.registry.get(dataset_id1)
        meta2 = self.registry.get(dataset_id2)
        if not meta1 or not meta2:
            raise ValueError("One or both dataset IDs could not be located in registry.")

        df1 = self.registry.load_clean_dataframe(dataset_id1)
        df2 = self.registry.load_clean_dataframe(dataset_id2)
        if df1 is None or df2 is None:
            raise ValueError("Failed to load DataFrame for joining.")

        check = self.check_join_compatibility(meta1, df1, meta2, df2, join_key=join_key)
        if not check.is_compatible:
            raise ValueError(f"Incompatible datasets for join: {'; '.join(check.errors)}")

        # Normalize join key format
        if join_key == "date":
            df1[join_key] = pd.to_datetime(df1[join_key])
            df2[join_key] = pd.to_datetime(df2[join_key])

        # Apply suffixes based on dataset names
        suffix1 = f"_{meta1.name[:6].strip().lower()}"
        suffix2 = f"_{meta2.name[:6].strip().lower()}"
        joined_df = pd.merge(df1, df2, on=join_key, how=how, suffixes=(suffix1, suffix2))

        if join_key == "date" and "date" in joined_df.columns:
            joined_df = joined_df.sort_values("date").reset_index(drop=True)
            joined_df["YEAR"] = joined_df["date"].dt.year

        meta_out, _ = self.registry.register_dataset(
            name=combined_name,
            df=joined_df,
            source=f"Join of: {meta1.name} + {meta2.name} on '{join_key}' ({how})",
            source_type="combined",
            description=description or f"Horizontal join of datasets {dataset_id1} and {dataset_id2}.",
            allow_duplicate=True,
            is_combined=True,
            parent_ids=[dataset_id1, dataset_id2],
        )

        return meta_out, joined_df
