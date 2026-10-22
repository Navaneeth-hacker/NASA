"""
tests/test_combiner.py
----------------------
Unit tests for dataset combination, compatibility checking, joins, and provenance preservation.
"""

import shutil
import tempfile
import unittest
import pandas as pd

from src.combiner import DatasetCombiner
from src.registry import DatasetRegistry


class TestDatasetCombiner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.registry = DatasetRegistry(base_dir=self.test_dir)
        self.combiner = DatasetCombiner(self.registry)

        # Dataset A: Kollam 2023 H1
        self.df_a = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-01-05"),
            "T2M": [24.0, 24.5, 25.0, 24.8, 25.2],
            "PRECTOTCORR": [0.0, 0.0, 1.2, 0.0, 5.0],
        })
        self.meta_a, _ = self.registry.register_dataset(
            name="Kollam Period 1",
            df=self.df_a,
            raw_bytes=b"raw_a",
            raw_filename="kollam_p1.csv",
        )

        # Dataset B: Kollam 2023 H2 (with 1 overlapping date: 2023-01-05)
        self.df_b = pd.DataFrame({
            "date": pd.date_range("2023-01-05", "2023-01-09"),
            "T2M": [25.2, 25.5, 26.0, 25.8, 26.1],
            "PRECTOTCORR": [5.0, 2.0, 0.0, 0.0, 0.0],
        })
        self.meta_b, _ = self.registry.register_dataset(
            name="Kollam Period 2",
            df=self.df_b,
            raw_bytes=b"raw_b",
            raw_filename="kollam_p2.csv",
        )

        # Dataset C: Non-climate unmatchable schema
        self.df_c = pd.DataFrame({
            "product_id": ["P1", "P2", "P3"],
            "unit_price": [19.99, 29.99, 49.99],
        })
        self.meta_c, _ = self.registry.register_dataset(
            name="Store Catalog",
            df=self.df_c,
            raw_bytes=b"raw_c",
            raw_filename="catalog.csv",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_concatenation_and_deduplication(self):
        """Test concatenating period 1 and period 2 with automatic overlap resolution."""
        check = self.combiner.check_concatenation_compatibility(
            self.meta_a, self.df_a, self.meta_b, self.df_b
        )
        self.assertTrue(check.is_compatible)
        self.assertEqual(check.common_date_count, 1)

        meta_comb, df_comb = self.combiner.concatenate_datasets(
            dataset_ids=[self.meta_a.dataset_id, self.meta_b.dataset_id],
            combined_name="Kollam Full Period",
        )

        # 5 rows + 5 rows - 1 overlapping date = 9 unique dates
        self.assertEqual(len(df_comb), 9)
        self.assertTrue(meta_comb.is_combined)
        self.assertIn(self.meta_a.dataset_id, meta_comb.parent_ids)
        self.assertIn(self.meta_b.dataset_id, meta_comb.parent_ids)

        # Verify original datasets remain unchanged (immutability)
        loaded_a = self.registry.load_clean_dataframe(self.meta_a.dataset_id)
        loaded_b = self.registry.load_clean_dataframe(self.meta_b.dataset_id)
        self.assertEqual(len(loaded_a), 5)
        self.assertEqual(len(loaded_b), 5)

    def test_rejection_of_incompatible_concatenation(self):
        """Verify incompatible schemas produce failure diagnostics and refuse concatenation."""
        check = self.combiner.check_concatenation_compatibility(
            self.meta_a, self.df_a, self.meta_c, self.df_c
        )
        self.assertFalse(check.is_compatible)
        self.assertTrue(len(check.errors) > 0)

        with self.assertRaises(ValueError):
            self.combiner.concatenate_datasets(
                dataset_ids=[self.meta_a.dataset_id, self.meta_c.dataset_id],
                combined_name="Invalid Concat",
            )

    def test_join_datasets_on_date(self):
        """Test horizontally joining two parameter streams on matching date keys."""
        # Create humidity stream for the overlapping dates
        df_humidity = pd.DataFrame({
            "date": pd.date_range("2023-01-03", "2023-01-06"),
            "RH2M": [75.0, 78.0, 82.0, 80.0],
            "WS2M": [2.5, 3.1, 2.8, 3.0],
        })
        meta_h, _ = self.registry.register_dataset("Humidity Stream", df_humidity, raw_bytes=b"rh_raw")

        check = self.combiner.check_join_compatibility(self.meta_a, self.df_a, meta_h, df_humidity, join_key="date")
        self.assertTrue(check.is_compatible)
        # Shared dates between 2023-01-01..05 and 2023-01-03..06 = 3 dates (03, 04, 05)
        self.assertEqual(check.common_date_count, 3)

        meta_joined, df_joined = self.combiner.join_datasets(
            dataset_id1=self.meta_a.dataset_id,
            dataset_id2=meta_h.dataset_id,
            combined_name="Kollam Temp + Humidity",
            join_key="date",
            how="inner",
        )

        self.assertEqual(len(df_joined), 3)
        self.assertIn("RH2M", df_joined.columns)
        self.assertIn("T2M", df_joined.columns)
        self.assertTrue(meta_joined.is_combined)
        self.assertEqual(len(meta_joined.parent_ids), 2)


if __name__ == "__main__":
    unittest.main()
