"""
tests/test_registry.py
----------------------
Unit and persistence tests for DatasetRegistry and DatasetMetadata.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.registry import DatasetMetadata, DatasetRegistry


class TestDatasetRegistry(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.registry = DatasetRegistry(base_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_register_and_retrieve_dataset(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "T2M": [22.0, 23.5, 24.1, 22.8, 23.0],
            "PRECTOTCORR": [0.0, 5.2, 1.1, 0.0, 0.0],
        })
        meta, is_new = self.registry.register_dataset(
            name="Station Alpha",
            df=df,
            raw_bytes=b"dummy,raw,bytes\n1,2,3",
            raw_filename="station_data.csv",
            source_type="file_upload",
            description="Test weather station dataset",
        )

        self.assertTrue(is_new)
        self.assertIsNotNone(meta.dataset_id)
        self.assertEqual(meta.name, "Station Alpha")
        self.assertEqual(meta.record_count, 5)

        # Retrieve and verify
        retrieved = self.registry.get(meta.dataset_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.dataset_id, meta.dataset_id)

        # Verify clean parquet loading
        loaded_df = self.registry.load_clean_dataframe(meta.dataset_id)
        self.assertIsNotNone(loaded_df)
        self.assertEqual(len(loaded_df), 5)
        self.assertIn("T2M", loaded_df.columns)

    def test_identical_filenames_coexist_without_overwrite(self):
        """Verify two datasets with identical filenames get distinct IDs and do not overwrite each other."""
        df1 = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "T2M": [10.0, 11.0, 12.0]})
        df2 = pd.DataFrame({"date": pd.date_range("2024-06-01", periods=3), "T2M": [30.0, 31.0, 32.0]})

        meta1, _ = self.registry.register_dataset(
            name="Station North",
            df=df1,
            raw_bytes=b"dataset_1_unique_content",
            raw_filename="climate_data.csv",
        )
        meta2, _ = self.registry.register_dataset(
            name="Station South",
            df=df2,
            raw_bytes=b"dataset_2_different_content",
            raw_filename="climate_data.csv",
        )

        self.assertNotEqual(meta1.dataset_id, meta2.dataset_id)
        self.assertEqual(len(self.registry.list_datasets()), 2)

        # Verify separate storage
        loaded1 = self.registry.load_clean_dataframe(meta1.dataset_id)
        loaded2 = self.registry.load_clean_dataframe(meta2.dataset_id)
        self.assertEqual(loaded1["T2M"].iloc[0], 10.0)
        self.assertEqual(loaded2["T2M"].iloc[0], 30.0)

    def test_deduplication_via_checksum(self):
        """Verify identical content returns existing dataset and prevents duplication."""
        df = pd.DataFrame({"val": [1, 2, 3]})
        content = b"fixed_identical_csv_content"

        meta1, is_new1 = self.registry.register_dataset("First Import", df, raw_bytes=content, raw_filename="test.csv")
        self.assertTrue(is_new1)

        meta2, is_new2 = self.registry.register_dataset("Second Import", df, raw_bytes=content, raw_filename="copy.csv")
        self.assertFalse(is_new2)
        self.assertEqual(meta1.dataset_id, meta2.dataset_id)
        self.assertEqual(len(self.registry.list_datasets()), 1)

    def test_persistence_across_reinstantiation(self):
        """Simulate application restart and verify state survives."""
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4), "val": [10, 20, 30, 40]})
        meta, _ = self.registry.register_dataset("Persist Test", df, raw_bytes=b"persisted_raw", raw_filename="persist.csv")

        # New instance pointing to same base directory
        new_registry = DatasetRegistry(base_dir=self.test_dir)
        retrieved = new_registry.get(meta.dataset_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Persist Test")
        self.assertEqual(retrieved.record_count, 4)

        loaded_df = new_registry.load_clean_dataframe(meta.dataset_id)
        self.assertEqual(len(loaded_df), 4)

    def test_search_filter_and_delete(self):
        df = pd.DataFrame({"x": [1, 2]})
        m1, _ = self.registry.register_dataset("Kollam Daily", df, raw_bytes=b"k1", source_type="local_file")
        m2, _ = self.registry.register_dataset("London Weather", df, raw_bytes=b"k2", source_type="nasa_api")

        # Search
        search_res = self.registry.list_datasets(search_query="kollam")
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0].name, "Kollam Daily")

        # Filter by source
        filter_res = self.registry.list_datasets(source_type="nasa_api")
        self.assertEqual(len(filter_res), 1)
        self.assertEqual(filter_res[0].name, "London Weather")

        # Delete m1
        deleted = self.registry.delete_dataset(m1.dataset_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.registry.get(m1.dataset_id))
        self.assertEqual(len(self.registry.list_datasets()), 1)


if __name__ == "__main__":
    unittest.main()
