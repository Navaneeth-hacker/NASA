"""
tests/test_validation.py
------------------------
Unit tests for data validation, format sniffing, and batch ingestion failure isolation.
"""

import shutil
import tempfile
import unittest
import pandas as pd

from src.registry import DatasetRegistry
from src.validation import DataValidator, IngestionService, sniff_content_format


class TestValidationAndIngestion(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.registry = DatasetRegistry(base_dir=self.test_dir)
        self.service = IngestionService(self.registry)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sniff_content_format(self):
        self.assertEqual(sniff_content_format(b""), "empty")
        self.assertEqual(sniff_content_format(b"PAR1somebinarydataPAR1"), "parquet")
        self.assertEqual(sniff_content_format(b'{"key": "value", "rows": [1, 2]}'), "json")
        self.assertEqual(sniff_content_format(b'{"row": 1}\n{"row": 2}\n'), "jsonl")
        self.assertEqual(sniff_content_format(b"-BEGIN HEADER-\nNASA DATA\n-END HEADER-\nYEAR,DOY,T2M"), "csv_nasa_power")
        self.assertEqual(sniff_content_format(b"col1\tcol2\tcol3\tcol4\n1\t2\t3\t4\n"), "tsv")
        self.assertEqual(sniff_content_format(b"date;temperature;rainfall;humidity\n2024-01-01;22.5;0.0;65"), "csv_semicolon")
        self.assertEqual(sniff_content_format(b"date,temperature,rainfall\n2024-01-01,22.5,0.0"), "csv")

    def test_empty_and_malformed_rejection(self):
        # Empty file rejection
        rep_empty = DataValidator.validate_raw(b"", "empty.csv")
        self.assertFalse(rep_empty.is_valid)
        self.assertIn("empty", rep_empty.errors[0].lower())

        # No numeric columns rejection
        df_no_num = pd.DataFrame({"names": ["Alice", "Bob"], "status": ["active", "pending"]})
        rep_df = DataValidator.validate_dataframe(df_no_num)
        self.assertFalse(rep_df.is_valid)
        self.assertTrue(any("numeric" in err for err in rep_df.errors))

    def test_batch_ingestion_failure_isolation(self):
        """
        Verify that in a batch with valid and invalid datasets:
        1. Valid datasets are registered successfully.
        2. Corrupted dataset is cleanly rejected with actionable explanation.
        3. Registry remains consistent and valid files are fully accessible.
        """
        batch = [
            (b"date,temp,rain\n2024-01-01,25.0,0.0\n2024-01-02,26.0,1.5\n", "station_1.csv"),
            (b"broken_header_with_no_rows", "corrupt_data.csv"),
            (b"date,T2M,PRECTOTCORR\n2024-01-01,22.0,10.0\n2024-01-02,23.0,0.0\n", "station_2.csv"),
        ]

        results = self.service.ingest_batch(batch)
        self.assertEqual(len(results), 3)

        # File 1: Success
        self.assertTrue(results[0].success)
        self.assertIsNotNone(results[0].dataset_id)

        # File 2: Clean Failure
        self.assertFalse(results[1].success)
        self.assertIsNone(results[1].dataset_id)
        self.assertIsNotNone(results[1].error_message)

        # File 3: Success despite File 2 failure
        self.assertTrue(results[2].success)
        self.assertIsNotNone(results[2].dataset_id)

        # Registry contains exactly 2 valid datasets
        all_d = self.registry.list_datasets()
        self.assertEqual(len(all_d), 2)
        valid_ids = [d.dataset_id for d in all_d]
        self.assertIn(results[0].dataset_id, valid_ids)
        self.assertIn(results[2].dataset_id, valid_ids)


if __name__ == "__main__":
    unittest.main()
