"""
tests/test_adapters.py
----------------------
Unit tests for the NASA POWER API Adapter contract and normalization.
"""

import unittest
import pandas as pd

from src.adapters.nasa_power import NasaPowerAdapter, NasaAdapterError


class TestNasaPowerAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = NasaPowerAdapter()

    def test_discovery_contract(self):
        disc = self.adapter.discover()
        self.assertEqual(disc["provider"], "nasa_power")
        self.assertIn("supported_parameters", disc)
        self.assertIn("T2M", disc["supported_parameters"])
        self.assertIn("PRECTOTCORR", disc["supported_parameters"])

    def test_coordinate_validation(self):
        # Invalid Latitude > 90
        _, err1 = self.adapter.fetch(latitude=95.0, longitude=76.0, start_date="20230101", end_date="20230102")
        self.assertIsNotNone(err1)
        self.assertEqual(err1.code, "INVALID_COORDINATES")

        # Invalid Longitude < -180
        _, err2 = self.adapter.fetch(latitude=8.0, longitude=-185.0, start_date="20230101", end_date="20230102")
        self.assertIsNotNone(err2)
        self.assertEqual(err2.code, "INVALID_COORDINATES")

    def test_payload_validation_and_normalization(self):
        sample_payload = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [76.61, 8.89, 211.5]},
            "header": {"title": "NASA/POWER Daily Data", "api_version": "v1.0"},
            "parameters": {"T2M": "Temperature at 2m", "PRECTOTCORR": "Precipitation"},
            "properties": {
                "parameter": {
                    "T2M": {"20240101": 25.5, "20240102": 26.1, "20240103": -999.0},
                    "PRECTOTCORR": {"20240101": 0.0, "20240102": 12.5, "20240103": 0.0},
                }
            }
        }

        # Validation passes
        val_err = self.adapter.validate(sample_payload)
        self.assertIsNone(val_err)

        # Normalization
        df = self.adapter.normalize(sample_payload)
        self.assertEqual(len(df), 3)
        self.assertIn("T2M", df.columns)
        self.assertIn("PRECTOTCORR", df.columns)
        self.assertIn("date", df.columns)
        self.assertIn("YEAR", df.columns)

        # Verify -999.0 was cleaned to NaN
        self.assertTrue(pd.isna(df["T2M"].iloc[2]))

        # Metadata extraction
        meta = self.adapter.metadata(sample_payload)
        self.assertEqual(meta["latitude"], 8.89)
        self.assertEqual(meta["longitude"], 76.61)
        self.assertEqual(meta["elevation"], 211.5)

    def test_malformed_payload_rejection(self):
        bad_payload = {"type": "Feature", "properties": {}}
        err = self.adapter.validate(bad_payload)
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "EMPTY_DATA")


if __name__ == "__main__":
    unittest.main()
