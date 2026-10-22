"""
tests/test_app_integration.py
-----------------------------
Integration tests for Streamlit application using AppTest.
"""

import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

from src.registry import get_registry


class TestAppIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure registry is initialized and seeded
        cls.registry = get_registry(base_dir="data")
        cls.app_path = str(Path(r"C:\Users\Navan\Downloads\NASA\app.py").resolve())

    def test_app_initial_render_and_metrics(self):
        at = AppTest.from_file(self.app_path)
        at.run(timeout=30)

        # Verify no unhandled Python exceptions
        self.assertEqual(len(at.exception), 0, f"App threw exceptions: {at.exception}")

        # Verify essential KPI metrics rendered
        metric_labels = [m.label for m in at.metric]
        self.assertIn("Total Records", metric_labels)
        self.assertIn("Date Span", metric_labels)

        # Check prediction metric presence
        has_prediction = any("chance of rain" in l.lower() for l in metric_labels)
        self.assertTrue(has_prediction, f"Expected rain prediction metric in {metric_labels}")

    def test_upload_new_csv_via_sidebar(self):
        at = AppTest.from_file(self.app_path)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        csv_content = b"date,temperature,rainfall,humidity\n2024-05-01,27.5,0.0,65.0\n2024-05-02,28.0,15.2,78.0\n2024-05-03,26.8,4.1,72.0\n"
        at.sidebar.file_uploader[0].upload("new_test_station.csv", csv_content)
        at.run(timeout=30)

        self.assertEqual(len(at.exception), 0)
        # Verify total records reflects new file or updated state
        metric_dict = {m.label: m.value for m in at.metric}
        self.assertIn("Total Records", metric_dict)

    def test_multi_dataset_tab_renders_inventory(self):
        at = AppTest.from_file(self.app_path)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        # Check that multiselect for datasets exists and can be queried
        multiselects = at.multiselect
        self.assertTrue(len(multiselects) > 0)


if __name__ == "__main__":
    unittest.main()
