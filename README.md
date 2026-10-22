# Climate Intelligence Dashboard

A flexible Streamlit dashboard for exploring climate, weather, and time-series data, with a Random Forest estimate of whether the day after the latest observation will receive more than 1 mm of rain.

**Universal Data Support:** Any CSV file thrown into the dashboard (via drag-and-drop upload or placed in the directory) will load and work seamlessly!

## Key Features

- **Universal CSV Ingestion:**
  - **Headers & Preambles:** Automatically skips NASA POWER headers (`-BEGIN HEADER-` ... `-END HEADER-`), metadata comments (`#`, `//`, `;`), or clean headerless CSVs.
  - **Delimiters:** Auto-detects comma (`,`), semicolon (`;`), tab (`\t`), and pipe (`|`).
  - **Encodings:** Supports UTF-8, UTF-8-BOM, CP1252, and Latin-1.
  - **Dates:** Resolves NASA `YEAR` + `DOY`, `YEAR` + `MO` + `DY`, ISO dates (`YYYY-MM-DD`), timestamps, or custom formats. If a CSV has no dates, a synthetic daily timeline is automatically generated so features and charts never break.
  - **Sentinels & Missing Values:** Cleans meteorological missing values (`-999`, `-999.0`, `-9999`, `9999`, `NA`, `null`, etc.) with time-series interpolation.
  - **Smart Column Detection & Mapping:** Automatically matches temperature, precipitation/rainfall, and relative humidity from common synonyms (e.g., `T2M`, `temp`, `tavg`, `PRECTOTCORR`, `rainfall`, `prcp`, `RH2M`, `humidity`), with an interactive sidebar mapping override.
- **Machine Learning Rain Prediction:**
  - Tomorrow's rain prediction using a trained Random Forest model (`model.pkl`).
  - If weather columns are present, builds rolling lag features and displays rain likelihood for the day after the last observation.
  - If weather columns are not present, gracefully displays general multi-metric exploration without crashing.
- **Interactive Visualizations:**
  - Configurable aggregation frequencies: Monthly, Weekly, or Daily.
  - Interactive Plotly charts for Rainfall, Temperature, or any custom numeric metrics in uploaded CSVs.
  - Filtering by Year or date range with Select All / Clear All controls.
- **Data Table & Export:**
  - Filtered table viewer, summary statistics (`describe()`), and one-click processed CSV download.

## Project Structure

- `app.py` — Streamlit dashboard with file upload, auto-column mapping, year filtering, charts, rain prediction, and CSV export.
- `model.py` — Script that trains and evaluates the Random Forest classifier on any weather CSV and saves to `model.pkl`.
- `model.pkl` — Serialized classifier used by the dashboard.
- `power.csv` — Default NASA POWER daily climate data (2005–2025).
- `src/` — Modular data pipeline:
  - `src/config.py`: Dataset configuration and canonical column mappings.
  - `src/load.py`: Universal CSV loader (`read_universal_csv`), column detector (`detect_weather_columns`), and API/NetCDF loaders.
  - `src/clean.py`: `UniversalDataCleaner` for missing value handling, date index enforcement, and calendar features.
  - `src/features.py`: Time-series feature engineering (lags, rolling stats, cyclical encoding).

## Running the Dashboard

Ensure dependencies are installed:

```sh
python -m pip install pandas plotly streamlit joblib scikit-learn
```

Launch the dashboard:

```sh
python -m streamlit run app.py
```

Then either:
1. Use the default `power.csv` dataset, OR
2. Drag and drop any CSV file into the **"Throw any CSV file here"** sidebar uploader, OR
3. Select any local `.csv` file from the workspace dropdown.

## Retraining the Model

To retrain the prediction model on `power.csv` or any other weather CSV:

```sh
# On default power.csv
python model.py

# Or on any other CSV file
python model.py path/to/your_weather_data.csv
```
