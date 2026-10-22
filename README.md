# Kollam Climate Dashboard

A Streamlit dashboard for exploring daily climate data for Kollam, Kerala, with a Random Forest estimate of whether the day after the latest observation will receive more than 1 mm of rain.

## Current project contents

- `app.py` — Streamlit dashboard with year filtering, monthly rainfall and temperature charts, a rain-probability metric, a data table, and CSV download.
- `model.py` — script that builds features, trains and evaluates a Random Forest classifier, and saves it to `model.pkl`.
- `model.pkl` — serialized classifier used by the dashboard.
- `power.csv` — daily NASA POWER data from January 1, 2005 through December 31, 2025.

The dataset is for latitude 8.89, longitude 76.61, and contains daily temperature at 2 m (`T2M`), corrected precipitation (`PRECTOTCORR`), and relative humidity at 2 m (`RH2M`). NASA POWER metadata is included at the beginning of the CSV; the application skips it before reading the tabular data.

## Run the dashboard

Use Python with the packages `pandas`, `plotly`, `streamlit`, `joblib`, and `scikit-learn` installed. From the project directory, run:

```sh
python -m pip install pandas plotly streamlit joblib scikit-learn
python -m streamlit run app.py
```

The dashboard defaults to all years in the dataset. Use the sidebar to select years; the selection filters the charts, table, and downloaded CSV.

## Model and prediction details

The training script sorts observations by date and predicts whether the following day's precipitation exceeds 1 mm. It uses a chronological 80/20 train/test split and prints the test-set rain-day share, a simple baseline accuracy, model accuracy, and confusion matrix before saving the trained model to `model.pkl`.

The dashboard creates rolling weather features and uses the saved model's feature names to select its inputs. Its rain probability is calculated for the day after the latest row in `power.csv`. **The current CSV ends on December 31, 2025, so the displayed estimate is for the day after that observation—not a live forecast for today's tomorrow.** Update the data and retrain or otherwise validate the model before treating the displayed value as a current forecast.

## Data source

NASA Prediction Of Worldwide Energy Resources (POWER), daily native-resolution data. The source metadata in `power.csv` includes the location, date range, parameters, and missing-data sentinel (`-999`).
