import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.load import detect_weather_columns, read_universal_csv

# Page configuration
st.set_page_config(
    page_title="Climate & Weather Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌦️ Climate & Weather Intelligence Dashboard")


# 1. Cached loaders
@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return read_universal_csv(file_bytes)


@st.cache_data(show_spinner=False)
def load_data_from_path(file_path: str) -> pd.DataFrame:
    return read_universal_csv(file_path)


# 2. Sidebar: File Ingestion ("Throw any CSV file here")
st.sidebar.header("📁 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Throw any CSV file here",
    type=["csv", "txt", "tsv"],
    help="Upload any CSV: NASA POWER data, weather observations, or any custom tabular CSV.",
)

# Discover local CSV files in the workspace
local_csvs = sorted([p.name for p in ROOT_DIR.glob("*.csv")] + [p.relative_to(ROOT_DIR).as_posix() for p in ROOT_DIR.glob("data/**/*.csv")])
if "power.csv" in local_csvs:
    local_csvs.remove("power.csv")
    local_csvs.insert(0, "power.csv")

active_source_name = "power.csv"
df = None

if uploaded_file is not None:
    try:
        active_source_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        df = load_data_from_bytes(file_bytes, active_source_name)
        st.sidebar.success(f"Loaded uploaded file: **{active_source_name}**")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded CSV: {e}")
elif local_csvs:
    selected_local = st.sidebar.selectbox(
        "Or choose a local dataset:",
        local_csvs,
        index=0,
        help="Select a local CSV file stored in this workspace."
    )
    active_source_name = selected_local
    try:
        df = load_data_from_path(str(ROOT_DIR / selected_local))
    except Exception as e:
        st.sidebar.error(f"Error reading {selected_local}: {e}")

# If still no DataFrame loaded, fallback to empty
if df is None or df.empty:
    st.warning("⚠️ No data available. Please upload or provide a valid CSV file.")
    st.stop()

# Caption & Metadata
is_inferred = df.attrs.get("date_inferred", False)
date_min_str = str(df["date"].min().date()) if "date" in df.columns else "N/A"
date_max_str = str(df["date"].max().date()) if "date" in df.columns else "N/A"

st.caption(
    f"Active Dataset: **{active_source_name}** | Observations: **{len(df):,}** | "
    f"Span: **{date_min_str}** to **{date_max_str}**"
    + (" *(synthetic timeline inferred)*" if is_inferred else "")
)

# 3. Dynamic Column Mapping
all_numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["YEAR", "month", "day", "day_of_year", "day_of_week", "is_weekend", "quarter"]]
detected = detect_weather_columns(df)

with st.sidebar.expander("⚙️ Column Mapping & Detection", expanded=False):
    st.write("The system auto-detects column names. You can adjust the mappings below:")
    
    # Temperature selection
    temp_default = detected["temperature"] if detected["temperature"] in df.columns else None
    temp_idx = ([None] + all_numeric_cols).index(temp_default) if temp_default in all_numeric_cols else 0
    selected_temp = st.selectbox("Temperature Column", [None] + all_numeric_cols, index=temp_idx)

    # Rainfall selection
    rain_default = detected["rainfall"] if detected["rainfall"] in df.columns else None
    rain_idx = ([None] + all_numeric_cols).index(rain_default) if rain_default in all_numeric_cols else 0
    selected_rain = st.selectbox("Rainfall / Precipitation Column", [None] + all_numeric_cols, index=rain_idx)

    # Humidity selection
    rh_default = detected["humidity"] if detected["humidity"] in df.columns else None
    rh_idx = ([None] + all_numeric_cols).index(rh_default) if rh_default in all_numeric_cols else 0
    selected_rh = st.selectbox("Relative Humidity Column", [None] + all_numeric_cols, index=rh_idx)

# 4. Sidebar: Filtering by Year or Date Range
st.sidebar.subheader("🔍 Filters")
years = sorted(df["YEAR"].dropna().unique().tolist()) if "YEAR" in df.columns else []

if len(years) > 1:
    col_btn1, col_btn2 = st.sidebar.columns(2)
    select_all = col_btn1.button("Select All", use_container_width=True)
    clear_all = col_btn2.button("Clear All", use_container_width=True)
    
    if select_all:
        default_years = years
    elif clear_all:
        default_years = [years[-1]]
    else:
        default_years = years

    selected_years = st.sidebar.multiselect("Select Years:", years, default=default_years)
    if not selected_years:
        selected_years = years
    df_filtered = df[df["YEAR"].isin(selected_years)].copy()
else:
    df_filtered = df.copy()

# Date range display
if df_filtered.empty:
    st.warning("No data matches the selected filter. Resetting to all records.")
    df_filtered = df.copy()

# 5. Top KPI Summary Metrics
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Total Records", f"{len(df_filtered):,}")
mcol2.metric("Date Span", f"{df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")

if selected_temp and selected_temp in df_filtered.columns:
    avg_t = df_filtered[selected_temp].mean()
    mcol3.metric("Avg Temperature", f"{avg_t:.1f} °C" if pd.notna(avg_t) else "N/A")
elif all_numeric_cols:
    col_name = all_numeric_cols[0]
    mcol3.metric(f"Mean ({col_name})", f"{df_filtered[col_name].mean():.2f}")
else:
    mcol3.metric("Numeric Metrics", "0")

if selected_rain and selected_rain in df_filtered.columns:
    tot_rain = df_filtered[selected_rain].sum()
    mcol4.metric("Total Rainfall", f"{tot_rain:.1f} mm" if pd.notna(tot_rain) else "N/A")
elif len(all_numeric_cols) > 1:
    col_name = all_numeric_cols[1]
    mcol4.metric(f"Mean ({col_name})", f"{df_filtered[col_name].mean():.2f}")
else:
    mcol4.metric("Total Columns", f"{len(df_filtered.columns)}")

st.divider()

# 6. Visualizations Section
st.subheader("📈 Visual Exploration")

freq_option = st.radio(
    "Aggregation Frequency:",
    ["Monthly", "Weekly", "Daily"],
    index=0,
    horizontal=True,
)
freq_map = {"Monthly": "MS", "Weekly": "W", "Daily": "D"}
freq_code = freq_map[freq_option]

# Check if we can display standard weather charts
has_weather_charts = bool(selected_rain and selected_temp and selected_rain in df_filtered.columns and selected_temp in df_filtered.columns)

if has_weather_charts:
    # Build resampled aggregated view
    agg_dict = {
        selected_temp: "mean",
        selected_rain: "sum",
    }
    if selected_rh and selected_rh in df_filtered.columns:
        agg_dict[selected_rh] = "mean"

    try:
        resampled = (
            df_filtered.set_index("date")
            .resample(freq_code)
            .agg(agg_dict)
            .reset_index()
            .dropna(subset=[selected_temp, selected_rain], how="all")
        )
    except Exception:
        resampled = df_filtered.copy()

    vcol1, vcol2 = st.columns(2)
    fig_rain = px.bar(
        resampled,
        x="date",
        y=selected_rain,
        title=f"{freq_option} Rainfall ({selected_rain}) [mm]",
        labels={"date": "Date", selected_rain: "Rainfall (mm)"},
        color_discrete_sequence=["#1f77b4"],
    )
    vcol1.plotly_chart(fig_rain, use_container_width=True)

    fig_temp = px.line(
        resampled,
        x="date",
        y=selected_temp,
        title=f"{freq_option} Average Temperature ({selected_temp}) [°C]",
        labels={"date": "Date", selected_temp: "Temperature (°C)"},
        color_discrete_sequence=["#ff7f0e"],
    )
    vcol2.plotly_chart(fig_temp, use_container_width=True)

else:
    # Generalized interactive exploration for ANY CSV
    st.info("💡 Weather metrics not fully mapped. Showing general multi-metric explorer for detected numeric columns.")
    if all_numeric_cols:
        picked_cols = st.multiselect(
            "Select columns to visualize:",
            options=all_numeric_cols,
            default=all_numeric_cols[:min(3, len(all_numeric_cols))],
        )
        if picked_cols:
            try:
                resampled = (
                    df_filtered.set_index("date")
                    .resample(freq_code)[picked_cols]
                    .mean()
                    .reset_index()
                )
            except Exception:
                resampled = df_filtered.copy()

            gcol1, gcol2 = st.columns(2)
            fig_line = px.line(
                resampled,
                x="date",
                y=picked_cols,
                title=f"{freq_option} Average Trends",
                labels={"date": "Date", "value": "Metric Value"},
            )
            gcol1.plotly_chart(fig_line, use_container_width=True)

            fig_bar = px.bar(
                resampled,
                x="date",
                y=picked_cols[0],
                title=f"{freq_option} Distribution: {picked_cols[0]}",
                labels={"date": "Date", picked_cols[0]: picked_cols[0]},
            )
            gcol2.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No numeric columns found in this CSV to chart.")

# 7. Machine Learning: Tomorrow's Rain Prediction
st.divider()
st.subheader("🤖 Tomorrow's Rain Prediction")

if selected_temp and selected_rain and selected_temp in df.columns and selected_rain in df.columns:
    try:
        model_path = ROOT_DIR / "model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            
            # Prepare chronological features from full unsegmented dataset
            full = df.sort_values("date").copy().reset_index(drop=True)
            full["T2M"] = pd.to_numeric(full[selected_temp], errors="coerce").interpolate().bfill().ffill()
            full["PRECTOTCORR"] = pd.to_numeric(full[selected_rain], errors="coerce").fillna(0.0)
            
            if selected_rh and selected_rh in full.columns:
                full["RH2M"] = pd.to_numeric(full[selected_rh], errors="coerce").interpolate().bfill().ffill()
            else:
                full["RH2M"] = 70.0  # Default neutral relative humidity
            
            # Lag and rolling features (safe with min_periods=1)
            full["T2M_yesterday"] = full["T2M"].shift(1).bfill().ffill()
            full["T2M_avg7"] = full["T2M"].rolling(7, min_periods=1).mean().bfill().ffill()
            full["rain_avg7"] = full["PRECTOTCORR"].rolling(7, min_periods=1).mean().bfill().ffill()
            full["rain_avg3"] = full["PRECTOTCORR"].rolling(3, min_periods=1).mean().bfill().ffill()
            full["rh_avg3"] = full["RH2M"].rolling(3, min_periods=1).mean().bfill().ffill()
            full["month"] = full["date"].dt.month

            expected_features = list(getattr(model, "feature_names_in_", [
                "T2M", "T2M_yesterday", "T2M_avg7", "PRECTOTCORR", "rain_avg7", "RH2M", "month"
            ]))

            last_row = full.iloc[[-1]]
            prob = model.predict_proba(last_row[expected_features])[0][1]
            last_date_str = str(last_row["date"].iloc[0].date())

            pcol1, pcol2 = st.columns([1, 2])
            with pcol1:
                st.metric(
                    label=f"Chance of rain after {last_date_str}",
                    value=f"{prob:.0%}",
                    delta="Rain Expected (>1mm)" if prob >= 0.5 else "Dry / Low Rain",
                    delta_color="normal" if prob >= 0.5 else "inverse",
                )
            with pcol2:
                if prob >= 0.5:
                    st.success(f"🌧️ **High Rain Likelihood ({prob:.1%})**: Precipitation (>1mm) is likely on the next day.")
                elif prob >= 0.25:
                    st.info(f"⛅ **Moderate Rain Likelihood ({prob:.1%})**: Possible light rain or isolated showers.")
                else:
                    st.write(f"☀️ **Low Rain Likelihood ({prob:.1%})**: Mostly dry weather anticipated.")
                
                if not (selected_rh and selected_rh in df.columns):
                    st.caption("ℹ️ *Relative humidity column not found; neutral estimate (70%) was used for inference.*")
        else:
            st.warning("Model file 'model.pkl' not found. Run 'python model.py' to train and save the model.")
    except Exception as e:
        st.error(f"Prediction could not be completed on this dataset: {e}")
else:
    st.info(
        "ℹ️ **Rain Prediction is disabled for this dataset.**\n\n"
        "The predictive model requires Temperature and Rainfall measurements. "
        "If your CSV contains them under different names, assign them under **'⚙️ Column Mapping & Detection'** in the sidebar."
    )

# 8. Data Explorer & Download
st.divider()
st.subheader("📋 Data Explorer")

tab1, tab2 = st.tabs(["Data Table", "Summary Statistics"])

with tab1:
    st.dataframe(df_filtered, use_container_width=True)
    clean_csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    export_filename = f"cleaned_{Path(active_source_name).stem}.csv"
    st.download_button(
        label="📥 Download Processed CSV",
        data=clean_csv_bytes,
        file_name=export_filename,
        mime="text/csv",
    )

with tab2:
    num_df = df_filtered.select_dtypes(include=[np.number])
    if not num_df.empty:
        st.dataframe(num_df.describe())
    else:
        st.dataframe(df_filtered.describe().astype(str))
