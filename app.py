import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

st.set_page_config(page_title="Kollam Climate", layout="wide")
st.title("Kollam Climate Dashboard")
st.caption("Source: NASA POWER (daily, 2021-2025)")


@st.cache_data
def load_data():
    with open("power.csv") as f:
        skip = next(i for i, line in enumerate(f) if "-END HEADER-" in line) + 1
    df = pd.read_csv("power.csv", skiprows=skip)
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
        format="%Y%j",
    )
    return df


df = load_data()

years = sorted(df["YEAR"].unique())
selected = st.sidebar.multiselect("Year", years, default=years)
df_filtered = df[df["YEAR"].isin(selected)]

monthly = df_filtered.set_index("date").resample("MS").agg(
    {"T2M": "mean", "PRECTOTCORR": "sum", "RH2M": "mean"}
).reset_index()

col1, col2 = st.columns(2)
col1.plotly_chart(
    px.bar(monthly, x="date", y="PRECTOTCORR", title="Monthly rainfall (mm)"),
    use_container_width=True,
)
col2.plotly_chart(
    px.line(monthly, x="date", y="T2M", title="Monthly avg temperature (°C)"),
    use_container_width=True,
)

st.subheader("Tomorrow's rain prediction")

full = load_data().sort_values("date").reset_index(drop=True)
full["T2M_yesterday"] = full["T2M"].shift(1)
full["T2M_avg7"] = full["T2M"].rolling(7).mean()
full["rain_avg7"] = full["PRECTOTCORR"].rolling(7).mean()
full["rain_avg3"] = full["PRECTOTCORR"].rolling(3).mean()
full["rh_avg3"] = full["RH2M"].rolling(3).mean()
full["month"] = full["date"].dt.month

model = joblib.load("model.pkl")

# Use the exact features the saved model was trained on
expected_features = list(model.feature_names_in_)

last = full.iloc[[-1]]
prob = model.predict_proba(last[expected_features])[0][1]

st.metric("Chance of rain after " + str(last["date"].iloc[0].date()), f"{prob:.0%}")

st.subheader("Data")
st.dataframe(df_filtered)
st.download_button("Download CSV", df_filtered.to_csv(index=False), "kollam.csv")