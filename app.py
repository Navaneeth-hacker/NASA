import pandas as pd
import plotly.express as px
import streamlit as st

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
df = df[df["YEAR"].isin(selected)]

monthly = df.set_index("date").resample("MS").agg(
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

st.subheader("Data")
st.dataframe(df)
st.download_button("Download CSV", df.to_csv(index=False), "kollam.csv")