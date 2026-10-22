import pandas as pd
import plotly.express as px

with open("power.csv") as f:
    skip = next(i for i, line in enumerate(f) if "-END HEADER-" in line ) + 1
df = pd.read_csv("power.csv",skiprows=skip)

df["date"] = pd.to_datetime(
    df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
    format="%Y%j",
)

fig = px.line(df, x="date", y="PRECTOTCORR", title="Kollam daily rainfall (mm)")
#fig.show()
monthly = df.set_index("date").resample("MS").agg(
    {"T2M": "mean", "PRECTOTCORR": "sum", "RH2M": "mean"}
).reset_index()

print(monthly.head(12))

fig2 = px.bar(monthly, x="date", y="PRECTOTCORR",
              title="Kollam monthly rainfall (mm)")
fig2.show()

fig3 = px.line(monthly, x="date", y="T2M",
               title="Kollam monthly avg temperature (°C)")
fig3.show()