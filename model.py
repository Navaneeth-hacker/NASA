import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from src.load import read_universal_csv, detect_weather_columns

# 1. Load from argument or default power.csv
csv_target = sys.argv[1] if len(sys.argv) > 1 else "power.csv"
print(f"Loading data from: {csv_target}")

df = read_universal_csv(csv_target)
df = df.sort_values("date").reset_index(drop=True)

# 2. Identify weather columns dynamically
cols = detect_weather_columns(df)
temp_col = cols["temperature"] or "T2M"
rain_col = cols["rainfall"] or "PRECTOTCORR"
rh_col = cols["humidity"] or ("RH2M" if "RH2M" in df.columns else None)

print(f"Mapped columns -> Temperature: '{temp_col}', Rainfall: '{rain_col}', Humidity: '{rh_col}'")

df["T2M"] = df[temp_col].astype(float)
df["PRECTOTCORR"] = df[rain_col].astype(float)
if rh_col and rh_col in df.columns:
    df["RH2M"] = df[rh_col].astype(float)
else:
    df["RH2M"] = 70.0

# Impute missing if any
df["T2M"] = df["T2M"].interpolate().bfill().ffill()
df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0.0)
df["RH2M"] = df["RH2M"].interpolate().bfill().ffill()

# 3. Features (innathe data vech naale predict cheyyaan)
df["T2M_yesterday"] = df["T2M"].shift(1)
df["T2M_avg7"] = df["T2M"].rolling(7, min_periods=1).mean()
df["rain_avg7"] = df["PRECTOTCORR"].rolling(7, min_periods=1).mean()
df["rain_avg3"] = df["PRECTOTCORR"].rolling(3, min_periods=1).mean()
df["rh_avg3"] = df["RH2M"].rolling(3, min_periods=1).mean()
df["month"] = df["date"].dt.month

# 4. Target = naale mazha (1 mm-il kooduthal) undo
df["rain_next"] = df["PRECTOTCORR"].shift(-1)
df = df.dropna(subset=["rain_next"]).bfill().ffill()
df["target"] = (df["rain_next"] > 1.0).astype(int)

features = ["T2M", "T2M_yesterday", "T2M_avg7",
            "PRECTOTCORR", "rain_avg7", "RH2M", "month"]

# 5. Split
split = max(1, int(len(df) * 0.8))
train, test = df.iloc[:split], df.iloc[split:]

# 6. Train
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(train[features], train["target"])

# 7. Evaluate
if len(test) > 0:
    pred = model.predict(test[features])
    baseline = (test["PRECTOTCORR"] > 1).astype(int)
    print("Rain days share:  ", round(test["target"].mean(), 2))
    print("Baseline accuracy:", round(accuracy_score(test["target"], baseline), 4))
    print("Model accuracy:   ", round(accuracy_score(test["target"], pred), 4))
    print("Confusion matrix:\n", confusion_matrix(test["target"], pred))

joblib.dump(model, "model.pkl")
print("Model saved successfully to model.pkl")
