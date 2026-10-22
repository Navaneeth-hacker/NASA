import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# 1. Load
with open("power.csv") as f:
    skip = next(i for i, line in enumerate(f) if "-END HEADER-" in line) + 1
df = pd.read_csv("power.csv", skiprows=skip)
df["date"] = pd.to_datetime(
    df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j"
)
df = df.sort_values("date").reset_index(drop=True)

# 2. Features (innathe data vech naale predict cheyyaan)
df["T2M_yesterday"] = df["T2M"].shift(1)
df["T2M_avg7"] = df["T2M"].rolling(7).mean()
df["rain_avg7"] = df["PRECTOTCORR"].rolling(7).mean()
df["month"] = df["date"].dt.month
df["rain_avg3"] = df["PRECTOTCORR"].rolling(3).mean()
df["rh_avg3"] = df["RH2M"].rolling(3).mean()

# 3. Target = naale mazha (1 mm-il kooduthal) undo
df["rain_next"] = df["PRECTOTCORR"].shift(-1)
df = df.dropna()
df["target"] = (df["rain_next"] > 1).astype(int)

features = ["T2M", "T2M_yesterday", "T2M_avg7",
            "PRECTOTCORR", "rain_avg7", "RH2M", "month"]

# 4. Split
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

# 5. Train
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(train[features], train["target"])

# 6. Evaluate
pred = model.predict(test[features])
baseline = (test["PRECTOTCORR"] > 1).astype(int)
print("Rain days share:", round(test["target"].mean(), 2))
print("Baseline accuracy:", accuracy_score(test["target"], baseline))
print("Model accuracy:   ", accuracy_score(test["target"], pred))