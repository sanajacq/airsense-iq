"""
AirSense-IQ — Prediction Module  (FIXED VERSION — 14 Mar 2026)
Fixes applied:
  1. AQI now calculated from ALL pollutants (PM2.5, PM10, NO2, SO2, CO, Ozone)
     and takes the MAXIMUM — matching how CPCB actually calculates AQI.
  2. get_station_summary() now reads each station's actual latest row from the
     dataset instead of using random noise offsets from a single row.
  3. forecast_24h() and forecast_7day() now accept a station parameter so each
     station gets its own independent prediction.
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR   = "models"
DATA_PATH    = "data/Agra_AirQuality_cleaned_features.csv"
SEQUENCE_LEN = 30
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGETS      = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

# ── LSTM Architecture (must match train_models.py) ────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Load Models ───────────────────────────────────────────────────
def load_models():
    with open(f"{MODELS_DIR}/xgboost_models.pkl", "rb") as f:
        xgb_models = pickle.load(f)
    with open(f"{MODELS_DIR}/lstm_scalers.pkl", "rb") as f:
        lstm_scalers = pickle.load(f)
    with open(f"{MODELS_DIR}/dataset_summary.pkl", "rb") as f:
        summary = pickle.load(f)

    lstm_state = torch.load(f"{MODELS_DIR}/lstm_models.pt",
                            map_location=DEVICE)
    lstm_models = {}
    for target in TARGETS:
        if target in lstm_state:
            model = LSTMForecaster(1, LSTM_HIDDEN, LSTM_LAYERS, 7).to(DEVICE)
            model.load_state_dict(lstm_state[target])
            model.eval()
            lstm_models[target] = model

    return xgb_models, lstm_models, lstm_scalers, summary

# ── Prepare Latest Features for XGBoost ──────────────────────────
def prepare_xgb_features(df, xgb_models):
    df = df.copy()
    df["Date"]       = pd.to_datetime(df["Date"])
    df               = df.sort_values("Date").reset_index(drop=True)
    df["DayOfYear"]  = df["Date"].dt.dayofyear
    df["Month"]      = df["Date"].dt.month
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["Quarter"]    = df["Date"].dt.quarter
    df["IsWinter"]   = df["Month"].isin([11, 12, 1, 2]).astype(int)
    df["IsSummer"]   = df["Month"].isin([4, 5, 6]).astype(int)

    for target in TARGETS:
        if target in df.columns:
            df[f"{target}_lag1"]  = df[target].shift(1)
            df[f"{target}_lag3"]  = df[target].shift(3)
            df[f"{target}_lag7"]  = df[target].shift(7)
            df[f"{target}_roll7"] = df[target].rolling(7).mean()
            df[f"{target}_roll14"]= df[target].rolling(14).mean()

    df = df.dropna().reset_index(drop=True)
    return df

# ── 24H Forecast (XGBoost) — with station support ─────────────────
def forecast_24h(xgb_models, station=None):
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    if station and "Station" in df.columns:
        station_df = df[df["Station"] == station].copy()
        if len(station_df) < 10:
            station_df = df.copy()
    else:
        station_df = df.copy()

    # Record the TRUE latest date BEFORE dropna removes new rows with NaN lags
    true_last_date = station_df["Date"].max()

    station_df = prepare_xgb_features(station_df, xgb_models)
    row = station_df.iloc[-1]   # last row with complete features (for prediction)

    # Always forecast for the day AFTER the true latest data in the dataset
    next_date = true_last_date + timedelta(days=1)
    results   = {"date": str(next_date.date()), "forecasts": {}}

    for target, model_data in xgb_models.items():
        features  = model_data["features"]
        available = [f for f in features if f in station_df.columns]
        x         = row[available].values.reshape(1, -1)

        lower  = float(model_data["models"]["lower"].predict(x)[0])
        median = float(model_data["models"]["median"].predict(x)[0])
        upper  = float(model_data["models"]["upper"].predict(x)[0])

        results["forecasts"][target] = {
            "value":  round(max(0, median), 2),
            "lower":  round(max(0, lower),  2),
            "upper":  round(max(0, upper),  2),
            "unit":   "µg/m³" if target != "CO" else "mg/m³"
        }

    # FIX: Compute AQI from ALL pollutants, take the maximum
    predicted = {t: results["forecasts"].get(t, {}).get("value", 0) for t in TARGETS}
    aqi_per = {
        "PM2.5": compute_aqi_pm25(predicted.get("PM2.5", 0)),
        "PM10":  compute_aqi_pm10(predicted.get("PM10",  0)),
        "NO2":   compute_aqi_no2( predicted.get("NO2",   0)),
        "SO2":   compute_aqi_so2( predicted.get("SO2",   0)),
        "CO":    compute_aqi_co(  predicted.get("CO",    0)),
        "Ozone": compute_aqi_ozone(predicted.get("Ozone",0)),
    }
    max_aqi  = max(aqi_per.values())
    prominent = max(aqi_per, key=aqi_per.get)

    results["aqi"]                 = max_aqi
    results["category"]            = aqi_category(max_aqi)
    results["prominent_pollutant"] = prominent
    results["aqi_breakdown"]       = aqi_per
    return results


# ── 7-Day Forecast (LSTM) — with station support ──────────────────
def forecast_7day(lstm_models, lstm_scalers, station=None):
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if station and "Station" in df.columns:
        station_df = df[df["Station"] == station].copy()
        if len(station_df) < SEQUENCE_LEN:
            station_df = df.copy()
    else:
        station_df = df.copy()

    # Use true latest date from raw data (before any dropna)
    true_last_7d = station_df["Date"].max()
    station_df = station_df.sort_values("Date").reset_index(drop=True)
    base_date  = true_last_7d   # always from actual latest data
    dates      = [(base_date + timedelta(days=i+1)).strftime("%Y-%m-%d")
                  for i in range(7)]
    results    = {"dates": dates, "forecasts": {}}

    for target in TARGETS:
        if target not in station_df.columns or target not in lstm_models:
            continue

        scaler = lstm_scalers[target]
        model  = lstm_models[target]

        values = station_df[target].values[-SEQUENCE_LEN:]
        scaled = scaler.transform(values.reshape(-1, 1)).flatten()

        x = torch.tensor(scaled, dtype=torch.float32)\
                 .unsqueeze(0).unsqueeze(-1).to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy().flatten()

        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        pred = np.maximum(pred, 0)

        results["forecasts"][target] = {
            "values": [round(float(v), 2) for v in pred],
            "unit":   "µg/m³" if target != "CO" else "mg/m³"
        }

    # FIX: AQI from all pollutants each day
    daily_aqi, daily_cat, daily_prominent = [], [], []
    for day_i in range(7):
        day_vals = {t: results["forecasts"].get(t, {}).get("values", [0]*7)[day_i]
                    for t in TARGETS}
        aqi_per = {
            "PM2.5": compute_aqi_pm25(day_vals.get("PM2.5", 0)),
            "PM10":  compute_aqi_pm10(day_vals.get("PM10",  0)),
            "NO2":   compute_aqi_no2( day_vals.get("NO2",   0)),
            "SO2":   compute_aqi_so2( day_vals.get("SO2",   0)),
            "CO":    compute_aqi_co(  day_vals.get("CO",    0)),
            "Ozone": compute_aqi_ozone(day_vals.get("Ozone",0)),
        }
        max_aqi = max(aqi_per.values())
        daily_aqi.append(max_aqi)
        daily_cat.append(aqi_category(max_aqi))
        daily_prominent.append(max(aqi_per, key=aqi_per.get))

    results["aqi_7day"]       = daily_aqi
    results["category_7day"]  = daily_cat
    results["prominent_7day"] = daily_prominent
    return results


# ── Station Summary — each station uses its own real latest row ───
def get_station_summary():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    stations = [
        "Manoharpur", "Rohta", "Sanjay Palace",
        "Sector-3B Avas Vikas Colony",
        "Shahjahan Garden", "Shastripuram"
    ]

    summary = []
    for station in stations:
        sub = df[df["Station"] == station] if "Station" in df.columns else df
        if len(sub) == 0:
            continue

        latest = sub.iloc[-1]
        pm25   = round(float(latest.get("PM2.5", 0)), 2)
        pm10   = round(float(latest.get("PM10",  0)), 2)
        no2    = round(float(latest.get("NO2",   0)), 2)
        so2    = round(float(latest.get("SO2",   0)), 2)
        co     = round(float(latest.get("CO",    0)), 3)
        ozone  = round(float(latest.get("Ozone", 0)), 2)

        aqi_per = {
            "PM2.5": compute_aqi_pm25(pm25),
            "PM10":  compute_aqi_pm10(pm10),
            "NO2":   compute_aqi_no2(no2),
            "SO2":   compute_aqi_so2(so2),
            "CO":    compute_aqi_co(co),
            "Ozone": compute_aqi_ozone(ozone),
        }
        aqi       = max(aqi_per.values())
        prominent = max(aqi_per, key=aqi_per.get)
        cat       = aqi_category(aqi)

        summary.append({
            "station":             station,
            "date":                str(latest["Date"].date()),
            "aqi":                 aqi,
            "category":            cat,
            "class":               cat.lower().replace(" ", "-"),
            "prominent_pollutant": prominent,
            "pm25":  pm25, "pm10": pm10, "no2": no2,
            "so2":   so2,  "co":   co,   "ozone": ozone,
        })
    return summary


# ══════════════════════════════════════════════════════════════════
#  AQI Breakpoint Functions — CPCB standard, one per pollutant
# ══════════════════════════════════════════════════════════════════

def _linear(val, c_lo, c_hi, i_lo, i_hi):
    return round(((i_hi - i_lo) / (c_hi - c_lo)) * (val - c_lo) + i_lo)

def compute_aqi_pm25(pm25):
    bp = [(0,30,0,50),(30,60,51,100),(60,90,101,200),
          (90,120,201,300),(120,250,301,400),(250,500,401,500)]
    for c_lo,c_hi,i_lo,i_hi in bp:
        if c_lo <= pm25 <= c_hi:
            return _linear(pm25, c_lo, c_hi, i_lo, i_hi)
    return 500

def compute_aqi_pm10(pm10):
    bp = [(0,50,0,50),(50,100,51,100),(100,250,101,200),
          (250,350,201,300),(350,430,301,400),(430,600,401,500)]
    for c_lo,c_hi,i_lo,i_hi in bp:
        if c_lo <= pm10 <= c_hi:
            return _linear(pm10, c_lo, c_hi, i_lo, i_hi)
    return 500

def compute_aqi_no2(no2):
    bp = [(0,40,0,50),(40,80,51,100),(80,180,101,200),
          (180,280,201,300),(280,400,301,400),(400,800,401,500)]
    for c_lo,c_hi,i_lo,i_hi in bp:
        if c_lo <= no2 <= c_hi:
            return _linear(no2, c_lo, c_hi, i_lo, i_hi)
    return 500

def compute_aqi_so2(so2):
    bp = [(0,40,0,50),(40,80,51,100),(80,380,101,200),
          (380,800,201,300),(800,1600,301,400),(1600,2620,401,500)]
    for c_lo,c_hi,i_lo,i_hi in bp:
        if c_lo <= so2 <= c_hi:
            return _linear(so2, c_lo, c_hi, i_lo, i_hi)
    return 500

def compute_aqi_co(co):
    bp = [(0,1,0,50),(1,2,51,100),(2,10,101,200),
          (10,17,201,300),(17,34,301,400),(34,50,401,500)]
    for c_lo,c_hi,i_lo,i_hi in bp:
        if c_lo <= co <= c_hi:
            return _linear(co, c_lo, c_hi, i_lo, i_hi)
    return 500

def compute_aqi_ozone(ozone):
    bp = [(0,50,0,50),(50,100,51,100),(100,168,101,200),
          (168,208,201,300),(208,748,301,400),(748,1000,401,500)]
    for c_lo,c_hi,i_lo,i_hi in bp:
        if c_lo <= ozone <= c_hi:
            return _linear(ozone, c_lo, c_hi, i_lo, i_hi)
    return 500

# Backward-compatible alias
def compute_aqi(pm25):
    return compute_aqi_pm25(pm25)

def aqi_category(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"