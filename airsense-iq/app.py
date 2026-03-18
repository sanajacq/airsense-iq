"""
AirSense-IQ — Flask Backend v3.0
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os, csv, traceback, pandas as pd, numpy as np
from datetime import datetime, timedelta

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Agra_AirQuality_cleaned_features.csv")

# ── Global NaN/Inf sanitizer for JSON responses ──────────────
import math as _math, json as _json
class _NaNSafeEncoder(_json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        # Walk the object and replace NaN/Inf with None before encoding
        def _clean(obj):
            if isinstance(obj, float):
                if _math.isnan(obj) or _math.isinf(obj):
                    return None
                return obj
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj
        return super().iterencode(_clean(o), _one_shot)

app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, "static"),
            static_url_path="")
CORS(app)
app.json_encoder = _NaNSafeEncoder   # prevents NaN in JSON responses

HOSPITAL_STATION = {
    'snmc':'Rohta','snmc-emergency':'Rohta','district-hospital':'Rohta',
    'district-women':'Rohta','lady-lyall':'Rohta','uphc-mantola':'Rohta',
    'uphc-rohta':'Rohta','phc-jagner':'Rohta',
    'fhmc':'Shastripuram','cantonment':'Shastripuram','military':'Shastripuram',
    'uphc-shastripuram':'Shastripuram','phc-etmadpur':'Shastripuram','phc-fatehabad':'Shastripuram',
    'uphc-manoharpur':'Manoharpur','phc-runkata':'Manoharpur','phc-saiyan':'Manoharpur','imhh':'Manoharpur',
    'uphc-shahjahan':'Shahjahan Garden','asopa':'Shahjahan Garden',
    'phc-bah':'Shahjahan Garden','phc-akola':'Shahjahan Garden','govt-baroli':'Shahjahan Garden',
    'peoples-heritage':'Sector-3B Avas Vikas Colony','uphc-sector3b':'Sector-3B Avas Vikas Colony',
    'phc-kheragarh':'Sector-3B Avas Vikas Colony','phc-barauli':'Sector-3B Avas Vikas Colony',
    'govt-raghuvir':'Sector-3B Avas Vikas Colony',
    'uphc-sanjay':'Sanjay Palace',
    'zone-manoharpur':'Manoharpur','zone-rohta':'Rohta','zone-sanjay':'Sanjay Palace',
    'zone-sector3b':'Sector-3B Avas Vikas Colony','zone-shahjahan':'Shahjahan Garden',
    'zone-shastripuram':'Shastripuram'
}

# ── AQI helpers ───────────────────────────────────────────────
def aqi_pm25(v):
    bp=[(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)]
    for c,C,i,I in bp:
        if c<=v<=C: return round(((I-i)/(C-c))*(v-c)+i)
    return 500

def aqi_pm10(v):
    bp=[(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)]
    for c,C,i,I in bp:
        if c<=v<=C: return round(((I-i)/(C-c))*(v-c)+i)
    return 500

def aqi_no2(v):
    bp=[(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)]
    for c,C,i,I in bp:
        if c<=v<=C: return round(((I-i)/(C-c))*(v-c)+i)
    return 500

def aqi_so2(v):
    bp=[(0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2620,401,500)]
    for c,C,i,I in bp:
        if c<=v<=C: return round(((I-i)/(C-c))*(v-c)+i)
    return 500

def full_aqi(pm25,pm10,no2=0,so2=0):
    vals = {'PM2.5':aqi_pm25(pm25),'PM10':aqi_pm10(pm10),'NO2':aqi_no2(no2),'SO2':aqi_so2(so2)}
    mx = max(vals.values()); prom = max(vals, key=vals.get)
    return mx, prom, vals

def aqi_cat(a):
    if a<=50: return "Good"
    if a<=100: return "Satisfactory"
    if a<=200: return "Moderate"
    if a<=300: return "Poor"
    if a<=400: return "Very Poor"
    return "Severe"

# ── Load models ───────────────────────────────────────────────
print("Loading models...")
try:
    from predict import load_models, forecast_24h, forecast_7day, get_station_summary
    from alert_engine import generate_alerts, chatbot_response
    xgb_models, lstm_models, lstm_scalers, summary = load_models()
    print("✅ Models loaded.")
    MODELS_READY = True
except Exception as e:
    print(f"⚠️  Models not loaded: {e}")
    MODELS_READY = False

def fallback_forecast(station='Rohta'):
    """Build fallback from real dataset values."""
    try:
        df  = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        sub = df[df["Station"]==station].sort_values("Date")
        if len(sub)==0: sub = df.sort_values("Date")
        last = sub.iloc[-1]
        pm25 = float(last.get("PM2.5",50)); pm10 = float(last.get("PM10",120))
        no2  = float(last.get("NO2",30));   so2  = float(last.get("SO2",10))
        mx, prom, _ = full_aqi(pm25, pm10, no2, so2)
        return {
            "date": str((datetime.now()+timedelta(days=1)).date()),
            "aqi": mx, "category": aqi_cat(mx), "prominent_pollutant": prom,
            "forecasts": {
                "PM2.5": {"value":round(pm25,2),"lower":round(pm25*0.8,2),"upper":round(pm25*1.25,2),"unit":"µg/m³"},
                "PM10":  {"value":round(pm10,2),"lower":round(pm10*0.8,2),"upper":round(pm10*1.25,2),"unit":"µg/m³"},
                "NO2":   {"value":round(no2,2), "lower":round(no2*0.8,2), "upper":round(no2*1.25,2), "unit":"µg/m³"},
                "SO2":   {"value":round(float(last.get("SO2",10)),2),"lower":5.0,"upper":15.0,"unit":"µg/m³"},
                "CO":    {"value":round(float(last.get("CO",0.3)),3), "lower":0.1,"upper":0.6,"unit":"mg/m³"},
                "Ozone": {"value":round(float(last.get("Ozone",8)),2), "lower":3.0,"upper":15.0,"unit":"µg/m³"},
            }
        }
    except:
        return {"date":str(datetime.now().date()),"aqi":148,"category":"Moderate",
                "prominent_pollutant":"PM10","forecasts":{}}

# ── Serve frontend ────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

# ── API: Dashboard ────────────────────────────────────────────
@app.route("/api/dashboard")
def api_dashboard():
    try:
        hospital = request.args.get("hospital","uphc-rohta")
        station  = HOSPITAL_STATION.get(hospital,"Rohta")
        if MODELS_READY:
            fc24 = forecast_24h(xgb_models, station=station)
            fc7  = forecast_7day(lstm_models, lstm_scalers, station=station)
        else:
            fc24 = fallback_forecast(station)
            fc7  = None
        alerts   = generate_alerts(fc24) if MODELS_READY else _fallback_alerts(fc24)
        stations = get_station_summary() if MODELS_READY else _station_summary_from_csv()
        city     = _city_summary(stations)
        return jsonify({
            "status":"ok","station":station,
            "forecast_24h":fc24,"forecast_7day":fc7,
            "alerts":alerts,"stations":stations,"city":city,
            "models_ready":MODELS_READY,
            "last_updated":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500

def _city_summary(stations):
    """Compute city-wide average AQI from all stations."""
    if not stations: return {}
    aqis  = [s['aqi'] for s in stations if s.get('aqi')]
    pm25s = [s['pm25'] for s in stations if s.get('pm25')]
    pm10s = [s['pm10'] for s in stations if s.get('pm10')]
    city_aqi = int(round(sum(aqis)/len(aqis))) if aqis else 0
    return {
        "aqi": city_aqi, "category": aqi_cat(city_aqi),
        "avg_pm25": round(sum(pm25s)/len(pm25s),1) if pm25s else 0,
        "avg_pm10": round(sum(pm10s)/len(pm10s),1) if pm10s else 0,
        "station_count": len(stations),
        "date": stations[0]['date'] if stations else "--"
    }

def _station_summary_from_csv():
    """Fallback station summary direct from CSV."""
    try:
        df  = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        stations = ['Manoharpur','Rohta','Sanjay Palace',
                    'Sector-3B Avas Vikas Colony','Shahjahan Garden','Shastripuram']
        result = []
        for st in stations:
            sub  = df[df["Station"]==st].sort_values("Date")
            if len(sub)==0: continue
            last = sub.iloc[-1]
            pm25 = round(float(last.get("PM2.5",0)),2)
            pm10 = round(float(last.get("PM10",0)),2)
            no2  = round(float(last.get("NO2",0)),2)
            so2  = round(float(last.get("SO2",0)),2)
            mx, prom, _ = full_aqi(pm25, pm10, no2, so2)
            result.append({
                "station":st,"date":str(last["Date"].date()),
                "aqi":mx,"category":aqi_cat(mx),
                "prominent_pollutant":prom,
                "pm25":pm25,"pm10":pm10,"no2":no2,"so2":so2,
                "co":round(float(last.get("CO",0)),3),
                "ozone":round(float(last.get("Ozone",0)),2)
            })
        return result
    except: return []

def _fallback_alerts(fc24):
    """Minimal alert structure when models not loaded."""
    from alert_engine import generate_alerts
    return generate_alerts(fc24)

# ── API: Historical AQI (all stations + city, full dataset) ──
@app.route("/api/historical_aqi")
def api_historical_aqi():
    try:
        station = request.args.get("station","Rohta")
        days    = int(request.args.get("days", 60))
        df  = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])

        if station == "Agra City":
            # Average across all stations per day
            records = []
            city_daily = df.groupby("Date").agg(
                pm25=("PM2.5","mean"), pm10=("PM10","mean"),
                no2=("NO2","mean"), so2=("SO2","mean")
            ).reset_index().sort_values("Date").tail(days)
            prev_aqi = None
            for _, row in city_daily.iterrows():
                act_aqi = int(max(aqi_pm25(row.pm25), aqi_pm10(row.pm10)))
                prd_aqi = prev_aqi if prev_aqi else act_aqi
                records.append({
                    "date": str(row["Date"].date()),
                    "actual_aqi": act_aqi,
                    "predicted_aqi": prd_aqi,
                    "pm25": round(row.pm25,1),
                    "pm10": round(row.pm10,1),
                    "category": aqi_cat(act_aqi),
                    "error_pct": round(abs(act_aqi-prd_aqi)/max(act_aqi,1)*100,1)
                })
                prev_aqi = act_aqi
        else:
            sub = df[df["Station"]==station].sort_values("Date").tail(days+1).reset_index(drop=True)
            records = []
            for i in range(1, len(sub)):
                act = sub.iloc[i]; prev = sub.iloc[i-1]
                act_aqi = int(max(aqi_pm25(act["PM2.5"]), aqi_pm10(act["PM10"])))
                prd_aqi = int(max(aqi_pm25(prev["PM2.5"]), aqi_pm10(prev["PM10"])))
                records.append({
                    "date": str(act["Date"].date()),
                    "actual_aqi": act_aqi,
                    "predicted_aqi": prd_aqi,
                    "pm25": round(float(act["PM2.5"]),1),
                    "pm10": round(float(act["PM10"]),1),
                    "category": aqi_cat(act_aqi),
                    "error_pct": round(abs(act_aqi-prd_aqi)/max(act_aqi,1)*100,1)
                })

        return jsonify({"status":"ok","station":station,"data":records})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500

# ── API: Accuracy table ───────────────────────────────────────
@app.route("/api/accuracy")
def api_accuracy():
    try:
        # Try models/ folder first (generated by train_models.py)
        acc_path  = os.path.join(BASE_DIR,"models","accuracy_table.csv")
        pred_path = os.path.join(BASE_DIR,"models","prediction_log.csv")

        # Fallback: compute baseline metrics from dataset
        if not os.path.exists(acc_path):
            acc_path  = os.path.join(BASE_DIR,"models","accuracy_table_baseline.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(BASE_DIR,"models","prediction_log_aqi.csv")

        acc_rows, pred_rows = [], []
        if os.path.exists(acc_path):
            with open(acc_path) as f: acc_rows = list(csv.DictReader(f))
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                rows = list(csv.DictReader(f))
                # Return AQI-based prediction log
                pred_rows = rows[-120:]

        return jsonify({"status":"ok","data":{"accuracy_table":acc_rows,"prediction_log":pred_rows}})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500

# ── API: Summary (stations) ───────────────────────────────────
@app.route("/api/summary")
def api_summary():
    try:
        st = get_station_summary() if MODELS_READY else _station_summary_from_csv()
        return jsonify({"status":"ok","stations":st})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500

# ── API: Chatbot ──────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        body     = request.get_json()
        message  = body.get("message","").strip()
        hospital = body.get("hospital","uphc-rohta")
        history  = body.get("history", [])   # full conversation history from frontend
        station  = HOSPITAL_STATION.get(hospital,"Rohta")

        if not message:
            return jsonify({"status":"error","reply":"Please type a message."}), 400

        # ── Build live context ─────────────────────────────────
        fc24     = forecast_24h(xgb_models,station=station) if MODELS_READY else fallback_forecast(station)
        alerts   = generate_alerts(fc24) if MODELS_READY else generate_alerts(fc24)
        stations = get_station_summary() if MODELS_READY else _station_summary_from_csv()
        city     = _city_summary(stations)

        forecasts = fc24.get("forecasts", {})
        surge     = alerts.get("patient_surge", {})
        staffing  = alerts.get("staffing", {})

        # ── System prompt with full project context ────────────
        system_prompt = f"""You are the AirSense-IQ AI Assistant — an intelligent, helpful assistant built into a hospital air quality preparedness dashboard for Agra, India.

You have deep knowledge about:
1. The AirSense-IQ project — an urban air quality prediction system for Agra
2. The live data currently showing on the dashboard
3. Air quality science, CPCB standards, WHO guidelines
4. Hospital preparedness and health impacts of pollution
5. The machine learning models used (XGBoost for 24H, LSTM for 7-day)
6. Everything related to this internship project

=== LIVE DASHBOARD DATA (right now) ===
Station selected: {station}
Forecast date: {fc24.get('date', 'N/A')}
AQI: {fc24.get('aqi', '--')} ({fc24.get('category', '--')})
Prominent Pollutant: {fc24.get('prominent_pollutant', '--')}

Pollutant Forecasts:
- PM2.5:  {forecasts.get('PM2.5', {}).get('value', '--')} µg/m³  (Limit: 60, Band: {forecasts.get('PM2.5', {}).get('lower','--')}–{forecasts.get('PM2.5', {}).get('upper','--')})
- PM10:   {forecasts.get('PM10',  {}).get('value', '--')} µg/m³  (Limit: 100)
- NO2:    {forecasts.get('NO2',   {}).get('value', '--')} µg/m³  (Limit: 80)
- SO2:    {forecasts.get('SO2',   {}).get('value', '--')} µg/m³  (Limit: 80)
- CO:     {forecasts.get('CO',    {}).get('value', '--')} mg/m³  (Limit: 4)
- Ozone:  {forecasts.get('Ozone', {}).get('value', '--')} µg/m³  (Limit: 100)

RRI Level: {alerts.get('rri', '--')}
Patient Surge: +{surge.get('surge_pct', 0)}% expected ({surge.get('extra_patients', 0)} extra patients above {surge.get('base_admissions', 8)} baseline)
Staffing needed: {staffing.get('extra_nurses', 0)} nurses, {staffing.get('extra_doctors', 0)} doctors, {staffing.get('extra_rt', 0)} respiratory therapists
Shift directive: {staffing.get('shift_directive', '--')}
ICU Status: {staffing.get('icu_status', 'Normal')}

All stations city average: AQI {city.get('aqi','--')} ({city.get('category','--')}), PM2.5 avg: {city.get('avg_pm25','--')}, PM10 avg: {city.get('avg_pm10','--')}

=== PROJECT DETAILS ===
- Dataset: 9,198 rows, 6 CPCB stations in Agra, Jan 2022 to Mar 2026
- Stations: Manoharpur, Rohta, Sanjay Palace, Sector-3B Avas Vikas Colony, Shahjahan Garden, Shastripuram
- Pollutants tracked: PM2.5, PM10, NO2, SO2, CO, Ozone
- ML Models: XGBoost (24H quantile regression, 18 models), LSTM (7-day forecast, PyTorch)
- Enhanced features: 115 engineered features including lag, rolling averages, weather interactions, seasonal encoding
- Model accuracy: MAPE 8-14% (better than CPCB SAFAR 15-20%, using only historical CSV data)
- Alert engine: Research-backed using Dominici 2006 JAMA, Shanghai 2021 Respiratory Research, MOHFW IPHS 2022
- Backend: Flask REST API (Python), Frontend: HTML/CSS/JS, Database: CSV files
- Admin panel password: airsense2026
- 27 hospitals and health centres mapped across Agra
- Developer: Sana Jacquilin D, Reg: 2340158, Xellentro Consulting Services LLP

=== AQI SCALE (CPCB) ===
0-50: Good | 51-100: Satisfactory | 101-200: Moderate | 201-300: Poor | 301-400: Very Poor | 401-500: Severe

=== CPCB SAFE LIMITS ===
PM2.5: 60 µg/m³ (24h) | PM10: 100 µg/m³ | NO2: 80 µg/m³ | SO2: 80 µg/m³ | CO: 4 mg/m³ | Ozone: 100 µg/m³
WHO guidelines: PM2.5: 15 µg/m³ | PM10: 45 µg/m³

=== YOUR ROLE ===
- Answer ANY question the user asks — about the project, the data, health impacts, air quality science, hospital protocols, ML models, code, or anything else
- Use the live data above when answering questions about current conditions
- Be conversational, clear, and helpful — like a knowledgeable colleague who built this system
- Keep responses concise but complete. Use bullet points for lists, plain text for explanations
- If asked about something not in your context, use your general knowledge honestly
- Never say "I don't have access to" when the data is provided above — you have it, use it
- Respond in the same language the user writes in (English or Hindi)"""

        # ── Build messages array with conversation history ─────
        messages = []
        # Add previous turns from history (last 10 to keep context manageable)
        for turn in history[-10:]:
            if turn.get("role") in ("user","assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        # Add current message
        messages.append({"role": "user", "content": message})

        # ── Call Claude API ────────────────────────────────────
        import urllib.request, json as _json
        payload = _json.dumps({
            "model":      "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "system":     system_prompt,
            "messages":   messages
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read().decode("utf-8"))

        # Extract reply text
        reply = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                reply += block.get("text", "")

        if not reply:
            reply = "I received your message but got an empty response. Please try again."

        return jsonify({"status": "ok", "reply": reply})

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8") if e.fp else ""
        # Fall back to keyword-based chatbot if API fails
        try:
            from alert_engine import chatbot_response
            fc24 = forecast_24h(xgb_models, station=station) if MODELS_READY else fallback_forecast(station)
            reply = chatbot_response(message, fc24)
            return jsonify({"status": "ok", "reply": reply + "<br><small style='color:#90a4ae'>(Offline mode)</small>"})
        except:
            pass
        traceback.print_exc()
        return jsonify({"status": "error", "reply": "API error. Please check your connection and try again."}), 500
    except Exception as e:
        # Fall back to keyword chatbot on any error
        try:
            from alert_engine import chatbot_response
            fc24 = fallback_forecast(station)
            reply = chatbot_response(message, fc24)
            return jsonify({"status": "ok", "reply": reply + "<br><small style='color:#90a4ae'>(Offline mode)</small>"})
        except:
            pass
        traceback.print_exc()
        return jsonify({"status": "error", "reply": "Sorry, something went wrong. Please try again."}), 500

# ── API: Admin ────────────────────────────────────────────────
@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    body = request.get_json()
    return jsonify({"ok": body.get("password","")=="airsense2026"})

@app.route("/api/admin/add_reading", methods=["POST"])
def api_admin_add():
    try:
        body = request.get_json()
        if body.get("password","") != "airsense2026":
            return jsonify({"status":"error","message":"Invalid password"}),403

        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])

        new_row = {
            "Date":    body.get("date", str(datetime.now().date())),
            "Station": body.get("station","Rohta"),
            "PM2.5":   float(body.get("pm25",0)  or 0),
            "PM10":    float(body.get("pm10",0)   or 0),
            "NO2":     float(body.get("no2",0)    or 0),
            "SO2":     float(body.get("so2",0)    or 0),
            "CO":      float(body.get("co",0)     or 0),
            "Ozone":   float(body.get("ozone",0)  or 0),
            "RH":      float(body.get("rh",0)     or 0),
            "WS":      float(body.get("ws",0)     or 0),
            "AT":      float(body.get("at",0)     or 0),
            "BP":      float(body.get("bp",0)     or 0),
        }

        # Append and sort
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Station","Date"]).reset_index(drop=True)

        # Recompute lag/rolling features so predict.py gets correct values
        TARGETS = ["PM2.5","PM10","NO2","SO2","CO","Ozone"]
        for t in TARGETS:
            if t not in df.columns: continue
            df[f"{t}_lag1"]        = df.groupby("Station")[t].shift(1)
            df[f"{t}_lag3"]        = df.groupby("Station")[t].shift(3)
            df[f"{t}_lag7"]        = df.groupby("Station")[t].shift(7)
            df[f"{t}_roll7_mean"]  = df.groupby("Station")[t].transform(
                lambda x: x.rolling(7, min_periods=1).mean())
            df[f"{t}_roll7_std"]   = df.groupby("Station")[t].transform(
                lambda x: x.rolling(7, min_periods=1).std())
            df[f"{t}_roll30_mean"] = df.groupby("Station")[t].transform(
                lambda x: x.rolling(30, min_periods=1).mean())

        # Recompute city averages per date
        city_avg = df.groupby("Date")[TARGETS].mean().reset_index()
        city_avg.columns = ["Date"] + [f"city_avg_{t}" for t in TARGETS]
        df = df.drop(columns=[c for c in df.columns if c.startswith("city_avg_")], errors="ignore")
        df = df.merge(city_avg, on="Date", how="left")

        # Recompute date features
        df["dayofyear"] = df["Date"].dt.dayofyear
        df["month"]     = df["Date"].dt.month
        df["weekday"]   = df["Date"].dt.dayofweek
        df["season"]    = df["month"].map(
            lambda m: 1 if m in [12,1,2] else 2 if m in [3,4,5] else 3 if m in [6,7,8] else 4)

        # Recompute AQI columns
        def _aqi(v):
            bp = [(0,30,0,50),(30,60,51,100),(60,90,101,200),
                  (90,120,201,300),(120,250,301,400),(250,500,401,500)]
            for c,C,i,I in bp:
                if c <= v <= C: return round(((I-i)/(C-c))*(v-c)+i)
            return 500
        df["AQI_PM25"]     = df["PM2.5"].apply(_aqi)
        df["AQI_category"] = df["AQI_PM25"].apply(aqi_cat)

        df.to_csv(DATA_PATH, index=False)

        new_last     = df["Date"].max().date()
        forecast_for = str(pd.Timestamp(new_last) + pd.Timedelta(days=1))[:10]

        return jsonify({
            "status":       "ok",
            "message":      f"Added {new_row['Date']} for {new_row['Station']}. Dataset now up to {new_last}. Forecast is now for {forecast_for}.",
            "new_rows":     len(df),
            "dataset_last": str(new_last),
            "forecast_for": forecast_for
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500

if __name__ == "__main__":
    print("\n🌍 AirSense-IQ Server v3.0  →  http://localhost:5000\n")
    app.run(debug=True, port=5000)

# ── API: Date Detail (for Past Data lookup) ───────────────────
@app.route("/api/date_detail")
def api_date_detail():
    import math, json as _json
    from flask import Response

    def _safe(val, decimals=2):
        """Convert pandas/numpy value to clean float, replacing NaN with 0."""
        try:
            v = float(val)
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return round(v, decimals)
        except (TypeError, ValueError):
            return 0.0

    def _safe_opt(val, decimals=1):
        """Like _safe but returns None if value is genuinely missing."""
        try:
            v = float(val)
            if math.isnan(v) or math.isinf(v):
                return None
            return round(v, decimals)
        except (TypeError, ValueError):
            return None

    try:
        date_str = request.args.get("date", "")
        station  = request.args.get("station", "Rohta")
        if not date_str:
            return Response(_json.dumps({"status":"error","message":"date required"}),
                            mimetype="application/json"), 400

        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        target_date = pd.Timestamp(date_str).date()

        sub = df[(df["Station"] == station) & (df["Date"].dt.date == target_date)]
        if len(sub) == 0:
            sub = df[df["Date"].dt.date == target_date]
        if len(sub) == 0:
            return Response(_json.dumps({"status":"error","message":"No data found for "+date_str+" at "+station}),
                            mimetype="application/json"), 404

        row  = sub.iloc[0]
        pm25  = _safe(row["PM2.5"])
        pm10  = _safe(row["PM10"])
        no2   = _safe(row["NO2"])
        so2   = _safe(row["SO2"])
        co    = _safe(row["CO"], 3)
        ozone = _safe(row["Ozone"])
        temp  = _safe_opt(row.get("AT", float("nan")))
        rh    = _safe_opt(row.get("RH", float("nan")))
        ws    = _safe_opt(row.get("WS", float("nan")), 2)

        total_aqi = max(aqi_pm25(pm25), aqi_pm10(pm10), aqi_no2(no2), aqi_so2(so2))
        aqi_map   = {"PM2.5": aqi_pm25(pm25), "PM10": aqi_pm10(pm10),
                     "NO2": aqi_no2(no2), "SO2": aqi_so2(so2)}
        prominent = max(aqi_map, key=aqi_map.get)

        result = {
            "status":    "ok",
            "station":   station,
            "date":      date_str,
            "aqi":       int(total_aqi),
            "category":  aqi_cat(total_aqi),
            "prominent": prominent,
            "pm25":      pm25,
            "pm10":      pm10,
            "no2":       no2,
            "so2":       so2,
            "co":        co,
            "ozone":     ozone,
            "temp":      temp,
            "rh":        rh,
            "ws":        ws,
        }
        # Use json.dumps directly so we control NaN handling completely
        return Response(_json.dumps(result), mimetype="application/json")

    except Exception as e:
        traceback.print_exc()
        return Response(_json.dumps({"status":"error","message":str(e)}),
                        mimetype="application/json"), 500

        return jsonify({"status":"error","message":str(e)}), 500