# app.py
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="Reliance Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")
model = load_model()

# Load full data to get actual R_Close values
@st.cache_data(ttl=1800)  # Refresh every 30 min
def load_full_data():
    return pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)

# Load features for prediction — exact same as training
@st.cache_data(ttl=1800)
def load_features():
    df = pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)
    cols_to_drop = ['Target', 'R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df_full = load_full_data()
df_features = load_features()

# === REAL-TIME LOGIC: Use TODAY's close if available ===
today = date.today()
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="5d")

# Check if today's close is available (after 3:30 PM IST)
actual_today = None
today_close_date = None
if not hist.empty:
    last_date = hist.index[-1].date()
    if last_date == today:
        actual_today = round(hist["Close"].iloc[-1], 2)
        today_close_date = today
    elif last_date == today - timedelta(days=1):
        actual_today = round(hist["Close"].iloc[-1], 2)
        today_close_date = last_date  # Fallback to yesterday

# If today's close exists → predict TOMORROW
if actual_today is not None:
    latest_features = df_features.iloc[-1:]  # Last row in training data
    prediction = round(float(model.predict(latest_features)[0]), 2)
    prediction_date = today + timedelta(days=1)
else:
    # Fallback: predict from yesterday → for today
    latest_features = df_features.iloc[-2:-1] if len(df_features) > 1 else df_features.iloc[-1:]
    prediction = round(float(model.predict(latest_features)[0]), 2)
    prediction_date = today

# Error calculation
error_text = "—"
if actual_today and prediction_date == today:
    error = abs(prediction - actual_today)
    error_pct = error / actual_today * 100
    error_text = f"₹{error:.2f} ({error_pct:.2f}%)"

# Display
col1, col2 = st.columns(2)
with col1:
    if actual_today:
        st.metric("Actual Close Today", f"{today:%d-%b-%Y}: ₹{actual_today}")
        st.metric("Today's Prediction Error", error_text)
    else:
        st.metric("Actual Close Today", "Market not closed yet")

with col2:
    st.metric("Predicted Close Tomorrow", f"{prediction_date + timedelta(days=1) if actual_today else prediction_date:%d-%b-%Y}: ₹{prediction}")

# Table: Recent actuals + next-day predictions
st.subheader("Recent Predictions")
n = 7
recent_full = df_full.tail(n)
recent_features = df_features.tail(n)
predicted = model.predict(recent_features).round(2)

table = pd.DataFrame({
    "Date": recent_full.index.strftime("%d-%b-%Y"),
    "Actual Close": recent_full["R_Close"].round(2).values,
    "Predicted Next-Day": predicted
})

# If today is in data → show prediction for tomorrow in table too
if actual_today:
    tomorrow_row = pd.DataFrame([{
        "Date": f"{(today + timedelta(days=1)):%d-%b-%Y}",
        "Actual Close": "—",
        "Predicted Next-Day": prediction
    }])
    table = pd.concat([table, tomorrow_row], ignore_index=True)

st.dataframe(table, use_container_width=True, hide_index=True)