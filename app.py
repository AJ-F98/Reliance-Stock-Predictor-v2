# app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import date

st.set_page_config(page_title="Reliance Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load the LATEST committed model and features (updated daily by GitHub Actions)
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")

@st.cache_data(ttl=3600)  # Refresh max once per hour
def load_features():
    df = pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)
    # Drop volume columns (same as training)
    cols_to_drop = ['R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol', 'Target']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

model = load_model()
df = load_features()

# Latest available row (yesterday) → predict today
latest_row = df.iloc[-1:]
prediction = round(float(model.predict(latest_row)[0]), 2)
latest_date = df.index[-1].strftime("%Y-%m-%d")

# Actual close today (live from yfinance)
actual_today = None
import yfinance as yf
rel = yf.Ticker("RELIANCE.NS")
hist = rel.history(period="2d")
if not hist.empty and hist.index[-1].date() == date.today():
    actual_today = round(hist["Close"].iloc[-1], 2)

# Display
col1, col2 = st.columns(2)
with col1:
    st.metric("Data up to", latest_date)
    st.metric("Predicted Close (Tomorrow)", f"₹{prediction}")

with col2:
    if actual_today:
        error = abs(prediction - actual_today)
        error_pct = error / actual_today * 100
        st.metric("Actual Close Today", f"₹{actual_today}")
        st.metric("Today's Prediction Error", f"₹{error:.2f} ({error_pct:.2f}%)")
    else:
        st.info("Actual close appears after 3:30 PM IST")

# Recent table
st.subheader("Recent Predictions vs Actual")
recent = df.tail(6).copy()
recent["Predicted"] = model.predict(recent).round(2)
recent["Actual Next Day"] = df["R_Close"].shift(-1)
st.dataframe(
    recent[["R_Close", "Predicted", "Actual Next Day"]].rename(columns={
        "R_Close": "Close (t)",
        "Predicted": "Predicted (t+1)",
        "Actual Next Day": "Actual (t+1)"
    }).round(2).style.format("₹{:.2f}"),
    use_container_width=True
)