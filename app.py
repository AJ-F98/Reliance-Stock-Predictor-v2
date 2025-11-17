# app.py
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import date

# Page config
st.set_page_config(page_title="Reliance Stock Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load model and latest feature data
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")

@st.cache_data
def load_features():
    return pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)

model = load_model()
df_features = load_features()

# Latest available data
latest_date = df_features.index[-1].date()
latest_row = df_features.iloc[-1:]

# Prediction
prediction = model.predict(latest_row)[0]
prediction_rounded = round(prediction, 2)

# Fetch actual close if market is closed
today = date.today()
actual_close = None
if latest_date <= today:
    ticker = yf.Ticker("RELIANCE.NS")
    hist = ticker.history(period="5d")
    if not hist.empty and hist.index[-1].date() == today:
        actual_close = round(hist["Close"].iloc[-1], 2)

# Display
col1, col2 = st.columns(2)
with col1:
    st.metric("Latest Date in Data", latest_date)
    st.metric("Predicted Next-Day Close", f"₹{prediction_rounded}")

with col2:
    if actual_close:
        error_pct = abs(prediction_rounded - actual_close) / actual_close * 100
        st.metric("Actual Close (Today)", f"₹{actual_close}")
        st.metric("Prediction Error", f"₹{abs(prediction_rounded - actual_close):.2f} ({error_pct:.2f}%)")
    else:
        st.info("Market open or data not updated yet – actual close will appear after 3:30 PM IST")

# Optional: show recent predictions table
if actual_close or latest_date < today:
    st.subheader("Recent Predictions")
    recent = df_features.tail(5).copy()
    recent["Predicted"] = model.predict(df_features.tail(5))
    recent["Predicted"] = recent["Predicted"].round(2)
    display_cols = ["R_Close", "Predicted"]
    st.dataframe(recent[display_cols].rename(columns={"R_Close": "Actual Close"}))