# app.py
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="Reliance Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load model & data
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")

@st.cache_data(ttl=3600)
def load_features():
    df = pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)
    cols_to_drop = ['R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol', 'Target']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

model = load_model()
df = load_features()

# Latest row → predict tomorrow
latest_row = df.iloc[-1:]
prediction = round(float(model.predict(latest_row)[0]), 2)
latest_date = df.index[-1].date()
tomorrow_date = latest_date + timedelta(days=1)

# Actual close today
actual_today = None
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="2d")
if not hist.empty and hist.index[-1].date() == date.today():
    actual_today = round(hist["Close"].iloc[-1], 2)

# Display metrics
col1, col2 = st.columns(2)
with col1:
    if actual_today:
        st.metric("Actual Close Today", f"{date.today():%d-%b-%Y}: ₹{actual_today}")
    else:
        st.metric("Actual Close Today", "Market not closed yet")

with col2:
    st.metric("Predicted Close Tomorrow", f"{tomorrow_date:%d-%b-%Y}: ₹{prediction}")

# Recent predictions table (clean & correct)
st.subheader("Recent Predictions")
recent = df.tail(8).copy()

# Predict using EXACT same columns/order as training
X_recent = recent.drop(columns=['R_Close'], errors='ignore')  # Keep all features except R_Close if present
predictions = model.predict(X_recent).round(2)

# Build clean display table
display_df = pd.DataFrame({
    "Date": recent.index.strftime("%d-%b-%Y"),
    "Predicted Close": predictions
})

# Remove the last row (tomorrow) — already shown in metric
display_df = display_df.iloc[:-1]

st.dataframe(
    display_df.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)