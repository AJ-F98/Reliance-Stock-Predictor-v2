# app.py
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="Reliance Predictor", layout="wide")
st.title("Reliance Industries Close Price Prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")
model = load_model()

# Load full data for actual closes
@st.cache_data(ttl=3600)
def load_full_data():
    return pd.read_csv("reliance_final_model_ready_live.csv", index_col=0, parse_dates=True)

# Load features for prediction (exact training columns)
@st.cache_data(ttl=3600)
def load_features():
    df = pd.read_csv("reliance_final_model_ready_live.csv", index_col=0, parse_dates=True)
    cols_to_drop = ['Target', 'R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df_full = load_full_data()
df_features = load_features()

# Latest prediction date
prediction_date = df_full.index[-1].date() + timedelta(days=1)  # 19-Nov if last close is 18-Nov
prediction = round(float(model.predict(df_features.iloc[-1:])[0]), 2)

# Actual close for today (18-Nov)
actual_today = None
today = date.today()
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="2d")
if not hist.empty and hist.index[-1].date() == today:
    actual_today = round(hist["Close"].iloc[-1], 2)

# Error (only if prediction was for today)
error_text = "—"
if actual_today and prediction_date == today:
    error = abs(prediction - actual_today)
    error_pct = error / actual_today * 100
    error_text = f"₹{error:.2f} ({error_pct:.2f}%)"

# Display
col1, col2 = st.columns(2)
st.metric("Close Price", f"{today:%d-%b-%Y}: ₹{actual_today if actual_today else 'Market open'}")
st.metric("Predicted Price", f"{prediction_date:%d-%b-%Y}: ₹{prediction}")
st.metric("Error", error_text)

# Table: Date + Close Price + Predicted Price
st.subheader("Predictions")
recent_full = df_full.tail(5)
recent_features = df_features.tail(5)
predicted_prices = model.predict(recent_features).round(2)

table = pd.DataFrame({
    "Date": recent_full.index.strftime("%d-%b-%Y"),
    "Close Price": recent_full["R_Close"].round(2).values,
    "Predicted Price": predicted_prices
})

# Add today's actual if available (for 18-Nov row)
if actual_today and today in df_full.index:
    table.loc[table["Date"] == today.strftime("%d-%b-%Y"), "Close Price"] = actual_today

# Add prediction row for tomorrow
tomorrow_row = pd.DataFrame({
    "Date": [prediction_date.strftime("%d-%b-%Y")],
    "Close Price": [actual_today if prediction_date == today else None],
    "Predicted Price": [prediction]
})
table = pd.concat([table, tomorrow_row], ignore_index=True)

st.dataframe(table, use_container_width=True, hide_index=True)