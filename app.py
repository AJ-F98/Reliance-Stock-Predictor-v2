# app.py — FINAL & PERFECT: Uses TODAY's close to predict TOMORROW
import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta

st.set_page_config(page_title="Reliance Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")
model = load_model()

# Load live data (includes 18-Nov close)
@st.cache_data(ttl=3600)
def load_live():
    df = pd.read_csv("reliance_final_model_ready_live.csv", index_col=0, parse_dates=True)
    cols_to_drop = ['Target', 'R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    features = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df, features

df_full, df_features = load_live()

# Today's close (18-Nov)
today_close = round(df_full['R_Close'].iloc[-1], 2)
today_date = df_full.index[-1].date()
tomorrow_date = today_date + timedelta(days=1)

# Fresh prediction for tomorrow using TODAY's data
tomorrow_prediction = round(float(model.predict(df_features.iloc[-1:])[0]), 2)

# Display big metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Close Price (Today)", f"{today_date:%d-%b-%Y}", f"₹{today_close}")
with col2:
    st.metric("Predicted Close (Tomorrow)", f"{tomorrow_date:%d-%b-%Y}", f"₹{tomorrow_prediction}")

# Table: Historical actuals + their next-day predictions + tomorrow's fresh prediction
st.subheader("Prediction History")
recent = df_full.tail(7).copy()

# Predict next-day for each historical row
historical_predictions = model.predict(df_features.tail(7)).round(2)

table = pd.DataFrame({
    "Date": recent.index.strftime("%d-%b-%Y"),
    "Close Price": recent["R_Close"].round(2).values,
    "Predicted Next Day": historical_predictions
})

# Add tomorrow's row
tomorrow_row = pd.DataFrame({
    "Date": [tomorrow_date.strftime("%d-%b-%Y")],
    "Close Price": ["—"],
    "Predicted Next Day": [tomorrow_prediction]
})
table = pd.concat([table, tomorrow_row], ignore_index=True)

st.dataframe(table.reset_index(drop=True), use_container_width=True, hide_index=True)