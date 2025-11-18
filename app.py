# app.py — Corrected for t+1 predictions
import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta

st.set_page_config(page_title="Reliance Predictor", layout="wide")
st.title("Reliance Industries – Daily Close Prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")
model = load_model()

# Load live data (includes latest day for prediction)
@st.cache_data(ttl=3600)
def load_full_data():
    return pd.read_csv("reliance_final_model_ready_live.csv", index_col=0, parse_dates=True)

# Load features for prediction
@st.cache_data(ttl=3600)
def load_features():
    df = pd.read_csv("reliance_final_model_ready_live.csv", index_col=0, parse_dates=True)
    cols_to_drop = ['Target', 'R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df_full = load_full_data()
df_features = load_features()

# Latest date and prediction
latest_date = df_full.index[-1].date()
next_date = latest_date + timedelta(days=1)  # 19-Nov-2025 tonight
prediction = round(float(model.predict(df_features.iloc[-1:])[0]), 2)

# Actual close today (18-Nov)
actual_today = df_full['R_Close'].iloc[-1] if latest_date == date.today() else None

# Display
col1, col2 = st.columns(2)
with col1:
    st.metric("Close Price", f"{latest_date:%d-%b-%Y}: ₹{actual_today if actual_today else 'Market open'}")
with col2:
    st.metric("Predicted Price", f"{next_date:%d-%b-%Y}: ₹{prediction}")

# Table with t+1 predictions
st.subheader("Recent Data & Predictions")
n_days = 7
recent_full = df_full.tail(n_days + 1)  # Extra day for t+1 prediction
recent_features = df_features.tail(n_days + 1)

# Generate predictions for each day (t+1)
predictions = pd.Series(model.predict(recent_features), index=recent_features.index).shift(-1).iloc[:-1]
# Shift predictions forward by 1 day, drop the last (future) prediction

table = pd.DataFrame({
    "Date": recent_full.index[:-1].strftime("%d-%b-%Y"),  # Exclude the last row (today's prediction)
    "Close Price": recent_full['R_Close'].iloc[:-1].round(2).values,
    "Predicted Price": predictions.round(2).values
})

# Add tomorrow's prediction row
tomorrow_row = pd.DataFrame({
    "Date": [next_date.strftime("%d-%b-%Y")],
    "Close Price": ["—"],
    "Predicted Price": [prediction]
})
table = pd.concat([table, tomorrow_row], ignore_index=True)

st.dataframe(table, use_container_width=True, hide_index=True)