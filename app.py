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

# Load features — EXACT same columns used in training
@st.cache_data(ttl=3600)
def load_features():
    df = pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)
    # These are the exact columns we dropped during training
    cols_to_drop = ['R_Close', 'R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol', 'Target']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df_features = load_features()  # This now has ONLY the features the model was trained on

# Prediction for tomorrow (from latest row = yesterday)
latest_row = df_features.iloc[-1:]
prediction = round(float(model.predict(latest_row)[0]), 2)
latest_date = df_features.index[-1].date()
tomorrow_date = latest_date + timedelta(days=1)

# Actual close today
actual_today = None
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="2d")
if not hist.empty and hist.index[-1].date() == date.today():
    actual_today = round(hist["Close"].iloc[-1], 2)

# Display
col1, col2 = st.columns(2)
with col1:
    if actual_today:
        st.metric("Actual Close Today", f"{date.today():%d-%b-%Y}: ₹{actual_today}")
    else:
        st.metric("Actual Close Today", "Market not closed yet")

with col2:
    st.metric("Predicted Close Tomorrow", f"{tomorrow_date:%d-%b-%Y}: ₹{prediction}")

# Recent predictions table
st.subheader("Recent Predictions")
recent_features = df_features.tail(8)

# Predict using EXACT same DataFrame (no column changes)
recent_predictions = model.predict(recent_features).round(2)

display_df = pd.DataFrame({
    "Date": recent_features.index.strftime("%d-%b-%Y"),
    "Predicted Close": recent_predictions
})

# Remove tomorrow's prediction (already shown above)
display_df = display_df.iloc[:-1].reset_index(drop=True)

st.dataframe(display_df, use_container_width=True, hide_index=True)