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

# Load full data (we need R_Close for actual prices)
@st.cache_data(ttl=3600)
def load_full_data():
    return pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)

# Load features for prediction (exact same as training)
@st.cache_data(ttl=3600)
def load_features():
    df = pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)
    cols_to_drop = ['Target', 'R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df_full = load_full_data()        # Has R_Close → for actual prices
df_features = load_features()     # Exact training features → for model.predict()

# Latest date in data (yesterday)
latest_date = df_features.index[-1].date()
tomorrow_date = latest_date + timedelta(days=1)

# Predict tomorrow
prediction = round(float(model.predict(df_features.iloc[-1:])[0]), 2)

# Actual close today (live)
actual_today = None
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="2d")
if not hist.empty and hist.index[-1].date() == date.today():
    actual_today = round(hist["Close"].iloc[-1], 2)

# Display
col1, col2 = st.columns(2)
with col1:
    if actual_today:
        error = abs(prediction - actual_today)
        error_pct = error / actual_today * 100
        st.metric("Actual Close Today", f"{date.today():%d-%b-%Y}: ₹{actual_today}")
        st.metric("Today's Prediction Error", f"₹{error:.2f} ({error_pct:.2f}%)")
    else:
        st.metric("Actual Close Today", "Market not closed yet")
        st.metric("Today's Prediction Error", "—")

with col2:
    st.metric("Predicted Close Tomorrow", f"{tomorrow_date:%d-%b-%Y}: ₹{prediction}")

# Recent table: Date + Actual Close + Predicted Close
st.subheader("Recent Predictions")
recent_full = df_full.tail(8)
recent_features = df_features.tail(8)

# Predict next-day for each recent row
predicted = model.predict(recent_features).round(2)

display_df = pd.DataFrame({
    "Date": recent_full.index.strftime("%d-%b-%Y"),
    "Actual Close": recent_full["R_Close"].round(2).values,
    "Predicted Next-Day": predicted
})

# Remove tomorrow's row from table (already shown above)
display_df = display_df.iloc[:-1].reset_index(drop=True)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)