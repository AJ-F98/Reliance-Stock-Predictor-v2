# app.py
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import date
import yfinance as yf

yf.set_tz_cache_location("/tmp")



st.set_page_config(page_title="Reliance Stock Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load model and training features once
@st.cache_resource
def load_model_and_features():
    model = joblib.load("reliance_model.pkl")

    df = pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)
    
    cols_to_drop = ['R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return model, df

model, df_features = load_model_and_features()


feature_columns = df_features.columns.drop("Target") 
latest_row = df_features[feature_columns].iloc[-1:].copy()

# Prediction – now guaranteed to match training
prediction = float(model.predict(latest_row)[0])
prediction_rounded = round(prediction, 2)

# Get actual close if available
today = date.today()
actual_close = None
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="2d")
if not hist.empty and hist.index[-1].date() == today:
    actual_close = round(hist["Close"].iloc[-1], 2)

# Display results
col1, col2 = st.columns(2)

with col1:
    st.metric("Data up to", df_features.index[-1].strftime("%Y-%m-%d"))
    st.metric("Predicted Next-Day Close", f"₹{prediction_rounded}")

with col2:
    if actual_close is not None:
        error = abs(prediction_rounded - actual_close)
        error_pct = error / actual_close * 100
        st.metric("Actual Close Today", f"₹{actual_close}")
        st.metric("Error", f"₹{error:.2f} ({error_pct:.2f}%)")
    else:
        st.info("Actual close will appear after market close")

# Recent performance table
st.subheader("Recent Predictions vs Actual")

# Take last 5 trading days that have actual next-day close
display_data = df_features[['R_Close', 'Target']].tail(5).copy()
display_data["Predicted"] = model.predict(df_features[feature_columns].tail(5)).round(2)
display_data = display_data.rename(columns={
    "R_Close": "Actual Close (t)",
    "Target": "Actual Close (t+1)"
})
display_data = display_data[["Actual Close (t)", "Actual Close (t+1)", "Predicted"]].round(2)

st.dataframe(
    display_data.style.format("₹{:.2f}"),
    use_container_width=True
)