# app.py
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import pandas_ta as ta
from datetime import date
import numpy as np

st.set_page_config(page_title="Reliance Stock Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")

model = load_model()

# Function to get live data up to today
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_live_data():
    tickers = ["RELIANCE.NS", "^NSEI", "CL=F", "INR=X"]
    data = yf.download(tickers, period="3mo", progress=False, group_by='ticker')
    clean_data = pd.DataFrame(index=data.index)
    prefixes = ["R", "N", "C", "FX"]
    for ticker, prefix in zip(tickers, prefixes):
        if ticker in data.columns.get_level_values(0):
            df_t = data[ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
            clean_data[f"{prefix}_Open"] = df_t['Open']
            clean_data[f"{prefix}_High"] = df_t['High']
            clean_data[f"{prefix}_Low"] = df_t['Low']
            clean_data[f"{prefix}_Close"] = df_t['Close']
            clean_data[f"{prefix}_Vol"] = df_t['Volume']
    clean_data = clean_data.dropna()
    price_cols = [col for col in clean_data.columns if col.endswith(('Open', 'High', 'Low', 'Close'))]
    clean_data[price_cols] = clean_data[price_cols].round(2)
    return clean_data

# Build features function (self-contained, no external imports)
def build_features(df):
    # Technical indicators
    rel = df[['R_Open', 'R_High', 'R_Low', 'R_Close', 'R_Vol']].tail(100).copy()
    rel.columns = ['open', 'high', 'low', 'close', 'volume']
    rel['RSI_14'] = ta.rsi(rel['close'], length=14)
    rel['RSI_7'] = ta.rsi(rel['close'], length=7)
    rel['EMA_12'] = ta.ema(rel['close'], length=12)
    rel['EMA_26'] = ta.ema(rel['close'], length=26)
    rel['SMA_20'] = ta.sma(rel['close'], length=20)
    rel['SMA_50'] = ta.sma(rel['close'], length=50)
    macd_df = ta.macd(rel['close'])
    rel = rel.join(macd_df.add_prefix('MACD_'))
    bb_df = ta.bbands(rel['close'], length=20)
    rel = rel.join(bb_df.add_prefix('BB_'))
    stoch_df = ta.stoch(rel['high'], rel['low'], rel['close'])
    rel = rel.join(stoch_df.add_prefix('STOCH_'))
    adx_df = ta.adx(rel['high'], rel['low'], rel['close'])
    rel = rel.join(adx_df.add_prefix('ADX_'))
    rel['ATR_14'] = ta.atr(rel['high'], rel['low'], rel['close'], length=14)
    rel['OBV'] = ta.obv(rel['close'], rel['volume'])
    rel['ROC_10'] = ta.roc(rel['close'], length=10)
    
    # Merge TA back
    ta_cols = [c for c in rel.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
    df_with_ta = df.join(rel[ta_cols].iloc[-len(df):]).dropna()
    
    # Lagged/rolling features
    df_with_ta['R_return_1d'] = df_with_ta['R_Close'].pct_change(1)
    df_with_ta['R_return_2d'] = df_with_ta['R_Close'].pct_change(2)
    df_with_ta['R_return_3d'] = df_with_ta['R_Close'].pct_change(3)
    df_with_ta['R_return_5d'] = df_with_ta['R_Close'].pct_change(5)
    df_with_ta['R_return_10d'] = df_with_ta['R_Close'].pct_change(10)
    df_with_ta['R_vol_5d'] = df_with_ta['R_Close'].pct_change().rolling(5).std()
    df_with_ta['R_vol_20d'] = df_with_ta['R_Close'].pct_change().rolling(20).std()
    df_with_ta['R_mom_10d'] = df_with_ta['R_Close'].pct_change(10)
    df_with_ta['Nifty_return'] = df_with_ta['N_Close'].pct_change()
    df_with_ta['Crude_return'] = df_with_ta['C_Close'].pct_change()
    df_with_ta['FX_return'] = df_with_ta['FX_Close'].pct_change()
    
    # Drop Target if exists, drop volumes, drop NaN
    if 'Target' in df_with_ta.columns:
        df_with_ta = df_with_ta.drop('Target', axis=1)
    cols_to_drop = ['R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    df_with_ta = df_with_ta.drop(columns=[col for col in cols_to_drop if col in df_with_ta.columns])
    df_features = df_with_ta.dropna()
    
    return df_features

# Get live data and features
live_df = get_live_data()
live_features = build_features(live_df)

# Use yesterday's row to predict today
if len(live_features) < 2:
    st.error("Not enough data for prediction. Try again later.")
    st.stop()

latest_date = live_features.index[-2].date()  # Yesterday
latest_row = live_features.iloc[-2: -1]  # Yesterday's features

prediction = model.predict(latest_row)[0]
prediction_rounded = round(prediction, 2)

# Actual close for today
today = date.today()
actual_close = None
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="5d")
if not hist.empty:
    today_hist = hist[hist.index.date == today]
    if not today_hist.empty:
        actual_close = round(today_hist["Close"].iloc[-1], 2)

# Display
col1, col2 = st.columns(2)
with col1:
    st.metric("Data up to (for prediction)", latest_date.strftime("%Y-%m-%d"))
    st.metric("Predicted Close (Today)", f"₹{prediction_rounded}")

with col2:
    if actual_close is not None:
        error = abs(prediction_rounded - actual_close)
        error_pct = (error / actual_close) * 100
        st.metric("Actual Close (Today)", f"₹{actual_close}")
        st.metric("Error", f"₹{error:.2f} ({error_pct:.2f}%)")
    else:
        st.info("Actual close updates after market close (3:30 PM IST)")

# Recent table
st.subheader("Recent Actuals & Predictions")
if len(live_features) >= 7:
    recent_features = live_features.tail(7)
    recent_actuals = live_df['R_Close'].tail(7).round(2)
    recent_df = pd.DataFrame({
        'Date': recent_features.index,
        'Actual Close': recent_actuals.values,
        'Predicted Next-Day': model.predict(recent_features).round(2)
    })
    recent_df = recent_df[['Date', 'Actual Close', 'Predicted Next-Day']]
    st.dataframe(recent_df, use_container_width=True)
else:
    st.info("Building recent history...")