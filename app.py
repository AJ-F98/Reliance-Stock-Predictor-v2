# app.py
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import date, timedelta
import pandas_ta as ta
from create_features_and_target import create_features  # Reuse your feature engineering logic

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
    data = yf.download(tickers, period="1mo", progress=False, group_by='ticker')
    # Flatten and clean columns (same as data_downloader.py)
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
    clean_data[ [col for col in clean_data.columns if col.endswith(('Open', 'High', 'Low', 'Close')) ] ] = clean_data[ [col for col in clean_data.columns if col.endswith(('Open', 'High', 'Low', 'Close')) ] ].round(2)
    return clean_data

# Build features for latest row (reuse logic from add_technical_features.py + create_features_and_target.py)
def build_latest_features(live_df):
    # Add TA indicators (same as add_technical_features.py)
    rel = live_df[['R_Open', 'R_High', 'R_Low', 'R_Close', 'R_Vol']].tail(100).copy()  # Need ~100 days for indicators
    rel.columns = ['open', 'high', 'low', 'close', 'volume']
    rel = rel.join(ta.rsi(rel['close'], length=14).rename('RSI_14'))
    rel = rel.join(ta.rsi(rel['close'], length=7).rename('RSI_7'))
    rel = rel.join(ta.ema(rel['close'], length=12).rename('EMA_12'))
    rel = rel.join(ta.ema(rel['close'], length=26).rename('EMA_26'))
    rel = rel.join(ta.sma(rel['close'], length=20).rename('SMA_20'))
    rel = rel.join(ta.sma(rel['close'], length=50).rename('SMA_50'))
    rel = rel.join(ta.macd(rel['close']).add_prefix('MACD_'))
    rel = rel.join(ta.bbands(rel['close'], length=20).add_prefix('BB_'))
    rel = rel.join(ta.stoch(rel['high'], rel['low'], rel['close']).add_prefix('STOCH_'))
    rel = rel.join(ta.adx(rel['high'], rel['low'], rel['close']).add_prefix('ADX_'))
    rel['ATR_14'] = ta.atr(rel['high'], rel['low'], rel['close'], length=14)
    rel['OBV'] = ta.obv(rel['close'], rel['volume'])
    rel['ROC_10'] = ta.roc(rel['close'], length=10)
    
    # Merge TA to live_df
    ta_cols = [c for c in rel.columns if c not in ['open','high','low','close','volume']]
    live_with_ta = live_df.join(rel[ta_cols].iloc[-len(live_df):]).dropna()
    
    # Add lagged/rolling features (same as create_features_and_target.py)
    live_with_ta['R_return_1d'] = live_with_ta['R_Close'].pct_change(1)
    live_with_ta['R_return_2d'] = live_with_ta['R_Close'].pct_change(2)
    live_with_ta['R_return_3d'] = live_with_ta['R_Close'].pct_change(3)
    live_with_ta['R_vol_5d'] = live_with_ta['R_Close'].pct_change().rolling(5).std()
    live_with_ta['R_vol_20d'] = live_with_ta['R_Close'].pct_change().rolling(20).std()
    live_with_ta['R_mom_10d'] = live_with_ta['R_Close'].pct_change(10)
    live_with_ta['Nifty_return'] = live_with_ta['N_Close'].pct_change()
    live_with_ta['Crude_return'] = live_with_ta['C_Close'].pct_change()
    live_with_ta['FX_return'] = live_with_ta['FX_Close'].pct_change()
    live_features = live_with_ta.dropna()
    
    # Drop volumes for model match
    cols_to_drop = ['R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    live_features = live_features.drop(columns=[col for col in cols_to_drop if col in live_features.columns])
    
    return live_features

# Get live data and build features
live_df = get_live_data()
live_features = build_latest_features(live_df)

# Latest row for prediction (yesterday's data → predict today)
latest_date = live_features.index[-2].date()  # -2 because -1 is today (no target yet)
latest_row = live_features.iloc[-2:].drop(columns=['Target'] if 'Target' in live_features.columns else [])

# Prediction for today
prediction = model.predict(latest_row.iloc[-1:])[0]
prediction_rounded = round(prediction, 2)

# Actual close for today (18-Nov)
today = date.today()
actual_close = None
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="2d")
if not hist.empty and hist.index[-1].date() == today:
    actual_close = round(hist["Close"].iloc[-1], 2)

# Display
col1, col2 = st.columns(2)
with col1:
    st.metric("Data up to (for prediction)", latest_date.strftime("%Y-%m-%d"))
    st.metric("Predicted Close (Today)", f"₹{prediction_rounded}")

with col2:
    if actual_close:
        error = abs(prediction_rounded - actual_close)
        error_pct = error / actual_close * 100
        st.metric("Actual Close (Today)", f"₹{actual_close}")
        st.metric("Error", f"₹{error:.2f} ({error_pct:.2f}%)")
    else:
        st.info("Actual close updates after 3:30 PM IST")

# Recent table (last 7 days, with predictions)
st.subheader("Recent Actuals vs Predictions")
recent_hist = ticker.history(period="10d")
if not recent_hist.empty:
    recent_df = recent_hist.tail(7)[['Close']].rename(columns={'Close': 'R_Close'})
    recent_df.index = pd.to_datetime(recent_df.index.date)
    recent_df = recent_df.join(live_features[['R_Close', 'Nifty_return', 'Crude_return', 'FX_return', 'RSI_14', 'MACD_MACD_12_26_9', 'BB_BBM_20_2.0', 'ATR_14', 'R_return_1d', 'R_vol_5d']])  # Sample key features
    recent_df = recent_df.dropna()
    recent_df["Predicted"] = model.predict(recent_df.drop('R_Close', axis=1)).round(2)
    recent_df = recent_df[['R_Close', 'Predicted']].round(2)
    recent_df.columns = ["Actual Close", "Predicted Next-Day"]
    st.dataframe(recent_df.style.format("₹{:.2f}"), use_container_width=True)
else:
    st.info("Fetching recent data...")