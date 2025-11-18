# app.py — FINAL & BULLETPROOF: ONLY ONE arrow (never both)
import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta

st.set_page_config(page_title="Reliance Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load model & data
@st.cache_resource
def load_model():
    return joblib.load("reliance_model.pkl")
model = load_model()

@st.cache_data(ttl=3600)
def load_live():
    df = pd.read_csv("reliance_final_model_ready_live.csv", index_col=0, parse_dates=True)
    cols_to_drop = ['Target', 'R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
    features = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df, features

df_full, df_features = load_live()

today_close   = round(df_full['R_Close'].iloc[-1], 2)
today_date    = df_full.index[-1].date()
tomorrow_date = today_date + timedelta(days=1)
tomorrow_pred = round(float(model.predict(df_features.iloc[-1:])[0]), 2)

change = tomorrow_pred - today_close

# Arrow + color logic
if change > 0:
    arrow = "↑"
    color = "normal"      # green
elif change < 0:
    arrow = "↓"
    color = "inverse"     # red
else:
    arrow = "→"
    color = "off"

col1, col2 = st.columns(2)

with col1:
    st.metric("Close Price (Today)", f"{today_date:%d-%b-%Y}", f"₹{today_close:.2f}")

with col2:
    st.metric(
        label="Predicted Close (Tomorrow)",
        value=f"{tomorrow_date:%d-%b-%Y}",
        delta=f"{arrow} ₹{tomorrow_pred:.2f}",
        delta_color=color,
        help=""        # ← THIS LINE REMOVES THE DEFAULT TINY ARROW
    )

# Table (unchanged)
st.subheader("Prediction History")
recent = df_full.tail(7).copy()
preds  = model.predict(df_features.tail(7)).round(2)

table = pd.DataFrame({
    "Date": recent.index.strftime("%d-%b-%Y"),
    "Close Price": recent["R_Close"].round(2).values,
    "Predicted Next Day": preds
})

tomorrow_row = pd.DataFrame({
    "Date": [tomorrow_date.strftime("%d-%b-%Y")],
    "Close Price": ["—"],
    "Predicted Next Day": [tomorrow_pred]
})
table = pd.concat([table, tomorrow_row], ignore_index=True)

st.dataframe(table.reset_index(drop=True), use_container_width=True, hide_index=True)