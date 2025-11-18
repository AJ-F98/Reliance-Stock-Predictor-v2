# app.py — FINAL & FLAWLESS: Only ONE clean arrow (no Streamlit default)
import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta

st.set_page_config(page_title="Reliance Predictor", layout="wide")
st.title("Reliance Industries – Next-Day Close Prediction")

# Load model & data
model = joblib.load("reliance_model.pkl")

df = pd.read_csv("reliance_final_model_ready_live.csv", index_col=0, parse_dates=True)
cols_to_drop = ['Target', 'R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
features = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# Today's values
today_close   = round(df['R_Close'].iloc[-1], 2)
today_date    = df.index[-1].date()
tomorrow_date = today_date + timedelta(days=1)
tomorrow_pred = round(float(model.predict(features.iloc[-1:])[0]), 2)

# Arrow logic
change = tomorrow_pred - today_close
if change > 0:
    arrow = "↑"
    color = "#00C853"   # pure green
elif change < 0:
    arrow = "↓"
    color = "#FF1744"   # pure red
else:
    arrow = "→"
    color = "#9E9E9E"   # grey

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Close Price (Today)**")
    st.markdown(f"### {today_date:%d-%b-%Y}")
    st.markdown(f"<h2 style='color:#E0E0E0;'>₹{today_close:.2f}</h2>", unsafe_allow_html=True)

with col2:
    st.markdown("**Predicted Close (Tomorrow)**")
    st.markdown(f"### {tomorrow_date:%d-%b-%Y}")
    st.markdown(
        f"<h2 style='color:{color}; margin:0;'>{arrow} ₹{tomorrow_pred:.2f}</h2>",
        unsafe_allow_html=True
    )

# Table
st.markdown("---")
st.subheader("Prediction History")

recent = df.tail(7).copy()
preds  = model.predict(features.tail(7)).round(2)

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