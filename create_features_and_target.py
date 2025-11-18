# create_features_and_target.py
import pandas as pd
from datetime import datetime

print("Loading data with technical indicators...")
df = pd.read_csv("reliance_with_features.csv", index_col=0, parse_dates=True)

# Target = next day's Reliance close price
df['Target'] = df['R_Close'].shift(-1)

# Lagged returns (very powerful)
for lag in [1, 2, 3, 5, 10]:
    df[f'R_return_{lag}d'] = df['R_Close'].pct_change(lag)

# Rolling statistics
df['R_vol_5d']  = df['R_Close'].pct_change().rolling(5).std()
df['R_vol_20d'] = df['R_Close'].pct_change().rolling(20).std()
df['R_mom_10d'] = df['R_Close'].pct_change(10)

# Market & macro returns
df['Nifty_return'] = df['N_Close'].pct_change()
df['Crude_return'] = df['C_Close'].pct_change()
df['FX_return']    = df['FX_Close'].pct_change()

# Create Target column
df['Target'] = df['R_Close'].shift(-1)

# 1. Training version → safe (drop last row with NaN Target)
df_training = df.dropna()
df_training.to_csv("reliance_final_model_ready.csv")

# 2. Live version → KEEP the last row (used by Streamlit app for today's prediction)
#    Target = NaN is totally fine for inference
df_live = df.copy()
df_live.to_csv("reliance_final_model_ready_live.csv")

print("Feature engineering complete!")
print(f"   → Training CSV  : {df_training.shape[0]:,} rows → up to {df_training.index[-1].date()}")
print(f"   → Live CSV      : {df_live.shape[0]:,} rows → up to {df_live.index[-1].date()} ← used for tonight's prediction")
print("   → Both files saved:")
print("        • reliance_final_model_ready.csv      (for model training)")
print("        • reliance_final_model_ready_live.csv (for live app)")