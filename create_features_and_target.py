# create_features_and_target.py
import pandas as pd

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

# Drop rows where target is NaN (last row) and any remaining NaN from rolling
df = df.dropna()

df.to_csv("reliance_final_model_ready.csv")
print(f"Feature engineering complete!")
print(f"   → Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   → Target column added: 'Target' = next day R_Close")
print(f"   → Saved as reliance_final_model_ready.csv")