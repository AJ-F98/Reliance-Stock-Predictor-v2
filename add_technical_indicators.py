import pandas as pd
import pandas_ta as ta

print("Loading & finalizing data...")
df = pd.read_csv("reliance_raw_2018_today.csv", index_col=0, parse_dates=True)

# Re-round prices (safety + consistency)
price_cols = [col for col in df.columns if col.endswith(('Open', 'High', 'Low', 'Close'))]
df[price_cols] = df[price_cols].round(2)

# Technical indicators on Reliance
rel = df[['R_Open', 'R_High', 'R_Low', 'R_Close', 'R_Vol']].copy()
rel.columns = ['open', 'high', 'low', 'close', 'volume']

print("Adding technical indicators...")
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
rel['OBV']    = ta.obv(rel['close'], rel['volume'])
rel['ROC_10'] = ta.roc(rel['close'], length=10)

ta_columns = [c for c in rel.columns if c not in ['open','high','low','close','volume']]
final = df.join(rel[ta_columns]).dropna()

final.to_csv("reliance_with_features.csv")

print(f"   → {final.shape[0]:,} rows × {final.shape[1]} columns")
print(f"   → {len(ta_columns)} powerful indicators added")
