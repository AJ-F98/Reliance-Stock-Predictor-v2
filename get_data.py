import yfinance as yf
import pandas as pd

tickers = ["RELIANCE.NS", "^NSEI", "CL=F", "INR=X"]
prefix  = ["R",          "N",       "C",     "FX"]

print("Downloading data...")

raw = yf.download(tickers, start="2018-01-01", progress=False, group_by='ticker')

clean_data = pd.DataFrame(index=raw.index)

for ticker, short in zip(tickers, prefix):
    if ticker in raw.columns.get_level_values(0):
        df = raw[ticker]
    else:
        df = raw[[col for col in raw.columns if ticker in str(col)]]
        if len(df.columns) > 0:
            df = df.iloc[:, :5]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if isinstance(df, pd.DataFrame) and len(df.columns) >= 5:
        clean_data[f"{short}_Open"]   = df['Open']
        clean_data[f"{short}_High"]   = df['High']
        clean_data[f"{short}_Low"]    = df['Low']
        clean_data[f"{short}_Close"]  = df['Close']
        clean_data[f"{short}_Vol"]    = df['Volume']

clean_data = clean_data.dropna()

# ROUND ALL PRICE COLUMNS TO 2 DECIMALS BEFORE SAVING
price_columns = [col for col in clean_data.columns if col.endswith(('Open', 'High', 'Low', 'Close'))]
clean_data[price_columns] = clean_data[price_columns].round(2)

# Save – now beautiful from the first moment
clean_data.to_csv("reliance_raw_2018_today.csv")

print(f"reliance_raw_2018_today.csv saved → {clean_data.shape[0]:,} rows × {clean_data.shape[1]} columns")