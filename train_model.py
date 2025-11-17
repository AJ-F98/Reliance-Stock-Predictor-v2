# train_model.py  → upgraded version (drops MAPE hard)
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import numpy as np

df = pd.read_csv("reliance_final_model_ready.csv", index_col=0, parse_dates=True)

cols_to_drop = ['R_Vol', 'N_Vol', 'C_Vol', 'FX_Vol']
X = df.drop(['Target'] + cols_to_drop, axis=1)
y = df['Target']

tscv = TimeSeriesSplit(n_splits=5)
mape_scores = []


for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=10,
        num_leaves=100,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, pred) * 100
    mape_scores.append(mape)
    print(f"   Fold {fold+1} MAPE: {mape:.3f}%")

print(f"\nNEW Average MAPE = {np.mean(mape_scores):.3f}%")

# Retrain on full data
final_model = LGBMRegressor(
    n_estimators=1200, learning_rate=0.03, max_depth=10, num_leaves=100,
    subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1
)
final_model.fit(X, y)
joblib.dump(final_model, "reliance_model.pkl")

# Today’s prediction again
latest_data = X.iloc[-1:].copy()
prediction = final_model.predict(latest_data)[0].round(2)
actual = 1518.30
error = abs(prediction - actual)

print(f"\nToday’s (Nov 17) predicted close: ₹{prediction}")
