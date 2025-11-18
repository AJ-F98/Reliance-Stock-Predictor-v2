# Reliance Industries – Next-Day Close Price Predictor

Live Demo: https://reliance-stock-predictor.streamlit.app

A complete end-to-end machine learning project that predicts the next trading day’s closing price for Reliance Industries (RELIANCE.NS) using multi-modal data and LightGBM.

## Performance (as of Nov 2025)
- Most recent validation fold (2024–2025 data): **2.86% MAPE**
- Real-time prediction error on 17-Nov-2025: **₹1.02 (0.07%)**

## Features
- Daily OHLCV data for Reliance, Nifty 50, Crude Oil, USD/INR
- 25+ technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ADX, OBV, etc.)
- Lagged returns, rolling volatility, momentum, and macro returns
- LightGBM regressor with time-series cross-validation