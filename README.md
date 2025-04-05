## This is 24 hour hackathon project
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Sample cash flow data (daily)
data = pd.read_csv("cashflow.csv", parse_dates=["date"], index_col="date")
model = ARIMA(data['net_cash_flow'], order=(3,1,2))  # ARIMA(p,d,q)
model_fit = model.fit()

# 30-day forecast
forecast = model_fit.forecast(steps=30)
forecast.to_csv("forecast_30_days.csv")

