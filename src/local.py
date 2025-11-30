import os
import pandas as pd
import rich
from src.main import run_optimisation

result = run_optimisation(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2025-01-01",
    end_date="2025-11-30"
)

print(f"Optimal Weights: {result['weights']}")
print(f"Predicted Returns: {result['predicted_returns']}")
print(f"Current Prices: {result['actual_prices_last_month']}")
print(f"Prediction Date: {pd.to_datetime(result['date'])+pd.Timedelta(days=1)}")