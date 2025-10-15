
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, r2_score
from darts.utils.timeseries_generation import datetime_attribute_timeseries


# -- 1. Load and Prepare Data --
def load_and_prepare_data(tickers, start_date, end_date):
    """
    Load stock data from yfinance and create a list of TimeSeries objects.
    """
    all_series = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True )
        series = TimeSeries.from_dataframe(stock_data, value_cols=['Close'], freq='B')
        all_series[ticker] = series
    return all_series

# -- 2. Create Covariates (Indicators) --
def create_covariates(series):
    """
    Create covariates from the original time series.
    - RSI
    - Moving Average
    - Time-based features
    """
    # RSI
    rsi_period = 14
    diff = series.to_series().diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_series = TimeSeries.from_series(rsi, freq='B')

    # Moving Average
    ma_period = 20
    ma_series = series.to_series().rolling(window=ma_period).mean()
    ma_series = TimeSeries.from_series(ma_series, freq='B')

    # Time-based features
    time_covariates = datetime_attribute_timeseries(series, attribute='month', one_hot=True)
    time_covariates = time_covariates.stack(datetime_attribute_timeseries(series, attribute='day_of_week', one_hot=True))

    # Align and stack all covariates
    covariates = rsi_series.stack(ma_series)
    covariates = covariates.stack(time_covariates)
    
    # Fill missing values that may have been introduced by indicators
    covariates_df = covariates.to_dataframe()
    covariates_df.ffill(inplace=True)
    covariates_df.bfill(inplace=True)
    covariates = TimeSeries.from_dataframe(covariates_df, freq='B')

    return covariates

# -- 3. Train, Predict, and Plot --
def train_and_forecast(ticker, series, covariates):
    """
    Train the N-BEATS model, make predictions, and plot the results.
    """
    # Split data
    train_series, val_series = series.split_before(pd.Timestamp('2024-01-01'))
    train_covariates, val_covariates = covariates.split_before(pd.Timestamp('2024-01-01'))

    # Scale data
    series_scaler = Scaler()
    train_series_scaled = series_scaler.fit_transform(train_series)
    val_series_scaled = series_scaler.transform(val_series)

    covariates_scaler = Scaler()
    covariates_scaled = covariates_scaler.fit_transform(covariates)
    train_covariates_scaled, _ = covariates_scaled.split_before(pd.Timestamp('2024-01-01'))

    # -- N-BEATS Model --
    # Optimized parameters for N-BEATS
    model = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=7,
        n_epochs=100,
        random_state=42,
        generic_architecture=True, # Generic architecture is often better for stock data
        num_stacks=2,
        num_blocks=3,
        num_layers=4,
        layer_widths=256,
        dropout=0.1,
        batch_size=32,
        pl_trainer_kwargs={"accelerator": "cpu", "devices": 1} # Use CPU
    )

    model.fit(train_series_scaled, past_covariates=train_covariates_scaled)

    # Make predictions
    pred_series = model.predict(n=len(val_series), series=train_series_scaled, past_covariates=covariates_scaled)
    pred_series = series_scaler.inverse_transform(pred_series)

    # Evaluate the model
    print(f"--- {ticker} Model Evaluation ---")
    print(f"MAPE: {mape(val_series, pred_series):.2f}%")
    print(f"R2 Score: {r2_score(val_series, pred_series):.2f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    series.plot(label='Actual')
    pred_series.plot(label='Forecast')
    plt.title(f'{ticker} Stock Price Forecast with N-BEATS')
    plt.legend()
    plt.show()

# -- Main Execution --
if __name__ == '__main__':
    TICKERS = ['MSFT', 'AAPL']
    START_DATE = '2020-01-01'
    END_DATE = '2024-10-14'

    all_stock_series = load_and_prepare_data(TICKERS, START_DATE, END_DATE)

    for ticker, series in all_stock_series.items():
        print(f"--- Processing {ticker} ---")
        covariates = create_covariates(series)
        
        # Align series and covariates
        start = max(series.start_time(), covariates.start_time())
        end = min(series.end_time(), covariates.end_time())
        series = series.slice(start, end)
        covariates = covariates.slice(start, end)
        
        train_and_forecast(ticker, series, covariates)
