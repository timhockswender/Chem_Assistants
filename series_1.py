
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, r2_score
from darts.utils.timeseries_generation import datetime_attribute_timeseries


def print_series_debug(series, label, sample_rows=5):
    df = series.to_dataframe()
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=np.float64, copy=False)
    total_values = values.size
    nan_count = int(np.isnan(values).sum()) if total_values else 0
    inf_count = int(np.isinf(values).sum()) if total_values else 0
    finite_mask = np.isfinite(values) if total_values else np.array([], dtype=bool)
    finite_values = values[finite_mask]

    print(f"{label} - length: {len(series)}, shape: {values.shape}")
    print(f"{label} - date range: {series.start_time()} to {series.end_time()}")
    components = getattr(series, "components", None)
    if components is not None:
        components = list(components)
        max_components_to_show = 10
        display_components = components[:max_components_to_show]
        if len(components) > max_components_to_show:
            display_components.append("...")
        print(f"{label} - components: {display_components}")
    print(
        f"{label} - NaNs: {nan_count}, Infs: {inf_count}, total values: {total_values}"
    )
    if finite_values.size > 0:
        print(
            f"{label} - min/mean/std/max: "
            f"{finite_values.min():.6f} / "
            f"{finite_values.mean():.6f} / "
            f"{finite_values.std(ddof=0):.6f} / "
            f"{finite_values.max():.6f}"
        )
    else:
        print(f"{label} - No finite values available for statistics.")
    if sample_rows and sample_rows > 0:
        print(f"{label} - sample values:\n{df.head(sample_rows)}")


# -- 1. Load and Prepare Data --
def load_and_prepare_data(tickers, start_date, end_date):
    """
    Load stock data from yfinance and create a list of TimeSeries objects.
    """
    all_series = {}
    for ticker in tickers:
        stock_data = yf.download(
            ticker, start=start_date, end=end_date, auto_adjust=True
        )

        if stock_data.empty:
            print(f"[Data Load] {ticker} returned no data.")
            continue

        series = TimeSeries.from_dataframe(stock_data, value_cols=['Close'], freq='B')
        close_values = stock_data['Close'].to_numpy()

        print(f"--- Data diagnostics for {ticker} ---")
        print(
            f"[Data Load] {ticker} raw dataframe shape: {stock_data.shape},"
            f" date range: {stock_data.index.min()} to {stock_data.index.max()}"
        )
        print(
            f"[Data Load] {ticker} Close NaNs: {int(np.isnan(close_values).sum())}, "
            f"Infs: {int(np.isinf(close_values).sum())}"
        )
        print(f"[Data Load] {ticker} Close sample:\n{stock_data['Close'].head()}")
        print_series_debug(series, f"[Data Load] {ticker} close price series")

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

    print_series_debug(covariates, "[Covariates] Combined covariates", sample_rows=5)

    return covariates

# -- 3. Train, Predict, and Plot --
def train_and_forecast(ticker, series, covariates):
    """
    Train the N-BEATS model, make predictions, and plot the results.
    """
    split_timestamp = pd.Timestamp('2024-01-01')

    # Split data
    train_series, val_series = series.split_before(split_timestamp)
    train_covariates, val_covariates = covariates.split_before(split_timestamp)

    print(f"--- {ticker} Train/Validation Split Diagnostics ---")
    print_series_debug(
        train_series, f"[Split] {ticker} train target series", sample_rows=5
    )
    print_series_debug(
        val_series, f"[Split] {ticker} validation target series", sample_rows=5
    )
    print_series_debug(
        train_covariates, f"[Split] {ticker} train covariates", sample_rows=5
    )
    print_series_debug(
        val_covariates, f"[Split] {ticker} validation covariates", sample_rows=5
    )

    # Scale data
    series_scaler = Scaler()
    train_series_scaled = series_scaler.fit_transform(train_series)
    val_series_scaled = series_scaler.transform(val_series)

    print(f"--- {ticker} Scaling Diagnostics (Target) ---")
    print_series_debug(
        train_series_scaled, f"[Scaling] {ticker} train series scaled", sample_rows=5
    )
    print_series_debug(
        val_series_scaled, f"[Scaling] {ticker} validation series scaled", sample_rows=5
    )

    covariates_scaler = Scaler()
    covariates_scaled = covariates_scaler.fit_transform(covariates)
    train_covariates_scaled, val_covariates_scaled = covariates_scaled.split_before(
        split_timestamp
    )

    print(f"--- {ticker} Scaling Diagnostics (Covariates) ---")
    print_series_debug(
        covariates_scaled, f"[Scaling] {ticker} all covariates scaled", sample_rows=5
    )
    print_series_debug(
        train_covariates_scaled,
        f"[Scaling] {ticker} train covariates scaled",
        sample_rows=5,
    )
    print_series_debug(
        val_covariates_scaled,
        f"[Scaling] {ticker} validation covariates scaled",
        sample_rows=5,
    )

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

    print(f"--- {ticker} Model Training Diagnostics ---")
    training_history = getattr(model, "training_loss_history", None)
    if training_history:
        print(
            f"[Model Fit] {ticker} training loss history length: {len(training_history)}"
        )
        print(
            f"[Model Fit] {ticker} last 5 training losses: {training_history[-5:]}"
        )
        print(f"[Model Fit] {ticker} final training loss: {training_history[-1]}")
    else:
        print(f"[Model Fit] {ticker} training loss history unavailable or empty.")
    print(f"[Model Fit] {ticker} _fit_called flag: {getattr(model, '_fit_called', 'N/A')}")
    model_module = getattr(model, "model", None)
    print(f"[Model Fit] {ticker} underlying module type: {type(model_module)}")
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        current_epoch = getattr(trainer, "current_epoch", None)
        max_epochs = getattr(trainer, "max_epochs", None)
        if current_epoch is not None:
            print(f"[Model Fit] {ticker} trainer current_epoch: {current_epoch}")
        if max_epochs is not None:
            print(f"[Model Fit] {ticker} trainer max_epochs: {max_epochs}")
        try:
            logged_metrics = trainer.callback_metrics
            if logged_metrics:
                print(
                    f"[Model Fit] {ticker} trainer callback metrics: {logged_metrics}"
                )
        except Exception as exc:
            print(f"[Model Fit] {ticker} trainer metrics unavailable: {exc}")
    else:
        print(f"[Model Fit] {ticker} trainer object unavailable.")

    # Make predictions
    pred_series = None
    pred_series_scaled = model.predict(
        n=len(val_series),
        series=train_series_scaled,
        past_covariates=covariates_scaled,
    )

    if pred_series_scaled is None or len(pred_series_scaled) == 0:
        print(f"[Prediction] {ticker} scaled prediction is None or empty.")
    else:
        print(
            f"[Prediction] {ticker} scaled prediction length vs validation (scaled): "
            f"{len(pred_series_scaled)} vs {len(val_series_scaled)}"
        )
        print(
            f"[Prediction] {ticker} scaled prediction date range: "
            f"{pred_series_scaled.start_time()} to {pred_series_scaled.end_time()}"
        )
        print_series_debug(
            pred_series_scaled,
            f"[Prediction] {ticker} scaled prediction",
            sample_rows=5,
        )
        try:
            pred_series = series_scaler.inverse_transform(pred_series_scaled)
        except Exception as exc:
            print(f"[Prediction] {ticker} inverse transform failed: {exc}")
            pred_series = None
        else:
            print(
                f"[Prediction] {ticker} prediction length vs validation: "
                f"{len(pred_series)} vs {len(val_series)}"
            )
            print_series_debug(
                pred_series,
                f"[Prediction] {ticker} inverse transformed prediction",
                sample_rows=5,
            )

    if pred_series is None or len(pred_series) == 0:
        print(
            f"[Prediction] {ticker} Forecast not generated; skipping evaluation and plotting."
        )
        return

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

        print_series_debug(series, f"[Alignment] {ticker} aligned target series", sample_rows=5)
        print_series_debug(
            covariates, f"[Alignment] {ticker} aligned covariates", sample_rows=5
        )

        train_and_forecast(ticker, series, covariates)
