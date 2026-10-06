"""Streamlit dashboard for an educational NSE/BSE stock forecasting project."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

# Keep TensorFlow startup messages concise in the student-facing terminal.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_utils import StockDataError, download_stock_data
from indicators import add_technical_indicators
from model_utils import (
    ModelTrainingError,
    TrainingResult,
    TrainingSettings,
    set_global_seed,
    train_or_load_models,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Wipro": "WIPRO.NS",
}
SECTIONS = [
    "Home",
    "Stock Analysis",
    "Model Training",
    "Prediction",
    "LSTM vs BiLSTM Comparison",
    "About Project",
]

st.set_page_config(
    page_title="NSE Stock Prediction | LSTM & BiLSTM",
    page_icon="📊",
    layout="wide",
)
set_global_seed(42)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Cache downloaded data for one hour to reduce repeat network requests."""
    return download_stock_data(ticker, start_date, end_date)


@st.cache_resource(show_spinner=False)
def cached_training(
    data: pd.DataFrame,
    ticker: str,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    units: int,
    patience: int,
    _progress_callback=None,
) -> TrainingResult:
    """Keep loaded Keras models in memory for the current Streamlit process."""
    settings = TrainingSettings(
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        units=units,
        patience=patience,
    )
    return train_or_load_models(
        data=data,
        ticker=ticker,
        settings=settings,
        model_root=MODEL_DIR,
        progress_callback=_progress_callback,
    )


def money(value: float) -> str:
    """Format a price as Indian rupees without assuming a future value."""
    return f"₹{value:,.2f}"


def metric_table(data: pd.DataFrame) -> pd.DataFrame:
    """Format metrics for a readable Streamlit table."""
    formatted = data.copy()
    for column in ["MAE", "RMSE"]:
        formatted[column] = formatted[column].map(money)
    formatted["MAPE (%)"] = formatted["MAPE (%)"].map(lambda value: f"{value:.2f}%")
    return formatted


def get_selected_data(
    ticker: str, start_date: date, end_date: date
) -> pd.DataFrame | None:
    """Download history and show friendly feedback when data is unavailable."""
    if start_date >= end_date:
        st.error("The start date must be earlier than the end date.")
        return None

    try:
        with st.spinner(f"Downloading historical data for {ticker}..."):
            return cached_stock_data(ticker, start_date, end_date)
    except StockDataError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("The data request could not be completed. Check your connection and retry.")
        with st.expander("Technical details"):
            st.code(str(exc))
    return None


def get_training_result(
    data: pd.DataFrame,
    ticker: str,
    settings: TrainingSettings,
) -> TrainingResult | None:
    """Train or reuse the selected models and provide progress feedback."""
    progress = st.progress(0, text="Preparing chronological training and test sets...")
    status = st.empty()
    last_value = {"value": 0}

    def report_progress(model_name: str, epoch: int, total_epochs: int) -> None:
        # Keras stops early when validation loss no longer improves.
        model_number = 1 if model_name == "LSTM" else 2
        progress_value = min(
            99,
            int(((model_number - 1) * settings.epochs + epoch) / (2 * settings.epochs) * 100),
        )
        last_value["value"] = max(last_value["value"], progress_value)
        progress.progress(
            last_value["value"],
            text=f"Training {model_name}: epoch {epoch} of up to {total_epochs}",
        )
        status.info(f"{model_name} training is in progress.")

    try:
        result = cached_training(
            data,
            ticker,
            settings.sequence_length,
            settings.epochs,
            settings.batch_size,
            settings.units,
            settings.patience,
            _progress_callback=report_progress,
        )
        progress.progress(100, text=result.cache_status)
        status.success(f"Both models are ready. {result.cache_status}.")
        return result
    except ModelTrainingError as exc:
        progress.empty()
        status.empty()
        st.error(str(exc))
        if exc.__cause__ is not None:
            with st.expander("Technical details"):
                st.code(str(exc.__cause__))
    except Exception as exc:
        progress.empty()
        status.empty()
        st.error("The models could not be loaded or trained. Try a wider date range.")
        with st.expander("Technical details"):
            st.code(str(exc))
    return None


def show_home(data: pd.DataFrame | None, ticker: str) -> None:
    """Show the project introduction and a real market snapshot if available."""
    st.title("Stock Price Prediction for NSE/BSE")
    st.subheader("LSTM and BiLSTM Deep Learning Models")
    st.write(
        "Explore historical Indian stock data, technical indicators, and a "
        "chronological comparison of two recurrent neural-network models."
    )
    st.info(
        "This is an educational experiment. Model forecasts are uncertain and "
        "are not financial advice."
    )

    if data is not None and not data.empty:
        latest = data.iloc[-1]
        latest_date = data.index[-1].strftime("%d %b %Y")
        previous_close = float(data["Close"].iloc[-2]) if len(data) > 1 else None
        change_text = None
        if previous_close not in (None, 0):
            change_pct = (float(latest["Close"]) - previous_close) / previous_close * 100
            change_text = f"{change_pct:+.2f}% from previous close"

        st.caption(f"Latest available observation: {latest_date} · {ticker}")
        columns = st.columns(4)
        columns[0].metric("Latest close", money(float(latest["Close"])), change_text)
        columns[1].metric("Open", money(float(latest["Open"])))
        columns[2].metric("Day high / low", f"{money(float(latest['High']))} / {money(float(latest['Low']))}")
        columns[3].metric("Volume", f"{float(latest['Volume']):,.0f}")
    else:
        st.warning("No market snapshot is available for the selected date range.")

    st.markdown("### Project overview")
    st.write(
        "The app downloads historical OHLCV data from Yahoo Finance, derives "
        "technical indicators, and trains LSTM and BiLSTM models on past "
        "observations. The newest 20% of the selected period is reserved for "
        "testing; the models are never trained on that test period."
    )
    st.markdown(
        "**Start here:** choose a company and date range in the sidebar, open "
        "**Stock Analysis** to inspect the data, then open **Model Training** "
        "to compare the models."
    )


def show_stock_analysis(data: pd.DataFrame, ticker: str) -> None:
    """Render historical pricing, volume, OHLCV, and technical indicators."""
    st.title("Stock Analysis")
    st.caption(f"{ticker} · {len(data):,} trading days in the selected period")
    indicator_data = add_technical_indicators(data)

    price_chart = go.Figure()
    price_chart.add_trace(
        go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close")
    )
    price_chart.add_trace(
        go.Scatter(
            x=indicator_data.index,
            y=indicator_data["SMA 20"],
            mode="lines",
            name="SMA 20",
        )
    )
    price_chart.add_trace(
        go.Scatter(
            x=indicator_data.index,
            y=indicator_data["SMA 50"],
            mode="lines",
            name="SMA 50",
        )
    )
    price_chart.update_layout(
        title="Historical closing price and moving averages",
        xaxis_title="Trading date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        legend_title="Series",
    )
    st.plotly_chart(price_chart, use_container_width=True)

    volume_chart = go.Figure(
        go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="#5470C6")
    )
    volume_chart.update_layout(
        title="Daily trading volume",
        xaxis_title="Trading date",
        yaxis_title="Shares traded",
        hovermode="x unified",
    )
    st.plotly_chart(volume_chart, use_container_width=True)

    indicator_columns = st.columns(2)
    with indicator_columns[0]:
        rsi_chart = go.Figure()
        rsi_chart.add_trace(
            go.Scatter(x=indicator_data.index, y=indicator_data["RSI"], name="RSI (14)")
        )
        rsi_chart.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_chart.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_chart.update_layout(
            title="Relative Strength Index (14-day)",
            xaxis_title="Trading date",
            yaxis_title="RSI",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(rsi_chart, use_container_width=True)

    with indicator_columns[1]:
        macd_chart = go.Figure()
        macd_chart.add_trace(
            go.Scatter(x=indicator_data.index, y=indicator_data["MACD"], name="MACD")
        )
        macd_chart.add_trace(
            go.Scatter(
                x=indicator_data.index,
                y=indicator_data["MACD Signal"],
                name="Signal",
            )
        )
        macd_chart.add_trace(
            go.Bar(
                x=indicator_data.index,
                y=indicator_data["MACD Histogram"],
                name="Histogram",
                opacity=0.35,
            )
        )
        macd_chart.update_layout(
            title="MACD (12, 26, 9)",
            xaxis_title="Trading date",
            yaxis_title="MACD value",
            hovermode="x unified",
        )
        st.plotly_chart(macd_chart, use_container_width=True)

    st.markdown("### Latest historical rows")
    display_data = indicator_data.reset_index().tail(10).copy()
    display_data["Date"] = pd.to_datetime(display_data["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_data, use_container_width=True, hide_index=True)
    st.caption(
        "SMA is a simple moving average. RSI is a 14-day momentum indicator. "
        "MACD uses 12- and 26-day exponential averages with a 9-day signal line."
    )


def show_model_training(result: TrainingResult, settings: TrainingSettings) -> None:
    """Show split information, model scores, and training loss curves."""
    st.title("Model Training")
    st.write(
        f"Training settings: **{settings.sequence_length} trading days per "
        f"sequence**, up to **{settings.epochs} epochs**, batch size "
        f"**{settings.batch_size}**, and **{settings.units}** LSTM units."
    )
    columns = st.columns(3)
    columns[0].metric("Training-period rows", f"{result.train_rows:,}")
    columns[1].metric("Test-period predictions", f"{result.test_rows:,}")
    columns[2].metric("Lower test RMSE", result.winner)

    st.markdown("### Test-set metrics")
    st.dataframe(metric_table(result.metrics), use_container_width=True, hide_index=True)
    st.caption(
        "Metrics are calculated in rupees on the held-out final 20% of dates. "
        "The lower RMSE model performed better on this selected test period."
    )

    loss_chart = go.Figure()
    for model_name, losses in result.histories.items():
        loss_chart.add_trace(
            go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=losses,
                mode="lines+markers",
                name=f"{model_name} training loss",
            )
        )
    loss_chart.update_layout(
        title="Training loss by epoch",
        xaxis_title="Epoch",
        yaxis_title="Mean squared error (scaled close)",
        hovermode="x unified",
    )
    st.plotly_chart(loss_chart, use_container_width=True)
    st.caption(f"Model bundle status: {result.cache_status}.")


def show_prediction(result: TrainingResult) -> None:
    """Plot date-aligned test forecasts and the experimental next-period estimate."""
    st.title("Prediction")
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=result.predictions["Date"],
            y=result.predictions["Actual"],
            mode="lines",
            name="Actual close",
            line={"width": 2},
        )
    )
    chart.add_trace(
        go.Scatter(
            x=result.predictions["Date"],
            y=result.predictions["LSTM"],
            mode="lines",
            name="LSTM predicted close",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=result.predictions["Date"],
            y=result.predictions["BiLSTM"],
            mode="lines",
            name="BiLSTM predicted close",
        )
    )
    chart.update_layout(
        title="Actual and model-predicted closing prices on the test period",
        xaxis_title="Trading date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        legend_title="Series",
    )
    st.plotly_chart(chart, use_container_width=True)

    st.warning("Experimental model forecast — not financial advice.")
    forecast_columns = st.columns(2)
    forecast_columns[0].metric(
        "LSTM next-period estimate", money(result.forecast["LSTM"])
    )
    forecast_columns[1].metric(
        "BiLSTM next-period estimate", money(result.forecast["BiLSTM"])
    )
    st.caption(
        "These experimental values are generated from the latest available "
        "60-day (or selected sequence-length) OHLCV window and are not reliable "
        "assurances of future prices."
    )


def show_comparison(result: TrainingResult) -> None:
    """Compare both architectures using genuine held-out predictions."""
    st.title("LSTM vs BiLSTM Comparison")
    st.dataframe(metric_table(result.metrics), use_container_width=True, hide_index=True)
    st.success(
        f"Lower test RMSE for this dataset and date range: **{result.winner}**."
    )
    st.caption(
        "This comparison applies only to this ticker, date range, training setup, "
        "and held-out test period. It does not imply dependable future performance."
    )

    comparison_chart = go.Figure()
    for model_name in ["LSTM", "BiLSTM"]:
        row = result.metrics.loc[result.metrics["Model"] == model_name].iloc[0]
        comparison_chart.add_trace(
            go.Bar(
                x=[model_name],
                y=[float(row["RMSE"])],
                name=model_name,
                text=[money(float(row["RMSE"]))],
                textposition="outside",
            )
        )
    comparison_chart.update_layout(
        title="Test RMSE comparison (lower is better)",
        xaxis_title="Model",
        yaxis_title="RMSE (INR)",
        showlegend=False,
    )
    st.plotly_chart(comparison_chart, use_container_width=True)


def show_about() -> None:
    """Explain the project's methods and limitations in student-friendly terms."""
    st.title("About Project")
    st.markdown(
        """
### Project title
**Stock Price Prediction for NSE/BSE Using LSTM and BiLSTM Deep Learning Models**

### Problem and objectives
Historical stock prices are time-series data: the order of observations matters.
This project studies whether recurrent neural networks can learn patterns from
past OHLCV observations and compares their test-set errors. Its objectives are
to collect historical market data, calculate common technical indicators,
build two deep-learning models, and evaluate them with a chronological holdout.

### Dataset and preprocessing
Daily Open, High, Low, Close, and Volume observations are retrieved from Yahoo
Finance through `yfinance`. The first 80% of dates are used as the training
period and the last 20% as the test period. No random shuffling is used.
Min-max scaling parameters are learned only from the training rows and then
applied to both periods. Each sample uses a configurable number of earlier
trading days to estimate the following day's close.

### Models and metrics
The LSTM model has two LSTM layers with dropout regularization, followed by
dense output layers. The BiLSTM wraps both recurrent layers in a bidirectional
layer. Early stopping monitors validation loss from the end of the training
sequences and restores the best weights. MAE, RMSE, and MAPE are calculated in
the original price scale.

### Limitations
Historical price patterns do not include every factor that moves markets.
News, macroeconomic events, liquidity, policy changes, and market shocks can
make future prices differ sharply from historical patterns. BiLSTM also reads
each supplied historical input window in both directions; it does not receive
future market observations, but its design is not a simulation of live trading.
The forecast is experimental, may be inaccurate, and is not financial advice.

### Future scope
Possible extensions include walk-forward validation, additional market and
fundamental features, model hyperparameter studies, comparison with simple
statistical baselines, and stronger uncertainty estimates.
"""
    )


def main() -> None:
    """Build the navigation and run the selected dashboard section."""
    st.sidebar.title("Project controls")
    section = st.sidebar.radio("Dashboard section", SECTIONS)
    company = st.sidebar.selectbox("Indian stock", list(STOCKS.keys()))
    ticker = STOCKS[company]

    today = date.today()
    default_start = today - timedelta(days=365 * 5)
    date_range = st.sidebar.date_input(
        "Historical date range",
        value=(default_start, today),
        min_value=date(2000, 1, 1),
        max_value=today,
    )

    st.sidebar.markdown("### Model settings")
    sequence_length = st.sidebar.slider(
        "Sequence length (trading days)", min_value=20, max_value=120, value=60, step=5
    )
    epochs = st.sidebar.slider("Maximum epochs", min_value=5, max_value=100, value=30)
    batch_size = st.sidebar.select_slider(
        "Batch size", options=[8, 16, 32, 64], value=32
    )
    units = st.sidebar.select_slider("LSTM units", options=[16, 32, 64, 96], value=64)
    patience = st.sidebar.slider("Early stopping patience", 2, 12, 5)
    st.sidebar.caption(
        "Use a wider date range and a shorter sequence if the selected history "
        "does not contain enough samples."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Educational project only · Not financial advice")

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.error("Select both a start date and an end date.")
        return
    start_date, end_date = date_range
    start_date, end_date = date(start_date.year, start_date.month, start_date.day), date(
        end_date.year, end_date.month, end_date.day
    )

    if section == "About Project":
        show_about()
        return

    data = get_selected_data(ticker, start_date, end_date)
    if data is None:
        return

    if section == "Home":
        show_home(data, ticker)
        return
    if section == "Stock Analysis":
        show_stock_analysis(data, ticker)
        return

    settings = TrainingSettings(
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        units=units,
        patience=patience,
    )
    result = get_training_result(data, ticker, settings)
    if result is None:
        return

    if section == "Model Training":
        show_model_training(result, settings)
    elif section == "Prediction":
        show_prediction(result)
    elif section == "LSTM vs BiLSTM Comparison":
        show_comparison(result)


if __name__ == "__main__":
    main()
