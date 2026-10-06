# Stock Price Prediction for NSE/BSE Using LSTM and BiLSTM Deep Learning Models

## Introduction

This final-year Computer Science / AI-ML project is an educational Streamlit
dashboard for exploring Indian stock data and comparing two recurrent neural
networks: Long Short-Term Memory (LSTM) and Bidirectional LSTM (BiLSTM).
Historical daily prices are downloaded from Yahoo Finance with `yfinance`.

The application displays Open, High, Low, Close, and Volume (OHLCV) data,
historical charts, technical indicators, evaluation metrics, and experimental
next-period estimates. Model results are computed from the selected stock
history; the project does not contain sample or hard-coded performance values.

> **Important:** Stock forecasts are uncertain. This project is for learning and
> demonstration only and is not financial advice.

## Problem statement

Stock prices are time-series observations whose order matters. The project
studies whether LSTM and BiLSTM models can learn patterns from earlier OHLCV
observations and compares their predictions on a later, held-out test period.
It also demonstrates correct chronological splitting and training-only scaling
to avoid data leakage.

## Objectives

- Download real historical NSE stock data.
- Visualize closing prices, trading volume, and common technical indicators.
- Form configurable sequences of previous trading days.
- Train LSTM and BiLSTM models using TensorFlow/Keras and EarlyStopping.
- Evaluate test predictions using MAE, RMSE, and MAPE.
- Compare the models and identify the one with lower test RMSE.
- Demonstrate the pipeline with a simple interactive dashboard.

## Technologies

- Python
- TensorFlow / Keras
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- yfinance

## Installation

Python 3.11 or newer is recommended. From a terminal, change to this project
directory and install the dependencies:

```bash
cd stock_prediction
pip install -r requirements.txt
```

TensorFlow is a large dependency; its first installation can take several
minutes. A compatible 64-bit Python environment and internet connection are
required.

## How to run

```bash
cd stock_prediction
streamlit run app.py
```

The dashboard opens in a browser. Use the sidebar to choose a stock, date range,
sequence length, and training settings. Open **Stock Analysis** for charts and
indicators, then **Model Training**, **Prediction**, or
**LSTM vs BiLSTM Comparison** to train/load and inspect the models.

Available example tickers include `RELIANCE.NS`, `TCS.NS`, `INFY.NS`,
`HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`, and `WIPRO.NS`.

## Project architecture

```text
stock_prediction/
├── app.py                 # Streamlit navigation, sidebar, charts, and results
├── data_utils.py          # yfinance download and OHLCV validation
├── indicators.py          # SMA, RSI, and MACD calculations
├── model_utils.py         # Leakage-safe sequences, models, metrics, and caching
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── models/                # Saved model bundles (created as models are trained)
```

### Data workflow

1. Download daily OHLCV data for the selected stock and date range.
2. Sort observations by date and remove incomplete OHLCV rows.
3. Keep the first 80% of dates for training and the last 20% for testing.
4. Fit separate `MinMaxScaler` objects for the five input features and the
   closing-price target using training rows only.
5. Transform the full chronological series with those training-fitted scalers.
6. Build sequences in which each input window contains only observations before
   its target date. Training targets remain in the training period; test targets
   are in the final 20%.
7. Train and evaluate both models; convert predictions back to rupees.

For test-period predictions, a window may include older training-period rows:
those observations would have been available at the prediction date. No later
test observation is used as an input to predict an earlier test date.

## LSTM explanation

An LSTM is a recurrent neural network designed to learn patterns across ordered
sequences. Its memory cell and gates regulate which information from earlier
steps is retained or discarded. This project feeds a window of OHLCV values
from previous trading days into two stacked LSTM layers. Dropout reduces
over-reliance on individual units, and dense layers produce the next close
estimate.

## BiLSTM explanation

A Bidirectional LSTM runs an LSTM over the supplied input window in both its
forward and reverse order. The two representations are combined before the
output layers, allowing the model to use context from both ends of that already
observed historical window. It does **not** use future market observations:
the window itself still ends before its target date.

## Dataset explanation

The dataset is downloaded when the app runs; no stock data file is bundled.
Yahoo Finance data is accessed through the `yfinance` Python package. With
`auto_adjust=True`, the OHLC fields use adjusted prices. The requested end date
is included in the app's date range.

The five model features are:

- **Open:** adjusted opening price for the day.
- **High:** adjusted highest price for the day.
- **Low:** adjusted lowest price for the day.
- **Close:** adjusted closing price and prediction target.
- **Volume:** number of shares traded.

SMA 20 and SMA 50 are moving averages. RSI is calculated over 14 periods.
MACD uses 12- and 26-period exponential moving averages and a 9-period signal.
These indicators are shown for analysis and are not model inputs in this
version.

## Evaluation metrics

- **MAE (Mean Absolute Error):** average absolute prediction error, in rupees.
- **RMSE (Root Mean Squared Error):** square-root of average squared error, in
  rupees; larger errors receive greater weight.
- **MAPE (Mean Absolute Percentage Error):** average absolute percentage error.

All three are calculated from actual versus predicted closing prices on the
chronological test period. The lower-RMSE model is highlighted for the selected
dataset and settings; it is not a promise of future performance.

## Limitations

- Historical prices cannot account for every market-moving event.
- yfinance depends on an external data service and internet access; availability
  and data may change.
- A single 80/20 holdout can vary substantially by ticker and selected period.
- Neural-network results depend on training settings and random behavior in
  numerical operations.
- The next-period output is a simple experimental estimate; it does not include
  uncertainty bounds or transaction costs.
- A lower test error for one model does not mean it can reliably predict future
  stock prices.

## Future scope

- Add walk-forward validation and compare performance across multiple periods.
- Compare against naïve, ARIMA, and other baseline forecasts.
- Explore additional market indices, sector data, or fundamentals.
- Add forecast uncertainty intervals and clearer model diagnostics.
- Study hyperparameter tuning and alternative sequence lengths.

## Model storage

Trained models and their fitted scalers are saved under `models/` in a
stock-and-settings-specific directory. Matching bundles are reused to avoid
retraining. If market data or training settings change, the app creates a
different bundle. Delete a bundle directory to force retraining for that exact
selection.
