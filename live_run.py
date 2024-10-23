import pandas as pd
import time
import csv
from loguru import logger
from polygon import RESTClient
from darts import TimeSeries as ts
from darts.models import NHiTSModel  # Import the correct model class
import torch
from dateutil.relativedelta import relativedelta

# Set up logger to print detailed steps in the console and file
logger.add("live_trading_log.log", rotation="500 MB")
logger.info("Starting Live Trading Session")

class LiveTrading:
    def __init__(self, model, data_fetch_func, update_interval=15, tolerance=0.0014, tcosts=0.000025):
        self.model = model
        self.data_fetch_func = data_fetch_func
        self.update_interval = update_interval
        self.tolerance = tolerance
        self.tcosts = tcosts
        self.last_price = None
        self.position = None
        self.trade_log = []
        self.predicted_prices = []
        self.actual_prices = []
        self.pnl = 0

        # Initialize CSV for logging trades, predictions, actual prices, and cumulative P&L
        with open('live_run.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Action', 'Price', 'Predicted Price', 'Profit', 'Cumulative P&L'])

    def log_trade(self, action, price, predicted_price, profit=None):
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        self.trade_log.append([current_time, action, price, predicted_price, profit, self.pnl])

        logger.info(f"Logging trade: {action} at {price}, predicted price: {predicted_price}, profit: {profit}, cumulative P&L: {self.pnl}")
        with open('live_run.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, action, price, predicted_price, profit, self.pnl])

    def run(self, accumulated_data):
        logger.info("Starting live trading loop")
        try:
            # Fetch live data
            logger.info("Fetching live data")
            new_data = self.data_fetch_func()

            # Ensure the DataFrame has a DatetimeIndex
            if not isinstance(new_data.index, pd.DatetimeIndex):
                raise ValueError("Fetched data must have a DatetimeIndex")

            # Convert to timezone-naive timestamps
            if new_data.index.tz:
                new_data.index = new_data.index.tz_convert(None)

            # Resample to 1-second intervals and forward-fill missing data
            new_data = new_data.resample('1s').ffill()

            # Keep track of the actual price
            actual_price = new_data['close'].iloc[-1]
            self.actual_prices.append(actual_price)

            # Predict next prices using the trained model
            logger.info(f"Accumulated data length: {len(accumulated_data)}")

            if len(accumulated_data) < 1000:
                logger.warning(f"Not enough data to make predictions. Waiting for more data.")
                return accumulated_data

            logger.info("Predicting prices using the trained model")

            # Ensure the model is of correct type and predict
            if isinstance(self.model, NHiTSModel):  # Check if it's an instance of NHiTSModel or the expected model
                prediction = self.model.predict(n=1, series=ts.from_dataframe(accumulated_data))
                logger.info("Prediction successful")
            else:
                logger.error("Loaded model is not an instance of the expected model type.")
                return accumulated_data

            predicted_price = prediction.univariate_component(0).pd_dataframe().iloc[-1, 0]
            self.predicted_prices.append(predicted_price)

            logger.info(f"Actual price: {actual_price}, Predicted price: {predicted_price}")

            # Decision-making based on predicted price
            if self.last_price is not None:
                if predicted_price > actual_price and self.position != 'long':
                    if self.position == 'short':
                        profit = self.last_price - actual_price - self.tcosts
                        self.pnl += profit
                        self.log_trade('Close Short', actual_price, predicted_price, profit)
                    self.log_trade('Buy', actual_price, predicted_price)
                    self.position = 'long'
                    self.last_price = actual_price

                elif predicted_price < actual_price and self.position != 'short':
                    if self.position == 'long':
                        profit = actual_price - self.last_price - self.tcosts
                        self.pnl += profit
                        self.log_trade('Close Long', actual_price, predicted_price, profit)
                    self.log_trade('Sell', actual_price, predicted_price)
                    self.position = 'short'
                    self.last_price = actual_price

            else:
                logger.info("Initializing first position")
                self.last_price = actual_price

            # Sleep for the update interval
            logger.info(f"Waiting {self.update_interval} seconds before next prediction")
            time.sleep(self.update_interval)

        except Exception as e:
            logger.error(f"Error in live trading loop: {e}")
        return accumulated_data


def fetch_live_btc_data():
    client = RESTClient('AAT1cWks1QORn39txz7O8pNJiUWeRasp')  # Your Polygon API key

    try:
        latest_trade = client.get_last_trade("X:BTCUSD")
        logger.info(f"Fetched latest trade data: {latest_trade}")

        # Check if the timestamp exists before conversion
        if latest_trade.sip_timestamp:
            now = pd.to_datetime(latest_trade.sip_timestamp, unit='ms', utc=True)
        elif latest_trade.trf_timestamp:
            now = pd.to_datetime(latest_trade.trf_timestamp, unit='ms', utc=True)
        else:
            # If both timestamps are None, use the current timestamp
            logger.warning("No valid timestamp available, using current time instead.")
            now = pd.Timestamp.now(tz='UTC')

        # Convert to timezone-naive timestamps for further operations
        now_no_tz = now.tz_localize(None)
        price = latest_trade.price

        live_data = pd.DataFrame({'close': [price]}, index=[now_no_tz])
        live_data = live_data.resample('1s').ffill()

        # Check for missing values and handle them
        if live_data['close'].isna().any():
            logger.warning("Missing live data detected, applying fallback strategy.")
            fallback_price = client.get_last_trade("X:BTCUSD").price  # Retry fetching the latest price
            live_data['close'].fillna(fallback_price, inplace=True)
            logger.info(f"Filled missing values with fallback price: {fallback_price}")

        return live_data

    except Exception as e:
        logger.error(f"Error fetching live BTC data: {e}")
        raise


def fetch_historical_data(ticker, start_date, end_date, freq="second"):
    client = RESTClient('AAT1cWks1QORn39txz7O8pNJiUWeRasp')

    logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date} with frequency: {freq}")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    datalist = []

    while start_date <= end_date:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = (start_date + relativedelta(days=1)).strftime('%Y-%m-%d')

        try:
            bars = client.get_aggs(ticker=ticker, multiplier=1, timespan=freq, from_=start_str, to=end_str, limit=50000)
            df = pd.DataFrame(bars)
            datalist.append(df)
            start_date += relativedelta(days=1)
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            start_date += relativedelta(days=1)

    datadf = pd.concat(datalist).drop_duplicates()
    datadf.timestamp = pd.to_datetime(datadf.timestamp, unit='ms', utc=True)
    datadf.timestamp = datadf.timestamp.dt.tz_convert('America/New_York')
    datadf.set_index('timestamp', inplace=True)

    # Convert historical data timestamps to timezone-naive
    if datadf.index.tz:
        datadf.index = datadf.index.tz_convert(None)

    logger.info(f"Returning processed historical data")
    return datadf[['close']]


def accumulate_live_data(live_data, accumulated_data, min_length=1000):
    # Concatenate new live data to the historical accumulated data
    accumulated_data = pd.concat([accumulated_data, live_data])

    # Convert to timezone-naive timestamps
    if accumulated_data.index.tz:
        accumulated_data.index = accumulated_data.index.tz_convert(None)

    # Resample to 1-second intervals and forward-fill missing values
    accumulated_data = accumulated_data.resample('1s').ffill()

    # Ensure we have enough data for the model (at least input_chunk_length)
    if len(accumulated_data) < min_length:
        logger.warning(f"Not enough data to make predictions. Accumulated length: {len(accumulated_data)}. Waiting for more data.")
        return None, accumulated_data
    else:
        # Only keep the most recent data points necessary for the model (window size of 1000)
        accumulated_data = accumulated_data[-min_length:]
        return accumulated_data, accumulated_data


if __name__ == "__main__":
    # Load the trained model using Darts native loading method
    logger.info("Loading the trained model")
    model = NHiTSModel.load('qqqmod_NHiTS_11.pth')  # Use the native Darts method to load the model

    # Fetch historical data to initialize with 1000 points
    logger.info("Fetching initial historical data to start predictions")
    historical_data = fetch_historical_data("X:BTCUSD", start_date=pd.Timestamp.now() - pd.Timedelta(days=1), end_date=pd.Timestamp.now(), freq="second")

    # Ensure we have at least 1000 data points for prediction
    historical_data = historical_data[-1000:]

    # Start live trading with historical data
    live_trader = LiveTrading(model, fetch_live_btc_data, update_interval=15)

    accumulated_data = historical_data  # Initialize with historical data

    while True:
        live_data = fetch_live_btc_data()
        live_data, accumulated_data = accumulate_live_data(live_data, accumulated_data)

        if live_data is not None:
            # Start live trading with accumulated data once enough data is present
            accumulated_data = live_trader.run(accumulated_data)
        time.sleep(15)  # Wait 15 seconds before fetching the next live data
