import pandas as pd
import numpy as np
from darts import TimeSeries as ts
from darts.models import NHiTSModel
from loguru import logger
from polygon import RESTClient
import time
import csv
import quantstats as qs
import torch

# Extend quantstats for pandas
qs.extend_pandas()

# Set up logger to print detailed steps in the console
logger.add("detailed_log.log", rotation="500 MB")
logger.info("Logging setup complete")

class LiveModeler:
    def __init__(self, model_file, window=1600, horizon=800, resample_rate='1s'):
        self.model_file = model_file
        self.window = window
        self.horizon = horizon
        self.resample_rate = resample_rate
        self.model = None

        logger.info(f"Initializing LiveModeler with window: {window}, horizon: {horizon}")
        self.init_model()

    def init_model(self):
        try:
            logger.info(f"Loading model from file: {self.model_file}")
            self.model = NHiTSModel.load(self.model_file)
            logger.success(f"Model loaded from {self.model_file}")
        except FileNotFoundError:
            logger.error(f"Model file {self.model_file} not found. Exiting.")
            raise

    def predict(self, live_data):
        live_data_resampled = live_data.resample(self.resample_rate).ffill()
        targetts = ts.from_dataframe(live_data_resampled, freq=self.resample_rate)

        # Extract the last window of data for prediction
        last_window = targetts[-self.window:]

        logger.info("Generating live forecasts")
        live_forecast = self.model.predict(
            n=self.horizon,
            series=last_window,
            verbose=True
        )
        return live_forecast

class LiveTrading:
    def __init__(self, modeler, data_fetch_func, update_interval=15, tolerance=0.0014, tcosts=0.000025):
        self.modeler = modeler
        self.data_fetch_func = data_fetch_func
        self.update_interval = update_interval
        self.tolerance = tolerance
        self.tcosts = tcosts
        self.last_price = None
        self.position = None
        self.cumulative_pnl = 0

        logger.info("Initializing CSV for logging trades")
        with open('live_trades.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Action', 'Price', 'Profit'])

    def log_trade(self, action, price, profit=None):
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Logging trade: {action} at {price}, profit: {profit}")
        with open('live_trades.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, action, price, profit])
        if profit:
            self.cumulative_pnl += profit
        logger.info(f"Cumulative PnL: {self.cumulative_pnl}")

    def run(self):
        logger.info("Starting live trading loop")
        accumulated_data = pd.DataFrame()

        while True:
            try:
                logger.info("Fetching live data")
                live_data = self.data_fetch_func()

                if accumulated_data.empty:
                    accumulated_data = live_data
                else:
                    accumulated_data = pd.concat([accumulated_data, live_data])

                if len(accumulated_data) >= self.modeler.window:
                    logger.info(f"Accumulated data length: {len(accumulated_data)}")
                    prediction = self.modeler.predict(accumulated_data)
                    predicted_price = prediction.univariate_component(0).pd_dataframe().iloc[-1, 0]
                    current_price = live_data['close'].iloc[-1]

                    logger.info(f"Current price: {current_price}, Predicted price: {predicted_price}")

                    if self.last_price is not None:
                        if predicted_price > current_price and self.position != 'long':
                            if self.position == 'short':
                                profit = self.last_price - current_price - self.tcosts
                                self.log_trade('Close Short', current_price, profit)
                            self.log_trade('Buy', current_price)
                            self.position = 'long'
                            self.last_price = current_price
                        elif predicted_price < current_price and self.position != 'short':
                            if self.position == 'long':
                                profit = current_price - self.last_price - self.tcosts
                                self.log_trade('Close Long', current_price, profit)
                            self.log_trade('Sell', current_price)
                            self.position = 'short'
                            self.last_price = current_price
                    else:
                        logger.info("Initializing first position")
                        self.last_price = current_price

                logger.info(f"Waiting {self.update_interval} seconds before next prediction")
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in live trading loop: {e}")

def fetch_live_btc_data():
    client = RESTClient('AAT1cWks1QORn39txz7O8pNJiUWeRasp')

    try:
        latest_trade = client.get_last_trade("X:BTCUSD")
        logger.info(f"Fetched latest trade data: {latest_trade}")

        if latest_trade.sip_timestamp is not None:
            now = pd.to_datetime(latest_trade.sip_timestamp, unit='ms', utc=True).tz_convert('America/New_York')
        else:
            now = pd.Timestamp.now(tz='America/New_York')

        price = latest_trade.price
        live_data = pd.DataFrame({'close': [price]}, index=[now])
        live_data.index = pd.to_datetime(live_data.index)

        return live_data

    except Exception as e:
        logger.error(f"Error fetching live BTC data: {e}")
        raise

# Initialize the model
modeler = LiveModeler(model_file='qqqmod_NHiTS_11.pth')

# Start live trading
live_trader = LiveTrading(modeler, fetch_live_btc_data, update_interval=15)
live_trader.run()
