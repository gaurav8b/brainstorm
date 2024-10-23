from flask import Flask,request,json
import time
from datetime import date, datetime,timezone,timedelta
import logging
from utils import *
from signal_processor import SignalProcessor
tz_utc = timezone.utc


dirR = f'logs/'
create_dir(dirR)
log_file = f'{dirR}bot_logs.log'

logging.Formatter.converter = timetz

logging.basicConfig(
    filename=log_file,
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def print_and_log(msg):
    log_msg = f'{msg}'
    print(log_msg)
    logging.info(log_msg)

app = Flask(__name__)

###################################################################################
def get_access_token():
    dt_now = datetime.now(tz_utc)
    dt_str = dt_now.strftime('%Y_%m_%d_%H_%M')
    msg = 'New alert received...'
    print_and_log(msg)
    signalProcessor = SignalProcessor(base_dir=dirR,logger=logging)
    return signalProcessor.get_access_token()

def Market_Order_Signal(Market, TradeType, Side, Size=1):
    alert_json_str = f'{{"AlertType": "TRADE","OrderType": "market", "Market":"{Market}", "TradeType": "{TradeType}", "Side": "{Side}", "Size": {Size}}}'
    return json.loads(alert_json_str)

def tradesignal(alert_json):
    dt_now = datetime.now(tz_utc)
    dt_str = dt_now.strftime('%Y_%m_%d_%H_%M')
    msg = 'New alert received...'
    print_and_log(msg)
    msg = f'Alert: {alert_json}'
    print_and_log(msg)

    if alert_json['AlertType'] == 'TRADE':
        signalProcessor = SignalProcessor(base_dir=dirR,logger=logging)
        signalProcessor.process_signal(alert_json)

    txt = 'Signal processed. Done!!'
    print_and_log(txt)
    return txt


#Call_Market_Order_Signal = Market_Order_Signal("ESZ4","long","buy")

#Call_Market_Order_Signal = Market_Order_Signal("ESZ4","short","sell")



###################################################################################
# alert_json_str = '{"AlertType": "TRADE", "LimitPrice": 5850, "Market": "ESZ4", "OrderType": "OSO", "Side": "buy", "Size": 1, "SLStopPrice": 5800,"TPLimitPrice": 5900, "TradeType": "long"}'
# tradesignal(json.loads(alert_json_str))

# @app.route('/')
# def welcome():
#     return ' Tradovate trading bot'

# @app.route('/tradesignal',methods=['GET', 'POST'])
# def tradesignal():
#     alert_json = request.json
#     dt_now = datetime.now(tz_utc)
#     dt_str = dt_now.strftime('%Y_%m_%d_%H_%M')
#     msg = 'New alert received...'
#     print_and_log(msg)
#     msg = f'Alert: {alert_json}'
#     print_and_log(msg)

#     if alert_json['AlertType'] == 'TRADE':
#         signalProcessor = SignalProcessor(base_dir=dirR,logger=logging)
#         signalProcessor.process_signal(alert_json)


#     txt = 'Signal processed. Done!!'
#     print_and_log(txt)
#     return txt

#  
# if __name__ == '__main__':
#     app.run(debug=True,port=6000,host='localhost')

##### Instructions #####
# ngrok config add-authtoken <auth-token>
# ngrok http 6000
# Webhook url will be displayed in terminal 

# https://<web_hook_base_url> -> http://localhost:6000

# web_hook_url = <web_hook_base_url>/tradesignal

###################################################################################

#!pip install pandas numpy polygon-api-client darts loguru torch dask[dataframe] torch scikit-learn optuna quantstats
#!pip install pandas_market_calendars
#!pip install darts
#!pip install polygon-api-client
#!pip install loguru
#!pip install quantstats
#!pip install pandas_market_calendars


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries as ts
from darts.utils.missing_values import fill_missing_values as fill_missing
from dateutil.relativedelta import relativedelta
from darts.models import NHiTSModel
import torch
torch.set_float32_matmul_precision('medium')
from darts.metrics import mape
from loguru import logger
from polygon import RESTClient
import time
import csv
import quantstats as qs
import pandas_market_calendars as mcal
from torch.utils.data import DataLoader

# Extend quantstats for pandas
qs.extend_pandas()

# Set up logger to print detailed steps in the console
logger.add("detailed_log.log", rotation="500 MB")
logger.info("Logging setup complete")

def sharpe(ser):
    try:
        result = ser.mean() / ser.std()
        logger.info(f"Sharpe ratio calculated: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return -1

def custom_sign(series, tolerance=1e-6):
    logger.info(f"Generating custom signal for series with tolerance {tolerance}")
    return np.where(np.abs(series) < tolerance, 0, np.sign(series))

class MyModeler:
    def __init__(self, data, modelfile, *, window=1600, horizon=800, resample_rate='1s', n_epochs=20, modeltype=NHiTSModel, **modelparams):
        self.data = data
        self.model_file = modelfile
        self.window = window
        self.horizon = horizon
        self.n_epochs = n_epochs
        self.fcast = {}
        self.fcastdf = {}
        self.mape = {}
        self.datadict = {}
        self.netrets = {}
        self.grossrets = {}
        self.resample_rate = resample_rate
        self.modeltype = modeltype
        self.modelparams = modelparams

        logger.info(f"Initializing MyModeler with model: {modeltype.__name__}, window: {window}, horizon: {horizon}")
        self.init_model()

    def save_model_weights(self):
        logger.info(f"Saving model weights to {self.model_file}")
        self.model.save(self.model_file)  # Save model using Darts' save method
        logger.success(f"Model weights saved to {self.model_file}")

    def init_model(self):
        try:
            logger.info(f"Loading model from file: {self.model_file}")
            self.model = self.modeltype.load(self.model_file)  # Use Darts' load method to load the model
            logger.success(f"Model loaded from {self.model_file}")
        except FileNotFoundError:
            logger.warning(f"Model file not found, initializing a new model of type {self.modeltype.__name__}")
            self.model = self.modeltype(
                input_chunk_length=self.window,
                output_chunk_length=2 * self.horizon,
                random_state=42,
                n_epochs=self.n_epochs,
                **self.modelparams,
            )
            logger.info("New model initialized")


    def day_fit(self, *, data=None, theday=None):
        logger.info(f"Starting day_fit process for day: {theday}")
        if data is None:
            data = self.data
        if theday is not None:
            subdata = data[data.index.normalize() == pd.to_datetime(theday)]
        else:
            subdata = data

        # Ensure the DataFrame has a DatetimeIndex
        if not isinstance(subdata.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        # Remove timezone to avoid xarray issues
        if subdata.index.tz:
            subdata.index = subdata.index.tz_convert(None)

        # Resample the data to fill any missing timestamps
        subdata = subdata.resample(self.resample_rate).ffill()

        # Convert resampled data into a Darts TimeSeries object
        targetts = ts.from_dataframe(subdata, freq=self.resample_rate)

        logger.info("Fitting model with new data")
        self.model.fit(targetts, verbose=True)  # Directly pass the TimeSeries object
        self.save_model_weights()


    def day_predict(self, *, data=None, theday=None):
        logger.info(f"Starting day_predict process for day: {theday}")
        if data is None:
            data = self.data
        if theday is not None:
            subdata = data[data.index.normalize() == pd.to_datetime(theday)]
        else:
            subdata = data

        # Ensure the DataFrame has a DatetimeIndex
        if not isinstance(subdata.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        # Remove timezone to avoid xarray issues
        if subdata.index.tz:
            subdata.index = subdata.index.tz_convert(None)

        # Resample data to fill missing values with forward-fill
        subdata = subdata.resample(self.resample_rate).ffill()

        logger.info(f"Checking for missing data before prediction: {subdata.isnull().sum()}")

        # Convert resampled data into a Darts TimeSeries object
        targetts = ts.from_dataframe(subdata, freq=self.resample_rate)

        # Extract the last window of data for prediction
        last_window = targetts[-self.window:]

        logger.info("Generating live forecasts")
        live_forecast = self.model.predict(
            n=self.horizon,
            series=last_window,
            verbose=True,
        )
        forecast_mape = mape(live_forecast, targetts[-self.horizon:])
        logger.info(f"Forecast generated with MAPE: {forecast_mape}")
        return live_forecast, forecast_mape

    def fit_span(self, beg_day=None, end_day=None):
        logger.info(f"Starting fit_span from {beg_day} to {end_day}")
        if beg_day is None:
            if end_day is None:
                subdata = self.data
            else:
                subdata = self.data.loc[:end_day]
        else:
            if end_day is None:
                subdata = self.data.loc[beg_day:]
            else:
                subdata = self.data.loc[beg_day:end_day]

        for d, df in subdata.groupby(subdata.index.date):
            logger.info(f"Fitting model for day: {d}")
            self.datadict[d] = subdata
            self.day_fit(data=df)

class LiveTrading:
    def __init__(self, modeler, data_fetch_func, update_interval=15, tolerance=0.00014, tcosts=0.000025):
        self.modeler = modeler
        self.data_fetch_func = data_fetch_func
        self.update_interval = update_interval
        self.tolerance = tolerance
        self.tcosts = tcosts
        self.last_price = None
        self.position = None
        self.trade_log = []

        logger.info("Initializing CSV for logging trades")
        with open('live_run.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Action', 'Price', 'Profit'])

    def log_trade(self, action, price, profit=None):
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        self.trade_log.append([current_time, action, price, profit])

        logger.info(f"Logging trade: {action} at {price}, profit: {profit}")
        with open('live_run.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, action, price, profit])

    def send_trade_signal(self, action, side):
        """Function to execute trades using Market_Order_Signal and tradesignal."""
        try:
            logger.info(f"Sending {action} signal.")
            tradesignal(Market_Order_Signal("BTCV4", action, side))
            logger.info(f"{action} signal sent successfully.")
        except Exception as e:
            logger.error(f"Error sending {action} signal: {e}")

    def run(self):
        logger.info("Starting live trading loop")
        while True:
            try:
                logger.info("Fetching live data")
                new_data = self.data_fetch_func()

                # Ensure the DataFrame has a DatetimeIndex
                if not isinstance(new_data.index, pd.DatetimeIndex):
                    raise ValueError("Fetched data must have a DatetimeIndex")

                # Remove timezone to avoid xarray issues
                if new_data.index.tz:
                    new_data.index = new_data.index.tz_convert(None)

                # Append the new live data to existing model data
                self.modeler.data = pd.concat([self.modeler.data, new_data])

                # Predict price using model
                logger.info("Generating price prediction")
                prediction, mape = self.modeler.day_predict(data=self.modeler.data)

                # Get the predicted and current prices
                predicted_price = prediction.univariate_component(0).pd_dataframe().iloc[-1, 0]
                current_price = new_data['close'].iloc[-1]

                logger.info(f"Current price: {current_price}, Predicted price: {predicted_price}")

                # Calculate price difference
                price_diff = abs(predicted_price - current_price)

                # Trading logic with tolerance
                if price_diff > self.tolerance:  # Only proceed if the price difference exceeds the tolerance
                    if self.last_price is not None:
                        if predicted_price > current_price and self.position != 'long':
                            # Send buy signal if we are not already long
                            self.send_trade_signal("long", "buy")

                            if self.position == 'short':
                                # Close short and log profit
                                profit = self.last_price - current_price - self.tcosts
                                self.log_trade('Close Short', current_price, profit)

                            # Open long position and log trade
                            self.log_trade('Buy', current_price)
                            self.position = 'long'
                            self.last_price = current_price

                        elif predicted_price < current_price and self.position != 'short':
                            # Send sell signal if we are not already short
                            self.send_trade_signal("short", "sell")

                            if self.position == 'long':
                                # Close long and log profit
                                profit = current_price - self.last_price - self.tcosts
                                self.log_trade('Close Long', current_price, profit)

                            # Open short position and log trade
                            self.log_trade('Sell', current_price)
                            self.position = 'short'
                            self.last_price = current_price
                    else:
                        # Initialize first position
                        logger.info("Initializing first position")
                        self.last_price = current_price
                else:
                    logger.info(f"Price difference {price_diff} is within tolerance {self.tolerance}. No trade executed.")

                logger.info(f"Waiting {self.update_interval} seconds before next prediction")
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in live trading loop: {e}")

def polygonprocess(ticker, start_date, end_date, freq="second"):
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

    logger.info(f"Resampling data to 1-second intervals")
    dflist = []
    for d, df in datadf.groupby(datadf.index.date):
        tmp = df.resample('1s').ffill()
        dflist.append(tmp)
    newdata = pd.concat(dflist)

    logger.info("Returning processed second-level data")
    return newdata[['close']]

# Fetch historical data
qqqdf = polygonprocess("X:BTCUSD", '2024-10-01', '2024-10-21', freq='second')

# Initialize model and fit on historical data
mymod5 = MyModeler(qqqdf, modelfile='btclivemod_NHiTS_11.pth', window=1600, horizon=800, modeltype=NHiTSModel)
#mymod5.fit_span(end_day='2024-10-21')
mymod5.fit_span(end_day='2024-10-05')

# Define a function to fetch real-time BTC data from the Polygon API
def fetch_live_btc_data():
    client = RESTClient('AAT1cWks1QORn39txz7O8pNJiUWeRasp')  # Your API key

    try:
        latest_trade = client.get_last_trade("X:BTCUSD")
        logger.info(f"Fetched latest trade data: {latest_trade}")

        if latest_trade.sip_timestamp is not None:
            now = pd.to_datetime(latest_trade.sip_timestamp, unit='ms', utc=True).tz_convert('America/New_York')
        else:
            logger.warning("sip_timestamp is None. Skipping timezone conversion.")
            now = pd.Timestamp.now(tz='America/New_York')

        price = latest_trade.price

        live_data = pd.DataFrame({'close': [price]}, index=[now])

        # Resample to 1-second intervals and forward-fill missing data
        live_data = live_data.resample('1s').ffill()

        # Check for NaN values
        if live_data['close'].isna().any():
            logger.warning("Missing live data detected, applying fallback strategy.")
            # Fallback strategy: Fetch previous price from historical data or use the latest available non-NaN value
            fallback_price = client.get_last_trade("X:BTCUSD").price  # Retry fetching latest price
            live_data['close'].fillna(fallback_price, inplace=True)  # Fill NaN with the fallback price
            logger.info(f"Filled missing values with fallback price: {fallback_price}")

        live_data.index = pd.to_datetime(live_data.index)
        return live_data

    except Exception as e:
        logger.error(f"Error fetching live BTC data: {e}")
        raise


# Start live trading with real-time data
live_trader = LiveTrading(mymod5, fetch_live_btc_data, update_interval=15)
live_trader.run()
