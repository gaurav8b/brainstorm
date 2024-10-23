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
    def __init__(self, data, modelfile, *, window=1600, horizon=800, resample_rate='1s', n_epochs=1, modeltype=NHiTSModel, **modelparams):
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
            # Load the model weights for fine-tuning instead of retraining from scratch
            self.model = self.modeltype.load(self.model_file)  # Load the entire model
            self.model.load_weights(self.model_file)  # Load only the weights for fine-tuning
            logger.success(f"Model weights loaded from {self.model_file}")
        except FileNotFoundError:
            logger.warning(f"Model file not found, initializing a new model of type {self.modeltype.__name__}")
            # Initialize the model from scratch if no saved model is found
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
    def __init__(self, modeler, data_fetch_func, update_interval=15, tolerance=0.0014, tcosts=0.000025):
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

    def run(self):
        logger.info("Starting live trading loop")
        while True:
            try:
                logger.info("Fetching live data")
                new_data = self.data_fetch_func()

                # Ensure the DataFrame has a DatetimeIndex
                logger.info(f"Fetched data index type: {type(new_data.index)}")
                logger.info(f"First few rows of fetched data: {new_data.head()}")

                if not isinstance(new_data.index, pd.DatetimeIndex):
                    raise ValueError("Fetched data must have a DatetimeIndex")

                # Remove timezone to avoid xarray issues and ensure uniformity with historical data
                if new_data.index.tz:
                    new_data.index = new_data.index.tz_convert(None)
                if self.modeler.data.index.tz:
                    self.modeler.data.index = self.modeler.data.index.tz_convert(None)

                # Concatenate the new live data with the existing historical data
                self.modeler.data = pd.concat([self.modeler.data, new_data])

                # Resample the combined data to fill any missing timestamps
                self.modeler.data = self.modeler.data.resample(self.modeler.resample_rate).ffill()

                logger.info("Generating price prediction")
                prediction, mape = self.modeler.day_predict(data=self.modeler.data)

                predicted_price = prediction.univariate_component(0).pd_dataframe().iloc[-1, 0]
                current_price = new_data['close'].iloc[-1]

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
mymod5 = MyModeler(qqqdf, modelfile='btcmod_NHiTS_11.pth', window=1600, horizon=800, modeltype=NHiTSModel)
mymod5.fit_span(end_day='2024-10-21')

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
