import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from polygon import RESTClient
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Polygon client with your API key
client = RESTClient("ELKyuDIy2c3CSW3eHCnuFhkq2yFmgMoE")

def fetch_polygon_data(ticker, start_date, end_date, timespan="day"):
    """
    Fetch data from Polygon.io
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    current_date = start_date
    all_data = []

    with tqdm(total=(end_date - start_date).days) as pbar:
        while current_date <= end_date:
            try:
                # Fetch data for a 30-day period (to avoid hitting rate limits)
                period_end = min(current_date + timedelta(days=30), end_date)
                bars = client.get_aggs(ticker=ticker, 
                                       multiplier=1, 
                                       timespan=timespan, 
                                       from_=current_date.strftime('%Y-%m-%d'), 
                                       to=period_end.strftime('%Y-%m-%d'),
                                       limit=50000)
                
                df = pd.DataFrame(bars)
                all_data.append(df)
                
                current_date = period_end + timedelta(days=1)
                pbar.update((period_end - current_date).days + 1)
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

    if not all_data:
        logger.error("No data fetched.")
        return None

    data_df = pd.concat(all_data).drop_duplicates()
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], unit='ms')
    data_df.set_index('timestamp', inplace=True)
    data_df = data_df.sort_index()

    return data_df

# Fetch QQQ data
start_date = '2023-01-01'  # Starting with a more recent date due to potential limitations
end_date = '2023-12-31'
qqqdf = fetch_polygon_data("QQQ", start_date, end_date)

if qqqdf is not None and not qqqdf.empty:
    # Basic data analysis
    print(qqqdf.head())
    print(qqqdf.info())
    
    # Plot closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(qqqdf.index, qqqdf['close'])
    plt.title('QQQ Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
else:
    print("No data available for analysis.")

# If you want to calculate returns
if qqqdf is not None and not qqqdf.empty:
    qqqdf['returns'] = qqqdf['close'].pct_change()
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = np.sqrt(252) * qqqdf['returns'].mean() / qqqdf['returns'].std()
    print(f"Sharpe Ratio: {sharpe_ratio}")

    # Plot returns distribution
    plt.figure(figsize=(12, 6))
    qqqdf['returns'].hist(bins=50)
    plt.title('QQQ Daily Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()