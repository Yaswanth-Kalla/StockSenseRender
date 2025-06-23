# backend/app/data_loader.py

import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
import ta
import os # Import os to potentially check for API key directly if config fails
import logging

# Set up logger for data_loader
logger = logging.getLogger(__name__)

# Import the API key from config.py
# config.py should now fetch this from an environment variable
from app.config import ALPHA_VANTAGE_API_KEY

def fetch_data(symbol: str) -> pd.DataFrame:
    """
    Fetches daily historical stock data from Alpha Vantage and computes technical indicators.

    Args:
        symbol (str): The stock symbol (e.g., "RELIANCE.BSE").

    Returns:
        pd.DataFrame: A DataFrame containing 'Open', 'High', 'Low', 'Close', 'Volume'
                      and calculated technical indicators, or an empty DataFrame if data fetch fails.
    """
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY is not set. Cannot fetch data from Alpha Vantage.")
        return pd.DataFrame() # Return empty DataFrame if API key is missing

    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        logger.info(f"Fetching daily data for {symbol} from Alpha Vantage...")
        
        # Use full outputsize to get all available historical data
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        logger.info(f"Data fetched successfully for {symbol}. Meta data: {meta_data}")

        # Introduce a delay to respect Alpha Vantage API rate limits (5 calls/minute for free tier)
        time.sleep(12) 

        # Rename columns and reverse order to have most recent data at the bottom
        df = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.iloc[::-1] # Reverse the DataFrame using .iloc[::-1] to keep index intact conceptually
        df.reset_index(inplace=True) # Reset index to get 'Date' as a column
        df.rename(columns={'index': 'Date'}, inplace=True) # Rename the new 'index' column to 'Date'
        
        # Ensure numeric types after loading
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute Technical Indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_diff'] = macd.macd_diff()
        df['SMA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
        df['RET1'] = df['Close'].pct_change(1) # 1-day return
        df['VOL'] = df['RET1'].rolling(window=10).std() # Volatility (standard deviation of returns)

        # Drop rows with NaN values (resulting from indicator calculations or initial data gaps)
        initial_rows = len(df)
        df.dropna(inplace=True)
        if len(df) < initial_rows:
            logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values for {symbol}.")
        
        if df.empty:
            logger.warning(f"DataFrame became empty after dropping NA for {symbol}. Check data quality or feature calculation.")
            return pd.DataFrame()

        logger.info(f"Successfully processed data for {symbol}. DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error fetching or processing data for {symbol}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
