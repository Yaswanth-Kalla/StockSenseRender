import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
import ta
from app.config import API_KEY

def fetch_data(symbol: str) -> pd.DataFrame:
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    time.sleep(12)

    df = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[::-1]
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_diff'] = macd.macd_diff()
    df['SMA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
    df['RET1'] = df['Close'].pct_change(1)
    df['VOL'] = df['RET1'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df
