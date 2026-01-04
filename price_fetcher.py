"""
Stock Price Data Fetcher
Fetches real-time and historical stock/crypto price data using yfinance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class PriceDataFetcher:
    """Fetches real-time and historical price data for stocks and cryptocurrencies"""
    
    def __init__(self):
        self.data = None
    
    def fetch_data(self, symbol, period="1y", interval="1d"):
        """
        Fetch historical price data
        
        Args:
            symbol (str): Stock ticker (e.g., 'AAPL') or crypto ticker (e.g., 'BTC-USD')
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pandas.DataFrame: Historical price data
        """
        try:
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(period=period, interval=interval)
            
            if self.data.empty:
                print(f"No data found for symbol: {symbol}")
                return None
            
            return self.data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol):
        """
        Get current/latest price for a symbol
        
        Args:
            symbol (str): Stock or crypto ticker
        
        Returns:
            dict: Current price information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_data = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                'previous_close': info.get('previousClose', 'N/A'),
                'open': info.get('open', 'N/A'),
                'day_high': info.get('dayHigh', 'N/A'),
                'day_low': info.get('dayLow', 'N/A'),
                'volume': info.get('volume', 'N/A'),
                'name': info.get('longName', info.get('shortName', symbol))
            }
            
            return current_data
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None
    
    def prepare_data_for_lstm(self, feature='Close'):
        """
        Prepare data for LSTM model
        
        Args:
            feature (str): Price feature to use (Close, Open, High, Low)
        
        Returns:
            pandas.DataFrame: Prepared data
        """
        if self.data is None or self.data.empty:
            print("No data available. Please fetch data first.")
            return None
        
        # Use the specified feature
        data = self.data[[feature]].copy()
        return data
