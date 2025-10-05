import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class CryptoDataFetcher:
    """Fetch real cryptocurrency data from free APIs"""
    
    def __init__(self, symbol="ethereum", vs_currency="usd"):
        self.symbol = symbol
        self.vs_currency = vs_currency
        self.base_url = "https://api.coingecko.com/api/v3"
        self.last_request_time = 0
        self.min_request_interval = 1.5  # CoinGecko free tier: 10-30 calls/min
        
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_current_price(self):
        """Get current price of cryptocurrency"""
        try:
            self._rate_limit()
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': self.symbol,
                'vs_currencies': self.vs_currency,
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if self.symbol in data:
                return {
                    'price': data[self.symbol][self.vs_currency],
                    'change_24h': data[self.symbol].get(f'{self.vs_currency}_24h_change', 0),
                    'volume_24h': data[self.symbol].get(f'{self.vs_currency}_24h_vol', 0),
                    'timestamp': datetime.now()
                }
            return None
            
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None
    
    def get_historical_data(self, days=30):
        """Get historical price data"""
        try:
            self._rate_limit()
            url = f"{self.base_url}/coins/{self.symbol}/market_chart"
            params = {
                'vs_currency': self.vs_currency,
                'days': days,
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = [v[1] for v in data.get('total_volumes', [])]
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def get_ohlc_data(self, days=7):
        """Get OHLC (candlestick) data"""
        try:
            self._rate_limit()
            url = f"{self.base_url}/coins/{self.symbol}/ohlc"
            params = {
                'vs_currency': self.vs_currency,
                'days': days
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            print(f"Error fetching OHLC data: {e}")
            return None


class BinanceDataFetcher:
    """Alternative: Fetch data from Binance public API (no auth needed)"""
    
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.base_url = "https://api.binance.com/api/v3"
    
    def get_current_price(self):
        """Get current price from Binance"""
        try:
            url = f"{self.base_url}/ticker/price"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'price': float(data['price']),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching Binance price: {e}")
            return None
    
    def get_24h_ticker(self):
        """Get 24h price statistics"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'volume_24h': float(data['volume']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching 24h ticker: {e}")
            return None
    
    def get_klines(self, interval='1h', limit=100):
        """Get candlestick/kline data"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': self.symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return None


class CryptoCompareDataFetcher:
    """Alternative: Fetch from CryptoCompare (free tier)"""
    
    def __init__(self, fsym="ETH", tsym="USD"):
        self.fsym = fsym
        self.tsym = tsym
        self.base_url = "https://min-api.cryptocompare.com/data"
    
    def get_current_price(self):
        """Get current price"""
        try:
            url = f"{self.base_url}/price"
            params = {
                'fsym': self.fsym,
                'tsyms': self.tsym
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'price': data.get(self.tsym, 0),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching CryptoCompare price: {e}")
            return None
    
    def get_historical_hourly(self, limit=168):
        """Get hourly historical data (max 2000 hours)"""
        try:
            url = f"{self.base_url}/v2/histohour"
            params = {
                'fsym': self.fsym,
                'tsym': self.tsym,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success':
                df = pd.DataFrame(data['Data']['Data'])
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volumefrom']]
            
            return None
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None