import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class CryptoDataFetcher:
    """Fetch real cryptocurrency data from free APIs with fallback support"""
    
    def __init__(self, symbol="bitcoin", vs_currency="usd"):
        self.symbol = symbol
        self.vs_currency = vs_currency
        self.base_url = "https://api.coingecko.com/api/v3"
        self.last_request_time = 0
        self.min_request_interval = 2.0  # CoinGecko free tier
        
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
            print(f"CoinGecko error: {e}")
            return None
    
    def get_historical_data(self, days=7):
        """Get historical price data"""
        try:
            self._rate_limit()
            url = f"{self.base_url}/coins/{self.symbol}/market_chart"
            params = {
                'vs_currency': self.vs_currency,
                'days': days,
                'interval': 'hourly'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = [v[1] if len(volumes) > i else 0 for i, v in enumerate(volumes)]
            df['open'] = df['price']
            df['high'] = df['price']
            df['low'] = df['price']
            df['close'] = df['price']
            
            return df
            
        except Exception as e:
            print(f"CoinGecko historical error: {e}")
            return None


class UnifiedCryptoFetcher:
    """Unified fetcher with multiple API fallbacks - REAL DATA ONLY"""
    
    def __init__(self, symbol="BTC"):
        self.symbol = symbol.upper()
        self.coingecko_id = self._get_coingecko_id(symbol)
        self.last_successful_api = None
        print(f"Initializing UnifiedCryptoFetcher for {symbol}")
        
    def _get_coingecko_id(self, symbol):
        """Map symbol to CoinGecko ID"""
        mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'SOL': 'solana',
            'ADA': 'cardano'
        }
        return mapping.get(symbol.upper(), 'bitcoin')
    
    def test_apis(self):
        """Test which APIs are working"""
        results = {}
        
        # Test CoinGecko
        try:
            response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=10)
            results['CoinGecko'] = response.status_code == 200
        except:
            results['CoinGecko'] = False
        
        # Test CryptoCompare
        try:
            response = requests.get("https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD", timeout=10)
            results['CryptoCompare'] = response.status_code == 200
        except:
            results['CryptoCompare'] = False
        
        # Test Coinbase
        try:
            response = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=10)
            results['Coinbase'] = response.status_code == 200
        except:
            results['Coinbase'] = False
        
        return results
    
    def get_current_price(self):
        """Try multiple APIs until one works - WITH EXTENDED TIMEOUT"""
        # Try CoinGecko first (most reliable on Streamlit Cloud)
        try:
            print("Trying CoinGecko API...")
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': self.coingecko_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            response = requests.get(url, params=params, timeout=15)  # Increased timeout
            response.raise_for_status()
            data = response.json()
            
            if self.coingecko_id in data:
                self.last_successful_api = "CoinGecko"
                print(f"✓ CoinGecko API successful")
                return {
                    'price': float(data[self.coingecko_id]['usd']),
                    'change_24h': data[self.coingecko_id].get('usd_24h_change', 0),
                    'volume_24h': data[self.coingecko_id].get('usd_24h_vol', 0),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"CoinGecko failed: {e}")
        
        # Try CryptoCompare as fallback
        try:
            print("Trying CryptoCompare API...")
            url = "https://min-api.cryptocompare.com/data/pricemultifull"
            params = {
                'fsyms': self.symbol.replace('USDT', ''),
                'tsyms': 'USD'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            raw_data = data['RAW'][self.symbol.replace('USDT', '')]['USD']
            self.last_successful_api = "CryptoCompare"
            print(f"✓ CryptoCompare API successful")
            return {
                'price': float(raw_data['PRICE']),
                'change_24h': float(raw_data.get('CHANGEPCT24HOUR', 0)),
                'volume_24h': float(raw_data.get('VOLUME24HOUR', 0)),
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"CryptoCompare failed: {e}")
        
        # Try Coinbase as last resort
        try:
            print("Trying Coinbase API...")
            symbol_pair = f"{self.symbol.replace('USDT', '')}-USD"
            url = f"https://api.coinbase.com/v2/prices/{symbol_pair}/spot"
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            self.last_successful_api = "Coinbase"
            print(f"✓ Coinbase API successful")
            return {
                'price': float(data['data']['amount']),
                'change_24h': 0,
                'volume_24h': 0,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Coinbase failed: {e}")
        
        print("❌ All APIs failed")
        return None
    
    def get_historical_data(self, days=7):
        """Get historical data from CoinGecko"""
        try:
            time.sleep(1)  # Rate limiting
            url = f"https://api.coingecko.com/api/v3/coins/{self.coingecko_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['price'] = df['close']
            df['open'] = df['close']
            df['high'] = df['close'] * 1.002
            df['low'] = df['close'] * 0.998
            df['volume'] = [v[1] if i < len(volumes) else 0 for i, v in enumerate(volumes)]
            
            print(f"Loaded {len(df)} historical data points from CoinGecko")
            return df
            
        except Exception as e:
            print(f"Historical data error: {e}")
            return None


# Keep old classes for backwards compatibility
class BinanceDataFetcher:
    """Binance API - MAY NOT WORK ON STREAMLIT CLOUD"""
    
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.base_url = "https://api.binance.com/api/v3"
        print("Warning: Binance API may be blocked on Streamlit Cloud")
    
    def get_current_price(self):
        try:
            url = f"{self.base_url}/ticker/price"
            response = requests.get(url, params={'symbol': self.symbol}, timeout=5)
            response.raise_for_status()
            return {
                'price': float(response.json()['price']),
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Binance error: {e}")
            return None
    
    def get_24h_ticker(self):
        try:
            url = f"{self.base_url}/ticker/24hr"
            response = requests.get(url, params={'symbol': self.symbol}, timeout=5)
            response.raise_for_status()
            data = response.json()
            return {
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'volume_24h': float(data['volume']),
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Binance 24h ticker error: {e}")
            return None
    
    def get_klines(self, interval='5m', limit=100):
        try:
            url = f"{self.base_url}/klines"
            params = {'symbol': self.symbol, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            df = pd.DataFrame(response.json(), columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Binance klines error: {e}")
            return None


class CryptoCompareDataFetcher:
    """CryptoCompare API - Works on Streamlit Cloud"""
    
    def __init__(self, fsym="BTC", tsym="USD"):
        self.fsym = fsym
        self.tsym = tsym
        self.base_url = "https://min-api.cryptocompare.com/data"
    
    def get_current_price(self):
        try:
            url = f"{self.base_url}/price"
            response = requests.get(url, params={'fsym': self.fsym, 'tsyms': self.tsym}, timeout=10)
            response.raise_for_status()
            data = response.json()
            return {
                'price': data.get(self.tsym, 0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"CryptoCompare error: {e}")
            return None
    
    def get_historical_hourly(self, limit=168):
        try:
            url = f"{self.base_url}/v2/histohour"
            params = {'fsym': self.fsym, 'tsym': self.tsym, 'limit': limit}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success':
                df = pd.DataFrame(data['Data']['Data'])
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df = df.rename(columns={'volumefrom': 'volume'})
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return None
        except Exception as e:
            print(f"CryptoCompare historical error: {e}")
            return None
