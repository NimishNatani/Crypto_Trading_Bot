import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
import warnings
warnings.filterwarnings('ignore')
from data.crypto_data_fetcher import BinanceDataFetcher

class CryptoTradingBot:
    def __init__(self, initial_balance=1000, leverage=5, symbol="ETHUSDT"):
        # Basic setup
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.symbol = symbol
        
        # Initialize data fetcher for REAL market data
        self.data_fetcher = BinanceDataFetcher(symbol=symbol)
        
        # Initialize empty structures
        self.positions = []
        self.closed_positions = []
        self.current_price = 0
        self.price_history = []
        
        # Performance tracking
        self.total_profit_loss = 0
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.largest_win = 0
        self.largest_loss = 0
        
        # Risk management parameters - OPTIMIZED FOR FASTER P&L
        self.max_position_size = 0.15      # 15% per trade
        self.max_open_positions = 3         # Up to 3 simultaneous
        self.leverage = 5                   # 5x leverage
        self.stop_loss_threshold = 0.03     # 3% stop loss
        self.base_target = 0.025            # 2.5% profit target
        
        # Trading state
        self.running = False
        self.last_update = datetime.now()
        
        # Load initial REAL data
        self._initialize_real_data()
    
    def _initialize_real_data(self):
        """Initialize with real historical data from Binance"""
        try:
            print("Fetching real market data from Binance...")
            
            # Get last 100 5-minute candles (8+ hours of data)
            df = self.data_fetcher.get_klines(interval='5m', limit=100)
            
            if df is not None and not df.empty:
                # Convert to price history format
                for idx, row in df.iterrows():
                    self.price_history.append({
                        'timestamp': row['timestamp'],
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'price': float(row['close']),
                        'volume': float(row['volume'])
                    })
                
                # Set current price
                if self.price_history:
                    self.current_price = self.price_history[-1]['price']
                    print(f"✓ Loaded {len(self.price_history)} real data points")
                    print(f"✓ Current {self.symbol} price: ${self.current_price:,.2f}")
                else:
                    self._fetch_current_price()
            else:
                print("Warning: Could not load historical data, fetching current price...")
                self._fetch_current_price()
                
        except Exception as e:
            print(f"Error initializing real data: {e}")
            print("Attempting to fetch current price as fallback...")
            self._fetch_current_price()
    
    def _fetch_current_price(self):
        """Fetch current real-time price from Binance"""
        try:
            data = self.data_fetcher.get_24h_ticker()
            
            if data and 'price' in data:
                self.current_price = data['price']
                
                # Add to price history
                self.price_history.append({
                    'timestamp': data.get('timestamp', datetime.now()),
                    'open': self.current_price,
                    'high': self.current_price,
                    'low': self.current_price,
                    'close': self.current_price,
                    'price': self.current_price,
                    'volume': data.get('volume_24h', 0)
                })
                
                # Keep only last 500 data points
                if len(self.price_history) > 500:
                    self.price_history = self.price_history[-500:]
                
                return True
            else:
                print("Failed to fetch price data")
                return False
                
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return False
    
    def _trading_step(self):
        """Execute one trading step with real data"""
        # Fetch new real-time price
        success = self._fetch_current_price()
        
        if not success and self.price_history:
            print("Using last known price")
        
        # Check existing positions
        self._check_positions()
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return success

    def _calculate_technical_indicators(self, lookback=20):
        """Calculate technical indicators for ML model"""
        if len(self.price_history) < lookback:
            return None

        prices = [p['price'] for p in self.price_history[-lookback:]]
        volumes = [p['volume'] for p in self.price_history[-lookback:]]

        # Moving averages
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)

        # RSI
        price_changes = np.diff(prices)
        gains = price_changes[price_changes > 0]
        losses = -price_changes[price_changes < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        # Price momentum
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        # Volatility
        volatility = np.std(prices) / np.mean(prices)

        # Volume trend
        volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-10:]) if len(volumes) >= 10 else 1

        return {
            'sma_5': sma_5,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'rsi': rsi,
            'momentum': momentum,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'price_to_sma_5': self.current_price / sma_5,
            'price_to_sma_10': self.current_price / sma_10,
            'price_to_sma_20': self.current_price / sma_20
        }

    def _should_open_position(self):
        """Determine if we should open a new position - OPTIMIZED FOR MORE TRADES"""
        if len(self.positions) >= self.max_open_positions:
            return False, None, None

        margin_used = sum([pos.get('margin_used', 0) for pos in self.positions])
        available_balance = self.balance

        if available_balance < 50:  # Reduced from 100 to 50
            return False, None, None

        prediction, confidence = self._predict_market_direction()

        if confidence < 0.50:  # Reduced from 0.52 to 0.50
            return False, None, None

        indicators = self._calculate_technical_indicators()
        if indicators is None:
            return False, None, None

        volatility = indicators['volatility']

        if volatility > 0.10:  # Increased from 0.08 to 0.10
            return False, None, None

        action = "LONG" if prediction == 1 else "SHORT"
        target = self._calculate_dynamic_target(confidence, volatility)

        return True, action, target

    def _predict_market_direction(self):
        """Predict market direction using technical analysis"""
        indicators = self._calculate_technical_indicators()
        if indicators is None:
            return random.choice([0, 1]), random.uniform(0.55, 0.8)

        rsi = indicators['rsi']
        momentum = indicators['momentum']
        price_trend = indicators['price_to_sma_5']

        if (rsi > 70 or momentum < -0.02 or price_trend < 0.98):
            prediction = 0  # SHORT
            confidence = random.uniform(0.65, 0.85)
        elif (rsi < 30 or momentum > 0.02 or price_trend > 1.02):
            prediction = 1  # LONG
            confidence = random.uniform(0.65, 0.85)
        else:
            prediction = random.choice([0, 1])
            confidence = random.uniform(0.55, 0.75)

            if random.random() < 0.4:
                prediction = 1 - prediction
                confidence = max(confidence, 0.6)

        return prediction, confidence

    def _calculate_dynamic_target(self, confidence, volatility):
        """Calculate dynamic profit target - OPTIMIZED FOR BETTER P&L"""
        base_target = 0.025  # Increased from 0.015 to 0.025 (2.5%)

        confidence_multiplier = 0.8 + confidence * 0.6
        volatility_multiplier = 0.8 + volatility * 8

        target = base_target * confidence_multiplier * volatility_multiplier
        target = max(0.012, min(target, 0.08))  # Between 1.2% and 8%

        return target

    def _calculate_position_size(self):
        """Calculate position size based on available balance"""
        available_balance = self.balance
        position_value = available_balance * self.max_position_size * self.leverage
        return position_value

    def _calculate_stop_loss(self, action, entry_price):
        """Calculate stop loss"""
        if action == "LONG":
            return entry_price * 0.97
        else:
            return entry_price * 1.03

    def _open_position(self, action, target):
        """Open a new position - OPTIMIZED"""
        position_size = self._calculate_position_size()
        margin_required = position_size * 0.20

        if margin_required > self.balance * 0.9:  # Increased from 0.8 to 0.9
            return

        if margin_required < 5:  # Reduced from 10 to 5
            return

        crypto_size = position_size / self.current_price

        trade_action = "BUY" if action == "LONG" else "SELL"
        target_price = self.current_price * (1 + target) if action == "LONG" else self.current_price * (1 - target)

        position = {
            'id': len(self.positions) + len(self.closed_positions) + 1,
            'type': action,
            'trade_action': trade_action,
            'entry_price': self.current_price,
            'size': crypto_size,
            'entry_time': datetime.now(),
            'target': target,
            'target_price': target_price,
            'stop_loss': self._calculate_stop_loss(action, self.current_price),
            'position_value': position_size,
            'margin_used': margin_required,
            'leverage': self.leverage
        }

        self.balance -= margin_required
        self.positions.append(position)

    def _check_positions(self):
        """Check and close positions"""
        positions_to_close = []
        triggered_positions = []

        for pos in self.positions:
            current_profit_pct = 0
            should_close = False
            close_reason = ""

            if pos['type'] == "LONG":
                current_profit_pct = (self.current_price - pos['entry_price']) / pos['entry_price']
                target_threshold = pos.get('target', 0.015)
                
                if current_profit_pct >= target_threshold:
                    should_close = True
                    close_reason = "🎯 Target reached"
                elif self.current_price <= pos['stop_loss']:
                    should_close = True
                    close_reason = "🛑 Stop loss"

            elif pos['type'] == "SHORT":
                current_profit_pct = (pos['entry_price'] - self.current_price) / pos['entry_price']
                target_threshold = pos.get('target', 0.015)

                if current_profit_pct >= target_threshold:
                    should_close = True
                    close_reason = "🎯 Target reached"
                elif self.current_price >= pos['stop_loss']:
                    should_close = True
                    close_reason = "🛑 Stop loss"

            position_age = (datetime.now() - pos['entry_time']).seconds
            if position_age > 180 and current_profit_pct < -0.01:
                should_close = True
                close_reason = "⏰ Time stop"

            if current_profit_pct < -0.05:
                should_close = True
                close_reason = "🚨 Max loss limit"

            if should_close:
                if pos['type'] == "LONG":
                    profit_loss = pos['size'] * (self.current_price - pos['entry_price'])
                else:
                    profit_loss = pos['size'] * (pos['entry_price'] - self.current_price)

                self.balance += pos['margin_used'] + profit_loss
                self.total_profit_loss += profit_loss
                self.realized_pnl += profit_loss

                closed_pos = pos.copy()
                closed_pos['close_price'] = self.current_price
                closed_pos['close_time'] = datetime.now()
                closed_pos['profit_loss'] = profit_loss
                closed_pos['close_reason'] = close_reason
                closed_pos['profit_pct'] = current_profit_pct

                self.closed_positions.append(closed_pos)
                positions_to_close.append(pos)
                triggered_positions.append((pos, close_reason, profit_loss))

                self.total_trades += 1
                if profit_loss > 0:
                    self.winning_trades += 1
                    self.largest_win = max(self.largest_win, profit_loss)
                else:
                    self.largest_loss = min(self.largest_loss, profit_loss)

        for pos in positions_to_close:
            self.positions.remove(pos)

        return triggered_positions

    def _calculate_unrealized_pnl(self):
        """Calculate unrealized P&L"""
        unrealized = 0
        for pos in self.positions:
            if pos['type'] == "LONG":
                unrealized += pos['size'] * (self.current_price - pos['entry_price'])
            else:
                unrealized += pos['size'] * (pos['entry_price'] - self.current_price)
        
        self.unrealized_pnl = unrealized
        return unrealized

    def _update_performance_metrics(self):
        """Update performance tracking"""
        current_equity = self.balance + sum([pos.get('margin_used', 0) for pos in self.positions])
        current_equity += self._calculate_unrealized_pnl()
        
        if current_equity > self.peak_balance:
            self.peak_balance = current_equity
        
        current_drawdown = (self.peak_balance - current_equity) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
