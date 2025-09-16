import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
import warnings
warnings.filterwarnings('ignore')

class CryptoTradingBot:
    def __init__(self, initial_balance=10000, crypto_symbol="BTC/USDT"):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto_symbol = crypto_symbol
        self.positions = []  # Start with NO positions
        self.closed_positions = []
        self.current_price = 50000  # Starting BTC price
        self.price_history = []
        self.total_profit_loss = 0
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        
        # Performance tracking
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.largest_win = 0
        self.largest_loss = 0
        
        # Risk management parameters - EXACT FROM ORIGINAL
        self.max_position_size = 0.05  # 5% of balance per trade
        self.max_open_positions = 3  # Maximum 3 positions
        self.stop_loss_threshold = 0.03  # 3% stop loss
        self.leverage = 5  # 5x leverage
        
        # Trading state
        self.running = False
        self.last_update = datetime.now()
        
        # Initialize with historical data only
        self._generate_initial_data()
        # Start with clean slate - no pre-existing positions

    def _generate_initial_data(self):
        """Generate initial price data for model training - EXACT FROM ORIGINAL"""
        np.random.seed(42)
        base_time = datetime.now() - timedelta(hours=2)
        
        for i in range(120):
            change = np.random.normal(0, 0.015)
            old_price = self.current_price
            self.current_price *= (1 + change)
            
            # Generate realistic OHLC
            high = max(old_price, self.current_price) * (1 + abs(np.random.normal(0, 0.008)))
            low = min(old_price, self.current_price) * (1 - abs(np.random.normal(0, 0.008)))
            
            self.price_history.append({
                'timestamp': base_time + timedelta(minutes=i),
                'open': old_price,
                'high': high,
                'low': low,
                'close': self.current_price,
                'price': self.current_price,
                'volume': np.random.uniform(500, 2000)
            })

    def _simulate_some_trades(self):
        """Simulate some historical trades for demo - EXACT FROM ORIGINAL"""
        trades_data = [
            {'type': 'LONG', 'entry': 49000, 'exit': 51000, 'size': 0.02},
            {'type': 'SHORT', 'entry': 52000, 'exit': 50500, 'size': 0.015},
            {'type': 'LONG', 'entry': 48500, 'exit': 47800, 'size': 0.025},
            {'type': 'LONG', 'entry': 47000, 'exit': 49500, 'size': 0.03},
            {'type': 'SHORT', 'entry': 51500, 'exit': 50000, 'size': 0.018},
        ]
        
        for i, trade in enumerate(trades_data):
            if trade['type'] == 'LONG':
                profit = trade['size'] * (trade['exit'] - trade['entry'])
            else:  # SHORT
                profit = trade['size'] * (trade['entry'] - trade['exit'])
            
            self.closed_positions.append({
                'id': i + 1,
                'type': trade['type'],
                'trade_action': 'BUY' if trade['type'] == 'LONG' else 'SELL',
                'entry_price': trade['entry'],
                'close_price': trade['exit'],
                'size': trade['size'],
                'position_value': trade['size'] * trade['entry'],
                'profit_loss': profit,
                'close_time': datetime.now() - timedelta(hours=random.randint(1, 24)),
                'close_reason': 'PROFIT_TARGET' if profit > 0 else 'STOP_LOSS',
                'entry_time': datetime.now() - timedelta(hours=random.randint(2, 25)),
                'profit_pct': profit / (trade['size'] * trade['entry'])
            })
            
            self.total_profit_loss += profit
            self.realized_pnl += profit
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
                self.largest_win = max(self.largest_win, profit)
            else:
                self.largest_loss = min(self.largest_loss, profit)
        
        # Add some open positions
        self.positions = [
            {
                'id': len(self.closed_positions) + 1,
                'type': 'LONG',
                'trade_action': 'BUY',
                'entry_price': 49800,
                'size': 0.02,
                'position_value': 0.02 * 49800,
                'target_price': 52000,
                'stop_loss': 47500,
                'entry_time': datetime.now() - timedelta(minutes=30),
                'margin_used': 0.02 * 49800 * 0.2,  # 20% margin for 5x leverage
                'leverage': 5
            },
            {
                'id': len(self.closed_positions) + 2,
                'type': 'SHORT',
                'trade_action': 'SELL',
                'entry_price': 50200,
                'size': 0.015,
                'position_value': 0.015 * 50200,
                'target_price': 48000,
                'stop_loss': 52000,
                'entry_time': datetime.now() - timedelta(minutes=15),
                'margin_used': 0.015 * 50200 * 0.2,  # 20% margin for 5x leverage
                'leverage': 5
            }
        ]
        
        # Update balance to reflect margin usage
        total_margin = sum([pos['margin_used'] for pos in self.positions])
        self.balance -= total_margin

    def _calculate_technical_indicators(self, lookback=20):
        """Calculate technical indicators for ML model - EXACT FROM ORIGINAL"""
        if len(self.price_history) < lookback:
            return None

        prices = [p['price'] for p in self.price_history[-lookback:]]
        volumes = [p['volume'] for p in self.price_history[-lookback:]]

        # Moving averages
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)

        # RSI-like indicator
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

    def _simulate_price_movement(self):
        """Simulate realistic price movement - EXACT FROM ORIGINAL LOGIC"""
        # Create more volatile and bidirectional price movement
        base_volatility = 0.012  # 1.2% base volatility

        # Add cyclical trends that go both up and down
        time_factor = len(self.price_history) % 60
        if time_factor < 15:
            trend_factor = 0.002  # Upward trend
        elif time_factor < 30:
            trend_factor = -0.003  # Strong downward trend (for SHORT opportunities)
        elif time_factor < 45:
            trend_factor = -0.001  # Mild downward trend
        else:
            trend_factor = 0.0015  # Mild upward trend

        # Add random volatility
        random_factor = np.random.normal(0, base_volatility)

        # Occasional strong movements in either direction
        if random.random() < 0.1:  # 10% chance of strong movement
            strong_move = np.random.choice([-0.02, 0.02])  # 2% strong move up or down
            random_factor += strong_move

        price_change = trend_factor + random_factor
        old_price = self.current_price
        self.current_price *= (1 + price_change)

        # Ensure price doesn't go negative or too extreme
        self.current_price = max(self.current_price, 1000)  # Minimum $1000
        self.current_price = min(self.current_price, 100000)  # Maximum $100,000

        # Generate OHLC data
        if price_change > 0:
            high = self.current_price * (1 + abs(np.random.normal(0, 0.003)))
            low = old_price * (1 - abs(np.random.normal(0, 0.002)))
        else:
            high = old_price * (1 + abs(np.random.normal(0, 0.002)))
            low = self.current_price * (1 - abs(np.random.normal(0, 0.003)))

        self.price_history.append({
            'timestamp': datetime.now(),
            'open': old_price,
            'high': max(old_price, self.current_price, high),
            'low': min(old_price, self.current_price, low),
            'close': self.current_price,
            'price': self.current_price,
            'volume': np.random.uniform(800, 1500)
        })
        
        # Keep only last 200 candles
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]

    def _calculate_position_size(self):
        """Calculate position size - EXACT FROM ORIGINAL"""
        margin_used = sum([pos.get('margin_used', 0) for pos in self.positions])
        available_balance = self.balance

        # Use 5% of available balance as margin per trade
        margin_per_trade = available_balance * 0.05  # 5% margin

        # With leverage
        position_value = margin_per_trade * self.leverage

        # Minimum and maximum position sizes
        min_position = 50  # Minimum $50 position
        max_position = available_balance * 0.20  # Never more than 20% of total balance

        position_size = max(min_position, min(position_value, max_position))
        return position_size

    def _should_open_position(self):
        """Determine if we should open a new position - EXACT LOGIC FROM ORIGINAL"""
        if len(self.positions) >= self.max_open_positions:
            return False, None, None

        # Check available balance for margin
        margin_used = sum([pos.get('margin_used', 0) for pos in self.positions])
        available_balance = self.balance

        if available_balance < 100:  # Need at least $100 available
            return False, None, None

        # Get technical prediction
        prediction, confidence = self._predict_market_direction()

        # Lower confidence threshold to get more trades
        if confidence < 0.52:  # 52% confidence threshold
            return False, None, None

        indicators = self._calculate_technical_indicators()
        if indicators is None:
            return False, None, None

        volatility = indicators['volatility']

        # Allow higher volatility for more trading opportunities
        if volatility > 0.08:  # 8% volatility threshold
            return False, None, None

        # Determine action
        action = "LONG" if prediction == 1 else "SHORT"
        target = self._calculate_dynamic_target(confidence, volatility)

        return True, action, target

    def _predict_market_direction(self):
        """Predict market direction - EXACT FROM ORIGINAL"""
        indicators = self._calculate_technical_indicators()
        if indicators is None:
            return random.choice([0, 1]), random.uniform(0.55, 0.8)

        # Technical analysis bias for better SHORT detection
        rsi = indicators['rsi']
        momentum = indicators['momentum']
        price_trend = indicators['price_to_sma_5']

        # Force SHORT bias when conditions are right
        if (rsi > 70 or momentum < -0.02 or price_trend < 0.98):  # Overbought/downward momentum
            prediction = 0  # SHORT
            confidence = random.uniform(0.65, 0.85)
        elif (rsi < 30 or momentum > 0.02 or price_trend > 1.02):  # Oversold/upward momentum
            prediction = 1  # LONG
            confidence = random.uniform(0.65, 0.85)
        else:
            # Random prediction but ensure both directions
            prediction = random.choice([0, 1])
            confidence = random.uniform(0.55, 0.75)

            # Add randomness to ensure SHORT positions happen
            if random.random() < 0.4:  # 40% chance to flip prediction
                prediction = 1 - prediction
                confidence = max(confidence, 0.6)

        return prediction, confidence

    def _calculate_dynamic_target(self, confidence, volatility):
        """Calculate dynamic profit target - EXACT FROM ORIGINAL"""
        base_target = 0.015  # 1.5% base target

        # Adjust based on confidence and volatility
        confidence_multiplier = 0.8 + confidence * 0.6
        volatility_multiplier = 0.8 + volatility * 8

        target = base_target * confidence_multiplier * volatility_multiplier
        target = max(0.008, min(target, 0.05))  # Between 0.8% and 5%

        return target

    def _calculate_stop_loss(self, action, entry_price):
        """Calculate stop loss - EXACT FROM ORIGINAL"""
        if action == "LONG":
            # LONG: Stop loss below entry price (3% down)
            return entry_price * 0.97
        else:
            # SHORT: Stop loss above entry price (3% up)
            return entry_price * 1.03

    def _open_position(self, action, target):
        """Open position - EXACT FROM ORIGINAL LOGIC"""
        position_size = self._calculate_position_size()
        margin_required = position_size * 0.20  # 20% for 5x leverage

        if margin_required > self.balance * 0.8:
            return

        if margin_required < 10:
            return

        # Calculate crypto size
        crypto_size = position_size / self.current_price

        trade_action = "BUY" if action == "LONG" else "SELL"
        target_price = self.current_price * (1 + target) if action == "LONG" else self.current_price * (1 - target)

        position = {
            'id': len(self.positions) + len(self.closed_positions) + 1,
            'type': action,  # LONG or SHORT
            'trade_action': trade_action,  # BUY or SELL
            'entry_price': self.current_price,
            'size': crypto_size,  # Amount of crypto
            'entry_time': datetime.now(),
            'target': target,
            'target_price': target_price,
            'stop_loss': self._calculate_stop_loss(action, self.current_price),
            'position_value': position_size,
            'margin_used': margin_required,
            'leverage': self.leverage
        }

        # Reserve margin from balance
        self.balance -= margin_required
        self.positions.append(position)

    def _check_positions(self):
        """Check and close positions - EXACT FROM ORIGINAL"""
        positions_to_close = []
        triggered_positions = []

        for pos in self.positions:
            current_profit_pct = 0
            should_close = False
            close_reason = ""

            if pos['type'] == "LONG":
                # LONG: Profit when price goes UP
                current_profit_pct = (self.current_price - pos['entry_price']) / pos['entry_price']
                
                # Use 'target' if available, otherwise calculate from target percentage
                target_threshold = pos.get('target', 0.015)  # Default 1.5% target
                
                if current_profit_pct >= target_threshold:
                    should_close = True
                    close_reason = "🎯 Target reached"
                elif self.current_price <= pos['stop_loss']:
                    should_close = True
                    close_reason = "🛑 Stop loss"

            elif pos['type'] == "SHORT":
                # SHORT: Profit when price goes DOWN
                current_profit_pct = (pos['entry_price'] - self.current_price) / pos['entry_price']
                
                # Use 'target' if available, otherwise calculate from target percentage
                target_threshold = pos.get('target', 0.015)  # Default 1.5% target

                if current_profit_pct >= target_threshold:
                    should_close = True
                    close_reason = "🎯 Target reached"
                elif self.current_price >= pos['stop_loss']:
                    should_close = True
                    close_reason = "🛑 Stop loss"

            # Time-based exit
            position_age = (datetime.now() - pos['entry_time']).seconds
            if position_age > 180 and current_profit_pct < -0.01:  # 3 minutes and -1%
                should_close = True
                close_reason = "⏰ Time stop"

            # Maximum loss protection
            if current_profit_pct < -0.05:  # -5% maximum loss
                should_close = True
                close_reason = "🚨 Max loss limit"

            if should_close:
                # Calculate profit/loss
                if pos['type'] == "LONG":
                    profit_loss = pos['size'] * (self.current_price - pos['entry_price'])
                else:
                    profit_loss = pos['size'] * (pos['entry_price'] - self.current_price)

                # Return margin plus profit/loss
                self.balance += pos['margin_used'] + profit_loss
                self.total_profit_loss += profit_loss
                self.realized_pnl += profit_loss

                # Record closed position
                closed_pos = pos.copy()
                closed_pos['close_price'] = self.current_price
                closed_pos['close_time'] = datetime.now()
                closed_pos['profit_loss'] = profit_loss
                closed_pos['close_reason'] = close_reason
                closed_pos['profit_pct'] = current_profit_pct

                self.closed_positions.append(closed_pos)
                positions_to_close.append(pos)
                triggered_positions.append((pos, close_reason, profit_loss))

                # Update statistics
                self.total_trades += 1
                if profit_loss > 0:
                    self.winning_trades += 1
                    self.largest_win = max(self.largest_win, profit_loss)
                else:
                    self.largest_loss = min(self.largest_loss, profit_loss)

        # Remove closed positions
        for pos in positions_to_close:
            self.positions.remove(pos)

        return triggered_positions

    def _calculate_unrealized_pnl(self):
        """Calculate unrealized P&L - EXACT FROM ORIGINAL"""
        unrealized = 0
        for pos in self.positions:
            if pos['type'] == "LONG":
                unrealized += pos['size'] * (self.current_price - pos['entry_price'])
            else:  # SHORT
                unrealized += pos['size'] * (pos['entry_price'] - self.current_price)
        
        self.unrealized_pnl = unrealized
        return unrealized

    def _trading_step(self):
        """Execute one trading step - EXACT FROM ORIGINAL"""
        # Simulate price movement
        self._simulate_price_movement()
        
        # Check existing positions
        self._check_positions()
        
        # Update performance metrics
        self._update_performance_metrics()

    def _update_performance_metrics(self):
        """Update performance tracking"""
        current_equity = self.balance + sum([pos.get('margin_used', 0) for pos in self.positions])
        current_equity += self._calculate_unrealized_pnl()
        
        # Update peak balance and drawdown
        if current_equity > self.peak_balance:
            self.peak_balance = current_equity
        
        current_drawdown = (self.peak_balance - current_equity) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)