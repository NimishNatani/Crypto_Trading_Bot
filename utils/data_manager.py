import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class DataManager:
    def __init__(self):
        self.data_dir = "data"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def export_trading_data(self, bot, filename=None):
        """Export trading data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_data_{timestamp}.csv"
        
        # Combine price history with trading data
        price_df = pd.DataFrame(bot.price_history)
        
        # Export price history
        price_file = os.path.join(self.data_dir, f"price_history_{filename}")
        price_df.to_csv(price_file, index=False)
        
        # Export closed positions
        if bot.closed_positions:
            positions_df = pd.DataFrame(bot.closed_positions)
            positions_file = os.path.join(self.data_dir, f"closed_positions_{filename}")
            positions_df.to_csv(positions_file, index=False)
        
        # Export current positions
        if bot.positions:
            current_positions_df = pd.DataFrame(bot.positions)
            current_file = os.path.join(self.data_dir, f"open_positions_{filename}")
            current_positions_df.to_csv(current_file, index=False)
        
        return filename
    
    def export_performance_report(self, bot, performance_metrics, filename=None):
        """Export comprehensive performance report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "bot_config": {
                "initial_balance": bot.initial_balance,
                "leverage": bot.leverage,
                "max_positions": bot.max_open_positions,
                "stop_loss_threshold": bot.stop_loss_threshold,
                "max_position_size": bot.max_position_size
            },
            "performance_metrics": performance_metrics,
            "current_status": {
                "current_price": bot.current_price,
                "balance": bot.balance,
                "total_pnl": bot.total_profit_loss,
                "unrealized_pnl": bot.unrealized_pnl,
                "open_positions": len(bot.positions),
                "total_trades": len(bot.closed_positions)
            }
        }
        
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename
    
    def load_trading_session(self, filename):
        """Load previous trading session data"""
        try:
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            return None
    
    def get_price_statistics(self, price_history):
        """Calculate price statistics"""
        if not price_history:
            return {}
        
        prices = [p['price'] for p in price_history]
        volumes = [p['volume'] for p in price_history]
        
        return {
            "current_price": prices[-1],
            "min_price": min(prices),
            "max_price": max(prices),
            "avg_price": np.mean(prices),
            "price_volatility": np.std(prices) / np.mean(prices),
            "avg_volume": np.mean(volumes),
            "total_candles": len(price_history),
            "time_range": {
                "start": price_history[0]['timestamp'],
                "end": price_history[-1]['timestamp']
            }
        }
    
    def calculate_drawdown_periods(self, bot):
        """Calculate drawdown periods"""
        if not bot.closed_positions:
            return []
        
        df = pd.DataFrame(bot.closed_positions)
        df = df.sort_values('close_time')
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        df['running_balance'] = bot.initial_balance + df['cumulative_pnl']
        
        # Find peak and drawdown periods
        df['peak'] = df['running_balance'].expanding().max()
        df['drawdown'] = (df['peak'] - df['running_balance']) / df['peak']
        
        # Identify drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for idx, row in df.iterrows():
            if row['drawdown'] > 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = idx
            elif row['drawdown'] == 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                period_data = df.loc[start_idx:idx]
                drawdown_periods.append({
                    'start_time': period_data.iloc[0]['close_time'],
                    'end_time': period_data.iloc[-1]['close_time'],
                    'max_drawdown': period_data['drawdown'].max(),
                    'duration_trades': len(period_data),
                    'recovery_trades': idx - start_idx
                })
        
        return drawdown_periods
    
    def export_backtest_results(self, bot, test_duration):
        """Export backtest results for analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backtest_data = {
            "test_info": {
                "start_time": (datetime.now() - timedelta(seconds=test_duration)).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": test_duration,
                "final_balance": bot.balance + sum([pos.get('margin_used', 0) for pos in bot.positions])
            },
            "trades": [
                {
                    "id": trade['id'],
                    "type": trade['type'],
                    "entry_price": trade['entry_price'],
                    "close_price": trade['close_price'],
                    "profit_loss": trade['profit_loss'],
                    "duration": str(trade['close_time'] - trade['entry_time']),
                    "close_reason": trade.get('close_reason', 'Unknown')
                }
                for trade in bot.closed_positions
            ],
            "price_data": [
                {
                    "timestamp": p['timestamp'].isoformat(),
                    "price": p['price'],
                    "volume": p['volume']
                }
                for p in bot.price_history[-100:]  # Last 100 candles
            ]
        }
        
        filename = f"backtest_results_{timestamp}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(backtest_data, f, indent=2, default=str)
        
        return filename
    
    def get_trading_summary(self, bot):
        """Get trading session summary"""
        total_margin_used = sum([pos.get('margin_used', 0) for pos in bot.positions])
        total_equity = bot.balance + total_margin_used + bot.unrealized_pnl
        
        long_trades = [t for t in bot.closed_positions if t['type'] == 'LONG']
        short_trades = [t for t in bot.closed_positions if t['type'] == 'SHORT']
        
        winning_trades = [t for t in bot.closed_positions if t['profit_loss'] > 0]
        losing_trades = [t for t in bot.closed_positions if t['profit_loss'] < 0]
        
        return {
            "account": {
                "initial_balance": bot.initial_balance,
                "current_balance": bot.balance,
                "total_equity": total_equity,
                "margin_used": total_margin_used,
                "free_margin": bot.balance,
                "equity_change": total_equity - bot.initial_balance,
                "equity_change_pct": ((total_equity - bot.initial_balance) / bot.initial_balance) * 100
            },
            "trading": {
                "total_trades": len(bot.closed_positions),
                "open_positions": len(bot.positions),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": (len(winning_trades) / len(bot.closed_positions) * 100) if bot.closed_positions else 0,
                "long_trades": len(long_trades),
                "short_trades": len(short_trades)
            },
            "pnl": {
                "realized_pnl": bot.realized_pnl,
                "unrealized_pnl": bot.unrealized_pnl,
                "total_pnl": bot.total_profit_loss + bot.unrealized_pnl,
                "largest_win": bot.largest_win,
                "largest_loss": bot.largest_loss,
                "avg_win": np.mean([t['profit_loss'] for t in winning_trades]) if winning_trades else 0,
                "avg_loss": np.mean([t['profit_loss'] for t in losing_trades]) if losing_trades else 0
            },
            "risk": {
                "max_drawdown": bot.max_drawdown * 100,
                "current_drawdown": ((bot.peak_balance - total_equity) / bot.peak_balance * 100) if bot.peak_balance > 0 else 0,
                "risk_per_trade": bot.max_position_size * 100,
                "leverage_used": bot.leverage
            }
        }