import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

class PerformanceCalculator:
    def __init__(self):
        pass
    
    def calculate_performance_metrics(self, bot) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not bot.closed_positions:
            return self._empty_metrics()
        
        # Basic metrics
        total_trades = len(bot.closed_positions)
        winning_trades = len([t for t in bot.closed_positions if t['profit_loss'] > 0])
        losing_trades = total_trades - winning_trades
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        profits = [t['profit_loss'] for t in bot.closed_positions if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in bot.closed_positions if t['profit_loss'] < 0]
        
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(profits) if profits else 0
        largest_loss = min(losses) if losses else 0
        
        # Profit factor
        total_profits = sum(profits) if profits else 0
        total_losses = abs(sum(losses)) if losses else 0.01
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Total return
        current_equity = bot.balance + sum([pos.get('margin_used', 0) for pos in bot.positions])
        current_equity += bot._calculate_unrealized_pnl()
        total_return = ((current_equity - bot.initial_balance) / bot.initial_balance) * 100
        
        # Risk metrics
        max_drawdown = bot.max_drawdown * 100
        
        # Sharpe ratio approximation
        if bot.closed_positions:
            returns = [t['profit_loss'] / bot.initial_balance for t in bot.closed_positions]
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0.01
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Position type analysis
        long_trades = [t for t in bot.closed_positions if t['type'] == 'LONG']
        short_trades = [t for t in bot.closed_positions if t['type'] == 'SHORT']
        
        long_profit = sum([t['profit_loss'] for t in long_trades])
        short_profit = sum([t['profit_loss'] for t in short_trades])
        
        long_win_rate = (len([t for t in long_trades if t['profit_loss'] > 0]) / len(long_trades) * 100) if long_trades else 0
        short_win_rate = (len([t for t in short_trades if t['profit_loss'] > 0]) / len(short_trades) * 100) if short_trades else 0
        
        # Trading frequency
        if len(bot.closed_positions) >= 2:
            first_trade = min(bot.closed_positions, key=lambda x: x['close_time'])
            last_trade = max(bot.closed_positions, key=lambda x: x['close_time'])
            time_diff = (last_trade['close_time'] - first_trade['close_time']).total_seconds()
            trades_per_hour = (total_trades / (time_diff / 3600)) if time_diff > 0 else 0
        else:
            trades_per_hour = 0
        
        # Risk-adjusted return
        risk_adjusted_return = total_return / max(max_drawdown, 1)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_profit': long_profit,
            'short_profit': short_profit,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'trades_per_hour': trades_per_hour,
            'risk_adjusted_return': risk_adjusted_return,
            'current_equity': current_equity,
            'total_profit_loss': bot.total_profit_loss + bot.unrealized_pnl
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no trades exist"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_profit': 0,
            'short_profit': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'trades_per_hour': 0,
            'risk_adjusted_return': 0,
            'current_equity': 0,
            'total_profit_loss': 0
        }
    
    def calculate_trade_statistics(self, closed_positions: List) -> Dict:
        """Calculate detailed trade statistics"""
        if not closed_positions:
            return {}
        
        # Duration analysis
        durations = [(t['close_time'] - t['entry_time']).total_seconds() 
                    for t in closed_positions if 'entry_time' in t]
        
        avg_duration = np.mean(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # P&L distribution
        pnl_values = [t['profit_loss'] for t in closed_positions]
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_sequences(closed_positions, True)
        consecutive_losses = self._calculate_consecutive_sequences(closed_positions, False)
        
        return {
            'avg_trade_duration_seconds': avg_duration,
            'min_trade_duration_seconds': min_duration,
            'max_trade_duration_seconds': max_duration,
            'pnl_std_dev': np.std(pnl_values),
            'pnl_skewness': self._calculate_skewness(pnl_values),
            'max_consecutive_wins': max(consecutive_wins) if consecutive_wins else 0,
            'max_consecutive_losses': max(consecutive_losses) if consecutive_losses else 0,
            'avg_consecutive_wins': np.mean(consecutive_wins) if consecutive_wins else 0,
            'avg_consecutive_losses': np.mean(consecutive_losses) if consecutive_losses else 0
        }
    
    def _calculate_consecutive_sequences(self, trades: List, winning: bool) -> List[int]:
        """Calculate consecutive winning or losing streaks"""
        sequences = []
        current_streak = 0
        
        for trade in trades:
            is_winner = trade['profit_loss'] > 0
            
            if (winning and is_winner) or (not winning and not is_winner):
                current_streak += 1
            else:
                if current_streak > 0:
                    sequences.append(current_streak)
                current_streak = 0
        
        # Add final streak if exists
        if current_streak > 0:
            sequences.append(current_streak)
        
        return sequences
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of P&L distribution"""
        if len(values) < 3:
            return 0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0
        
        skewness = np.mean([((x - mean_val) / std_val) ** 3 for x in values])
        return skewness
    
    def calculate_risk_metrics(self, bot) -> Dict:
        """Calculate risk management metrics"""
        total_equity = bot.balance + sum([pos.get('margin_used', 0) for pos in bot.positions])
        total_equity += bot._calculate_unrealized_pnl()
        
        # Value at Risk (simplified)
        if bot.closed_positions:
            returns = [t['profit_loss'] / bot.initial_balance for t in bot.closed_positions]
            var_95 = np.percentile(returns, 5) * total_equity if len(returns) > 10 else 0
        else:
            var_95 = 0
        
        # Current risk exposure
        margin_used = sum([pos.get('margin_used', 0) for pos in bot.positions])
        risk_exposure = margin_used / total_equity if total_equity > 0 else 0
        
        # Position concentration
        if bot.positions:
            position_sizes = [pos.get('position_value', 0) for pos in bot.positions]
            max_position = max(position_sizes)
            position_concentration = max_position / total_equity if total_equity > 0 else 0
        else:
            position_concentration = 0
        
        return {
            'value_at_risk_95': var_95,
            'risk_exposure_pct': risk_exposure * 100,
            'position_concentration_pct': position_concentration * 100,
            'leverage_ratio': bot.leverage,
            'max_drawdown_pct': bot.max_drawdown * 100,
            'current_drawdown_pct': ((bot.peak_balance - total_equity) / bot.peak_balance * 100) if bot.peak_balance > 0 else 0
        }
    
    def generate_performance_grade(self, metrics: Dict) -> Dict:
        """Generate overall performance grade"""
        score = 0
        
        # Return score (25 points)
        if metrics['total_return'] > 20:
            score += 25
        elif metrics['total_return'] > 10:
            score += 20
        elif metrics['total_return'] > 5:
            score += 15
        elif metrics['total_return'] > 0:
            score += 10
        
        # Win rate score (25 points)
        if metrics['win_rate'] > 70:
            score += 25
        elif metrics['win_rate'] > 60:
            score += 20
        elif metrics['win_rate'] > 50:
            score += 15
        elif metrics['win_rate'] > 40:
            score += 10
        
        # Profit factor score (25 points)
        if metrics['profit_factor'] > 2.5:
            score += 25
        elif metrics['profit_factor'] > 2.0:
            score += 20
        elif metrics['profit_factor'] > 1.5:
            score += 15
        elif metrics['profit_factor'] > 1.0:
            score += 10
        
        # Risk score (25 points) - lower drawdown is better
        if metrics['max_drawdown'] < 5:
            score += 25
        elif metrics['max_drawdown'] < 10:
            score += 20
        elif metrics['max_drawdown'] < 15:
            score += 15
        elif metrics['max_drawdown'] < 25:
            score += 10
        
        # Determine grade
        if score >= 85:
            grade = "A+"
            description = "Excellent Performance"
        elif score >= 75:
            grade = "A"
            description = "Very Good Performance"
        elif score >= 65:
            grade = "B+"
            description = "Good Performance"
        elif score >= 55:
            grade = "B"
            description = "Average Performance"
        elif score >= 45:
            grade = "C+"
            description = "Below Average"
        elif score >= 35:
            grade = "C"
            description = "Poor Performance"
        else:
            grade = "D"
            description = "Very Poor Performance"
        
        return {
            'grade': grade,
            'score': score,
            'max_score': 100,
            'description': description
        }