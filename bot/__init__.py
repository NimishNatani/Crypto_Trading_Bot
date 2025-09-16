"""
AI Crypto Trading Bot - Core Trading Engine Package
"""

from .trading_engine import CryptoTradingBot
from .ml_model import MLPredictor

__version__ = "2.1.0"
__author__ = "AI Crypto Trading Bot Team"

__all__ = [
    'CryptoTradingBot',
    'MLPredictor'
]