class TradingConfig:
    """Main trading configuration"""
    
    # Bot Settings
    DEFAULT_INITIAL_BALANCE = 10000
    DEFAULT_CRYPTO_SYMBOL = "BTC/USDT"
    DEFAULT_LEVERAGE = 5
    DEFAULT_MAX_POSITIONS = 3
    
    # Risk Management
    DEFAULT_STOP_LOSS_THRESHOLD = 0.03  # 3%
    DEFAULT_MAX_POSITION_SIZE = 0.05    # 5% of balance
    DEFAULT_TAKE_PROFIT_TARGET = 0.015  # 1.5%
    MAX_DRAWDOWN_LIMIT = 0.20           # 20%
    
    # ML Model Settings
    ML_LOOKBACK_PERIOD = 20
    ML_CONFIDENCE_THRESHOLD = 0.52
    ML_RETRAIN_FREQUENCY = 50  # Retrain every 50 new data points
    
    # UI Settings
    CHART_UPDATE_INTERVAL = 3  # seconds
    MAX_ALERTS_DISPLAY = 5
    MAX_PRICE_HISTORY = 200
    
    # Colors (for UI)
    COLORS = {
        'primary': '#00d4ff',
        'success': '#00ff88',
        'danger': '#ff4444',
        'warning': '#ffb347',
        'background': 'rgba(26, 26, 46, 0.6)',
        'card_bg': 'rgba(26, 26, 46, 0.8)'
    }

class SimulationConfig:
    """Price simulation configuration"""
    
    BASE_VOLATILITY = 0.012      # 1.2% base volatility
    TREND_CYCLE_LENGTH = 60      # Price trend cycle in steps
    STRONG_MOVE_PROBABILITY = 0.1  # 10% chance of strong move
    STRONG_MOVE_SIZE = 0.02      # 2% strong move
    
    # Price limits
    MIN_PRICE = 1000
    MAX_PRICE = 100000