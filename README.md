README_MD = """# AI Crypto Trading Bot Pro

A sophisticated real-time cryptocurrency trading bot with machine learning predictions, technical analysis, and professional UI built with Streamlit.

## Features

### 🤖 Advanced AI Trading
- **Machine Learning Predictions**: Random Forest classifier with technical indicators
- **Real-time Market Analysis**: RSI, Moving Averages, Bollinger Bands, Momentum
- **Both LONG and SHORT Positions**: Complete bidirectional trading support
- **Dynamic Position Sizing**: Risk-based position calculation with leverage

### 📊 Professional Dashboard
- **Real-time Price Charts**: Candlestick charts with technical indicators
- **Live P&L Tracking**: Unrealized and realized profit/loss monitoring
- **Portfolio Analytics**: Comprehensive performance metrics
- **Risk Management**: Stop-loss, take-profit, and drawdown protection

### 🎯 Risk Management
- **5x Leverage Trading**: Margin-based position management
- **Stop Loss & Take Profit**: Automated position closure
- **Position Limits**: Maximum position size and count controls
- **Drawdown Protection**: Real-time risk monitoring

### 📈 Performance Analytics
- **Win Rate Analysis**: Detailed success rate tracking
- **Sharpe Ratio Calculation**: Risk-adjusted return metrics
- **Drawdown Analysis**: Maximum and current drawdown tracking
- **Trade Statistics**: Comprehensive trading history analysis

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or Create Project Structure**:
   ```
   trading_bot/
   ├── app.py                          # Main Streamlit application
   ├── bot/
   │   ├── __init__.py
   │   ├── trading_engine.py           # Core trading logic
   │   └── ml_model.py                 # Machine learning predictor
   ├── ui/
   │   ├── __init__.py
   │   └── dashboard_components.py     # UI components
   ├── utils/
   │   ├── __init__.py
   │   ├── data_manager.py            # Data export/import
   │   └── performance_calculator.py   # Performance metrics
   ├── data/                          # Data storage directory
   ├── config.py                      # Configuration settings
   ├── requirements.txt
   └── README.md
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage Guide

### Starting the Bot

1. **Configure Parameters** (in sidebar):
   - Initial Balance: Set your starting capital
   - Leverage: Choose leverage ratio (1x to 10x)
   - Risk Management: Set stop-loss and position size limits

2. **Start Trading**:
   - Click "START" button to begin live trading
   - The bot will automatically analyze market conditions
   - AI model will make LONG/SHORT predictions
   - Positions will be opened/closed automatically

3. **Monitor Performance**:
   - Watch real-time P&L in the main dashboard
   - Check open positions and their current status
   - Review trading alerts and notifications

### Dashboard Sections

#### 📈 Live Trading
- Real-time price chart with technical indicators
- Current open positions with live P&L
- Recent trading alerts and notifications
- Market analysis and AI predictions

#### 📊 Analytics  
- Technical indicator values (RSI, Momentum, Volatility)
- AI model predictions and confidence levels
- Market sentiment analysis
- Model performance metrics

#### 💼 Portfolio
- Portfolio overview with balance breakdown
- Detailed position information
- Recent trade history
- Margin usage and available balance

#### 📋 Performance
- Comprehensive performance report
- Win rate and profit factor analysis
- Risk metrics and drawdown analysis
- Performance grading system

## Trading Strategy

### AI-Powered Decision Making
The bot uses a Random Forest classifier trained on:
- Moving averages (SMA 5, 10, 20)
- RSI (Relative Strength Index)
- Price momentum indicators
- Volatility measurements
- Volume trend analysis

### Position Management
- **LONG Positions**: Opened when AI predicts upward price movement
- **SHORT Positions**: Opened when AI predicts downward price movement
- **Risk Controls**: Each position has stop-loss and take-profit levels
- **Leverage**: Configurable leverage up to 10x with margin management

### Risk Management Features
- Maximum 3 open positions simultaneously
- 5% maximum position size per trade
- 3% stop-loss threshold
- Dynamic take-profit targets based on market conditions
- Real-time drawdown monitoring

## Performance Metrics

### Key Performance Indicators
- **Total Return**: Overall portfolio performance percentage
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Maximum Drawdown**: Largest peak-to-trough decline

### Advanced Analytics
- Separate analysis for LONG vs SHORT trades
- Trade duration statistics
- Consecutive win/loss streak analysis
- Risk-adjusted performance grading

## Configuration

### Bot Parameters
Edit `config.py` to customize:
- Default balance and leverage settings
- Risk management thresholds
- ML model parameters
- UI update intervals

### Risk Settings
- `DEFAULT_STOP_LOSS_THRESHOLD`: Stop-loss percentage
- `DEFAULT_MAX_POSITION_SIZE`: Maximum position size
- `MAX_DRAWDOWN_LIMIT`: Portfolio protection limit

## Important Notes

### ⚠️ Risk Disclaimer
This is a **simulation/educational tool** for learning algorithmic trading concepts. Key points:

- **Paper Trading Only**: No real money or live exchange connections
- **Educational Purpose**: Designed for learning trading strategies and risk management
- **Simulated Data**: Uses generated price data, not real market feeds
- **No Financial Advice**: Not intended for actual trading decisions

### 🔒 Safety Features
- All trading is simulated with virtual money
- No external API connections to real exchanges
- Risk limits prevent excessive losses in simulation
- Complete control over all trading parameters

### 📚 Learning Objectives
- Understand algorithmic trading concepts
- Learn risk management principles
- Experience ML-based trading strategies
- Practice portfolio management skills

## Technical Details

### Architecture
- **Frontend**: Streamlit web application
- **Backend**: Python with sklearn for ML
- **Data Visualization**: Plotly for interactive charts
- **Real-time Updates**: Automatic refresh system

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Features**: 10 technical indicators
- **Training**: Continuous learning with new data
- **Prediction**: Market direction with confidence scores

### Performance
- **Update Frequency**: 3-second intervals
- **Data Retention**: Last 200 price candles
- **Memory Efficient**: Optimized for long-running sessions
- **Export Capability**: CSV/JSON data export

## Troubleshooting

### Common Issues
1. **Slow Performance**: Reduce update frequency in config
2. **Memory Usage**: Restart app after extended sessions  
3. **Chart Loading**: Check browser compatibility
4. **Data Export**: Ensure data/ directory exists

### Support
For issues or questions:
- Check configuration settings in `config.py`
- Review console output for error messages
- Ensure all dependencies are installed correctly

---

**Version**: 2.1.0 Pro  
**License**: Educational Use Only  
**Author**: AI Crypto Trading Bot Team
"""

# Create __init__.py files content
INIT_PY_CONTENT = """# AI Crypto Trading Bot Package"""

if __name__ == "__main__":
    # Write requirements.txt
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS_TXT.strip())
    
    # Write README.md
    with open("README.md", "w") as f:
        f.write(README_MD.strip())
    
    # Create __init__.py files
    import os
    
    dirs = ["bot", "ui", "utils"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        with open(f"{dir_name}/__init__.py", "w") as f:
            f.write(INIT_PY_CONTENT)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    print("Configuration files created successfully!")
    print("\nProject structure:")
    print("├── app.py")
    print("├── config.py") 
    print("├── requirements.txt")
    print("├── README.md")
    print("├── bot/")
    print("│   ├── __init__.py")
    print("│   ├── trading_engine.py")
    print("│   └── ml_model.py")
    print("├── ui/")
    print("│   ├── __init__.py")
    print("│   └── dashboard_components.py")
    print("├── utils/")
    print("│   ├── __init__.py") 
    print("│   ├── data_manager.py")
    print("│   └── performance_calculator.py")
    print("└── data/")
    print("\nTo run: streamlit run app.py")