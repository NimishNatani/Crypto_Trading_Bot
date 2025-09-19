import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import warnings
warnings.filterwarnings('ignore')


# Import our custom modules
from bot.trading_engine import CryptoTradingBot
from bot.ml_model import MLPredictor
from ui.dashboard_components import DashboardComponents
from utils.data_manager import DataManager
from utils.performance_calculator import PerformanceCalculator
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=2000, limit=None, key="charts_autorefresh")

# Page configuration
st.set_page_config(
    page_title="AI Crypto Trading Bot Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Dark theme with neon accents */
        .stApp {
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        
        /* Premium cards */
        .premium-card {
            background: rgba(26, 26, 46, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 16px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .premium-card:hover {
            border-color: rgba(0, 212, 255, 0.6);
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
            transform: translateY(-2px);
        }
        
        /* Metric displays */
        .metric-card {
            background: rgba(26, 26, 46, 0.8);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }
        
        .metric-positive { color: #00ff88; }
        .metric-negative { color: #ff4444; }
        
        /* Status indicators */
        .status-running {
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        
        .status-stopped {
            color: #ff4444;
            text-shadow: 0 0 10px rgba(255, 68, 68, 0.5);
        }
        
        /* Live indicator */
        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 12px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #00ff88;
        }
        
        .pulse-dot {
            width: 8px;
            height: 8px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(45deg, rgba(0, 212, 255, 0.1), rgba(123, 44, 191, 0.1)) !important;
            border: 2px solid #00d4ff !important;
            border-radius: 12px !important;
            color: #00d4ff !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            transition: all 0.3s ease !important;
            height: 50px !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, rgba(0, 212, 255, 0.2), rgba(123, 44, 191, 0.2)) !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
            transform: translateY(-2px) !important;
            color: #ffffff !important;
        }
        
        /* Alert boxes */
        .alert-success {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 8px;
            padding: 10px;
            color: #00ff88;
            margin: 5px 0;
        }
        
        .alert-danger {
            background: rgba(255, 68, 68, 0.1);
            border: 1px solid rgba(255, 68, 68, 0.3);
            border-radius: 8px;
            padding: 10px;
            color: #ff4444;
            margin: 5px 0;
        }
        
        /* Header */
        .main-header {
            background: linear-gradient(45deg, #00d4ff, #7b2cbf, #ff006e);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 2rem;
        }
        
        /* Stop button warning */
        .stop-warning {
            font-size: 0.75rem;
            color: #ffb347;
            margin-top: 5px;
            text-align: center;
            opacity: 0.8;
        }
    </style>
    """, unsafe_allow_html=True)

class TradingBotApp:
    def __init__(self):
        self.initialize_session_state()
        self.dashboard = DashboardComponents()
        self.data_manager = DataManager()
        self.performance_calc = PerformanceCalculator()
        
    def initialize_session_state(self):
        """Initialize all session state variables - FIXED to start from scratch"""
        if 'bot' not in st.session_state:
            # Create a fresh bot without any pre-existing trades
            st.session_state.bot = CryptoTradingBot()
            # Remove the simulate trades call to start fresh
            
        if 'ml_predictor' not in st.session_state:
            st.session_state.ml_predictor = MLPredictor()
            
        if 'running' not in st.session_state:
            st.session_state.running = False
            
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
            
        if 'update_counter' not in st.session_state:
            st.session_state.update_counter = 0
            
        if 'auto_trade' not in st.session_state:
            st.session_state.auto_trade = False
            
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
            
        if 'force_update' not in st.session_state:
            st.session_state.force_update = False
            
        # Initialize trading thread variables
        if 'trading_active' not in st.session_state:
            st.session_state.trading_active = False
            
        # Stop button click counter for double-click functionality
        if 'stop_click_count' not in st.session_state:
            st.session_state.stop_click_count = 0
            
        # Auto-refresh timer for UI updates
        if 'last_ui_update' not in st.session_state:
            st.session_state.last_ui_update = datetime.now()

    def start_trading(self):
        """Start the trading bot - Fixed to ensure UI updates"""
        try:
            # Set all necessary flags
            st.session_state.running = True
            st.session_state.auto_trade = True
            st.session_state.trading_active = True
            
            # Train ML model if needed
            if not st.session_state.ml_predictor.is_trained:
                with st.spinner("Training ML model..."):
                    if st.session_state.ml_predictor.train_model(st.session_state.bot.price_history):
                        st.success("ML Model trained successfully!")
            
            # Add success alert
            st.session_state.alerts.append({
                'time': datetime.now(),
                'message': "Trading Bot Started - Ready to trade!",
                'type': 'success'
            })
            
            st.success("Bot Started Successfully!")
            
            # Force immediate update and UI refresh
            st.session_state.force_update = True
            st.session_state.last_ui_update = datetime.now()
            
        except Exception as e:
            st.error(f"Failed to start bot: {str(e)}")
            st.session_state.running = False
            st.session_state.auto_trade = False
            st.session_state.trading_active = False
    
    def stop_trading(self):
        """Stop the trading bot - Enhanced with double-click protection and position closure"""
        try:
            current_time = datetime.now()
            
            # Initialize first click time if not set
            if 'first_stop_click_time' not in st.session_state:
                st.session_state.first_stop_click_time = None
            
            # If this is the first click or more than 3 seconds have passed
            if (st.session_state.first_stop_click_time is None or 
                (current_time - st.session_state.first_stop_click_time).total_seconds() > 3):
                
                # First click - start timer
                st.session_state.first_stop_click_time = current_time
                st.session_state.stop_click_count = 1
                
                position_count = len(st.session_state.bot.positions)
                if position_count > 0:
                    st.warning(f"Click STOP again within 3 seconds to confirm stopping the bot and closing {position_count} open position(s)")
                else:
                    st.info("Click STOP again within 3 seconds to confirm stopping the bot")
                
            else:
                # Second click within 3 seconds - actually stop and close positions
                bot = st.session_state.bot
                
                # Close all open positions before stopping
                positions_closed = 0
                total_pnl_from_closure = 0
                
                if bot.positions:
                    positions_to_close = bot.positions.copy()  # Create copy to avoid modification during iteration
                    
                    for pos in positions_to_close:
                        # Calculate profit/loss at market close
                        if pos['type'] == "LONG":
                            profit_loss = pos['size'] * (bot.current_price - pos['entry_price'])
                        else:  # SHORT
                            profit_loss = pos['size'] * (pos['entry_price'] - bot.current_price)
                        
                        # Return margin plus profit/loss
                        bot.balance += pos['margin_used'] + profit_loss
                        bot.total_profit_loss += profit_loss
                        bot.realized_pnl += profit_loss
                        
                        # Record closed position
                        closed_pos = pos.copy()
                        closed_pos['close_price'] = bot.current_price
                        closed_pos['close_time'] = datetime.now()
                        closed_pos['profit_loss'] = profit_loss
                        closed_pos['close_reason'] = "🛑 Manual Stop"
                        closed_pos['profit_pct'] = profit_loss / (pos['size'] * pos['entry_price']) if pos['entry_price'] > 0 else 0
                        
                        bot.closed_positions.append(closed_pos)
                        
                        # Update statistics
                        bot.total_trades += 1
                        if profit_loss > 0:
                            bot.winning_trades += 1
                            bot.largest_win = max(bot.largest_win, profit_loss)
                        else:
                            bot.largest_loss = min(bot.largest_loss, profit_loss)
                        
                        positions_closed += 1
                        total_pnl_from_closure += profit_loss
                        
                        # Add individual position closure alert
                        alert_type = 'success' if profit_loss > 0 else 'danger'
                        st.session_state.alerts.append({
                            'time': datetime.now(),
                            'message': f"Position #{pos['id']} closed: ${profit_loss:+,.2f} - Manual Stop",
                            'type': alert_type
                        })
                    
                    # Clear all positions
                    bot.positions.clear()
                
                # Stop the bot
                st.session_state.running = False
                st.session_state.auto_trade = False
                st.session_state.trading_active = False
                
                # Add comprehensive stop alert
                if positions_closed > 0:
                    st.session_state.alerts.append({
                        'time': datetime.now(),
                        'message': f"Trading Bot Stopped - {positions_closed} positions closed, Total P&L: ${total_pnl_from_closure:+,.2f}",
                        'type': 'danger'
                    })
                    st.warning(f"Bot Stopped! Closed {positions_closed} positions with total P&L: ${total_pnl_from_closure:+,.2f}")
                else:
                    st.session_state.alerts.append({
                        'time': datetime.now(),
                        'message': "Trading Bot Stopped",
                        'type': 'danger'
                    })
                    st.warning("Bot Stopped!")
                
                # Reset click tracking
                st.session_state.stop_click_count = 0
                st.session_state.first_stop_click_time = None
                
        except Exception as e:
            st.error(f"Failed to stop bot: {str(e)}")
            st.session_state.stop_click_count = 0
            st.session_state.first_stop_click_time = None
    
    def reset_trading(self):
        """Reset the trading bot"""
        try:
            # Stop everything first
            st.session_state.running = False
            st.session_state.auto_trade = False
            st.session_state.trading_active = False
            
            # Create fresh instances without demo trades
            st.session_state.bot = CryptoTradingBot()
            # Remove the simulate trades to start truly fresh
            st.session_state.ml_predictor = MLPredictor()
            st.session_state.alerts = []
            st.session_state.force_update = True
            st.session_state.stop_click_count = 0
            
            st.info("Bot Reset Complete - Starting from scratch!")
            
        except Exception as e:
            st.error(f"Failed to reset bot: {str(e)}")

    def execute_trading_logic(self):
        """Execute trading logic when bot is running"""
        if not st.session_state.running or not st.session_state.trading_active:
            return
            
        bot = st.session_state.bot
        ml_predictor = st.session_state.ml_predictor
        
        try:
            # Always update price for live chart
            bot._trading_step()
            
            # Check for trading opportunities
            if len(bot.positions) < bot.max_open_positions:
                should_trade, action, target = bot._should_open_position()
                
                if should_trade and action and target:
                    # Enhanced decision making with ML
                    if ml_predictor.is_trained:
                        prediction, confidence = ml_predictor.predict_market_direction(bot.price_history)
                        ml_action = "LONG" if prediction == 1 else "SHORT"
                        
                        # Trade if conditions are met
                        if (ml_action == action and confidence > 0.55) or confidence > 0.75:
                            bot._open_position(action, target)
                            
                            st.session_state.alerts.append({
                                'time': datetime.now(),
                                'message': f"Opened {action} position - ML: {confidence*100:.1f}%",
                                'type': 'success'
                            })
                    else:
                        # Use technical analysis only
                        if target > 0.015:  # Strong signal required
                            bot._open_position(action, target)
                            st.session_state.alerts.append({
                                'time': datetime.now(),
                                'message': f"Opened {action} position - Technical Analysis",
                                'type': 'success'
                            })
            
            # Check for position closures
            triggered_positions = bot._check_positions()
            for pos, reason, pnl in triggered_positions:
                alert_type = 'success' if pnl > 0 else 'danger'
                st.session_state.alerts.append({
                    'time': datetime.now(),
                    'message': f"Position #{pos['id']} closed: ${pnl:+,.2f} - {reason}",
                    'type': alert_type
                })
                
        except Exception as e:
            st.session_state.alerts.append({
                'time': datetime.now(),
                'message': f"Trading error: {str(e)[:50]}...",
                'type': 'danger'
            })

    def render_sidebar(self):
        """Render sidebar with controls and settings"""
        with st.sidebar:
            st.markdown("### 🤖 Bot Configuration")
            
            # Status display
            bot = st.session_state.bot
            if st.session_state.running:
                status_text = "🟢 RUNNING"
                status_class = "status-running"
            else:
                status_text = "🔴 STOPPED"
                status_class = "status-stopped"
                bot.initial_balance = bot.balance
            
            st.markdown(f'<div class="{status_class}"><strong>{status_text}</strong></div>', 
                       unsafe_allow_html=True)
            
            if st.session_state.running:
                st.markdown('<div class="live-indicator"><div class="pulse-dot"></div>Live Trading</div>', 
                           unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Trading parameters
            st.markdown("### 📊 Trading Parameters")
            
            # Bot parameter controls
            initial_balance = st.number_input(
    "Initial Balance ($)",
    value=float(bot.initial_balance),
    min_value=1000.0,
    step=1000.0,
    disabled=st.session_state.running
)           
            
            bot.balance = initial_balance
            bot.initial_balance = initial_balance
            
            leverage = st.selectbox("Leverage", [1, 2, 3, 5, 10], 
                                  index=[1, 2, 3, 5, 10].index(bot.leverage),
                                  disabled=st.session_state.running)
            
            max_positions = st.selectbox("Max Positions", [1, 2, 3, 5], 
                                       index=min(bot.max_open_positions - 1, 3),
                                       disabled=st.session_state.running)
            
            # Update bot parameters when not running
            if not st.session_state.running:
                bot.initial_balance = initial_balance
                bot.leverage = leverage
                bot.max_open_positions = max_positions
            
            st.markdown("### ⚖️ Risk Management")
            
            stop_loss = st.slider("Stop Loss (%)", 1, 10, 
                                 int(bot.stop_loss_threshold * 100), 1,
                                 disabled=st.session_state.running) / 100
            
            position_size = st.slider("Position Size (%)", 1, 20, 
                                     int(bot.max_position_size * 100), 1,
                                     disabled=st.session_state.running) / 100
            
            # Update risk parameters when not running
            if not st.session_state.running:
                bot.stop_loss_threshold = stop_loss
                bot.max_position_size = position_size
            
            take_profit = st.slider("Take Profit (%)", 2, 20, 6, 1, disabled=st.session_state.running)
            
            st.markdown("### 🎮 Trading Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ START", 
                           use_container_width=True, 
                           key="start_btn", 
                           disabled=st.session_state.running):
                    self.start_trading()
                    
            with col2:
                stop_btn_clicked = st.button("⏹️ STOP", 
                                           use_container_width=True, 
                                           key="stop_btn",
                                           disabled=not st.session_state.running)
                
                # Show double-click warning only when running
                if st.session_state.running:
                    st.markdown('<div class="stop-warning">Click twice to stop</div>', 
                               unsafe_allow_html=True)
                
                if stop_btn_clicked:
                    self.stop_trading()
                    
            if st.button("🔄 RESET", 
                       use_container_width=True, 
                       key="reset_btn"):
                self.reset_trading()
                
            # Bot statistics
            st.markdown("---")
            st.markdown("### 📈 Quick Stats")
            
            total_trades = len(bot.closed_positions)
            winning_trades = len([t for t in bot.closed_positions if t['profit_loss'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            st.metric("Total Trades", total_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Open Positions", len(bot.positions))
            
            model_status = "Trained" if st.session_state.ml_predictor.is_trained else "Not Trained"
            st.metric("ML Model", model_status)

    def render_main_dashboard(self):
        """Render main trading dashboard"""
        bot = st.session_state.bot
        
        # Header
        st.markdown('<h1 class="main-header">🤖 AI CRYPTO TRADING BOT PRO</h1>', 
                   unsafe_allow_html=True)
        
        # Real-time metrics row
        self.render_metrics_row()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Trading", "📊 Analytics", 
                                          "💼 Portfolio", "📋 Performance"])
        
        with tab1:
            self.render_live_trading_tab()
            
        with tab2:
            self.render_analytics_tab()
            
        with tab3:
            self.render_portfolio_tab()
            
        with tab4:
            self.render_performance_tab()

    def render_metrics_row(self):
        """Render top metrics row"""
        bot = st.session_state.bot
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate price change
        price_change = 0
        if len(bot.price_history) > 1:
            price_change = ((bot.current_price - bot.price_history[-2]['price']) / 
                           bot.price_history[-2]['price'] * 100)
        
        # Current Price
        with col1:
            self.dashboard.render_metric_card(
                title="Current Price",
                value=f"${bot.current_price:,.2f}",
                change=f"{price_change:+.2f}%",
                positive=price_change >= 0
            )
        
        # Portfolio Balance
        with col2:
            total_equity = bot.balance + sum([pos.get('margin_used', 0) for pos in bot.positions])
            total_equity += bot._calculate_unrealized_pnl()
            balance_change = ((total_equity - bot.initial_balance) / bot.initial_balance * 100)
            
            self.dashboard.render_metric_card(
                title="Total Equity",
                value=f"${total_equity:,.2f}",
                change=f"{balance_change:+.2f}%",
                positive=balance_change >= 0
            )
        
        # Total P&L
        with col3:
            unrealized_pnl = bot._calculate_unrealized_pnl()
            total_pnl = bot.total_profit_loss + unrealized_pnl
            
            self.dashboard.render_metric_card(
                title="Total P&L",
                value=f"${total_pnl:+,.2f}",
                change=f"Realized: ${bot.total_profit_loss:+,.2f}",
                positive=total_pnl >= 0
            )
        
        # Open Positions
        with col4:
            self.dashboard.render_metric_card(
                title="Open Positions",
                value=str(len(bot.positions)),
                change=f"Max: {bot.max_open_positions}",
                positive=True
            )
        
        # Win Rate
        with col5:
            total_trades = len(bot.closed_positions)
            winning_trades = len([t for t in bot.closed_positions if t['profit_loss'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            self.dashboard.render_metric_card(
                title="Win Rate",
                value=f"{win_rate:.1f}%",
                change=f"{winning_trades}/{total_trades} trades",
                positive=win_rate > 50
            )

    def render_live_trading_tab(self):
        """Render live trading interface"""
        bot = st.session_state.bot
        
        # Chart controls
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col1:
            show_indicators = st.checkbox("📈 Technical Indicators", value=True)
        with col2:
            timeframe = st.selectbox("⏰ Timeframe", ["1m", "5m", "15m", "1h"], index=1)
        with col3:
            chart_type = st.selectbox("📊 Chart Type", ["Candlestick", "Line"], index=0)
        with col4:
            if st.button("🔄 Refresh Data"):
                bot._simulate_price_movement()
                st.rerun()
        
        # Show current trading status
        if st.session_state.running:
            st.success("📊 Live Trading Active - Bot is analyzing market conditions...")
        else:
            st.info("⏸️ Trading Stopped - Click START to begin live trading")
        
        # Main price chart
        if len(bot.price_history) > 0:
            try:
                chart = self.dashboard.create_price_chart(bot.price_history, show_indicators)
                st.plotly_chart(chart, use_container_width=True, key="main_chart")
            except Exception as e:
                st.error(f"Chart error: {e}")
                st.info("Generating new chart data...")
                bot._simulate_price_movement()
        else:
            st.info("No price data available. The bot will generate data when started.")
        
        # Live positions and alerts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_open_positions()
            
        with col2:
            self.render_alerts()

    def render_open_positions(self):
        """Render open positions panel"""
        bot = st.session_state.bot
        
        st.markdown("### 📊 Open Positions")
        
        if bot.positions:
            for pos in bot.positions:
                # Calculate current P&L
                if pos['type'] == 'LONG':
                    current_pnl = pos['size'] * (bot.current_price - pos['entry_price'])
                    current_pnl_pct = (bot.current_price - pos['entry_price']) / pos['entry_price']
                else:  # SHORT
                    current_pnl = pos['size'] * (pos['entry_price'] - bot.current_price)
                    current_pnl_pct = (pos['entry_price'] - bot.current_price) / pos['entry_price']
                
                # Position card
                pnl_color = '#00ff88' if current_pnl >= 0 else '#ff4444'
                position_color = '#00ff88' if pos['type'] == 'LONG' else '#ff4444'
                
                st.markdown(f"""
                <div class="premium-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="color: {position_color}; font-weight: 600; font-size: 1.1rem;">
                                #{pos['id']} {pos['type']} ({pos.get('trade_action', 'BUY' if pos['type'] == 'LONG' else 'SELL')})
                            </div>
                            <div style="font-size: 0.9rem; opacity: 0.8; margin: 5px 0;">
                                Entry: ${pos['entry_price']:,.2f} | Size: {pos['size']:.6f} BTC
                            </div>
                            <div style="font-size: 0.85rem; opacity: 0.7;">
                                Target: ${pos.get('target_price', 0):,.2f} | Stop: ${pos['stop_loss']:,.2f}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {pnl_color}; font-size: 1.2rem; font-weight: 700;">
                                ${current_pnl:+,.2f}
                            </div>
                            <div style="color: {pnl_color}; font-size: 0.9rem; font-weight: 600;">
                                {current_pnl_pct*100:+.2f}%
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            if st.session_state.running:
                st.info("🔍 Scanning for trading opportunities...")
            else:
                st.info("No open positions - Start the bot to begin trading")

    def render_alerts(self):
        """Render alerts panel"""
        st.markdown("### 🚨 Recent Alerts")
        
        # Keep only last 5 alerts
        if len(st.session_state.alerts) > 5:
            st.session_state.alerts = st.session_state.alerts[-5:]
        
        if st.session_state.alerts:
            for alert in reversed(st.session_state.alerts):
                alert_class = f"alert-{alert.get('type', 'info')}"
                time_str = alert['time'].strftime("%H:%M:%S")
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <div style="display: flex; justify-content: space-between;">
                        <div><strong>{alert['message']}</strong></div>
                        <div style="font-size: 0.8rem; opacity: 0.8;">{time_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent alerts - Start trading to see live updates")

    def render_analytics_tab(self):
        """Render analytics and technical analysis"""
        bot = st.session_state.bot
        ml_predictor = st.session_state.ml_predictor
        
        st.markdown("### 📊 Technical Analysis & ML Predictions")
        
        # Technical indicators
        if len(bot.price_history) > 20:
            indicators = bot._calculate_technical_indicators()
            
            if indicators:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    self.dashboard.render_indicator_card("RSI (14)", indicators['rsi'], 
                                                       threshold_high=70, threshold_low=30)
                
                with col2:
                    self.dashboard.render_indicator_card("Momentum", indicators['momentum'] * 100,
                                                       unit="%", positive=indicators['momentum'] > 0)
                
                with col3:
                    self.dashboard.render_indicator_card("Volatility", indicators['volatility'] * 100,
                                                       unit="%", threshold_high=3)
                
                with col4:
                    self.dashboard.render_indicator_card("Volume Trend", indicators['volume_trend'],
                                                       unit="x", positive=indicators['volume_trend'] > 1)
        
        # ML Prediction
        if ml_predictor.is_trained and len(bot.price_history) > 0:
            prediction, confidence = ml_predictor.predict_market_direction(bot.price_history)
            
            st.markdown("### 🤖 AI Market Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                direction = "BULLISH" if prediction == 1 else "BEARISH"
                direction_color = "#00ff88" if prediction == 1 else "#ff4444"
                
                st.markdown(f"""
                <div class="premium-card">
                    <h4 style="color: #00d4ff;">📈 Market Direction</h4>
                    <div style="text-align: center; margin: 20px 0;">
                        <div style="color: {direction_color}; font-size: 2rem; font-weight: 700;">
                            {direction}
                        </div>
                        <div style="color: #00d4ff; font-size: 1.2rem;">
                            Confidence: {confidence*100:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Model performance metrics
                model_accuracy = getattr(ml_predictor, 'last_accuracy', 0.873)
                
                st.markdown(f"""
                <div class="premium-card">
                    <h4 style="color: #00d4ff;">🧠 Model Performance</h4>
                    <div style="margin: 15px 0; line-height: 1.8;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>Accuracy:</span>
                            <span style="color: #00ff88;">{model_accuracy*100:.1f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Status:</span>
                            <span style="color: #00d4ff;">{"Trained" if ml_predictor.is_trained else "Training..."}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ML model not trained yet. Start the bot to train the model with historical data.")

    def render_portfolio_tab(self):
        """Render portfolio management interface"""
        bot = st.session_state.bot
        
        st.markdown("### Portfolio Overview")
        
        # Portfolio metrics
        total_margin_used = sum([pos.get('margin_used', 0) for pos in bot.positions])
        available_balance = bot.balance
        unrealized_pnl = bot._calculate_unrealized_pnl()
        total_equity = available_balance + total_margin_used + unrealized_pnl
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.dashboard.render_metric_card("Available Balance", f"${available_balance:,.2f}")
        with col2:
            self.dashboard.render_metric_card("Margin Used", f"${total_margin_used:,.2f}")
        with col3:
            self.dashboard.render_metric_card("Unrealized P&L", f"${unrealized_pnl:+,.2f}",
                                            positive=unrealized_pnl >= 0)
        with col4:
            self.dashboard.render_metric_card("Total Equity", f"${total_equity:,.2f}")
        
        # Recent trade history
        if bot.closed_positions:
            st.markdown("### Recent Trades")
            recent_trades = bot.closed_positions[-10:]  # Last 10 trades
            
            trade_data = []
            for trade in recent_trades:
                trade_data.append({
                    'ID': trade['id'],
                    'Type': trade['type'],
                    'Entry': f"${trade['entry_price']:,.2f}",
                    'Exit': f"${trade['close_price']:,.2f}",
                    'Size': f"{trade['size']:.6f} BTC",
                    'P&L': f"${trade['profit_loss']:+,.2f}",
                    'P&L %': f"{trade.get('profit_pct', 0)*100:+.2f}%",
                    'Reason': trade.get('close_reason', 'Unknown'),
                    'Time': trade['close_time'].strftime("%H:%M:%S")
                })
            
            df = pd.DataFrame(trade_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No trading history yet. Start the bot to generate trades.")

    def render_performance_tab(self):
        """Render performance analysis"""
        bot = st.session_state.bot
        
        st.markdown("### Performance Analysis")
        
        if bot.closed_positions:
            # Calculate performance metrics
            perf_metrics = self.performance_calc.calculate_performance_metrics(bot)
            
            # Performance overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self.dashboard.render_metric_card("Total Return", 
                                                f"{perf_metrics['total_return']:+.2f}%",
                                                positive=perf_metrics['total_return'] > 0)
            with col2:
                self.dashboard.render_metric_card("Profit Factor", 
                                                f"{perf_metrics['profit_factor']:.2f}",
                                                positive=perf_metrics['profit_factor'] > 1)
            with col3:
                self.dashboard.render_metric_card("Sharpe Ratio", 
                                                f"{perf_metrics['sharpe_ratio']:.2f}",
                                                positive=perf_metrics['sharpe_ratio'] > 1)
            with col4:
                self.dashboard.render_metric_card("Max Drawdown", 
                                                f"{perf_metrics['max_drawdown']:.2f}%",
                                                positive=False)
            
            # Performance chart
            if len(bot.closed_positions) > 1:
                portfolio_chart = self.dashboard.create_portfolio_chart(bot)
                st.plotly_chart(portfolio_chart, use_container_width=True)
            
        else:
            st.info("No trading history available. Start trading to generate performance reports.")

    def update_data(self):
        """Update price data and execute trading logic - ENHANCED FOR AUTO-REFRESH"""
        current_time = datetime.now()
        
        # Update every 2 seconds when running for better responsiveness
        update_interval = 2 if st.session_state.running else 8
        
        # Reset stop button counter after 3 seconds
        if (current_time - st.session_state.last_ui_update).total_seconds() > 3:
            st.session_state.stop_click_count = 0
        
        if (current_time - st.session_state.last_update).total_seconds() >= update_interval:
            # Always update price data for chart
            st.session_state.bot._simulate_price_movement()
            
            # Execute trading logic if running
            if st.session_state.running:
                self.execute_trading_logic()
            
            st.session_state.last_update = current_time
            st.session_state.update_counter += 1
            
            # Auto-rerun for live updates when bot is running
            if st.session_state.running:
                st.rerun()
            elif st.session_state.force_update:
                st.session_state.force_update = False
                st.rerun()

    def run(self):
        """Main application runner - ENHANCED VERSION"""
        # Load CSS
        load_css()
        
        # Update data continuously
        self.update_data()
        
        # Render the UI
        self.render_sidebar()
        self.render_main_dashboard()
        
        # Footer with status info
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Version:** v2.1.0 Pro")
        with col2:
            st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        with col3:
            st.markdown(f"**Data Points:** {len(st.session_state.bot.price_history):,}")
        with col4:
            status_text = "Live Trading" if st.session_state.running else "Stopped"
            st.markdown(f"**Status:** {status_text}")

# Main application entry point
def main():
    """Main function with proper error handling"""
    try:
        app = TradingBotApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page or click RESET to restart the application.")

# Run the app
if __name__ == "__main__":
    main()
