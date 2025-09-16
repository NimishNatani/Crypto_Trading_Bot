import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DashboardComponents:
    def __init__(self):
        self.colors = {
            'primary': '#00d4ff',
            'success': '#00ff88', 
            'danger': '#ff4444',
            'warning': '#ffb347',
            'background': 'rgba(26, 26, 46, 0.6)',
            'card_bg': 'rgba(26, 26, 46, 0.8)'
        }
    
    def render_metric_card(self, title, value, change=None, positive=None):
        """Render a metric card with optional change indicator"""
        change_color = ""
        change_text = ""
        
        if change is not None:
            if positive is not None:
                change_color = self.colors['success'] if positive else self.colors['danger']
            change_text = f'<div style="color: {change_color}; font-size: 0.8rem; margin-top: 5px;">{change}</div>'
        
        st.markdown(f"""
        <div class="premium-card metric-card">
            <div style="font-size: 0.8rem; opacity: 0.8; margin-bottom: 8px;">{title}</div>
            <div class="metric-value" style="margin-bottom: 5px;">{value}</div>
            {change_text}
        </div>
        """, unsafe_allow_html=True)
    
    def render_indicator_card(self, title, value, unit="", threshold_high=None, threshold_low=None, positive=None):
        """Render technical indicator card with threshold coloring"""
        # Determine color based on thresholds or positive flag
        if threshold_high and value > threshold_high:
            color = self.colors['danger']
            status = "High"
        elif threshold_low and value < threshold_low:
            color = self.colors['success']
            status = "Low"
        elif positive is not None:
            color = self.colors['success'] if positive else self.colors['danger']
            status = "Positive" if positive else "Negative"
        else:
            color = self.colors['primary']
            status = "Neutral"
        
        st.markdown(f"""
        <div class="premium-card">
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 10px;">{title}</div>
                <div style="color: {color}; font-size: 2rem; font-weight: 700; margin-bottom: 5px;">
                    {value:.2f}{unit}
                </div>
                <div style="font-size: 0.8rem; opacity: 0.7;">{status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_price_chart(self, price_history, show_indicators=True):
        """Create enhanced price chart with technical indicators"""
        if len(price_history) < 10:
            return go.Figure()
        
        # Get recent data
        recent_data = price_history[-60:] if len(price_history) > 60 else price_history
        df = pd.DataFrame(recent_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('BTC/USDT Live Price', 'Volume', 'RSI'),
            row_heights=[0.6, 0.25, 0.15]
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'], 
                low=df['low'],
                close=df['close'],
                name="OHLC",
                increasing_line_color=self.colors['success'],
                decreasing_line_color=self.colors['danger'],
                increasing_fillcolor='rgba(0, 255, 136, 0.6)',
                decreasing_fillcolor='rgba(255, 68, 68, 0.6)'
            ),
            row=1, col=1
        )
        
        # Technical indicators
        if show_indicators and len(df) >= 20:
            # Moving averages
            df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
            df['sma_10'] = df['close'].rolling(10, min_periods=1).mean()
            df['sma_5'] = df['close'].rolling(5, min_periods=1).mean()
            
            # Bollinger Bands
            df['bb_upper'] = df['sma_20'] + (df['close'].rolling(20).std() * 2)
            df['bb_lower'] = df['sma_20'] - (df['close'].rolling(20).std() * 2)
            
            # Add Bollinger Bands
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['bb_upper'],
                          line=dict(color='rgba(0, 212, 255, 0.3)', width=1),
                          showlegend=False), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['bb_lower'],
                          fill='tonexty', fillcolor='rgba(0, 212, 255, 0.1)',
                          line=dict(color='rgba(0, 212, 255, 0.3)', width=1),
                          name='Bollinger Bands'), row=1, col=1
            )
            
            # Moving averages
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['sma_20'],
                          line=dict(color=self.colors['primary'], width=2),
                          name='SMA 20'), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['sma_10'], 
                          line=dict(color=self.colors['warning'], width=2),
                          name='SMA 10'), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['sma_5'],
                          line=dict(color='#ff006e', width=1.5),
                          name='SMA 5'), row=1, col=1
            )
        
        # Volume chart
        colors = [self.colors['success'] if close >= open else self.colors['danger'] 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'],
                   marker_color=colors, opacity=0.7, name='Volume'),
            row=2, col=1
        )
        
        # RSI indicator
        if len(df) >= 14:
            price_changes = df['close'].diff()
            gains = price_changes.where(price_changes > 0, 0)
            losses = -price_changes.where(price_changes < 0, 0)
            avg_gains = gains.rolling(14, min_periods=1).mean()
            avg_losses = losses.rolling(14, min_periods=1).mean()
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=rsi,
                          line=dict(color=self.colors['primary'], width=2),
                          name='RSI'), row=3, col=1
            )
            
            # RSI threshold lines
            fig.add_hline(y=70, line_dash="dash", 
                         line_color="rgba(255, 68, 68, 0.5)", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", 
                         line_color="rgba(0, 255, 136, 0.5)", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10, 10, 15, 0.8)',
            font=dict(color='#ffffff'),
            title=dict(
                text="Live Trading Chart",
                font=dict(size=20, color=self.colors['primary'])
            ),
            xaxis_rangeslider_visible=False,
            legend=dict(
                bgcolor=self.colors['card_bg'],
                bordercolor='rgba(0, 212, 255, 0.3)',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=60, b=0)
        )
        
        # Grid styling
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
        
        return fig
    
    def create_portfolio_chart(self, bot):
        """Create portfolio performance chart"""
        if not bot.closed_positions:
            return go.Figure()
        
        df = pd.DataFrame(bot.closed_positions)
        df = df.sort_values('close_time')
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        df['running_balance'] = bot.initial_balance + df['cumulative_pnl']
        
        fig = go.Figure()
        
        # Portfolio equity curve
        fig.add_trace(
            go.Scatter(
                x=df['close_time'],
                y=df['running_balance'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color=self.colors['primary'], width=3),
                fill='tonexty',
                fillcolor='rgba(0, 212, 255, 0.1)'
            )
        )
        
        # Benchmark line
        fig.add_hline(y=bot.initial_balance, line_dash="dash",
                     line_color="rgba(255, 255, 255, 0.3)",
                     annotation_text="Initial Balance")
        
        # Winning and losing trades
        wins = df[df['profit_loss'] > 0]
        losses = df[df['profit_loss'] < 0]
        
        if not wins.empty:
            fig.add_trace(
                go.Scatter(
                    x=wins['close_time'],
                    y=wins['running_balance'],
                    mode='markers',
                    name='Wins',
                    marker=dict(color=self.colors['success'], size=8, symbol='triangle-up')
                )
            )
        
        if not losses.empty:
            fig.add_trace(
                go.Scatter(
                    x=losses['close_time'],
                    y=losses['running_balance'],
                    mode='markers', 
                    name='Losses',
                    marker=dict(color=self.colors['danger'], size=8, symbol='triangle-down')
                )
            )
        
        fig.update_layout(
            title="Portfolio Performance",
            height=450,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10, 10, 15, 0.8)',
            font=dict(color='#ffffff'),
            xaxis_title="Time",
            yaxis_title="Portfolio Value ($)",
            legend=dict(
                bgcolor=self.colors['card_bg'],
                bordercolor='rgba(0, 212, 255, 0.3)',
                borderwidth=1
            )
        )
        
        return fig
    
    def render_position_card(self, position, current_price, pnl, pnl_pct):
        """Render individual position card"""
        pnl_color = self.colors['success'] if pnl >= 0 else self.colors['danger']
        position_color = self.colors['success'] if position['type'] == 'LONG' else self.colors['danger']
        
        st.markdown(f"""
        <div class="premium-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: {position_color}; font-weight: 600; font-size: 1.1rem;">
                        #{position['id']} {position['type']} 
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.8; margin: 5px 0;">
                        Entry: ${position['entry_price']:,.2f} | Size: {position['size']:.6f} BTC
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.7;">
                        Current: ${current_price:,.2f} | Target: ${position.get('target_price', 0):,.2f}
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.6;">
                        Stop: ${position['stop_loss']:,.2f} | Leverage: {position.get('leverage', 1)}x
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: {pnl_color}; font-size: 1.2rem; font-weight: 700;">
                        ${pnl:+,.2f}
                    </div>
                    <div style="color: {pnl_color}; font-size: 0.9rem;">
                        {pnl_pct*100:+.2f}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_alert_box(self, message, alert_type="info", timestamp=None):
        """Render alert notification box"""
        if timestamp is None:
            timestamp = datetime.now()
        
        time_str = timestamp.strftime("%H:%M:%S")
        alert_class = f"alert-{alert_type}"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <div style="display: flex; justify-content: space-between;">
                <div><strong>{message}</strong></div>
                <div style="font-size: 0.8rem; opacity: 0.8;">{time_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)