import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_accuracy = 0.0
        self.feature_names = [
            'sma_5', 'sma_10', 'sma_20', 'rsi', 'momentum', 
            'volatility', 'volume_trend', 'price_to_sma_5', 
            'price_to_sma_10', 'price_to_sma_20'
        ]
    
    def _calculate_technical_indicators(self, price_history, index, lookback=20):
        """Calculate technical indicators for a specific point in history"""
        if index < lookback:
            return None
            
        end_idx = index + 1
        start_idx = max(0, end_idx - lookback)
        
        segment = price_history[start_idx:end_idx]
        prices = [p['price'] for p in segment]
        volumes = [p['volume'] for p in segment]
        
        if len(prices) < 10:
            return None
            
        # Moving averages
        sma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else np.mean(prices)
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else np.mean(prices)
        sma_20 = np.mean(prices) if len(prices) < 20 else np.mean(prices[-20:])
        
        # RSI calculation
        if len(prices) > 1:
            price_changes = np.diff(prices)
            gains = price_changes[price_changes > 0]
            losses = -price_changes[price_changes < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            rsi = 50
        
        # Momentum
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        # Volatility
        volatility = np.std(prices) / np.mean(prices) if len(prices) > 1 else 0
        
        # Volume trend
        volume_trend = (np.mean(volumes[-5:]) / np.mean(volumes[-10:])) if len(volumes) >= 10 else 1
        
        current_price = prices[-1]
        
        return {
            'sma_5': sma_5,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'rsi': rsi,
            'momentum': momentum,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'price_to_sma_5': current_price / sma_5,
            'price_to_sma_10': current_price / sma_10,
            'price_to_sma_20': current_price / sma_20
        }
    
    def train_model(self, price_history):
        """Train the ML model with historical data"""
        if len(price_history) < 50:
            print("Not enough data to train model")
            return False
            
        features = []
        labels = []
        
        # Create training data
        for i in range(30, len(price_history) - 10):
            # Calculate indicators at time i
            indicators = self._calculate_technical_indicators(price_history, i)
            if indicators is None:
                continue
                
            # Create feature vector
            feature_vector = [indicators[name] for name in self.feature_names]
            
            # Check for valid features
            if any(np.isnan(feature_vector)) or any(np.isinf(feature_vector)):
                continue
            
            # Label: future price direction (1 for up, 0 for down)
            current_price = price_history[i]['price']
            future_price = price_history[i + 10]['price']
            label = 1 if future_price > current_price else 0
            
            features.append(feature_vector)
            labels.append(label)
        
        if len(features) < 20:
            print("Not enough valid features for training")
            return False
            
        # Convert to arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test_scaled)
        self.last_accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        print(f"Model trained with {len(features)} samples, accuracy: {self.last_accuracy:.3f}")
        
        return True
    
    def predict_market_direction(self, price_history):
        """Predict market direction using trained model"""
        if not self.is_trained:
            # Return random prediction with moderate confidence
            return random.choice([0, 1]), random.uniform(0.55, 0.75)
            
        # Calculate current indicators
        indicators = self._calculate_technical_indicators(price_history, len(price_history) - 1)
        if indicators is None:
            return random.choice([0, 1]), random.uniform(0.55, 0.75)
            
        # Create feature vector
        feature_vector = [indicators[name] for name in self.feature_names]
        
        # Check for valid features
        if any(np.isnan(feature_vector)) or any(np.isinf(feature_vector)):
            return random.choice([0, 1]), random.uniform(0.55, 0.75)
            
        # Make prediction
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        confidence = self.model.predict_proba(X_scaled)[0].max()
        
        # Add technical analysis enhancement
        rsi = indicators['rsi']
        momentum = indicators['momentum']
        price_trend = indicators['price_to_sma_5']
        
        # Enhance prediction with technical analysis
        if rsi > 70 or momentum < -0.02 or price_trend < 0.98:
            # Bearish conditions - favor SHORT
            prediction = 0
            confidence = max(confidence, 0.65)
        elif rsi < 30 or momentum > 0.02 or price_trend > 1.02:
            # Bullish conditions - favor LONG
            prediction = 1
            confidence = max(confidence, 0.65)
        
        # Ensure minimum confidence for trading
        confidence = max(confidence, 0.52)
        
        return prediction, confidence
    
    def get_model_performance(self):
        """Get model performance metrics"""
        return {
            'accuracy': self.last_accuracy,
            'is_trained': self.is_trained,
            'features': len(self.feature_names)
        }
    
    def retrain_if_needed(self, price_history):
        """Retrain model if enough new data is available"""
        if len(price_history) > 100 and len(price_history) % 50 == 0:
            print("Retraining model with new data...")
            return self.train_model(price_history)
        return False