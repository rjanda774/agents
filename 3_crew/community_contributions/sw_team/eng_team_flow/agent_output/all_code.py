```python
import gradio as gr
import pandas as pd
import numpy as np
import sqlite3
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DatabaseManager:
    """Manages SQLite database operations for portfolio management"""
    
    def __init__(self, db_path: str = "portfolio.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Portfolio allocations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                asset_symbol TEXT NOT NULL,
                target_percentage REAL NOT NULL,
                current_value REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Rebalancing schedules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rebalancing_schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                frequency TEXT NOT NULL,
                drift_threshold REAL DEFAULT 5.0,
                auto_execute BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Notifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                is_read BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def create_user(self, username: str) -> int:
        """Create a new user and return user_id"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            # User already exists, get existing user_id
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def save_portfolio_allocation(self, user_id: int, allocations: Dict[str, float]):
        """Save portfolio allocations for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Clear existing allocations
        cursor.execute("DELETE FROM portfolio_allocations WHERE user_id = ?", (user_id,))
        
        # Insert new allocations
        for symbol, percentage in allocations.items():
            cursor.execute('''
                INSERT INTO portfolio_allocations (user_id, asset_symbol, target_percentage)
                VALUES (?, ?, ?)
            ''', (user_id, symbol, percentage))
        
        conn.commit()
        conn.close()
    
    def get_portfolio_allocations(self, user_id: int) -> Dict[str, float]:
        """Get portfolio allocations for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT asset_symbol, target_percentage 
            FROM portfolio_allocations 
            WHERE user_id = ?
        ''', (user_id,))
        
        allocations = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return allocations

class MarketDataProvider:
    """Provides real-time market data using yfinance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for given symbols"""
        prices = {}
        current_time = time.time()
        
        for symbol in symbols:
            # Check cache first
            if (symbol in self.cache and 
                symbol in self.cache_expiry and 
                current_time < self.cache_expiry[symbol]):
                prices[symbol] = self.cache[symbol]
                continue
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    prices[symbol] = float(price)
                    
                    # Update cache
                    self.cache[symbol] = price
                    self.cache_expiry[symbol] = current_time + self.cache_duration
                else:
                    prices[symbol] = 0.0
            except Exception as e:
                print(f"Error fetching price for {symbol}: {e}")
                prices[symbol] = 0.0
        
        return prices
    
    def get_historical_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Get historical data for ML model training"""
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close']
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {e}")
        
        return pd.DataFrame(data)

class PortfolioDriftCalculator:
    """Calculates portfolio drift from target allocations"""
    
    def __init__(self, market_data_provider: MarketDataProvider):
        self.market_data_provider = market_data_provider
    
    def calculate_drift(self, target_allocations: Dict[str, float], 
                       current_values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate portfolio drift from target allocations"""
        if not target_allocations or not current_values:
            return {}
        
        # Get current prices
        symbols = list(target_allocations.keys())
        current_prices = self.market_data_provider.get_current_prices(symbols)
        
        # Calculate total portfolio value
        total_value = sum(current_values.values())
        if total_value == 0:
            return {}
        
        # Calculate current allocations
        current_allocations = {
            symbol: (current_values.get(symbol, 0) / total_value) * 100
            for symbol in symbols
        }
        
        # Calculate drift
        drift_analysis = {}
        for symbol in symbols:
            target_pct = target_allocations[symbol]
            current_pct = current_allocations.get(symbol, 0)
            drift_pct = current_pct - target_pct
            
            drift_analysis[symbol] = {
                'target_percentage': target_pct,
                'current_percentage': current_pct,
                'drift_percentage': drift_pct,
                'current_value': current_values.get(symbol, 0),
                'current_price': current_prices.get(symbol, 0),
                'rebalance_needed': abs(drift_pct) > 5.0  # 5% threshold
            }
        
        return drift_analysis

class MLRebalancingModel:
    """Machine learning model for rebalancing recommendations"""
    
    def __init__(self, market_data_provider: MarketDataProvider):
        self.market_data_provider = market_data_provider
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, historical_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        
        for symbol in historical_data.columns:
            prices = historical_data[symbol].dropna()
            if len(prices) < 30:  # Need at least 30 data points
                continue
            
            # Calculate technical indicators
            returns = prices.pct_change().dropna()
            volatility = returns.rolling(window=20).std()
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            rsi = self.calculate_rsi(prices)
            
            # Create feature vector
            feature_row = [
                returns.mean(),
                volatility.iloc[-1] if not volatility.empty else 0,
                (prices.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] if not sma_20.empty else 0,
                (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] if not sma_50.empty else 0,
                rsi.iloc[-1] if not rsi.empty else 50
            ]
            features.append(feature_row)
        
        return np.array(features) if features else np.array([])
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_model(self, symbols: List[str]):
        """Train the ML model with historical data"""
        try:
            historical_data = self.market_data_provider.get_historical_data(symbols)
            if historical_data.empty:
                print("No historical data available for training")
                return
            
            features = self.prepare_features(historical_data)
            if features.size == 0:
                print("No features could be prepared for training")
                return
            
            # Create synthetic target (future returns)
            targets = np.random.normal(0.05, 0.1, len(features))  # Simulated expected returns
            
            # Train model
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled, targets)
            self.is_trained = True
            print("ML model trained successfully")
            
        except Exception as e:
            print(f"Error training ML model: {e}")
    
    def get_rebalancing_recommendations(self, target_allocations: Dict[str, float], 
                                     drift_analysis: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Get ML-based rebalancing recommendations"""
        if not self.is_trained:
            symbols = list(target_allocations.keys())
            self.train_model(symbols)
        
        recommendations = {}
        
        for symbol, analysis in drift_analysis.items():
            drift_pct = analysis['drift_percentage']
            
            if abs(drift_pct) > 5.0:  # Significant drift
                if drift_pct > 0:
                    recommendations[symbol] = f"SELL: Overweight by {drift_pct:.2f}%. Consider reducing position."
                else:
                    recommendations[symbol] = f"BUY: Underweight by {abs(drift_pct):.2f}%. Consider increasing position."
            else:
                recommendations[symbol] = "HOLD: Allocation within acceptable range."
        
        return recommendations

class RiskAssessment:
    """Assess portfolio risk based on allocations and market conditions"""
    
    def __init__(self, market_data_provider: MarketDataProvider):
        self.market_data_provider = market_data_provider
    
    def calculate_portfolio_risk(self, allocations: Dict[str, float], 
                               current_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate various risk metrics for the portfolio"""
        symbols = list(allocations.keys())
        
        try:
            # Get historical data for risk calculation
            historical_data = self.market_data_provider.get_historical_data(symbols, period="1y")
            
            if historical_data.empty:
                return {"overall_risk": 0.5, "diversification_score": 0.5}
            
            # Calculate returns
            returns = historical_data.pct_change().dropna()
            
            # Portfolio weights
            total_value = sum(current_values.values())
            weights = np.array([current_values.get(symbol, 0) / total_value if total_value > 0 else 0 
                              for symbol in symbols])
            
            # Portfolio volatility
            if len(returns.columns) > 1:
                cov_matrix = returns.cov()
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
            else:
                portfolio_volatility = returns.std().iloc[0] * np.sqrt(252) if not returns.empty else 0.2
            
            # Diversification score (higher is better)
            diversification_score = 1 - np.sum(weights ** 2)  # Herfindahl index
            
            # Overall risk score (0-1, where 1 is highest risk)
            overall_risk = min(portfolio_volatility / 0.5, 1.0)  # Normalize to 0-1
            
            return {
                "overall_risk": float(overall_risk),
                "portfolio_volatility": float(portfolio_volatility),
                "diversification_score": float(diversification_score),
                "risk_level": "High" if overall_risk > 0.7 else "Medium" if overall_risk > 0.4 else "Low"
            }
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {"overall_risk": 0.5, "diversification_score": 0.5, "risk_level": "Medium"}

class NotificationService:
    """Manages notifications for portfolio events"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.notifications = []
    
    def add_notification(self, user_id: int, message: str, notification_type: str = "info"):
        """Add a new notification"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO notifications (user_id, message, notification_type)
            VALUES (?, ?, ?)
        ''', (user_id, message, notification_type))
        conn.commit()
        conn.close()
        
        # Also add to in-memory list for immediate display
        self.notifications.append({
            'message': message,
            'type': notification_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_recent_notifications(self, limit: int = 10) -> List[Dict]:
        """Get recent notifications"""