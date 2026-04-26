import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """Custom Environment for Stock Trading using Gymnasium"""
    
    def __init__(self, df=None, window_size=30, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        # Load sample data if none provided
        if df is None:
            # Generate sample stock data
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=1000)
            prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
            self.df = pd.DataFrame({'price': prices}, index=dates)
        else:
            self.df = df
            
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price history + portfolio info
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size + 3,),  # price window + balance + shares + current_price
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.total_value = self.initial_balance
        self.trades = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Get price window
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        price_window = self.df['price'].iloc[start_idx:end_idx].values
        
        # Normalize prices
        price_window = (price_window - price_window.mean()) / (price_window.std() + 1e-8)
        
        # Current portfolio state
        current_price = self.df['price'].iloc[self.current_step]
        portfolio_info = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares,
            current_price / 100  # Normalized price
        ])
        
        return np.concatenate([price_window, portfolio_info]).astype(np.float32)
    
    def step(self, action):
        current_price = self.df['price'].iloc[self.current_step]
        
        # Execute action
        reward = 0
        if action == 1:  # Buy
            if self.balance >= current_price:
                shares_to_buy = self.balance // current_price
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.trades.append(('buy', shares_to_buy, current_price))
        
        elif action == 2:  # Sell
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.trades.append(('sell', self.shares, current_price))
                self.shares = 0
        
        # Calculate reward (portfolio value change)
        new_total_value = self.balance + self.shares * current_price
        reward = new_total_value - self.total_value
        self.total_value = new_total_value
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Additional info
        info = {
            'balance': self.balance,
            'shares': self.shares,
            'total_value': self.total_value,
            'trades': len(self.trades)
        }
        
        return self._get_observation(), reward, done, truncated, info