import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback

# Custom trading environment
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.max_steps = len(data) - 1
        
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: price features + balance + position
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self._get_observation(), {}
    
    def _get_observation(self):
        current_price = self.data.iloc[self.current_step]['close']
        return np.array([
            current_price,
            self.data.iloc[self.current_step]['volume'],
            self.balance,
            self.position,
            self.current_step / self.max_steps
        ], dtype=np.float32)
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            shares = self.balance / current_price
            self.balance = 0
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            self.balance = current_price * (self.balance if self.balance > 0 else 1)
            reward = self.balance - self.initial_balance
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, False, {}

def train_rl_model():
    # Remove wandb initialization for now
    # wandb.init(
    #     project="rl-trading",
    #     config={
    #         "algorithm": "PPO",
    #         "env": "TradingEnv",
    #     }
    # )
    
    # Generate sample data (replace with your actual data)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
    volumes = np.random.randint(1000, 10000, len(dates))
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes
    })
    
    # Create environment
    env = make_vec_env(lambda: TradingEnv(data), n_envs=1)
    
    # Create model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        # Remove wandb callback for now
        # callback=WandbCallback()
    )
    
    print("Starting training...")
    # Train the model
    model.learn(total_timesteps=10000)
    
    # Save the model
    model.save("ppo_trading_model")
    print("Model saved as 'ppo_trading_model'")
    
    # Test the trained model
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            break
    
    print("Training completed!")

if __name__ == "__main__":
    train_rl_model()