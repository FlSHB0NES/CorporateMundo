import gym
from gym import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self, stock_data, initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        
        # Stock data (prices, technical indicators, etc.)
        self.stock_data = stock_data
        
        # Portfolio initial conditions
        self.initial_balance = initial_balance
        self.current_balance = self.initial_balance
        self.current_step = 0
        
        # Number of assets
        num_assets = len(stock_data.columns)
        
        # Define action space
        # First num_assets entries are stock allocations (0 to 1 for each stock)
        # Next 2 entries are RSI thresholds (e.g., [RSI_low, RSI_high] in a range)
        self.action_space = spaces.Box(low=0, high=1, shape=(num_assets + 2,), dtype=np.float32)

        # Observation space: stock data + current portfolio balance
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_assets * 2,), dtype=np.float32)

    def reset(self):
        # Reset portfolio balance and starting point in the stock data
        self.current_balance = self.initial_balance
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        # Get the stock data for the current step and combine it with the current portfolio balance
        obs = np.concatenate((self.stock_data.iloc[self.current_step].values, [self.current_balance]))
        return obs

    def step(self, action):
        # Split the action into stock allocations and RSI thresholds
        stock_allocations = action[:len(self.stock_data.columns)]
        rsi_low_threshold = action[-2] * 100  # Scale between 0 and 100
        rsi_high_threshold = action[-1] * 100  # Scale between 0 and 100
        
        # Normalize stock allocations (optional, to make sure they sum to 1)
        stock_allocations /= np.sum(stock_allocations)
        
        # Portfolio management logic using stock allocations
        allocation = stock_allocations * self.current_balance
        portfolio_return = np.dot(self.stock_data.iloc[self.current_step].pct_change().fillna(0), allocation)
        
        # Update balance based on portfolio return
        self.current_balance += portfolio_return
        
        # Use RSI thresholds for trading signals (you'd calculate the RSI here based on price data)
        rsi_value = self.calculate_rsi(self.stock_data.iloc[:self.current_step])
        if rsi_value > rsi_high_threshold:
            # Apply selling logic based on the threshold
            pass
        elif rsi_value < rsi_low_threshold:
            # Apply buying logic based on the threshold
            pass
        
        # Calculate reward (based on portfolio return)
        reward = portfolio_return
        
        # Proceed to next step
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1
        obs = self._next_observation()
        
        return obs, reward, done, {}

    def calculate_rsi(self, prices):
        # Placeholder for RSI calculation (implement your RSI calculation here)
        delta = prices.diff()
        gain = delta[delta > 0].mean()
        loss = -delta[delta < 0].mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def render(self, mode='human'):
        # Print current portfolio balance (optional)
        print(f"Step: {self.current_step}, Balance: {self.current_balance}")
