import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketFeed:
    """Base class for market data feeds."""
    def get_spot_price(self, symbol: str) -> float:
        raise NotImplementedError

    def get_options_chain(self, symbol: str, date: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_time(self) -> datetime:
        raise NotImplementedError

class LiveFeed(MarketFeed):
    """Simulates a live market feed."""
    def __init__(self, initial_spot=18000.0):
        self.current_spot = initial_spot
        self.current_time = datetime.now()
        # In a real scenario, this would connect to a live data API

    def get_spot_price(self, symbol: str) -> float:
        # Simulate price fluctuation
        self.current_spot += np.random.normal(0, 5)
        return self.current_spot

    def get_options_chain(self, symbol: str, date: str) -> pd.DataFrame:
        # Simulate options chain data
        chain_data = []
        strikes = np.arange(self.current_spot - 200, self.current_spot + 200, 50)
        for strike in strikes:
            for option_type in ["CE", "PE"]:
                chain_data.append({
                    "symbol": f"{symbol}{date}",
                    "strike": strike,
                    "option_type": option_type,
                    "expiry_date": date,
                    "bid_price": max(0, (strike - self.current_spot) * 0.01 + np.random.normal(0, 0.5) if option_type == "CE" else (self.current_spot - strike) * 0.01 + np.random.normal(0, 0.5)),
                    "ask_price": max(0, (strike - self.current_spot) * 0.01 + np.random.normal(0, 0.5) + 0.1 if option_type == "CE" else (self.current_spot - strike) * 0.01 + np.random.normal(0, 0.5) + 0.1),
                    "implied_volatility": np.random.uniform(0.1, 0.5),
                    "last_trade_price": max(0, (strike - self.current_spot) * 0.01 + np.random.normal(0, 0.6) if option_type == "CE" else (self.current_spot - strike) * 0.01 + np.random.normal(0, 0.6))
                })
        return pd.DataFrame(chain_data)

    def get_time(self) -> datetime:
        self.current_time += timedelta(minutes=1) # Simulate time moving forward
        return self.current_time

class HistoricalFeed(MarketFeed):
    """Loads market data from a CSV file."""
    def __init__(self, csv_path: str, symbol: str = "NIFTY50"):
        self.data = pd.read_csv(csv_path)
        # Ensure data has a datetime index and necessary columns
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("CSV must contain a 'timestamp' column.")

        self.symbol = symbol
        self.current_index = 0
        self.data.sort_index(inplace=True) # Ensure sorted by time

    def get_spot_price(self, symbol: str) -> float:
        if self.current_index < len(self.data):
            return self.data.iloc[self.current_index]['spot_price'] # Assuming 'spot_price' column exists
        return np.nan # Or raise error

    def get_options_chain(self, symbol: str, date: str) -> pd.DataFrame:
        # This is a simplified simulation. A real implementation would parse options data
        # from the historical file or query a pre-processed structure.
        # For now, we'll return dummy data that might align with the current spot price.
        current_spot = self.get_spot_price(symbol)
        if pd.isna(current_spot):
            return pd.DataFrame()

        # Placeholder for options chain data; assumes columns like in LiveFeed
        chain_data = []
        strikes = np.arange(current_spot - 200, current_spot + 200, 50)
        for strike in strikes:
            for option_type in ["CE", "PE"]:
                chain_data.append({
                    "symbol": f"{symbol}{date}",
                    "strike": strike,
                    "option_type": option_type,
                    "expiry_date": date,
                    "bid_price": max(0, (strike - current_spot) * 0.01 + np.random.normal(0, 0.3)),
                    "ask_price": max(0, (strike - current_spot) * 0.01 + np.random.normal(0, 0.3) + 0.1),
                    "implied_volatility": np.random.uniform(0.1, 0.5),
                    "last_trade_price": max(0, (strike - current_spot) * 0.01 + np.random.normal(0, 0.4))
                })
        return pd.DataFrame(chain_data)

    def get_time(self) -> datetime:
        if self.current_index < len(self.data):
            return self.data.index[self.current_index]
        return self.data.index[-1] # Return last time if index is out of bounds

    def next_step(self):
        self.current_index += 1
        if self.current_index >= len(self.data):
            self.current_index = len(self.data) - 1 # Stay at the last point

class MockFeed(MarketFeed):
    """A mock feed for testing, returns static data."""
    def __init__(self):
        self.spot_price = 18500.0
        self.options_chain_data = {
            "NIFTY50_2023-12-28": pd.DataFrame({
                "symbol": "NIFTY50_2023-12-28", "strike": 18500, "option_type": "CE", "expiry_date": "2023-12-28", "bid_price": 100, "ask_price": 101, "implied_volatility": 0.2, "last_trade_price": 100.5
            })
        }
        self.current_time = datetime(2023, 12, 28, 9, 15)

    def get_spot_price(self, symbol: str) -> float:
        return self.spot_price

    def get_options_chain(self, symbol: str, date: str) -> pd.DataFrame:
        return self.options_chain_data.get(f"{symbol}_{date}", pd.DataFrame())

    def get_time(self) -> datetime:
        return self.current_time
