import pandas as pd
import numpy as np

# Import from our newly created modules
from data.feeds import MarketFeed
from data.processors import CostCalculator
# from data.feeds import HistoricalFeed # if needed

class OrderSimulator:
    """Simulates order execution and tracks portfolio."""
    def __init__(self, config: dict, feed: MarketFeed, cost_calculator: CostCalculator):
        self.config = config
        self.feed = feed
        self.cost_calculator = cost_calculator
        
        self.initial_capital = config.get("execution", {}).get("initial_capital", 100000.0)
        self.capital = self.initial_capital
        self.portfolio = {} # e.g., {'options': [{'strike': 18500, 'type': 'CE', 'quantity': 10, ...}]}
        self.transactions = []

    def place_order(self, order_details: dict):
        """Simulates placing an order."""
        # order_details could include: symbol, strike, option_type, action (buy/sell), quantity, price
        
        order_value = order_details.get('quantity', 1) * order_details.get('price', 0)
        is_option_sell = order_details.get('option_type') is not None and order_details.get('action') == 'sell'
        
        transaction_cost = self.cost_calculator.calculate_transaction_cost(order_value, is_option_sell=is_option_sell)
        
        # Check if sufficient capital
        if self.capital < order_value + transaction_cost:
            print(f"Order failed: Insufficient capital. Need {order_value + transaction_cost:.2f}, have {self.capital:.2f}")
            return False
        
        # Update capital
        if order_details.get('action') == 'buy':
            self.capital -= (order_value + transaction_cost)
        elif order_details.get('action') == 'sell':
            self.capital += (order_value - transaction_cost)
            
        # Log transaction
        self.transactions.append({
            "timestamp": self.feed.get_time(),
            "order": order_details,
            "order_value": order_value,
            "transaction_cost": transaction_cost,
            "new_capital": self.capital
        })
        
        print(f"Order placed: {order_details}. Cost: {transaction_cost:.2f}. New Capital: {self.capital:.2f}")
        return True

    def update_portfolio(self, order_details):
        # Logic to update self.portfolio based on trades
        pass

    def get_current_pnl(self):
        """Calculates unrealized PnL based on current portfolio and market."""
        # This would need access to current market prices from the feed
        pass


# Example of how to call this
if __name__ == "__main__":
    print("Running standalone order_simulator.py test (placeholder)...")
    
    # Dummy setup
    dummy_config = {
        "execution": {"initial_capital": 50000.0},
        "costs": {"stt_rate_buy": 0.000625, "stt_rate_sell_option": 0.000125, "brokerage_rate": 0.0003}
    }
    
    # Need a feed and cost calculator
    from data.feeds import MockFeed
    mock_feed = MockFeed()
    cost_calc = CostCalculator(**dummy_config["costs"])
    
    simulator = OrderSimulator(dummy_config, mock_feed, cost_calc)
    
    # Simulate a buy order
    buy_order = {'symbol': 'NIFTY50', 'strike': 18500, 'option_type': 'CE', 'action': 'buy', 'quantity': 10, 'price': 100.0}
    simulator.place_order(buy_order)
    
    # Simulate a sell order (e.g., selling an option)
    sell_order = {'symbol': 'NIFTY50', 'strike': 18500, 'option_type': 'CE', 'action': 'sell', 'quantity': 10, 'price': 120.0}
    simulator.place_order(sell_order)
