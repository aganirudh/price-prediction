import os
import argparse
import yaml
import pandas as pd
import numpy as np

# Import from our newly created modules
from data.feeds import HistoricalFeed, LiveFeed, MockFeed
from data.processors import PCPCalculator, CostCalculator

# Placeholder for other necessary imports and code
# from pcp_arb_env.environment import PcpArbEnv
# from training.train import train_model
# from backtest.backtester import Backtester
# from execution.order_simulator import OrderSimulator

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="AI Trading System Runner")
    parser.add_argument("mode", choices=["train", "backtest", "paper", "alpha"], help="Run mode: train, backtest, paper, or alpha")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Starting in mode: {args.mode}")

    # Initialize feed and processor based on config
    feed_type = config.get("data_feed", {}).get("type", "mock")
    
    if feed_type == "historical":
        feed_path = config.get("data_feed", {}).get("path", "data/nifty50_historical_data.csv")
        # Assuming HistoricalFeed needs symbol and date, which might come from config or be defaults
        feed = HistoricalFeed(csv_path=feed_path, symbol="NIFTY50")
        print(f"Using HistoricalFeed from {feed_path}")
    elif feed_type == "live":
        feed = LiveFeed()
        print("Using LiveFeed")
    else: # default to mock
        feed = MockFeed()
        print("Using MockFeed")

    # Initialize PCP Calculator. This might need more parameters like TTE, risk-free rate.
    # We'll use defaults for now and they might be updated by config or during runtime.
    # Placeholder: Actual TTE and risk rate would come from config or market data.
    pcp_calculator = PCPCalculator(
        spot_price=feed.get_spot_price("NIFTY50") if hasattr(feed, 'get_spot_price') else 18000.0,
        time_to_expiry_days=config.get("model_params", {}).get("time_to_expiry_days", 30)
    )
    
    cost_calculator = CostCalculator(
        stt_rate_buy=config.get("costs", {}).get("stt_rate_buy", 0.000625),
        stt_rate_sell_option=config.get("costs", {}).get("stt_rate_sell_option", 0.000125),
        brokerage_rate=config.get("costs", {}).get("brokerage_rate", 0.0003)
    )

    if args.mode == "train":
        print("Training mode selected. (Placeholder for training logic)")
        # Example of using the feed and calculator
        current_spot = feed.get_spot_price("NIFTY50")
        print(f"Current Spot Price: {current_spot}")
        
        # For historical feed, advance it
        if isinstance(feed, HistoricalFeed):
            feed.next_step()
            print(f"Advanced historical feed to next step.")

        # Example of using pcp_calculator (needs actual options data)
        # options_data = feed.get_options_chain("NIFTY50", "2023-12-28") # Example date
        # if not options_data.empty:
        #     arbitrage_opportunities = pcp_calculator.detect_arbitrage(options_data, tolerance=0.5)
        #     print(f"Detected Arbitrage Opportunities: {arbitrage_opportunities}")
        
        # train_model(config, feed, pcp_calculator, cost_calculator) # Placeholder

    elif args.mode == "backtest":
        print("Backtesting mode selected. (Placeholder for backtesting logic)")
        # backtester = Backtester(config, feed, pcp_calculator, cost_calculator)
        # backtester.run()

    elif args.mode == "paper":
        print("Paper trading mode selected. (Placeholder for paper trading logic)")
        # simulator = OrderSimulator(config, feed, pcp_calculator, cost_calculator)
        # simulator.run()

    elif args.mode == "alpha":
        print("Alpha analysis mode selected. (Placeholder for alpha analysis logic)")
        # Analyze market for arbitrage opportunities
        pass

if __name__ == "__main__":
    # Create a dummy config.yaml for testing purposes if it doesn't exist
    if not os.path.exists("config.yaml"):
        print("Creating dummy config.yaml")
        dummy_config = {
            "data_feed": {"type": "historical", "path": "data/nifty50_historical_data.csv"},
            "model_params": {"time_to_expiry_days": 30, "rl_algo": "GRPO", "llm_model": "Qwen2.5-1.5B"},
            "costs": {"stt_rate_buy": 0.000625, "stt_rate_sell_option": 0.000125, "brokerage_rate": 0.0003},
            "training": {"epochs": 100, "learning_rate": 0.001},
            "backtest": {"start_date": "2023-01-01", "end_date": "2023-12-31"}
        }
        with open("config.yaml", "w") as f:
            yaml.dump(dummy_config, f)

    # Create dummy data directory and a placeholder CSV if needed for testing
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/nifty50_historical_data.csv"):
        print("Creating dummy data/nifty50_historical_data.csv")
        dummy_data = pd.DataFrame({
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='min')),
            'spot_price': np.linspace(18000, 18500, 100) + np.random.normal(0, 20, 100),
            # Add other columns if your HistoricalFeed expects them, e.g. Option Chain related data, though we stubbed that out
        })
        dummy_data.to_csv("data/nifty50_historical_data.csv", index=False)
        
    main()
