import pandas as pd
import numpy as np

class PCPCalculator:
    """Calculates Put-Call Parity and arbitrage opportunities."""
    def __init__(self, spot_price: float, risk_free_rate: float = 0.05, time_to_expiry_days: int = 30):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.time_to_expiry_days = time_to_expiry_days
        self.time_to_expiry_years = time_to_expiry_days / 365.0

    def update_params(self, spot_price: float, time_to_expiry_days: int):
        self.spot_price = spot_price
        self.time_to_expiry_days = time_to_expiry_days
        self.time_to_expiry_years = time_to_expiry_days / 365.0

    def calculate_fair_value_call(self, strike_price: float, put_price: float) -> float:
        """Calculates the theoretical fair value of a call option using Put-Call Parity."""
        # C + PV(K) = P + S
        # C = P + S - PV(K)
        pv_strike = strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry_years)
        fair_call_price = put_price + self.spot_price - pv_strike
        return fair_call_price

    def calculate_fair_value_put(self, strike_price: float, call_price: float) -> float:
        """Calculates the theoretical fair value of a put option using Put-Call Parity."""
        # P = C - S + PV(K)
        pv_strike = strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry_years)
        fair_put_price = call_price - self.spot_price + pv_strike
        return fair_put_price

    def detect_arbitrage(self, option_data: pd.DataFrame, tolerance: float = 0.1) -> list:
        """
        Detects arbitrage opportunities by comparing theoretical prices with market prices.
        Assumes option_data is a DataFrame with columns: 'strike', 'option_type', 'bid_price', 'ask_price'.
        """
        arbitrage_opportunities = []
        
        for strike in option_data['strike'].unique():
            call_options = option_data[(option_data['strike'] == strike) & (option_data['option_type'] == 'CE')]
            put_options = option_data[(option_data['strike'] == strike) & (option_data['option_type'] == 'PE')]

            if not call_options.empty and not put_options.empty:
                call_bid = call_options['bid_price'].iloc[0]
                call_ask = call_options['ask_price'].iloc[0]
                put_bid = put_options['bid_price'].iloc[0]
                put_ask = put_options['ask_price'].iloc[0]

                # Theoretical Call Price calculation (using market Put)
                theoretical_call_from_put = self.calculate_fair_value_call(strike, put_bid)
                
                # Arbitrage if market call price is significantly different from theoretical call price
                # Long Call if market call_ask < theoretical_call_from_put - tolerance
                if call_ask < theoretical_call_from_put - tolerance:
                    arbitrage_opportunities.append({
                        "type": "long_call_arbitrage", "strike": strike, 
                        "market_call_ask": call_ask, "theoretical_call": theoretical_call_from_put,
                        "strategy": "buy call, sell put, sell spot" # simplified
                    })
                # Short Call if market call_bid > theoretical_call_from_put + tolerance
                if call_bid > theoretical_call_from_put + tolerance:
                     arbitrage_opportunities.append({
                        "type": "short_call_arbitrage", "strike": strike, 
                        "market_call_bid": call_bid, "theoretical_call": theoretical_call_from_put,
                        "strategy": "sell call, buy put, buy spot" # simplified
                    })

                # Theoretical Put Price calculation (using market Call)
                theoretical_put_from_call = self.calculate_fair_value_put(strike, call_ask)

                # Arbitrage if market put price is significantly different from theoretical put price
                # Long Put if market put_ask < theoretical_put_from_call - tolerance
                if put_ask < theoretical_put_from_call - tolerance:
                    arbitrage_opportunities.append({
                        "type": "long_put_arbitrage", "strike": strike,
                        "market_put_ask": put_ask, "theoretical_put": theoretical_put_from_call,
                        "strategy": "buy put, sell call, buy spot" # simplified
                    })
                # Short Put if market put_bid > theoretical_call_from_put + tolerance
                if put_bid > theoretical_call_from_put + tolerance:
                    arbitrage_opportunities.append({
                        "type": "short_put_arbitrage", "strike": strike,
                        "market_put_bid": put_bid, "theoretical_put": theoretical_put_from_call,
                        "strategy": "sell put, buy call, sell spot" # simplified
                    })

        return arbitrage_opportunities

class CostCalculator:
    """Calculates trading costs like STT, brokerage, etc."""
    def __init__(self, stt_rate_buy: float = 0.000625, stt_rate_sell_option: float = 0.000125, brokerage_rate: float = 0.0003):
        self.stt_rate_buy = stt_rate_buy # STT on buy side (e.g., for futures/options selling)
        self.stt_rate_sell_option = stt_rate_sell_option # STT on sell side for options
        self.brokerage_rate = brokerage_rate # Brokerage as a percentage

    def calculate_transaction_cost(self, order_value: float, is_option_sell: bool = False) -> float:
        """Calculates total transaction cost for an order."""
        stt = 0
        brokerage = order_value * self.brokerage_rate

        if is_option_sell:
            # STT applies differently for selling options
            stt = order_value * self.stt_rate_sell_option
        else:
            # STT on buy side (e.g., buying a Futures or selling an Option F&O segment)
            # This is simplified; actual STT rules can be complex.
            stt = order_value * self.stt_rate_buy 

        return brokerage + stt

# Example usage (optional, for testing)
if __name__ == '__main__':
    print("Testing Data Processors...")
    
    # Test PCPCalculator
    spot = 18500.0
    tte_days = 20
    calculator = PCPCalculator(spot_price=spot, time_to_expiry_days=tte_days)
    
    # Mock option data for a specific strike
    mock_option_df = pd.DataFrame({
        "symbol": ["NIFTY50_2023-12-28"] * 4,
        "strike": [18500, 18500, 18500, 18500],
        "option_type": ["CE", "CE", "PE", "PE"],
        "bid_price": [100.5, np.nan, 98.0, np.nan],
        "ask_price": [101.0, 102.0, np.nan, 99.0],
        "implied_volatility": [0.2, 0.2, 0.2, 0.2],
        "last_trade_price": [100.7, 101.5, 98.5, 98.8]
    })

    # Fill NaNs for calculation purposes
    mock_option_df.loc[(mock_option_df['option_type'] == 'CE') & (mock_option_df['bid_price'].isna()), 'bid_price'] = mock_option_df.loc[(mock_option_df['option_type'] == 'CE') & (mock_option_df['ask_price'].notna()), 'ask_price'] - 0.1
    mock_option_df.loc[(mock_option_df['option_type'] == 'PE') & (mock_option_df['ask_price'].isna()), 'ask_price'] = mock_option_df.loc[(mock_option_df['option_type'] == 'PE') & (mock_option_df['bid_price'].notna()), 'bid_price'] + 0.1


    print("\n--- PCPCalculator Test ---")
    arbitrages = calculator.detect_arbitrage(mock_option_df, tolerance=0.5)
    print(f"Spot Price: {spot}, TTE: {tte_days} days")
    print("Detected Arbitrage Opportunities:", arbitrages)

    # Test CostCalculator
    print("\n--- CostCalculator Test ---")
    cost_calc = CostCalculator()
    order_value = 100000  # Example order value
    transaction_cost = cost_calc.calculate_transaction_cost(order_value, is_option_sell=True)
    print(f"Order Value: {order_value}, Transaction Cost (Option Sell): {transaction_cost:.2f}")
