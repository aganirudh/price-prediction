import torch
import numpy as np
import pandas as pd
import time
import json
import logging
from datetime import datetime, timedelta
from stock_rl_env import StockTradingEnv
from train_rl_trading import DQN
import matplotlib.pyplot as plt
import yfinance as yf

class LivePerformanceEvaluator:
    def __init__(self, model_path, stock_symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']):
        self.stock_symbols = stock_symbols
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained model
        self.load_model()
        
        # Setup logging
        self.setup_logging()
        
        # Performance tracking
        self.performance_history = []
        self.live_portfolio_value = 10000
        self.live_holdings = {symbol: 0 for symbol in stock_symbols}
        self.live_balance = 10000
        
    def setup_logging(self):
        """Setup live evaluation logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'live_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Live performance evaluator initialized")
    
    def load_model(self):
        """Load the trained DQN model"""
        # Create environment to get state size
        temp_env = StockTradingEnv(stock_symbols=self.stock_symbols)
        state_size = temp_env.observation_space.shape[0]
        action_size = 7 * len(self.stock_symbols)
        
        # Load model
        self.model = DQN(state_size, action_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        self.logger.info(f"Model loaded from {self.model_path}")
    
    def get_model_prediction(self, state):
        """Get model prediction for current state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            
            # Get action for each stock
            actions = []
            action_confidences = []
            
            for i in range(len(self.stock_symbols)):
                stock_q_values = q_values[0][i*7:(i+1)*7]
                action = stock_q_values.argmax().item()
                confidence = torch.softmax(stock_q_values, dim=0)[action].item()
                
                actions.append(action)
                action_confidences.append(confidence)
            
            return np.array(actions), action_confidences
    
    def run_live_evaluation(self, evaluation_days=15, check_interval=3600):
        """Run continuous live evaluation"""
        self.logger.info(f"Starting live evaluation for {evaluation_days} days")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(days=evaluation_days)
        
        while datetime.now() < end_time:
            try:
                # Create fresh environment with recent data
                env = StockTradingEnv(stock_symbols=self.stock_symbols)
                
                # Run evaluation episode
                accuracy_metrics = self.evaluate_single_episode(env)
                
                # Log results
                self.log_evaluation_results(accuracy_metrics)
                
                # Wait before next evaluation
                self.logger.info(f"Sleeping for {check_interval} seconds until next evaluation...")
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in live evaluation: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        self.logger.info("Live evaluation completed")
        self.generate_final_report()
    
    def evaluate_single_episode(self, env):
        """Evaluate model on a single episode"""
        state = env.reset()
        total_reward = 0
        episode_trades = []
        correct_predictions = 0
        total_predictions = 0
        
        while True:
            # Get model prediction
            actions, confidences = self.get_model_prediction(state)
            
            # Execute actions in environment
            next_state, reward, done, info = env.step(actions)
            
            # Analyze prediction accuracy
            prediction_analysis = self.analyze_prediction_accuracy(
                env, actions, confidences, reward, info
            )
            
            if prediction_analysis:
                correct_predictions += prediction_analysis['correct']
                total_predictions += prediction_analysis['total']
                episode_trades.append({
                    'step': env.current_step,
                    'actions': actions.tolist(),
                    'confidences': confidences,
                    'reward': reward,
                    'analysis': prediction_analysis,
                    'market_data': self.get_current_market_snapshot(env)
                })
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Calculate accuracy metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_reward': total_reward,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'episode_trades': episode_trades,
            'final_portfolio_value': env._calculate_portfolio_value(),
            'performance_metrics': env.get_performance_metrics()
        }
    
    def analyze_prediction_accuracy(self, env, actions, confidences, reward, info):
        """Analyze if the model's predictions were accurate"""
        analysis = {
            'correct': 0,
            'total': 0,
            'trade_analysis': {}
        }
        
        # Check each stock's action
        for i, (action, confidence) in enumerate(zip(actions, confidences)):
            symbol = env.stock_symbols[i]
            
            if action == 0:  # Hold action
                continue
            
            analysis['total'] += 1
            
            # Get current and next price to evaluate decision
            current_price = env._get_current_price(symbol)
            
            # Check if it was a good trade based on immediate reward
            if symbol in info.get('trades', {}):
                trade_info = info['trades'][symbol]
                
                # Simple accuracy check: positive reward for the trade
                if reward > 0:
                    analysis['correct'] += 1
                
                analysis['trade_analysis'][symbol] = {
                    'action': trade_info['action'],
                    'confidence': confidence,
                    'reasoning': trade_info['reasoning'],
                    'price': current_price,
                    'successful': reward > 0
                }
        
        return analysis if analysis['total'] > 0 else None
    
    def get_current_market_snapshot(self, env):
        """Get current market data snapshot"""
        snapshot = {}
        
        for symbol in env.stock_symbols:
            current_data = env.data[env.data['Symbol'] == symbol].iloc[env.current_step]
            snapshot[symbol] = {
                'price': float(current_data['Close']),
                'volume': float(current_data['Volume']),
                'rsi': float(current_data['RSI']) if pd.notna(current_data['RSI']) else None,
                'sma_5': float(current_data['SMA_5']) if pd.notna(current_data['SMA_5']) else None,
                'volatility': float(current_data['Volatility']) if pd.notna(current_data['Volatility']) else None
            }
        
        return snapshot
    
    def log_evaluation_results(self, results):
        """Log detailed evaluation results"""
        self.performance_history.append(results)
        
        # Log summary
        self.logger.info(f"Evaluation - Reward: {results['total_reward']:.2f}, "
                        f"Accuracy: {results['accuracy']*100:.1f}%, "
                        f"Portfolio: ${results['final_portfolio_value']:.2f}")
        
        # Log detailed trade analysis
        for trade in results['episode_trades']:
            if trade['analysis']['trade_analysis']:
                self.logger.info(f"Trade analysis: {json.dumps(trade['analysis']['trade_analysis'], indent=2)}")
        
        # Save to file periodically
        if len(self.performance_history) % 5 == 0:
            self.save_evaluation_data()
    
    def save_evaluation_data(self):
        """Save evaluation data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to JSON
        with open(f'live_evaluation_data_{timestamp}.json', 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        # Create summary CSV
        summary_data = []
        for result in self.performance_history:
            summary_data.append({
                'timestamp': result['timestamp'],
                'total_reward': result['total_reward'],
                'accuracy': result['accuracy'],
                'portfolio_value': result['final_portfolio_value'],
                'num_trades': len(result['episode_trades'])
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(f'live_evaluation_summary_{timestamp}.csv', index=False)
        
        self.logger.info("Evaluation data saved")
    
    def generate_final_report(self):
        """Generate comprehensive final evaluation report"""
        if not self.performance_history:
            self.logger.warning("No evaluation data to report")
            return
        
        # Calculate aggregate statistics
        total_rewards = [r['total_reward'] for r in self.performance_history]
        accuracies = [r['accuracy'] for r in self.performance_history]
        portfolio_values = [r['final_portfolio_value'] for r in self.performance_history]
        
        report = {
            'evaluation_period': {
                'start': self.performance_history[0]['timestamp'],
                'end': self.performance_history[-1]['timestamp'],
                'total_evaluations': len(self.performance_history)
            },
            'performance_summary': {
                'avg_reward': np.mean(total_rewards),
                'std_reward': np.std(total_rewards),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'avg_portfolio_value': np.mean(portfolio_values),
                'final_portfolio_value': portfolio_values[-1],
                'total_return': (portfolio_values[-1] - 10000) / 10000
            },
            'trade_analysis': self.analyze_all_trades()
        }
        
        # Save detailed report
        with open(f'final_evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        self.logger.info("=== FINAL EVALUATION REPORT ===")
        self.logger.info(f"Average Reward: {report['performance_summary']['avg_reward']:.2f}")
        self.logger.info(f"Average Accuracy: {report['performance_summary']['avg_accuracy']*100:.1f}%")
        self.logger.info(f"Final Portfolio Value: ${report['performance_summary']['final_portfolio_value']:.2f}")
        self.logger.info(f"Total Return: {report['performance_summary']['total_return']*100:.1f}%")
        
        # Create visualization
        self.create_evaluation_plots()
        
        return report
    
    def analyze_all_trades(self):
        """Analyze all trades made during evaluation"""
        all_trades = []
        stock_performance = {symbol: {'total': 0, 'successful': 0} for symbol in self.stock_symbols}
        
        for result in self.performance_history:
            for trade in result['episode_trades']:
                all_trades.extend(trade['analysis']['trade_analysis'].items())
                
                for symbol, trade_data in trade['analysis']['trade_analysis'].items():
                    stock_performance[symbol]['total'] += 1
                    if trade_data['successful']:
                        stock_performance[symbol]['successful'] += 1
        
        # Calculate success rates per stock
        for symbol in stock_performance:
            total = stock_performance[symbol]['total']
            if total > 0:
                stock_performance[symbol]['success_rate'] = stock_performance[symbol]['successful'] / total
            else:
                stock_performance[symbol]['success_rate'] = 0
        
        return {
            'total_trades': len(all_trades),
            'stock_performance': stock_performance
        }
    
    def create_evaluation_plots(self):
        """Create comprehensive evaluation plots"""
        if len(self.performance_history) < 2:
            return
        
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in self.performance_history]
        rewards = [r['total_reward'] for r in self.performance_history]
        accuracies = [r['accuracy'] * 100 for r in self.performance_history]
        portfolio_values = [r['final_portfolio_value'] for r in self.performance_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Rewards over time
        ax1.plot(timestamps, rewards, marker='o')
        ax1.set_title('Model Rewards Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45)
        
        # Accuracy over time
        ax2.plot(timestamps, accuracies, marker='o', color='green')
        ax2.set_title('Model Accuracy Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45)
        
        # Portfolio value over time
        ax3.plot(timestamps, portfolio_values, marker='o', color='purple')
        ax3.set_title('Portfolio Value Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.grid(True)
        ax3.tick_params(axis='x', rotation=45)
        
        # Distribution of accuracies
        ax4.hist(accuracies, bins=10, alpha=0.7, color='orange')
        ax4.set_title('Distribution of Model Accuracies')
        ax4.set_xlabel('Accuracy (%)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'live_evaluation_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Evaluation plots saved")

def main():
    """Main function to run live evaluation"""
    # Make sure you have a trained model file
    model_path = 'final_dqn_model.pth'
    
    try:
        evaluator = LivePerformanceEvaluator(model_path)
        
        # Run evaluation for 15 days, checking every hour
        evaluator.run_live_evaluation(
            evaluation_days=15,
            check_interval=3600  # 1 hour
        )
        
    except FileNotFoundError:
        print("Model file not found. Please train the model first by running train_rl_trading.py")
    except Exception as e:
        print(f"Error running live evaluation: {e}")

if __name__ == "__main__":
    main()