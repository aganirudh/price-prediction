"""
Quick demo — runs in 2 minutes on CPU, no data download needed.
Shows: ensemble training → checkpoint saving → mock inference
"""
import os, json
from pathlib import Path
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

CHECKPOINT_DIR = "demo_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 60)
print("PCP Arbitrage RL System — Quick Demo")
print("=" * 60)

print("\n[1/3] Training PPO + A2C ensemble (500 steps each)...")
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

agents = {}
for Cls, name in [(PPO, "PPO"), (A2C, "A2C")]:
    m = Cls("MlpPolicy", env, verbose=0)
    m.learn(500)
    path = os.path.join(CHECKPOINT_DIR, name)
    m.save(path)
    agents[name] = path
    print(f"  ✓ {name} trained and saved")

print("\n[2/3] Simulating rolling Sharpe selector...")
sharpes = {"PPO": 1.42, "A2C": 0.87}  # mock values
best = max(sharpes, key=sharpes.get)
print(f"  Sharpe scores: {sharpes}")
print(f"  ✓ Best agent selected: {best}")

print("\n[3/3] Mock PCP violation check (no GPU needed)...")
violation = {"underlying": "NIFTY", "pcp_deviation": 0.73, "stt_cost": 0.125,
             "net_edge": 0.73 - 0.125, "decision": "ENTER" if 0.73 > 0.125 else "SKIP"}
print(f"  Violation detected: {violation}")
print(f"  ✓ Decision: {violation['decision']} (net edge after STT = {violation['net_edge']:.3f}%)")

print("\n✅ Demo complete! Checkpoints in:", CHECKPOINT_DIR)
print("   Next: python main.py --mode train-ensemble --timesteps 50000")
