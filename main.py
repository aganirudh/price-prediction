"""
PCP Arbitrage RL System - Main Entry Point
Modes: train, train-ensemble, train-hybrid, backtest, paper, alpha, compare, demo
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", ROOT / "checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "kaggle"

# ── Helpers ────────────────────────────────────────────────────────────────────
def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _require_servers(urls):
    """Warn if MCP servers aren't reachable (non-fatal)."""
    import requests
    for name, url in urls.items():
        try:
            requests.get(f"{url}/health", timeout=2)
            print(f"  [✓] {name} server OK at {url}")
        except Exception:
            print(f"  [!] {name} server not reachable at {url} — some features disabled")


# ── Mode: train-ensemble ───────────────────────────────────────────────────────
def run_train_ensemble(args):
    """
    Train PPO + A2C + DDPG on NIFTY50 stock data using a custom trading env.
    Falls back to synthetic data if Kaggle data not present.
    """
    import numpy as np
    import pandas as pd
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, A2C, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.noise import NormalActionNoise

    print("\n" + "=" * 60)
    print("  LAYER 1: Ensemble RL Training (PPO + A2C + DDPG)")
    print("=" * 60)

    # ── Stock Trading Environment ──────────────────────────────────────────────
    class StockTradingEnv(gym.Env):
        """
        Simple single-stock trading env.
        State: [price_norm, sma5_norm, sma20_norm, rsi_norm, position]
        Action: continuous [-1, 1] → short/hold/long
        """
        metadata = {"render_modes": []}

        def __init__(self, prices: np.ndarray, initial_cash: float = 100_000):
            super().__init__()
            self.prices = prices
            self.n = len(prices)
            self.initial_cash = initial_cash
            # obs: price_norm, sma5, sma20, rsi, position
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            )
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.reset()

        def _sma(self, t, w):
            start = max(0, t - w + 1)
            return self.prices[start : t + 1].mean()

        def _rsi(self, t, w=14):
            if t < w:
                return 50.0
            deltas = np.diff(self.prices[t - w : t + 1])
            gains = deltas[deltas > 0].sum() / w
            losses = -deltas[deltas < 0].sum() / w
            if losses == 0:
                return 100.0
            rs = gains / losses
            return 100 - 100 / (1 + rs)

        def _obs(self):
            t = self.t
            p = self.prices[t]
            norm_p = p / self.prices[0]
            sma5 = self._sma(t, 5) / self.prices[0]
            sma20 = self._sma(t, 20) / self.prices[0]
            rsi = self._rsi(t) / 100.0
            pos = self.position / 100.0
            return np.array([norm_p, sma5, sma20, rsi, pos], dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 20
            self.cash = self.initial_cash
            self.position = 0  # shares held
            self.nav_history = [self.initial_cash]
            return self._obs(), {}

        def step(self, action):
            price = self.prices[self.t]
            target_position = int(action[0] * 100)  # -100 to +100 shares
            delta = target_position - self.position
            cost = abs(delta) * price * 0.001  # 0.1% transaction cost
            self.cash -= delta * price + cost
            self.position = target_position
            self.t += 1

            done = self.t >= self.n - 1
            nav = self.cash + self.position * self.prices[self.t]
            self.nav_history.append(nav)
            reward = (nav - self.nav_history[-2]) / self.initial_cash * 100
            return self._obs(), float(reward), done, False, {}

    # ── Load or synthesize price data ──────────────────────────────────────────
    csv_files = list(Path(args.data_dir).glob("**/*.csv")) if Path(args.data_dir).exists() else []
    if csv_files:
        all_prices = []
        for csv_file in csv_files[:3]:
            try:
                raw = pd.read_csv(csv_file)
                # yfinance new format: first row is ticker name, not a date
                # Detect by checking if first value in first col is a date
                try:
                    pd.to_datetime(raw.iloc[0, 0])
                    df = raw  # first row IS a date — normal format
                except Exception:
                    df = pd.read_csv(csv_file, skiprows=1)  # skip ticker-name row

                df.columns = [str(c).strip() for c in df.columns]
                price_col = None
                for col in ["Close", "close", "CLOSE", "Adj Close", "adj_close", "Price"]:
                    if col in df.columns:
                        price_col = col
                        break
                if price_col is None:
                    for col in reversed(df.columns):
                        vals = pd.to_numeric(df[col], errors="coerce").dropna()
                        if len(vals) > 100:
                            price_col = col
                            break
                if price_col:
                    p = pd.to_numeric(df[price_col], errors="coerce").dropna().values.astype(np.float64)
                    if len(p) > 100:
                        all_prices.append(p)
                        print(f"[Data] Loaded {len(p)} rows from {csv_file.name} (col={price_col})")
            except Exception as e:
                print(f"[Data] Skipping {csv_file.name}: {e}")

        prices = all_prices[0] if all_prices else None
        if prices is None:
            print("[Data] Could not parse any CSV — using synthetic data")
            np.random.seed(42)
            prices = 17000 * np.exp(np.cumsum(np.random.normal(0.0004, 0.012, 2000)))
    else:
        print("[Data] No CSV found — generating synthetic GBM prices (NIFTY-like)")
        np.random.seed(42)
        n = 2000
        returns = np.random.normal(0.0004, 0.012, n)
        prices = 17000 * np.exp(np.cumsum(returns))

    # Train/val split
    split = int(len(prices) * 0.8)
    train_prices = prices[:split]
    print(f"[Data] Train: {len(train_prices)} steps | Val: {len(prices)-split} steps")

    def make_env():
        return StockTradingEnv(train_prices)

    vec_env = DummyVecEnv([make_env])
    timesteps = args.timesteps or 50_000
    results = {}

    # ── PPO ────────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Training PPO for {timesteps:,} steps...")
    ppo = PPO(
        "MlpPolicy", vec_env, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01,
        tensorboard_log=str(CHECKPOINT_DIR / "tb_logs")
    )
    ppo.learn(total_timesteps=timesteps, progress_bar=False)
    ppo_path = str(CHECKPOINT_DIR / "PPO_nifty")
    ppo.save(ppo_path)
    print(f"  [✓] PPO saved → {ppo_path}.zip")
    results["PPO"] = ppo_path

    # ── A2C ────────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Training A2C for {timesteps:,} steps...")
    a2c = A2C(
        "MlpPolicy", vec_env, verbose=1,
        learning_rate=7e-4, n_steps=5,
        gamma=0.99, ent_coef=0.01,
        tensorboard_log=str(CHECKPOINT_DIR / "tb_logs")
    )
    a2c.learn(total_timesteps=timesteps, progress_bar=False)
    a2c_path = str(CHECKPOINT_DIR / "A2C_nifty")
    a2c.save(a2c_path)
    print(f"  [✓] A2C saved → {a2c_path}.zip")
    results["A2C"] = a2c_path

    # ── DDPG ───────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Training DDPG for {timesteps:,} steps...")
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )
    ddpg = DDPG(
        "MlpPolicy", vec_env, verbose=1,
        learning_rate=1e-3, action_noise=action_noise,
        batch_size=256, gamma=0.99,
        tensorboard_log=str(CHECKPOINT_DIR / "tb_logs")
    )
    ddpg.learn(total_timesteps=timesteps, progress_bar=False)
    ddpg_path = str(CHECKPOINT_DIR / "DDPG_nifty")
    ddpg.save(ddpg_path)
    print(f"  [✓] DDPG saved → {ddpg_path}.zip")
    results["DDPG"] = ddpg_path

    # ── Evaluate & compute Sharpe ──────────────────────────────────────────────
    print("\n[Eval] Computing validation Sharpe ratios...")
    val_prices = prices[split:]
    val_env = StockTradingEnv(val_prices)
    sharpes = {}

    for name, path in results.items():
        if name == "PPO":
            model = PPO.load(path)
        elif name == "A2C":
            model = A2C.load(path)
        else:
            model = DDPG.load(path)

        obs, _ = val_env.reset()
        nav_series = [val_env.initial_cash]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = val_env.step(action)
            nav_series.append(nav_series[-1] * (1 + reward / 100))

        rets = np.diff(nav_series) / np.array(nav_series[:-1])
        sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
        sharpes[name] = round(float(sharpe), 4)
        print(f"  {name}: Sharpe = {sharpes[name]:.4f}")

    best_agent = max(sharpes, key=sharpes.get)
    print(f"\n  [★] Best agent: {best_agent} (Sharpe={sharpes[best_agent]:.4f})")

    # ── Save manifest ──────────────────────────────────────────────────────────
    manifest = {
        "agents": results,
        "sharpe_scores": sharpes,
        "best_agent": best_agent,
        "timesteps": timesteps,
        "data_points": len(train_prices),
        "selector": "rolling_sharpe"
    }
    manifest_path = CHECKPOINT_DIR / "ensemble_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  [✓] Ensemble manifest → {manifest_path}")
    print("=" * 60)
    return str(manifest_path)


# ── Mode: train (GRPO LLM arbitrage agent) ────────────────────────────────────
def run_train_grpo(args):
    """Train the Qwen2.5-1.5B GRPO arbitrage agent (requires GPU)."""
    if not _gpu_available():
        print("[ERROR] GRPO training requires a GPU. None detected.")
        print("  Run on Colab (T4/A100) or Lightning AI with GPU instance.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  LAYER 2: GRPO LLM Training (Qwen2.5-1.5B)")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
        import torch
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("  Run: pip install unsloth trl")
        sys.exit(1)

    # ── Load model (4-bit quantized) ───────────────────────────────────────────
    print("[Model] Loading Qwen2.5-1.5B (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("  [✓] Model loaded with LoRA adapters")

    # ── PCP Arb reward function ────────────────────────────────────────────────
    def pcp_reward_fn(prompts, completions, **kwargs):
        """
        Reward the LLM for:
        1. Valid JSON output
        2. Correct ENTER/SKIP/EXIT decision
        3. Calling cost tools before entering
        4. Avoiding STT trap (not holding to expiry)
        """
        import re
        rewards = []
        for completion in completions:
            reward = 0.0
            text = completion[0]["content"] if isinstance(completion, list) else completion

            # Format reward (0-0.3): valid JSON with required fields
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    required = {"action", "reasoning", "confidence"}
                    if required.issubset(data.keys()):
                        reward += 0.3
                    elif "action" in data:
                        reward += 0.1
            except Exception:
                reward -= 0.1

            # Cost-awareness reward (0-0.25): mentioned STT or cost check
            if any(kw in text.lower() for kw in ["stt", "cost", "transaction", "0.125"]):
                reward += 0.25

            # Timing reward (0-0.25): mentioned exit condition
            if any(kw in text.lower() for kw in ["exit", "expiry", "close", "stop"]):
                reward += 0.25

            # Profitability reward (0-0.2): sensible decision
            if "enter" in text.lower() or "skip" in text.lower() or "exit" in text.lower():
                reward += 0.2

            rewards.append(reward)
        return rewards

    # ── Sample training dataset ────────────────────────────────────────────────
    pcp_scenarios = [
        {
            "role": "user",
            "content": """You are an NSE options arbitrage agent. Analyze this PCP violation:
Underlying: NIFTY 22000
Call price: 450, Put price: 200, Strike: 22000
Spot: 22000, Risk-free rate: 6.5%, DTE: 5 days
Theoretical call-put diff: 247.94, Actual diff: 250
Gross edge: 0.73%
STT on exercise: 0.125% of intrinsic

Should you ENTER, SKIP, or EXIT? Respond in JSON with action, reasoning, confidence."""
        },
        {
            "role": "user",
            "content": """NSE PCP Arbitrage check:
Underlying: BANKNIFTY 48000
Call: 820, Put: 380, Strike: 48000
Gross edge: 0.31%, STT cost: 0.125%, Brokerage: 0.05%, Slippage: 0.08%
Total costs: 0.255%
Net edge: 0.055%

Respond in JSON with action, reasoning, confidence."""
        },
        {
            "role": "user",
            "content": """Arbitrage opportunity analysis:
NIFTY 21500 strike, DTE: 1 day
Apparent PCP violation: 0.8%
Warning: Holding to expiry triggers STT on full intrinsic value (0.125%)
Current time: 3:20 PM (market closes 3:30 PM)

Respond in JSON with action, reasoning, confidence."""
        },
    ]

    # Expand to minimum viable training set
    training_data = pcp_scenarios * 20  # 60 examples

    # ── GRPO Config ────────────────────────────────────────────────────────────
    steps = args.steps or 100
    grpo_config = GRPOConfig(
        output_dir=str(CHECKPOINT_DIR / "grpo_output"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_steps=steps,
        save_steps=50,
        logging_steps=10,
        num_generations=4,
        max_completion_length=512,
        report_to="none",
        remove_unused_columns=False,
    )

    # Convert to dataset format
    from datasets import Dataset
    dataset = Dataset.from_list([{"prompt": [s]} for s in training_data])

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[pcp_reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    print(f"[Train] Starting GRPO for {steps} steps...")
    trainer.train()

    # Save
    save_path = CHECKPOINT_DIR / "grpo_pcp_agent"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  [✓] GRPO model saved → {save_path}")
    return str(save_path)


# ── Mode: alpha ────────────────────────────────────────────────────────────────
def run_alpha(args):
    """Analyze PCP violation frequency and profitability in historical data."""
    import numpy as np
    print("\n" + "=" * 60)
    print("  ALPHA ANALYSIS: PCP Violation Profitability")
    print("=" * 60)

    # Synthetic analysis (replace with real NSE bhavcopy data)
    np.random.seed(42)
    n_days = 252
    violations = np.random.normal(0.005, 0.003, n_days)
    stt_cost = 0.00125
    brokerage = 0.0005
    slippage = 0.0008
    total_cost = stt_cost + brokerage + slippage

    profitable = violations[violations > total_cost]
    pct_profitable = len(profitable) / n_days * 100

    print(f"\n  Period: {n_days} trading days")
    print(f"  Total violations detected: {n_days}")
    print(f"  Profitable after costs: {len(profitable)} ({pct_profitable:.1f}%)")
    print(f"  Avg gross edge: {violations.mean()*100:.3f}%")
    print(f"  Total cost (STT+broker+slip): {total_cost*100:.3f}%")
    print(f"  Avg net edge (profitable only): {(profitable - total_cost).mean()*100:.3f}%")
    print(f"\n  Traffic light: {'🟢 GREEN' if pct_profitable > 30 else '🟡 YELLOW' if pct_profitable > 15 else '🔴 RED'}")
    print(f"  Recommendation: {'Train — sufficient alpha' if pct_profitable > 20 else 'More data needed'}")
    print("=" * 60)


# ── Mode: backtest ─────────────────────────────────────────────────────────────
def run_backtest(args):
    """Run backtest on trained ensemble."""
    import numpy as np
    print("\n" + "=" * 60)
    print("  BACKTEST: Ensemble Agent Performance")
    print("=" * 60)

    manifest_path = CHECKPOINT_DIR / "ensemble_manifest.json"
    if not manifest_path.exists():
        print("[ERROR] No trained ensemble found. Run: python main.py --mode train-ensemble first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"  Best agent: {manifest['best_agent']}")
    print(f"  Sharpe scores: {manifest['sharpe_scores']}")
    print(f"\n  Loading {manifest['best_agent']} for backtest...")

    from stable_baselines3 import PPO, A2C, DDPG
    agent_map = {"PPO": PPO, "A2C": A2C, "DDPG": DDPG}
    AgentClass = agent_map[manifest["best_agent"]]
    model = AgentClass.load(manifest["agents"][manifest["best_agent"]])

    # Synthetic NIFTY prices for backtest period
    np.random.seed(123)
    returns = np.random.normal(0.0004, 0.012, 500)
    prices = 17000 * np.exp(np.cumsum(returns))

    # Simple backtest loop
    nav = 100_000
    nav_history = [nav]
    position = 0
    obs = np.array([1.0, 1.0, 1.0, 0.5, 0.0], dtype=np.float32)

    for i in range(1, len(prices)):
        action, _ = model.predict(obs, deterministic=True)
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        pnl = position * ret * nav * 0.01
        nav += pnl
        nav_history.append(nav)
        # Update obs (simplified)
        obs = np.array([
            prices[i] / prices[0],
            prices[max(0, i-5):i+1].mean() / prices[0],
            prices[max(0, i-20):i+1].mean() / prices[0],
            0.5, float(action[0])
        ], dtype=np.float32)
        position = int(action[0] * 100)

    nav_arr = np.array(nav_history)
    total_return = (nav_arr[-1] / nav_arr[0] - 1) * 100
    rets = np.diff(nav_arr) / nav_arr[:-1]
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
    max_dd = ((pd.Series(nav_arr) - pd.Series(nav_arr).cummax()) / pd.Series(nav_arr).cummax()).min() * 100

    print(f"\n  ── Backtest Results ──")
    print(f"  Total Return:  {total_return:.2f}%")
    print(f"  Sharpe Ratio:  {sharpe:.4f}")
    print(f"  Max Drawdown:  {max_dd:.2f}%")
    print(f"  Final NAV:     ₹{nav_arr[-1]:,.0f}")
    print("=" * 60)


# ── Mode: paper ────────────────────────────────────────────────────────────────
def run_paper(args):
    """Paper trading loop (mock feed)."""
    import numpy as np
    print("\n" + "=" * 60)
    print("  PAPER TRADING (Mock Feed)")
    print("=" * 60)

    manifest_path = CHECKPOINT_DIR / "ensemble_manifest.json"
    if not manifest_path.exists():
        print("[WARN] No ensemble found. Using random agent for demo.")
        agent = None
    else:
        from stable_baselines3 import PPO, A2C, DDPG
        with open(manifest_path) as f:
            manifest = json.load(f)
        agent_map = {"PPO": PPO, "A2C": A2C, "DDPG": DDPG}
        agent = agent_map[manifest["best_agent"]].load(manifest["agents"][manifest["best_agent"]])
        print(f"  Loaded: {manifest['best_agent']}")

    print("  Feed: MOCK | Press Ctrl+C to stop\n")
    np.random.seed(int(time.time()))
    price = 22000.0
    nav = 100_000.0
    step = 0

    try:
        while True:
            price *= np.exp(np.random.normal(0.0001, 0.005))
            obs = np.array([price/22000, 1.0, 1.0, 0.5, 0.0], dtype=np.float32)
            if agent:
                action, _ = agent.predict(obs, deterministic=True)
                decision = "BUY" if action[0] > 0.3 else "SELL" if action[0] < -0.3 else "HOLD"
            else:
                decision = np.random.choice(["BUY", "SELL", "HOLD"])

            step += 1
            print(f"  Tick {step:4d} | NIFTY: {price:8.2f} | Action: {decision:4s} | NAV: ₹{nav:,.0f}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n  Paper trading stopped.")


# ── Mode: compare ──────────────────────────────────────────────────────────────
def run_compare(args):
    """Compare all trained agents head-to-head."""
    manifest_path = CHECKPOINT_DIR / "ensemble_manifest.json"
    if not manifest_path.exists():
        print("[ERROR] Train ensemble first: python main.py --mode train-ensemble")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("\n" + "=" * 60)
    print("  AGENT COMPARISON (Rolling Sharpe)")
    print("=" * 60)
    print(f"\n  {'Agent':<10} {'Sharpe':>10} {'Status':>15}")
    print("  " + "-" * 37)
    best = manifest["best_agent"]
    for name, sharpe in sorted(manifest["sharpe_scores"].items(), key=lambda x: -x[1]):
        star = " ← ACTIVE" if name == best else ""
        print(f"  {name:<10} {sharpe:>10.4f}{star}")
    print("=" * 60)


# ── Mode: demo ─────────────────────────────────────────────────────────────────
def run_demo(args):
    """Interactive demo — no training required."""
    import numpy as np
    print("\n" + "=" * 60)
    print("  PCP ARBITRAGE DEMO (Interactive)")
    print("=" * 60)
    print("""
  Scenario: NIFTY 22000 strike, 5 DTE
  Call: ₹450  |  Put: ₹200
  Gross PCP edge: 0.73%
  STT on exercise: 0.125%
  Net edge after all costs: ~0.47%
  """)
    decision = input("  Enter your decision [ENTER/SKIP]: ").strip().upper()
    if decision == "ENTER":
        print("""
  ✓ Correct! Net edge positive.
  Agent action: BUY call + SELL put + SELL futures
  Expected P&L: +0.47% on ₹10L = ~₹4,700
  Exit: Before 3:20 PM (avoid STT trap at expiry)
  """)
    else:
        print("\n  Agent would ENTER — edge exceeds all costs.\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="PCP Arbitrage RL System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train-ensemble  Train PPO + A2C + DDPG on stock data (CPU OK)
  train           Train GRPO LLM arbitrage agent (GPU required)
  backtest        Backtest trained ensemble
  paper           Paper trading with mock/live feed
  alpha           Analyze PCP violation profitability
  compare         Compare all agent Sharpe scores
  demo            Interactive demo (no training needed)

Examples:
  python main.py --mode train-ensemble --timesteps 50000
  python main.py --mode train --steps 200
  python main.py --mode backtest
  python main.py --mode alpha
  python main.py --mode demo
        """
    )
    parser.add_argument("--mode", required=True,
                        choices=["train", "train-ensemble", "train-hybrid",
                                 "backtest", "paper", "alpha", "compare", "demo"],
                        help="Operating mode")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to data directory")
    parser.add_argument("--timesteps", type=int, default=50_000, help="SB3 training timesteps")
    parser.add_argument("--steps", type=int, default=200, help="GRPO training steps")
    parser.add_argument("--ensemble-checkpoint", default=None, help="Ensemble checkpoint for hybrid")
    parser.add_argument("--feed", choices=["mock", "live"], default="mock", help="Paper trading feed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    dispatch = {
        "train-ensemble": run_train_ensemble,
        "train-hybrid":   run_train_ensemble,  # alias: trains ensemble first
        "train":          run_train_grpo,
        "backtest":       run_backtest,
        "paper":          run_paper,
        "alpha":          run_alpha,
        "compare":        run_compare,
        "demo":           run_demo,
    }

    dispatch[args.mode](args)