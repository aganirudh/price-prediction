"""
PCP Arbitrage RL — Colab/Lightning AI Orchestrator
====================================================
Run: python colab_train.py --step all
Or step by step: --step install / data / ensemble / grpo / backtest / push_hf
"""
import os, sys, subprocess, time, argparse, json, requests
from pathlib import Path

ROOT = Path(__file__).parent
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", ROOT / "checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd, check=True):
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str))
    if check and result.returncode != 0:
        print(f"[ERROR] Exit code {result.returncode}")
    return result.returncode == 0


def _wait_for_server(url, name, timeout=30):
    print(f"  Waiting for {name}...", end="", flush=True)
    for _ in range(timeout):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                print(" OK"); return True
        except Exception:
            pass
        print(".", end="", flush=True); time.sleep(1)
    print(" TIMEOUT"); return False


def step_install():
    print("\n=== STEP 1: Install Dependencies ===")
    run([sys.executable, "-m", "pip", "install", "-q",
         "stable-baselines3>=2.3.0", "gymnasium>=0.29.0", "shimmy>=1.3.0",
         "yfinance", "pandas", "numpy", "rich", "fastapi", "uvicorn",
         "requests", "huggingface_hub", "wandb"])

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "torch"])
        import torch; has_gpu = torch.cuda.is_available()

    if has_gpu:
        print(f"  GPU: {torch.cuda.get_device_name(0)} — installing LLM deps...")
        run('pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
            check=False)
        run([sys.executable, "-m", "pip", "install", "-q",
             "trl>=0.8.6", "transformers>=4.40.0", "peft>=0.10.0",
             "accelerate", "bitsandbytes>=0.43.0", "datasets"], check=False)

    print("  [OK] Dependencies installed")
    return True


def step_data():
    print("\n=== STEP 2: Download NIFTY50 Data ===")
    data_dir = ROOT / "data" / "kaggle"
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
        import yfinance as yf, pandas as pd

    tickers = {
        "^NSEI": "NIFTY50_index", "RELIANCE.NS": "Reliance",
        "TCS.NS": "TCS", "HDFCBANK.NS": "HDFC_Bank",
        "INFY.NS": "Infosys", "ICICIBANK.NS": "ICICI_Bank",
    }
    downloaded = 0
    for ticker, name in tickers.items():
        out = data_dir / f"{name}.csv"
        if out.exists():
            print(f"  [skip] {name}"); downloaded += 1; continue
        try:
            df = yf.download(ticker, start="2018-01-01", end="2024-12-31", progress=False)
            if len(df) > 0:
                df.to_csv(out); print(f"  [OK] {name}: {len(df)} rows"); downloaded += 1
        except Exception as e:
            print(f"  [!] {name}: {e}")

    print(f"  Downloaded {downloaded}/{len(tickers)} files")
    return downloaded > 0


def step_ensemble(timesteps=50000):
    print("\n=== STEP 3: Train Ensemble (PPO+A2C+DDPG) ===")
    ok = run([sys.executable, str(ROOT/"main.py"),
              "--mode", "train-ensemble",
              "--data-dir", str(ROOT/"data"/"kaggle"),
              "--timesteps", str(timesteps),
              "--no-wandb"])
    if ok:
        manifest = CHECKPOINT_DIR / "ensemble_manifest.json"
        if manifest.exists():
            m = json.load(open(manifest))
            print(f"  Best: {m['best_agent']} | Sharpe: {m['sharpe_scores']}")
    return ok


def step_grpo(steps=300):
    print("\n=== STEP 4: Train GRPO LLM Agent ===")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [SKIP] No GPU. Enable T4 in Colab: Runtime → Change runtime type"); return False
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  [SKIP] torch not installed"); return False

    return run([sys.executable, str(ROOT/"main.py"),
                "--mode", "train", "--steps", str(steps), "--no-wandb"])


def step_backtest():
    print("\n=== STEP 5: Backtest ===")
    return run([sys.executable, str(ROOT/"main.py"), "--mode", "backtest", "--no-wandb"])


def step_push_hf(hf_token=None, hf_user="aganirudh"):
    print("\n=== STEP 6: Push to Hugging Face ===")
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import HfApi, login

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("  Set HF_TOKEN env var. Get at: https://huggingface.co/settings/tokens"); return False

    login(token=token)
    api = HfApi()

    # Upload ensemble checkpoints
    repo = f"{hf_user}/pcp-arb-ensemble"
    api.create_repo(repo_id=repo, exist_ok=True)
    api.upload_folder(folder_path=str(CHECKPOINT_DIR), repo_id=repo, repo_type="model",
                      ignore_patterns=["grpo_pcp_agent/**", "tb_logs/**"])
    print(f"  [OK] Ensemble → https://huggingface.co/{repo}")

    # Upload GRPO if trained
    grpo = CHECKPOINT_DIR / "grpo_pcp_agent"
    if grpo.exists():
        grepo = f"{hf_user}/pcp-arb-grpo-qwen2.5"
        api.create_repo(repo_id=grepo, exist_ok=True)
        api.upload_folder(folder_path=str(grpo), repo_id=grepo, repo_type="model")
        print(f"  [OK] GRPO → https://huggingface.co/{grepo}")

    # Upload app.py as a Space
    app = ROOT / "app.py"
    if app.exists():
        space = f"{hf_user}/pcp-arb-demo"
        api.create_repo(repo_id=space, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_file(path_or_fileobj=str(app), path_in_repo="app.py",
                        repo_id=space, repo_type="space")
        reqs = ROOT / "requirements.txt"
        if reqs.exists():
            api.upload_file(path_or_fileobj=str(reqs), path_in_repo="requirements.txt",
                            repo_id=space, repo_type="space")
        print(f"  [OK] Demo → https://huggingface.co/spaces/{space}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCP Arb RL Pipeline")
    parser.add_argument("--step", default="all",
                        choices=["install","data","ensemble","grpo","backtest","push_hf","all"])
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--grpo-steps", type=int, default=300)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-user", default="aganirudh")
    args = parser.parse_args()

    steps_map = {
        "install":  step_install,
        "data":     step_data,
        "ensemble": lambda: step_ensemble(args.timesteps),
        "grpo":     lambda: step_grpo(args.grpo_steps),
        "backtest": step_backtest,
        "push_hf":  lambda: step_push_hf(args.hf_token, args.hf_user),
    }

    pipeline = list(steps_map.keys())[:-1] if args.step == "all" else [args.step]

    print("\n" + "="*60)
    print("  PCP Arbitrage RL — Training Pipeline")
    print(f"  Steps: {' → '.join(pipeline)}")
    print("="*60)

    results = {}
    for step in pipeline:
        ok = steps_map[step]()
        results[step] = "OK" if ok else "FAIL"
        if not ok and step == "ensemble":
            print(f"Critical step failed. Stopping."); break

    print("\n=== SUMMARY ===")
    for s, r in results.items():
        print(f"  {'✅' if r=='OK' else '❌'} {s}")