import os
import time
import requests
import subprocess
import sys
from pathlib import Path

# Assume main.py and other necessary modules are in the same directory or accessible
# You might need to adjust these imports based on your project structure
# For example, if main.py is in the root and servers are in a 'servers' directory:
# from servers.market_data_server import app as market_data_app
# from servers.risk_server import app as risk_app
# from servers.cost_server import app as cost_app
# import main as training_main # Assuming your main logic is in main.py

# --- Configuration ---
# Use environment variable for checkpoint directory, default to 'checkpoints' if not set
# This will be set to /teamspace/studios/this_studio/checkpoints when running on Lightning
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Ensure directory exists

# Define servers with their names, modules, ports, and any required arguments
# Ensure these server names match your actual FastAPI server files/apps
SERVERS = [
    {"name": "Market Data Server", "module": "market_data_server", "port": 8001, "command": ["uvicorn", "market_data_server:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]},
    {"name": "Risk Server", "module": "risk_server", "port": 8002, "command": ["uvicorn", "risk_server:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info"]},
    {"name": "Cost Server", "module": "cost_server", "port": 8003, "command": ["uvicorn", "cost_server:app", "--host", "0.0.0.0", "--port", "8003", "--log-level", "info"]},
]

def start_servers_with_health_check():
    """
    Starts the MCP servers and waits for them to be ready using health checks.
    """
    print("Starting MCP servers...")
    processes = []
    for server in SERVERS:
        print(f"Starting {server['name']} on port {server['port']}...")
        # Use subprocess.Popen to start servers in the background
        # stdout/stderr are captured or redirected if needed for debugging
        process = subprocess.Popen(
            server["command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True # Ensure output is text
        )
        processes.append((server, process))

    print("Waiting for servers to become ready...")
    max_wait_time = 30  # seconds
    for server, process in processes:
        server_ready = False
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                # Use localhost as the servers are running in the same Studio environment
                health_url = f"http://localhost:{server['port']}/health"
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    print(f"  {server['name']} is ready.")
                    server_ready = True
                    break
            except requests.exceptions.RequestException:
                # Server not ready yet, continue waiting
                time.sleep(1)
            except Exception as e:
                print(f"  Error checking health for {server['name']}: {e}")
                time.sleep(1)

        if not server_ready:
            print(f"Error: {server['name']} did not become ready within {max_wait_time} seconds.")
            # Optionally, you could kill other processes or exit here
            # For now, we'll just warn and continue to the next server

    if not all(server_ready for server, _ in processes):
        print("One or more servers failed to start. Please check logs.")
        # Depending on requirements, you might want to exit or raise an exception

def run_training(mode, timesteps=None, data_dir=None):
    """
    Main training orchestrator.
    """
    print(f"Starting training with mode: {mode}")

    # Load WandB API Key from environment variables
    # This assumes you've set WANDB_API_KEY in your Lightning Studio secrets
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        print("Warning: WANDB_API_KEY not found. WandB logging may fail.")
        # If WandB is critical, you might want to stop here or ensure --no-wandb is used if not available.
        # For this example, we assume it should be enabled if the key is present.

    # --- Step 1: Ensemble Training ---
    if mode == "train-ensemble" or mode == "train-all":
        print("\n--- Starting Ensemble Training (Layer 1) ---")
        if not data_dir:
            # Default data directory if not provided
            data_dir = "data/kaggle"
        
        ensemble_command = [
            sys.executable, "main.py",
            "--mode", "train-ensemble",
            f"--data-dir={data_dir}",
            f"--timesteps={timesteps}" if timesteps else "--timesteps=50000", # Default timesteps if not specified
        ]
        
        # Removed --no-wandb flag
        # Check if wandb_api_key is set to decide on wandb args.
        # Assuming main.py handles wandb flags based on env var or command-line args.
        # If not, you might need to pass --wandb explicitly or remove --no-wandb.
        # For now, we assume it's handled by removing --no-wandb.
        
        # If you need to explicitly pass wandb args, you might add them like:
        # if wandb_api_key:
        #     ensemble_command.extend(["--wandb"]) # Or other wandb-related args main.py expects
        
        print(f"Running ensemble training with command: {' '.join(ensemble_command)}")
        # Execute ensemble training
        # If main.py runs servers, this part might be different.
        # Assuming main.py is the script that orchestrates training steps.
        try:
            # This subprocess call might need adjustment if main.py also starts servers.
            # If main.py handles server startup, this might just be a direct call.
            # For now, assuming main.py is called to perform the training step.
            # We'll start the servers separately before calling main.py if needed.
            
            # Let's assume servers are started *before* main.py is called for training steps
            # Or if main.py *internally* calls the servers, this is fine.
            # For now, we'll call `main.py` directly for its training logic.
            
            # To run the servers, we'll use start_servers_with_health_check()
            # before calling the main training logic.

            # If main.py is supposed to run the servers itself, then this part needs adjustment.
            # Assuming for now, we call main.py for the training *logic* and manage servers separately.

            # If main.py needs to access the servers, they need to be running *before* it's called.
            # Let's adjust the flow to start servers first.
            
            # This section needs clarification based on how main.py is structured.
            # For now, let's assume the actual training execution will be done in a separate step after server setup.
            print("Executing ensemble training logic via main.py...")
            # Example: Replace with actual execution if needed
            # subprocess.run(ensemble_command, check=True)
            # For now, just printing the command as a placeholder for actual execution.
            print(f"Simulating execution of: {' '.join(ensemble_command)}")
            print("Ensemble training step simulated.")

        except Exception as e:
            print(f"Error during ensemble training: {e}")
            # sys.exit(1) # Exit if ensemble training fails

    # --- Step 2: Hybrid Training (LLM Arbitrage Agent) ---
    if mode == "train-hybrid" or mode == "train-all":
        print("\n--- Starting Hybrid Training (Layer 2 - LLM Agent) ---")
        
        # Ensure GPU is available for this step
        if not torch.cuda.is_available():
            print("Error: GPU not available for Layer 2 training. Please switch to a GPU instance.")
            # sys.exit(1) # Exit if GPU is required but not found
            # For now, just printing the error.

        hybrid_command = [
            sys.executable, "main.py",
            "--mode", "train-hybrid",
            "--data-dir", data_dir if data_dir else "data/kaggle", # Use provided or default data dir
            f"--checkpoint-dir={CHECKPOINT_DIR}", # Use the persistent storage path
            # Removed --no-wandb flag
            # Add --wandb if needed and API key is set.
        ]
        
        # Add GRPO-specific args if needed, e.g., model name, fine-tuning params
        # e.g., "--model-name", "Qwen2.5-1.5B-4bit", "--fine-tune-args", "..."

        print(f"Running hybrid training with command: {' '.join(hybrid_command)}")
        try:
            print("Executing hybrid training logic via main.py...")
            # Example: Replace with actual execution if needed
            # subprocess.run(hybrid_command, check=True)
            # For now, just printing the command as a placeholder for actual execution.
            print(f"Simulating execution of: {' '.join(hybrid_command)}")
            print("Hybrid training step simulated.")
        except Exception as e:
            print(f"Error during hybrid training: {e}")
            # sys.exit(1) # Exit if hybrid training fails

# --- Main execution logic for colab_train.py ---
if __name__ == "__main__":
    import argparse
    
    # Mocking torch.cuda.is_available for demonstration if torch isn't fully set up yet
    # In a real Studio with GPU, this will work correctly.
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. GRPO training might fail if GPU is required.")
    except ImportError:
        print("Warning: PyTorch not installed. CUDA availability check will be skipped.")
        torch = type('obj', (object,), {'cuda': type('obj', (object,), {'is_available': lambda: False})})()


    parser = argparse.ArgumentParser(description="Training script for RL trading system.")
    parser.add_argument("--mode", type=str, default="train-all", choices=["train-ensemble", "train-hybrid", "train-all"], help="Training mode: train-ensemble, train-hybrid, or train-all")
    parser.add_argument("--timesteps", type=int, help="Number of timesteps for ensemble training")
    parser.add_argument("--data-dir", type=str, default="data/kaggle", help="Directory for dataset")
    # Add other arguments your main.py script might need

    args = parser.parse_args()

    # Start servers first
    start_servers_with_health_check()

    # Then run the training logic based on mode
    run_training(mode=args.mode, timesteps=args.timesteps, data_dir=args.data_dir)

    print("\nTraining script finished. Check logs for details.")
    # In a real scenario, you might want to keep servers running if they are needed for inference.
    # For training, they might be stopped after the training step.