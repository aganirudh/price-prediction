from fastapi import FastAPI
import uvicorn
import time
import requests # Make sure requests is installed

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/cost")
def get_cost():
    # Placeholder for actual cost calculation logic (e.g., transaction fees)
    return {"transaction_cost_factor": 0.0001, "slippage_estimate": 0.0005}

if __name__ == "__main__":
    print("Starting Cost Server on port 8003")
    # uvicorn.run(app, host="0.0.0.0", port=8003)