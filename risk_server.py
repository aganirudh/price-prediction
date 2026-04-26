from fastapi import FastAPI
import uvicorn
import time
import requests # Make sure requests is installed

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/risk")
def get_risk():
    # Placeholder for actual risk assessment logic
    return {"portfolio_risk": 0.05, "option_risk_metrics": "..."}

if __name__ == "__main__":
    print("Starting Risk Server on port 8002")
    # uvicorn.run(app, host="0.0.0.0", port=8002)