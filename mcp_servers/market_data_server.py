"""
MCP Server 1: Market Data Server (Port 8001)
Provides real-time and historical NIFTY/BANKNIFTY option chain data.
"""
import asyncio
import math
import random
import time
from datetime import datetime, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Market Data MCP Server", version="1.0.0")


# ── Data Models ───────────────────────────────────────────────────────────────
class OptionChainRequest(BaseModel):
    underlying: str = "NIFTY"
    strike: float = 22000
    expiry_days: int = 5


class PCPCheckRequest(BaseModel):
    underlying: str = "NIFTY"
    strike: float = 22000
    expiry_days: int = 5


# ── Mock Market State ─────────────────────────────────────────────────────────
_state = {
    "NIFTY": {"spot": 22000.0, "last_update": time.time()},
    "BANKNIFTY": {"spot": 48000.0, "last_update": time.time()},
}


def _gbm_price(base: float, dt: float = 1/252) -> float:
    """Single GBM step."""
    mu, sigma = 0.12, 0.18
    return base * math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * random.gauss(0, 1))


def _bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    from math import log, sqrt, exp
    try:
        import scipy.stats as st
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * st.norm.cdf(d1) - K * exp(-r * T) * st.norm.cdf(d2)
    except ImportError:
        # Simple approximation without scipy
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        # Approximation of N(x)
        def N(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        return S * N(d1) - K * math.exp(-r * T) * N(d2)


def _get_option_chain(underlying: str, strike: float, dte: int):
    """Generate realistic option chain with occasional PCP violations."""
    state = _state.get(underlying, {"spot": 22000.0})
    spot = state["spot"]

    # Occasionally inject GBM drift
    spot = _gbm_price(spot, dt=1/(252*78))  # 1 tick (5-min bar)
    _state[underlying]["spot"] = spot

    T = dte / 365.0
    r = 0.065
    sigma = 0.18 + random.gauss(0, 0.02)  # vol uncertainty

    call_bs = _bs_call(spot, strike, T, r, sigma)
    put_bs = call_bs - spot + strike * math.exp(-r * T)

    # Inject PCP violation occasionally (20% of ticks)
    violation = 0.0
    if random.random() < 0.20:
        violation = random.uniform(0.001, 0.010) * spot
        if random.random() < 0.5:
            call_bs += violation
        else:
            put_bs -= violation

    # Add bid-ask spread
    spread_call = call_bs * 0.002
    spread_put = put_bs * 0.002

    return {
        "underlying": underlying,
        "spot": round(spot, 2),
        "strike": strike,
        "expiry_days": dte,
        "timestamp": datetime.now().isoformat(),
        "call": {
            "bid": round(call_bs - spread_call, 2),
            "ask": round(call_bs + spread_call, 2),
            "mid": round(call_bs, 2),
            "iv": round(sigma * 100, 2),
            "delta": round(0.5 + (spot - strike) / (spot * sigma * math.sqrt(T + 0.01)), 3),
        },
        "put": {
            "bid": round(put_bs - spread_put, 2),
            "ask": round(put_bs + spread_put, 2),
            "mid": round(put_bs, 2),
            "iv": round((sigma + 0.01) * 100, 2),
            "delta": round(-0.5 + (spot - strike) / (spot * sigma * math.sqrt(T + 0.01)), 3),
        },
        "pcp_violation_pct": round(violation / spot * 100, 4),
        "theoretical_cp_diff": round(spot - strike * math.exp(-r * T), 2),
        "actual_cp_diff": round(call_bs - put_bs, 2),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "server": "market_data", "port": 8001}


@app.get("/spot/{underlying}")
def get_spot(underlying: str):
    state = _state.get(underlying.upper(), {"spot": 22000.0})
    spot = _gbm_price(state["spot"], dt=1/(252*78))
    _state[underlying.upper()] = {"spot": spot, "last_update": time.time()}
    return {"underlying": underlying.upper(), "spot": round(spot, 2), "timestamp": datetime.now().isoformat()}


@app.post("/option_chain")
def get_option_chain(req: OptionChainRequest):
    return _get_option_chain(req.underlying.upper(), req.strike, req.expiry_days)


@app.post("/pcp_check")
def pcp_check(req: PCPCheckRequest):
    chain = _get_option_chain(req.underlying.upper(), req.strike, req.expiry_days)
    gross_edge = chain["pcp_violation_pct"]
    return {
        "underlying": req.underlying,
        "strike": req.strike,
        "gross_edge_pct": gross_edge,
        "has_violation": gross_edge > 0.1,
        "chain": chain,
    }


@app.get("/market_open")
def market_open():
    now = datetime.now()
    # NSE hours: 9:15 AM – 3:30 PM IST Mon–Fri
    is_open = (
        now.weekday() < 5 and
        (9 * 60 + 15) <= (now.hour * 60 + now.minute) <= (15 * 60 + 30)
    )
    minutes_to_close = max(0, 15 * 60 + 30 - (now.hour * 60 + now.minute))
    return {
        "is_open": is_open,
        "current_time": now.strftime("%H:%M:%S"),
        "minutes_to_close": minutes_to_close,
        "stt_danger_zone": minutes_to_close <= 10,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
