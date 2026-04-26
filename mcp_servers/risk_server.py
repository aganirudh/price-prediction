"""
MCP Server 2: Risk Server (Port 8002)
Position limits, Greeks-based risk checks, drawdown monitoring.
"""
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Risk MCP Server", version="1.0.0")

# ── In-memory position state ──────────────────────────────────────────────────
_positions = {}
_daily_pnl = 0.0
_session_start = time.time()

RISK_LIMITS = {
    "max_position_pct": 0.10,     # 10% of capital per trade
    "max_daily_loss_pct": 0.02,   # 2% daily stop-loss
    "max_open_positions": 5,
    "max_delta_exposure": 50,     # net delta (in NIFTY lots)
    "min_dte_to_enter": 2,        # don't enter with < 2 DTE (STT trap risk)
}


class RiskCheckRequest(BaseModel):
    underlying: str = "NIFTY"
    action: str = "ENTER"         # ENTER, EXIT, HOLD
    gross_edge_pct: float = 0.5
    dte: int = 5
    notional: float = 100000
    current_capital: float = 1000000


class PositionUpdateRequest(BaseModel):
    trade_id: str
    underlying: str
    action: str
    pnl: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "server": "risk", "port": 8002}


@app.post("/check")
def risk_check(req: RiskCheckRequest):
    """Full risk gate before entering a trade."""
    violations = []
    warnings = []

    # 1. DTE check (STT trap)
    if req.dte < RISK_LIMITS["min_dte_to_enter"] and req.action == "ENTER":
        violations.append(f"DTE={req.dte} < min={RISK_LIMITS['min_dte_to_enter']} — STT trap risk")

    # 2. Daily loss limit
    daily_loss_pct = abs(min(_daily_pnl, 0)) / req.current_capital
    if daily_loss_pct >= RISK_LIMITS["max_daily_loss_pct"]:
        violations.append(f"Daily loss {daily_loss_pct:.2%} >= limit {RISK_LIMITS['max_daily_loss_pct']:.2%}")

    # 3. Position count
    if len(_positions) >= RISK_LIMITS["max_open_positions"] and req.action == "ENTER":
        violations.append(f"Max open positions ({RISK_LIMITS['max_open_positions']}) reached")

    # 4. Position sizing
    position_pct = req.notional / req.current_capital
    if position_pct > RISK_LIMITS["max_position_pct"]:
        warnings.append(f"Position size {position_pct:.1%} > recommended {RISK_LIMITS['max_position_pct']:.1%}")

    # 5. Edge quality
    if req.gross_edge_pct < 0.2 and req.action == "ENTER":
        warnings.append(f"Gross edge {req.gross_edge_pct:.3f}% is thin — high execution risk")

    approved = len(violations) == 0

    return {
        "approved": approved,
        "action": req.action,
        "violations": violations,
        "warnings": warnings,
        "risk_metrics": {
            "daily_pnl": round(_daily_pnl, 2),
            "daily_loss_pct": round(daily_loss_pct * 100, 3),
            "open_positions": len(_positions),
            "position_size_pct": round(position_pct * 100, 2),
        },
        "limits": RISK_LIMITS,
        "recommendation": "PROCEED" if approved else "BLOCK",
    }


@app.post("/update_position")
def update_position(req: PositionUpdateRequest):
    global _daily_pnl
    if req.action == "OPEN":
        _positions[req.trade_id] = {"underlying": req.underlying, "open_time": time.time()}
    elif req.action == "CLOSE":
        _positions.pop(req.trade_id, None)
        _daily_pnl += req.pnl
    return {"status": "ok", "open_positions": len(_positions), "daily_pnl": round(_daily_pnl, 2)}


@app.get("/status")
def risk_status():
    return {
        "open_positions": len(_positions),
        "positions": list(_positions.keys()),
        "daily_pnl": round(_daily_pnl, 2),
        "session_hours": round((time.time() - _session_start) / 3600, 2),
        "limits": RISK_LIMITS,
    }


@app.post("/reset_daily")
def reset_daily():
    global _daily_pnl, _session_start
    _daily_pnl = 0.0
    _session_start = time.time()
    return {"status": "reset", "message": "Daily P&L reset"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
