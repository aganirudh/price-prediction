"""
MCP Server 3: Cost Server (Port 8003)
Computes all-in transaction costs for NSE options arbitrage.
The STT trap is the #1 reason apparent arb opportunities are unprofitable.
"""
import math
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Cost MCP Server", version="1.0.0")


class CostRequest(BaseModel):
    underlying: str = "NIFTY"
    strike: float = 22000
    spot: float = 22000
    call_price: float = 450
    put_price: float = 200
    expiry_days: int = 5
    notional: float = 100000  # ₹1L notional
    hold_to_expiry: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "server": "cost", "port": 8003}


@app.post("/compute_costs")
def compute_costs(req: CostRequest):
    """
    Full cost breakdown for a PCP arbitrage trade.
    
    Strategy: BUY call + SELL put + SELL futures (or vice versa)
    4 legs total.
    """
    intrinsic = max(req.spot - req.strike, 0)
    T = req.expiry_days / 365.0

    # ── STT ────────────────────────────────────────────────────────────────────
    # NSE STT rates (2024):
    # Options buy: 0.0625% of premium
    # Options sell (on exercise): 0.125% of intrinsic value  ← THE TRAP
    # Futures: 0.01% of turnover
    stt_buy_call = req.call_price * 0.000625      # on premium
    stt_sell_put = req.put_price * 0.000625       # on premium
    stt_futures = req.spot * 0.0001               # futures leg
    stt_exercise_trap = intrinsic * 0.00125 if req.hold_to_expiry else 0.0

    total_stt = stt_buy_call + stt_sell_put + stt_futures + stt_exercise_trap

    # ── Brokerage ──────────────────────────────────────────────────────────────
    # Zerodha: ₹20/order or 0.03% whichever lower, flat for options
    brokerage_per_leg = 20  # ₹20 per leg
    total_brokerage = brokerage_per_leg * 4  # 4 legs

    # ── Exchange charges ───────────────────────────────────────────────────────
    nse_charge_options = (req.call_price + req.put_price) * 0.00053
    nse_charge_futures = req.spot * 0.0002
    total_exchange = nse_charge_options + nse_charge_futures

    # ── GST ───────────────────────────────────────────────────────────────────
    gst = (total_brokerage + total_exchange) * 0.18

    # ── SEBI charges ──────────────────────────────────────────────────────────
    sebi = req.notional * 0.000001

    # ── Slippage (bid-ask) ─────────────────────────────────────────────────────
    slippage_call = req.call_price * 0.002  # 0.2% of premium
    slippage_put = req.put_price * 0.002
    slippage_futures = req.spot * 0.0001
    total_slippage = slippage_call + slippage_put + slippage_futures

    # ── Total ──────────────────────────────────────────────────────────────────
    total_costs = total_stt + total_brokerage + total_exchange + gst + sebi + total_slippage
    total_costs_pct = total_costs / req.spot * 100

    return {
        "underlying": req.underlying,
        "cost_breakdown": {
            "stt_buy_call": round(stt_buy_call, 4),
            "stt_sell_put": round(stt_sell_put, 4),
            "stt_futures": round(stt_futures, 4),
            "stt_exercise_trap": round(stt_exercise_trap, 4),
            "total_stt": round(total_stt, 4),
            "brokerage": round(total_brokerage, 4),
            "exchange_charges": round(total_exchange, 4),
            "gst": round(gst, 4),
            "sebi": round(sebi, 4),
            "slippage": round(total_slippage, 4),
        },
        "total_cost_inr": round(total_costs, 2),
        "total_cost_pct": round(total_costs_pct, 4),
        "stt_trap_active": req.hold_to_expiry and intrinsic > 0,
        "stt_trap_cost_pct": round(stt_exercise_trap / req.spot * 100, 4),
        "minimum_edge_needed_pct": round(total_costs_pct + 0.05, 4),  # + 5bps profit margin
        "warning": (
            "⚠️ STT TRAP: Holding ITM options to expiry triggers 0.125% STT on intrinsic value"
            if req.hold_to_expiry and intrinsic > 0
            else "✅ No STT trap risk (not holding to expiry)"
        )
    }


@app.get("/quick_cost/{underlying}/{gross_edge_pct}")
def quick_cost_check(underlying: str, gross_edge_pct: float):
    """Quick profitability check given a gross edge percentage."""
    # Typical all-in costs for NSE options arb
    typical_stt = 0.125
    typical_brokerage = 0.05
    typical_slippage = 0.08
    total = typical_stt + typical_brokerage + typical_slippage

    net = gross_edge_pct - total
    return {
        "gross_edge_pct": gross_edge_pct,
        "typical_total_cost_pct": total,
        "net_edge_pct": round(net, 4),
        "profitable": net > 0,
        "decision": "ENTER" if net > 0.1 else "MARGINAL" if net > 0 else "SKIP"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
