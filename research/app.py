from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data_sources import refresh_all
from .db import init_db, list_sources, save_backtest
from .model import PUBLIC_PRIORS, assess_risk, run_public_backtest

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
MODEL_VERSION = "0.1.0"
app = FastAPI(title="Risk Pulse Lab", version=MODEL_VERSION)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
BACKTEST = None
REFRESH_LOCK = threading.Lock()

class AssessmentRequest(BaseModel):
    objective_value: float = Field(gt=0)
    exposure: float = Field(gt=0)
    archetype: str = "conduct_life"
    control_strength: float = Field(default=50, ge=0, le=100)
    change_intensity: float = Field(default=50, ge=0, le=100)
    recent_events: int | None = Field(default=None, ge=0)
    typical_impact: float | None = Field(default=None, ge=0)
    materiality_pct: float = Field(default=5.0, gt=0, le=100)
    unit: str = "IDR"

@app.on_event("startup")
def startup():
    global BACKTEST
    init_db(); BACKTEST = run_public_backtest()
    try: save_backtest(BACKTEST, MODEL_VERSION)
    except Exception: pass
    threading.Thread(target=_safe_refresh, daemon=True).start()

def _safe_refresh():
    if not REFRESH_LOCK.acquire(blocking=False): return
    try: refresh_all()
    finally: REFRESH_LOCK.release()

@app.get("/")
def index(): return FileResponse(STATIC / "index.html")

@app.get("/api/health")
def health(): return {"status":"ok","model_version":MODEL_VERSION,"time":datetime.now(timezone.utc).isoformat()}

@app.get("/api/model-info")
def model_info():
    return {
        "name":"Risk Pulse Lab","version":MODEL_VERSION,
        "structure":["Objective / KPI reference","Exposure","Public or scenario prior","Company evidence (recent events)","Transparent control + change scenario adjustment","Gamma-Poisson posterior frequency","Severity distribution","Posterior predictive Monte Carlo","Objective-at-Risk / materiality probability"],
        "priors":PUBLIC_PRIORS,
        "guardrails":["No black-box number: assumptions and evidence mode are shown.","No false precision: scenario-only priors and missing severity evidence force a low-confidence label.","No model without challenger: ML must beat a simple temporal benchmark before promotion.","Complaint and breach databases are not treated as unbiased operational-loss samples."],
    }

@app.get("/api/backtest")
def backtest(): return BACKTEST or run_public_backtest()

@app.get("/api/sources")
def sources(): return {"sources":list_sources()}

@app.post("/api/refresh")
def trigger_refresh(background_tasks: BackgroundTasks):
    background_tasks.add_task(_safe_refresh)
    return {"status":"queued","message":"Public-source refresh queued. Existing observations remain available while the refresh runs."}

@app.post("/api/assess")
def assess(req: AssessmentRequest):
    if req.archetype not in PUBLIC_PRIORS: raise HTTPException(status_code=400, detail="Unknown risk archetype")
    return assess_risk(req.model_dump())
