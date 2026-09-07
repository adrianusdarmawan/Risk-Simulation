from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "fca_product_panel.csv"

# Empirical frequency anchors available from FCA contextualised complaints data.
# These are complaint-frequency proxies, NOT operational-loss frequencies.
PUBLIC_PRIORS = {
    "conduct_life": {
        "label": "Life insurance conduct / servicing",
        "rate_per_1000": 0.8,
        "effective_exposure": 5000,
        "evidence": "FCA contextualised complaints: whole of life / term assurance / critical illness, 2025 H2",
        "confidence": "moderate",
        "mode": "evidence-backed frequency proxy",
    },
    "conduct_general": {
        "label": "General insurance conduct / servicing",
        "rate_per_1000": 1.3,
        "effective_exposure": 5000,
        "evidence": "FCA contextualised complaints: other general insurance, 2025 H2",
        "confidence": "moderate",
        "mode": "evidence-backed frequency proxy",
    },
    "process": {
        "label": "Process / human error",
        "rate_per_1000": 0.5,
        "effective_exposure": 500,
        "evidence": "Scenario prior only; replace with internal near-miss / event data when available",
        "confidence": "low",
        "mode": "scenario prior",
    },
    "cyber": {
        "label": "Technology / cyber incident",
        "rate_per_1000": 0.2,
        "effective_exposure": 250,
        "evidence": "VCDB informs causal patterns, but has no unbiased company-exposure denominator; absolute rate remains scenario-based",
        "confidence": "low",
        "mode": "scenario prior",
    },
}


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.maximum(0, pred)
    denom = np.abs(y) + np.abs(pred) + 1e-9
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "smape_pct": float(np.mean(2 * np.abs(pred - y) / denom) * 100),
        "wape_pct": float(np.sum(np.abs(pred - y)) / np.sum(np.abs(y)) * 100),
        "log_mae": float(mean_absolute_error(np.log1p(y), np.log1p(pred))),
        "log_r2": float(r2_score(np.log1p(y), np.log1p(pred))),
    }


def run_public_backtest() -> dict[str, Any]:
    """Pilot one-step temporal backtest on FCA product-level complaints panel."""
    df = pd.read_csv(DATA_PATH)

    def features(prev2: str, prev: str):
        return pd.DataFrame({
            "group": df["group"],
            "log_prev": np.log1p(df[prev]),
            "log_prev2": np.log1p(df[prev2]),
            "log_ratio": np.log((df[prev] + 10) / (df[prev2] + 10)),
        })

    x_train = features("2024H1", "2024H2")
    x_test = features("2024H2", "2025H1")
    y_train_log = np.log1p(df["2025H1"].to_numpy())
    y_test = df["2025H2"].to_numpy(dtype=float)

    persistence = df["2025H1"].to_numpy(dtype=float)
    half_trend = np.maximum(0, df["2025H1"].to_numpy(dtype=float) + 0.5 * (df["2025H1"].to_numpy(dtype=float) - df["2024H2"].to_numpy(dtype=float)))

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["group"]),
        ("num", StandardScaler(), ["log_prev", "log_prev2", "log_ratio"]),
    ])
    candidates = {
        "Ridge": Ridge(alpha=10),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=2, learning_rate=0.05, loss="huber"),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=300, min_samples_leaf=2, max_features=0.8),
    }

    output = {"Persistence": _metrics(y_test, persistence), "Half-trend": _metrics(y_test, half_trend)}
    for name, estimator in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        pipe.fit(x_train, y_train_log)
        pred = np.maximum(0, np.expm1(pipe.predict(x_test)))
        output[name] = _metrics(y_test, pred)

    ranked = sorted(output.items(), key=lambda kv: kv[1]["wape_pct"])
    winner = ranked[0][0]
    best_ml = min((x for x in ranked if x[0] not in {"Persistence", "Half-trend"}), key=lambda kv: kv[1]["wape_pct"])
    promoted = best_ml[1]["wape_pct"] < output["Persistence"]["wape_pct"]

    return {
        "dataset": "FCA product-level complaints panel",
        "rows": int(len(df)),
        "train_window": "2024 H1 + 2024 H2 -> 2025 H1",
        "test_window": "2024 H2 + 2025 H1 -> 2025 H2",
        "target": "Complaint count by product (frequency proxy)",
        "metrics": output,
        "winner": winner,
        "best_ml": best_ml[0],
        "ml_promoted": promoted,
        "decision": "ML challenger NOT promoted: the simple persistence benchmark is more accurate on this pilot temporal holdout." if not promoted else "ML challenger promoted for the frequency proxy only.",
        "limitations": [
            "Only one out-of-time half-year holdout is available in the bundled product panel.",
            "Complaint counts are a conduct/service frequency proxy, not operational-loss amounts.",
            "The result does not validate severity, tail loss, or Objective-at-Risk.",
            "Older complaint history has structural breaks (including PPI), so regime-aware validation is required before pooling long history.",
        ],
    }


def _control_change_multiplier(control_strength: float, change_intensity: float) -> float:
    control = np.clip(control_strength, 0, 100)
    change = np.clip(change_intensity, 0, 100)
    log_mult = -0.65 * ((control - 50) / 50) + 0.55 * ((change - 50) / 50)
    return float(np.clip(np.exp(log_mult), 0.35, 2.8))


def assess_risk(payload: dict[str, Any], simulations: int = 12000) -> dict[str, Any]:
    archetype = payload.get("archetype", "conduct_life")
    prior = PUBLIC_PRIORS.get(archetype, PUBLIC_PRIORS["conduct_life"])
    objective = max(float(payload.get("objective_value", 1)), 1e-9)
    exposure = max(float(payload.get("exposure", 1)), 1)
    control = float(payload.get("control_strength", 50))
    change = float(payload.get("change_intensity", 50))
    recent_raw = payload.get("recent_events", None)
    recent_events = None if recent_raw in (None, "") else max(int(recent_raw), 0)
    materiality_pct = float(payload.get("materiality_pct", 5.0))
    impact_raw = payload.get("typical_impact")
    unit = payload.get("unit", "IDR")

    prior_mean = prior["rate_per_1000"] / 1000.0
    alpha0 = max(prior_mean * prior["effective_exposure"], 0.1)
    beta0 = float(prior["effective_exposure"])
    if recent_events is None:
        alpha_post, beta_post = alpha0, beta0
    else:
        alpha_post = alpha0 + recent_events
        beta_post = beta0 + exposure

    modifier = _control_change_multiplier(control, change)
    rng = np.random.default_rng(42)
    rate_draw = rng.gamma(shape=alpha_post, scale=1 / beta_post, size=simulations)
    event_count = rng.poisson(exposure * rate_draw * modifier)

    impact_is_user = impact_raw not in (None, "", 0, "0")
    if impact_is_user:
        typical_impact = max(float(impact_raw), objective * 1e-9)
        impact_note = "Severity anchored to the user's rough impact-per-event input."
    else:
        typical_impact = objective * 0.0002
        impact_note = "No severity evidence supplied: uses a training-only scenario median equal to 0.02% of the reference value. Replace this with company evidence before decision use."

    sigma = 0.75
    total_loss = np.zeros(simulations)
    for n in np.unique(event_count):
        if n <= 0:
            continue
        ix = np.where(event_count == n)[0]
        draws = rng.lognormal(mean=np.log(typical_impact), sigma=sigma, size=(len(ix), int(n)))
        total_loss[ix] = draws.sum(axis=1)

    materiality_floor = objective * materiality_pct / 100.0
    expected_loss = float(np.mean(total_loss))
    p95 = float(np.quantile(total_loss, 0.95))
    confidence = prior["confidence"] if impact_is_user else "low"

    control_driver = max(0, 50 - control) / 50
    change_driver = max(0, change - 50) / 50
    evidence_driver = 0.25 if recent_events is not None else 0.05
    raw = np.array([0.35, control_driver + 0.05, change_driver + 0.05, evidence_driver])
    shares = raw / raw.sum()

    return {
        "archetype": archetype,
        "archetype_label": prior["label"],
        "unit": unit,
        "expected_events": float(np.mean(event_count)),
        "probability_any_event": float(np.mean(event_count > 0)),
        "expected_loss": expected_loss,
        "p50_loss": float(np.quantile(total_loss, 0.50)),
        "p95_loss": p95,
        "p99_loss": float(np.quantile(total_loss, 0.99)),
        "materiality_floor": materiality_floor,
        "probability_material_loss": float(np.mean(total_loss >= materiality_floor)),
        "loss_to_objective_pct": expected_loss / objective * 100,
        "p95_to_objective_pct": p95 / objective * 100,
        "confidence": confidence,
        "evidence_mode": prior["mode"],
        "prior_evidence": prior["evidence"],
        "control_change_multiplier": modifier,
        "impact_note": impact_note,
        "drivers": [
            {"name": "Public / scenario baseline", "share": float(shares[0])},
            {"name": "Control condition", "share": float(shares[1])},
            {"name": "Change intensity", "share": float(shares[2])},
            {"name": "Recent-event evidence", "share": float(shares[3])},
        ],
        "distribution": {"bins": [float(x) for x in np.quantile(total_loss, np.linspace(0, 1, 41))]},
        "horizon": "next 6 months",
        "model_guardrail": "This is an explainable training / decision-support estimate. It is not a regulatory capital number. Where public exposure denominators or severity data are unavailable, the model labels the estimate as scenario-based instead of pretending precision.",
    }
