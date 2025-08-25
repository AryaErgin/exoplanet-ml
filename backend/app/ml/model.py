from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from joblib import load
from uuid import uuid4

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "current"
FEATURE_KEYS = ["median","std","log_depth","log_snr","log_n"]

def _safe_load():
    cfg_path = MODELS_DIR / "config.json"
    mdl_path = MODELS_DIR / "model.joblib"
    if cfg_path.exists() and mdl_path.exists():
        return load(mdl_path), json.loads(cfg_path.read_text())
    return None, {"features": FEATURE_KEYS, "version": "heuristic"}

clf, cfg = _safe_load()

def sanitize_series(t: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray,np.ndarray,int]:
    t = np.asarray(t, dtype="float64")
    f = np.asarray(f, dtype="float64")
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 200:
        raise ValueError("too_few_points")
    idx = np.argsort(t)
    t, f = t[idx], f[idx]
    n_raw = int(f.size)
    med = np.median(f)
    if not np.isfinite(med) or med == 0:
        raise ValueError("bad_median")
    f = f / med
    # keep up to 20k points (not 5k) to retain variability
    max_pts = 20000
    if f.size > max_pts:
        step = int(math.ceil(f.size / max_pts))
        t, f = t[::step], f[::step]
    # wide clip to avoid NaNs but not flatten features
    f = np.clip(f, 0.1, 3.0)
    return t, f, n_raw

def _mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))

def simple_features(t: np.ndarray, f: np.ndarray, n_raw: int) -> Dict:
    med = float(np.median(f))
    # smooth baseline
    win = min(max(51, (f.size // 200) * 2 + 1), 1001)
    sm = np.convolve(f, np.ones(win) / win, mode="same")
    resid = f - sm
    rstd = float(1.4826 * _mad(resid) + 1e-12)

    # robust depth & SNR (no hard caps; compress with log1p)
    p50 = float(np.percentile(sm, 50))
    p01 = float(np.percentile(sm, 1))
    depth_ppm = max(0.0, (p50 - p01) * 1e6)
    snr = max(0.0, (p50 - float(np.min(sm))) / rstd)

    return {
        "median": med,
        "std": rstd,
        "log_depth": float(np.log1p(depth_ppm)),
        "log_snr": float(np.log1p(snr)),
        "log_n": float(np.log1p(max(1, n_raw))),
    }

def predict(time: List[float], flux: List[float]) -> Dict:
    t, f, n_raw = sanitize_series(np.asarray(time), np.asarray(flux))
    feats = simple_features(t, f, n_raw)
    model_features = cfg.get("features", FEATURE_KEYS)
    X = np.array([[feats[k] for k in model_features]], dtype="float64")
    try:
        prob = float(clf.predict_proba(X)[0, 1]) if clf is not None else float(min(1.0, feats["log_snr"] / 5.0))
    except Exception:
        prob = float(min(1.0, feats["log_snr"] / 5.0))
    return {
        "id": f"EV-{uuid4().hex[:12]}",
        "probability": prob,
        "dipsAt": [],
        "periodDays": None,
        "t0": None,
        "depthPpm": None,
        "durationHr": None,
        "snr": float(np.expm1(feats["log_snr"])),  # human-friendly SNR
        "topPeriods": [],
        "vetting": {},
    }