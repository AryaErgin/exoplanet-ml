from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import load
from uuid import uuid4
from astropy.timeseries import BoxLeastSquares

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "current"
FEATURE_KEYS = [
    "median",
    "std",
    "mad",
    "min_depth_ppm",
    "snr",
    "cdpp_ppm",
    "flicker_ppm",
    "noise_autocorr",
    "bls_period",
    "bls_power",
    "bls_depth_ppm",
    "bls_duration_hr",
    "bls_snr",
    "stellar_teff",
    "stellar_logg",
    "stellar_radius",
    "log_depth",
    "log_snr",
    "log_n",
]


def _safe_load():
    cfg_path = MODELS_DIR / "config.json"
    mdl_path = MODELS_DIR / "model.joblib"
    if cfg_path.exists() and mdl_path.exists():
        data = load(mdl_path)
        model = data["model"] if isinstance(data, dict) and "model" in data else data
        cfg = json.loads(cfg_path.read_text())
        feats = data.get("features", cfg.get("features", FEATURE_KEYS)) if isinstance(data, dict) else cfg.get("features", FEATURE_KEYS)
        cfg.setdefault("features", feats)
        return model, cfg
    return None, {"features": FEATURE_KEYS, "version": "heuristic"}


clf, cfg = _safe_load()


def sanitize_series(t: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
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
    max_pts = 20000
    if f.size > max_pts:
        step = int(math.ceil(f.size / max_pts))
        t, f = t[::step], f[::step]
    f = np.clip(f, 0.1, 3.0)
    return t, f, n_raw


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _calc_noise_metrics(residual: np.ndarray, cadence_days: float) -> Tuple[float, float, float]:
    if residual.size == 0:
        return (np.nan, np.nan, np.nan)

    window = int(max(1, round((0.25 / max(cadence_days, 1e-6)))))
    if window <= 1 or window > residual.size:
        cdpp = float(np.std(residual) * 1e6)
    else:
        kernel = np.ones(window) / window
        smoothed = np.convolve(residual, kernel, mode="same")
        cdpp = float(np.sqrt(np.mean((residual - smoothed) ** 2)) * 1e6)

    flick_window = int(max(1, round((0.33 / max(cadence_days, 1e-6)))))
    if flick_window <= 1 or flick_window > residual.size:
        flicker = float(np.std(residual) * 1e6)
    else:
        kern = np.ones(flick_window) / flick_window
        flick = residual - np.convolve(residual, kern, mode="same")
        flicker = float(np.sqrt(np.mean(flick ** 2)) * 1e6)

    if residual.size < 2:
        autocorr = np.nan
    else:
        r0, r1 = residual[:-1], residual[1:]
        denom = np.sqrt(np.sum(r0 ** 2) * np.sum(r1 ** 2)) + 1e-12
        autocorr = float(np.sum(r0 * r1) / denom)

    return cdpp, flicker, autocorr


def _compute_bls(t: np.ndarray, f: np.ndarray) -> Tuple[Dict[str, float], List[Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]:
    if t.size < 200:
        return (
            {
                "bls_period": np.nan,
                "bls_power": np.nan,
                "bls_depth_ppm": np.nan,
                "bls_duration_hr": np.nan,
                "bls_snr": np.nan,
                "bls_t0": np.nan,
            },
            [],
            {"top_periods": [], "periodogram": []},
        )

    flux = f / np.nanmedian(f)
    flux = flux - np.nanmedian(flux)
    baseline = np.linspace(0.5, 40.0, 2000)
    durations = np.linspace(0.05, 0.3, 20)
    try:
        bls = BoxLeastSquares(t, flux)
        res = bls.power(baseline, durations)
    except Exception:
        return (
            {
                "bls_period": np.nan,
                "bls_power": np.nan,
                "bls_depth_ppm": np.nan,
                "bls_duration_hr": np.nan,
                "bls_snr": np.nan,
                "bls_t0": np.nan,
            },
            [],
            {"top_periods": [], "periodogram": []},
        )

    if np.all(~np.isfinite(res.power)):
        return (
            {
                "bls_period": np.nan,
                "bls_power": np.nan,
                "bls_depth_ppm": np.nan,
                "bls_duration_hr": np.nan,
                "bls_snr": np.nan,
                "bls_t0": np.nan,
            },
            [],
            {"top_periods": [], "periodogram": []},
        )

    idx = int(np.nanargmax(res.power))
    sorter = np.argsort(res.power)[-10:]
    top_periods = [(float(res.period[i]), float(res.power[i])) for i in sorter[::-1]]
    step = max(1, len(res.period) // 200)
    periodogram = [
        (float(res.period[i]), float(res.power[i])) for i in range(0, len(res.period), step)
    ]

    stats = {
        "bls_period": float(res.period[idx]),
        "bls_power": float(res.power[idx]),
        "bls_depth_ppm": float(abs(res.depth[idx]) * 1e6),
        "bls_duration_hr": float(res.duration[idx] * 24.0),
        "bls_snr": float(res.snr[idx]) if hasattr(res, "snr") else float(res.power[idx]),
        "bls_t0": float(res.transit_time[idx]) if hasattr(res, "transit_time") else float(t[0]),
    }

    aux = {"top_periods": top_periods, "periodogram": periodogram}

    return stats, periodogram, aux


def simple_features(t: np.ndarray, f: np.ndarray, n_raw: int, meta: Optional[dict]) -> Tuple[Dict, Dict]:
    med = float(np.median(f))
    win = min(max(51, (f.size // 200) * 2 + 1), 1001)
    sm = np.convolve(f, np.ones(win) / win, mode="same")
    resid = f - sm
    mad = _mad(resid)
    rstd = float(1.4826 * mad + 1e-12)

    depth_ppm = max(0.0, (med - float(np.min(sm))) * 1e6)
    snr = max(0.0, (med - float(np.min(sm))) / (rstd + 1e-12))

    cadence = float(np.median(np.diff(t))) if t.size > 1 else 0.020833
    cdpp, flicker, autocorr = _calc_noise_metrics(resid, cadence)
    bls_stats, periodogram, aux = _compute_bls(t, f)

    features = {
        "median": med,
        "std": rstd,
        "mad": float(mad),
        "min_depth_ppm": depth_ppm,
        "snr": snr,
        "cdpp_ppm": cdpp,
        "flicker_ppm": flicker,
        "noise_autocorr": autocorr,
        "bls_period": bls_stats["bls_period"],
        "bls_power": bls_stats["bls_power"],
        "bls_depth_ppm": bls_stats["bls_depth_ppm"],
        "bls_duration_hr": bls_stats["bls_duration_hr"],
        "bls_snr": bls_stats["bls_snr"],
        "stellar_teff": np.nan,
        "stellar_logg": np.nan,
        "stellar_radius": np.nan,
        "log_depth": float(np.log1p(depth_ppm)),
        "log_snr": float(np.log1p(snr)),
        "log_n": float(np.log1p(max(1, n_raw))),
    }

    meta = meta or {}
    for key in ("stellar_teff", "stellar_logg", "stellar_radius"):
        if key in meta and meta[key] is not None:
            features[key] = float(meta[key])

    interpret = {
        "cdpp_ppm": cdpp,
        "flicker_ppm": flicker,
        "noise_autocorr": autocorr,
        "periodogram": aux.get("periodogram", []) if aux else [],
        "top_periods": aux.get("top_periods", []) if aux else [],
        "bls": bls_stats,
    }

    return features, interpret


def predict(time: List[float], flux: List[float], meta: Optional[dict] = None) -> Dict:
    t, f, n_raw = sanitize_series(np.asarray(time), np.asarray(flux))
    feats, interpret = simple_features(t, f, n_raw, meta)
    model_features = cfg.get("features", FEATURE_KEYS)
    X = np.array([[feats.get(k, np.nan) for k in model_features]], dtype="float64")
    try:
        prob = float(clf.predict_proba(X)[0, 1]) if clf is not None else float(min(1.0, feats["log_snr"] / 5.0))
    except Exception:
        prob = float(min(1.0, feats["log_snr"] / 5.0))

    bls_stats = interpret.get("bls", {})
    period = bls_stats.get("bls_period")
    t0 = bls_stats.get("bls_t0")
    dips: List[float] = []
    if period is not None and t0 is not None and np.isfinite(period) and np.isfinite(t0):
        end = t[-1]
        current = t0
        while current <= end and len(dips) < 5:
            if current >= t[0]:
                dips.append(float(current))
            current += period

    result_periods = [p for p, _ in interpret.get("top_periods", [])]
    result_power = [pow for _, pow in interpret.get("top_periods", [])]

    return {
        "id": f"EV-{uuid4().hex[:12]}",
        "probability": prob,
        "dipsAt": dips,
        "periodDays": period if period is not None and np.isfinite(period) else None,
        "t0": t0 if t0 is not None and np.isfinite(t0) else None,
        "depthPpm": bls_stats.get("bls_depth_ppm"),
        "durationHr": bls_stats.get("bls_duration_hr"),
        "snr": float(np.expm1(feats["log_snr"])),
        "topPeriods": result_periods,
        "vetting": {
            "power": result_power,
            "cdpp_ppm": interpret.get("cdpp_ppm"),
            "flicker_ppm": interpret.get("flicker_ppm"),
            "noise_autocorr": interpret.get("noise_autocorr"),
        },
        "periodogram": interpret.get("periodogram", []),
    }
