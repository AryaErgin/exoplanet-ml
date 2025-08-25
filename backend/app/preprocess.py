import numpy as np
from .adapters.base import LightCurve

def normalize_flux(lc: LightCurve) -> LightCurve:
    fx = np.asarray(lc.flux, float)
    if fx.size == 0:
        return lc
    med = float(np.nanmedian(fx)) or 1.0
    norm = (fx / med).tolist()
    meta = {**(lc.meta or {}), "norm": "median"}
    return LightCurve(lc.target_id, lc.time, norm, lc.quality, lc.cadence_s, meta)

def profile(lc: LightCurve):
    t = np.asarray(lc.time, float)
    fx = np.asarray(lc.flux, float)
    if t.size == 0:
        return {"rows": 0, "span_days": 0.0, "cadence_est_s": None, "nan_rate": 1.0}
    dt = np.diff(np.sort(t))
    cadence = float(np.nanmedian(dt)) * 24 * 3600 if dt.size else None
    nan_rate = float(np.mean(~np.isfinite(fx))) if fx.size else 1.0
    return {"rows": int(t.size), "span_days": float(t[-1] - t[0]), "cadence_est_s": cadence, "nan_rate": nan_rate}