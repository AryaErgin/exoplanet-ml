import numpy as np

def sanitize_series(time: np.ndarray, flux: np.ndarray):
    mask = np.isfinite(time) & np.isfinite(flux)
    t = time[mask]
    f = flux[mask]
    if t.size == 0:
        return t, f
    order = np.argsort(t)
    t = t[order]
    f = f[order]
    med = np.median(f)
    if np.isfinite(med) and med != 0:
        f = f / med
    return t, f