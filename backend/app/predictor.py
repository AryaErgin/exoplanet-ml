import numpy as np

def moving_average(x: np.ndarray, window: int = 11) -> np.ndarray:
    if x.size == 0: return x
    w = max(1, int(window)); k = w // 2
    out = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        s = slice(max(0, i-k), min(x.size, i+k+1))
        out[i] = float(np.mean(x[s]))
    return out

def normalize_flux(flux: np.ndarray) -> np.ndarray:
    if flux.size == 0: return flux
    med = float(np.median(flux)) or 1e-9
    return flux / med

def simple_dip_detector(time: np.ndarray, flux: np.ndarray):
    if time.size < 20 or time.size != flux.size:
        return [], 0.0
    nf = normalize_flux(flux)
    sm = moving_average(nf, 11)
    resid = nf - sm
    m = float(np.mean(resid))
    sd = float(np.std(resid)) or 1e-9
    z = (resid - m) / sd
    idx = (np.where((z[1:-1] < z[:-2]) & (z[1:-1] < z[2:]) & (z[1:-1] < -2.0))[0] + 1).tolist()
    score = 0.0
    if idx:
        depths = (-z[idx]).tolist()
        avg_depth = float(np.mean(depths))
        score = min(0.99, max(0.0, 0.2 + 0.1*len(idx) + 0.05*avg_depth))
    return idx, float(score)

def mock_stats(time: np.ndarray, flux: np.ndarray, dip_idx: list):
    rng = np.random.default_rng(42)  # deterministic for tests/demos
    t = time; f = normalize_flux(flux)
    if not dip_idx:
        span = float(t[-1] - t[0]) if t.size else 1.0
        return {
            "period": max(0.5, span/5),
            "t0": float(t[0]) if t.size else 0.0,
            "depth_ppm": 120.0,
            "duration_hr": 2.0,
            "snr": 3.0,
            "top_periods": [max(0.5, span/6), max(0.6, span/4), max(0.7, span/3)],
            "vetting": {"odd_even_ratio": 1.0, "secondary_snr": 0.5, "rms_oot": float(np.std(f))}
        }
    dips_t = t[np.array(dip_idx, dtype=int)]
    period = float(np.median(np.diff(dips_t))) if dips_t.size >= 2 else max(0.5, (t[-1]-t[0])/4)
    t0 = float(dips_t[0])
    win = max(3, int(0.02 * t.size))
    local = []
    for i in dip_idx[: min(5, len(dip_idx))]:
        s = slice(max(0, i-win), min(t.size, i+win))
        local.append(float(np.median(f[s]) - np.min(f[s])))
    depth = float(np.median(local)) if local else 0.001
    depth_ppm = max(50.0, depth * 1e6)
    dur_days = period * 0.03
    duration_hr = max(0.5, float(dur_days * 24.0))
    sd = float(np.std(f - moving_average(f, 11))) or 1e-6
    snr = float((depth or 1e-6) / sd)
    top_periods = [period, period*0.5, period*2.0][:3]
    vetting = {"odd_even_ratio": 1.0 + float(rng.normal(0, 0.03)), "secondary_snr": max(0.0, snr * 0.2), "rms_oot": float(np.std(f))}
    return {
        "period": max(0.2, period),
        "t0": t0,
        "depth_ppm": depth_ppm,
        "duration_hr": duration_hr,
        "snr": snr,
        "top_periods": [float(x) for x in top_periods],
        "vetting": vetting,
    }