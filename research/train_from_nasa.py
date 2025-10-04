from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from itertools import zip_longest
from astropy.timeseries import BoxLeastSquares

# --------- CONFIG ---------
POS_N = int(os.getenv("EV_POS_N", "500"))
NEG_N = int(os.getenv("EV_NEG_N", "500"))
CHECKPOINT_EVERY = int(os.getenv("EV_CKPT_EVERY", "50"))
SLEEP_SEC = float(os.getenv("EV_SLEEP_SEC", "0.02"))

BASE = Path(__file__).resolve().parent                   # research/
WORK_DIR = BASE / "work"
OUT_DIR = BASE.parent / "backend" / "app" / "models" / "current"
FEATURES_CSV = WORK_DIR / "features_incremental.csv"

FEATURE_COLUMNS = [
    "kepid",
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
    "n",
    "log_depth",
    "log_snr",
    "log_n",
    "label",
]

MODEL_FEATURES = [
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

CLIP = {
    "min_depth_ppm": (0.0, 1e5),   # 0..100k ppm
    "snr": (0.0, 1e4),
    "n": (1, 20000),
}

def _clip(v, lo, hi):  # unchanged
    return float(np.minimum(hi, np.maximum(lo, v)))

def _mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))


# --------- NASA HELPERS ---------
def fetch_koi(n: int) -> List[int]:
    NasaExoplanetArchive.ROW_LIMIT = -1
    tbl = NasaExoplanetArchive.query_criteria(
        table="q1_q17_dr25_sup_koi",
        select="kepid,koi_disposition",
        where="koi_disposition='CONFIRMED'",
    )
    df = tbl.to_pandas().dropna(subset=["kepid"]).drop_duplicates("kepid")
    if df.empty:
        return []
    return df.sample(min(n, len(df)), random_state=42)["kepid"].astype(int).tolist()


def fetch_nonplanets(n: int) -> List[int]:
    NasaExoplanetArchive.ROW_LIMIT = -1
    tbl = NasaExoplanetArchive.query_criteria(
        table="q1_q17_dr25_sup_koi",
        select="kepid,koi_disposition",
        where="koi_disposition='FALSE POSITIVE'",
    )
    df = tbl.to_pandas().dropna(subset=["kepid"]).drop_duplicates("kepid")
    if df.empty:
        return []
    return df.sample(min(n, len(df)), random_state=123)["kepid"].astype(int).tolist()


def fetch_stellar_params(kepids: List[int]) -> Dict[int, Dict[str, float]]:
    """Fetch stellar parameters for the provided KEPIDs."""
    if not kepids:
        return {}

    NasaExoplanetArchive.ROW_LIMIT = -1
    params: Dict[int, Dict[str, float]] = {}
    unique_ids = sorted(set(int(k) for k in kepids))
    chunk_size = 300
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i : i + chunk_size]
        where = "kepid in (" + ",".join(str(int(k)) for k in chunk) + ")"
        tbl = NasaExoplanetArchive.query_criteria(
            table="q1_q17_dr25_sup_koi",
            select="kepid,koi_steff,koi_slogg,koi_srad",
            where=where,
        )
        if tbl is None:
            continue
        df = tbl.to_pandas()
        for row in df.itertuples():
            params[int(row.kepid)] = {
                "stellar_teff": float(getattr(row, "koi_steff", np.nan)),
                "stellar_logg": float(getattr(row, "koi_slogg", np.nan)),
                "stellar_radius": float(getattr(row, "koi_srad", np.nan)),
            }
    return params


# --------- IO ---------
def download_kepler_lc(kepid: int):
    try:
        sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long")
        if len(sr) == 0:
            return None
        sr = sr[:2]
        lcs = [lc for lc in (entry.download() for entry in sr) if lc is not None]
        if not lcs:
            return None
        lc = lcs[0].normalize().remove_nans()
        return lc.time.value, lc.flux.value
    except Exception:
        return None

# --------- Transform (must match backend) ---------
def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe contains the expected feature schema."""
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURE_COLUMNS]
    return df


def _transform_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_schema(df)
    # clips for robustness
    df["min_depth_ppm"] = df["min_depth_ppm"].clip(lower=0, upper=5e5)
    df["snr"] = df["snr"].clip(lower=0, upper=1e5)
    df["cdpp_ppm"] = df["cdpp_ppm"].clip(lower=0, upper=1e5)
    df["flicker_ppm"] = df["flicker_ppm"].clip(lower=0, upper=5e5)
    df["bls_depth_ppm"] = df["bls_depth_ppm"].clip(lower=0, upper=5e5)
    df["bls_duration_hr"] = df["bls_duration_hr"].clip(lower=0, upper=1000)
    df["bls_snr"] = df["bls_snr"].clip(lower=0, upper=1e5)
    for col in ("stellar_teff", "stellar_logg", "stellar_radius"):
        df[col] = df[col].fillna(np.nan)
    # logs to compress dynamic range
    df["log_depth"] = np.log1p(df["min_depth_ppm"].fillna(0))
    df["log_snr"] = np.log1p(df["snr"].fillna(0))
    df["log_n"] = np.log1p(df["n"].fillna(1))
    return df

# --------- FEATURES (must match backend) ---------
def _calc_noise_metrics(residual: np.ndarray, cadence_days: float) -> Tuple[float, float, float]:
    if residual.size == 0:
        return (np.nan, np.nan, np.nan)

    # approximate 6-hour CDPP using rolling std
    window = int(max(1, round((0.25 / max(cadence_days, 1e-6)))))
    if window > residual.size:
        window = residual.size
    if window <= 1:
        cdpp = float(np.std(residual) * 1e6)
    else:
        kernel = np.ones(window) / window
        smoothed = np.convolve(residual, kernel, mode="same")
        cdpp = float(np.sqrt(np.mean((residual - smoothed) ** 2)) * 1e6)

    # stellar flicker as RMS of residual on 8hr baseline (0.33 days)
    flick_window = int(max(1, round((0.33 / max(cadence_days, 1e-6)))))
    if flick_window > residual.size:
        flick_window = residual.size
    if flick_window <= 1:
        flicker = float(np.std(residual) * 1e6)
    else:
        kern = np.ones(flick_window) / flick_window
        flick = residual - np.convolve(residual, kern, mode="same")
        flicker = float(np.sqrt(np.mean(flick ** 2)) * 1e6)

    # lag-1 autocorrelation as noise persistence proxy
    if residual.size < 2:
        autocorr = np.nan
    else:
        r0 = residual[:-1]
        r1 = residual[1:]
        denom = np.sqrt(np.sum(r0 ** 2) * np.sum(r1 ** 2)) + 1e-12
        autocorr = float(np.sum(r0 * r1) / denom)

    return cdpp, flicker, autocorr


def _compute_bls(t: np.ndarray, f: np.ndarray) -> Tuple[Dict[str, float], List[Tuple[float, float]], float]:
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
            np.nan,
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
            np.nan,
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
            np.nan,
        )

    idx = int(np.nanargmax(res.power))
    top_period = float(res.period[idx])
    top_power = float(res.power[idx])
    top_depth = float(abs(res.depth[idx]) * 1e6)
    top_duration = float(res.duration[idx] * 24.0)
    top_snr = float(res.snr[idx]) if hasattr(res, "snr") else float(top_power)
    t0 = float(res.transit_time[idx]) if hasattr(res, "transit_time") else float(t[0])

    sorter = np.argsort(res.power)[-10:]
    top_periods = [(float(res.period[i]), float(res.power[i])) for i in sorter[::-1]]
    step = max(1, len(res.period) // 200)
    periodogram = [
        (float(res.period[i]), float(res.power[i])) for i in range(0, len(res.period), step)
    ]

    stats = {
        "bls_period": top_period,
        "bls_power": top_power,
        "bls_depth_ppm": top_depth,
        "bls_duration_hr": top_duration,
        "bls_snr": top_snr,
        "bls_t0": t0,
    }

    return stats, periodogram, t0


def simple_features(t: np.ndarray, f: np.ndarray, stellar: Dict[str, float] | None = None) -> Dict:
    med = float(np.median(f))
    win = 51 if f.size >= 51 else max(5, (f.size // 20) * 2 + 1)
    kernel = np.ones(win) / win
    sm = np.convolve(f, kernel, mode="same")
    resid = f - sm
    mad = _mad(resid)
    rstd = float(1.4826 * mad + 1e-12)

    min_depth_ppm = (med - float(np.min(sm))) * 1e6
    snr = (med - float(np.min(sm))) / (rstd + 1e-12)
    n = int(_clip(f.size, *CLIP["n"]))

    cadence = float(np.median(np.diff(t))) if t.size > 1 else 0.020833
    cdpp, flicker, autocorr = _calc_noise_metrics(resid, cadence)

    bls_stats, _, _ = _compute_bls(t, f)

    feats = {
        "median": med,
        "std": rstd,
        "mad": float(mad),
        "min_depth_ppm": _clip(min_depth_ppm, *CLIP["min_depth_ppm"]),
        "snr": _clip(snr, *CLIP["snr"]),
        "cdpp_ppm": cdpp,
        "flicker_ppm": flicker,
        "noise_autocorr": autocorr,
        "bls_period": bls_stats["bls_period"],
        "bls_power": bls_stats["bls_power"],
        "bls_depth_ppm": bls_stats["bls_depth_ppm"],
        "bls_duration_hr": bls_stats["bls_duration_hr"],
        "bls_snr": bls_stats["bls_snr"],
        "n": n,
        "log_depth": np.log1p(max(0.0, min_depth_ppm)),
        "log_snr": np.log1p(max(0.0, snr)),
        "log_n": np.log1p(max(1, n)),
    }

    if stellar:
        feats.update({
            "stellar_teff": stellar.get("stellar_teff", np.nan),
            "stellar_logg": stellar.get("stellar_logg", np.nan),
            "stellar_radius": stellar.get("stellar_radius", np.nan),
        })

    return feats


# --------- DATASET APPEND/RESUME ---------
def load_processed_kepids(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["kepid"])
        return set(df["kepid"].astype(int).tolist())
    except Exception:
        return set()

def refresh_cached_negatives(csv_path: Path, allowed_negatives: set[int]) -> int:
    """Ensure cached negatives align with the currently fetched set."""
    if not csv_path.exists():
        return 0

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return 0

    if df.empty or "kepid" not in df or "label" not in df:
        return 0

    neg_mask = df["label"] == 0
    keep_mask = ~neg_mask | df["kepid"].isin(list(allowed_negatives))
    removed = int((~keep_mask).sum())
    if removed > 0:
        df.loc[keep_mask].to_csv(csv_path, index=False)
    return removed

def refresh_cached_negatives(csv_path: Path, allowed_negatives: set[int]) -> int:
    """Ensure cached negatives align with the currently fetched set."""
    if not csv_path.exists():
        return 0

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return 0

    if df.empty or "kepid" not in df or "label" not in df:
        return 0

    neg_mask = df["label"] == 0
    keep_mask = ~neg_mask | df["kepid"].isin(list(allowed_negatives))
    removed = int((~keep_mask).sum())
    if removed > 0:
        cleaned = _ensure_schema(df.loc[keep_mask])
        cleaned.to_csv(csv_path, index=False)
    return removed


def refresh_cached_negatives(csv_path: Path, allowed_negatives: set[int]) -> int:
    """Ensure cached negatives align with the currently fetched set."""
    if not csv_path.exists():
        return 0

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return 0

    if df.empty or "kepid" not in df or "label" not in df:
        return 0

    neg_mask = df["label"] == 0
    keep_mask = ~neg_mask | df["kepid"].isin(list(allowed_negatives))
    removed = int((~keep_mask).sum())
    if removed > 0:
        cleaned = _ensure_schema(df.loc[keep_mask])
        cleaned.to_csv(csv_path, index=False)
    return removed


def append_row(csv_path: Path, row: Dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists()
    df = pd.DataFrame([row])
    df = _ensure_schema(df)
    df.to_csv(csv_path, mode="a", header=header, index=False)


# --------- TRAIN & SAVE ---------
def train_and_export(features_csv: Path, out_dir: Path) -> dict:
    df = pd.read_csv(features_csv)
    df = _transform_frame(df)
    df = df.dropna(subset=["label"])  # allow feature NaNs for imputer
    if df.empty:
        return {"status": "skipped", "reason": "no data", "rows": 0}

    counts = df["label"].value_counts()
    if len(counts) < 2 or counts.min() < 10:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "training_sample.csv", index=False)
        return {"status": "skipped", "reason": "class imbalance", "rows": int(len(df))}

    X = df[MODEL_FEATURES]
    y = df["label"].values

    stratify = y if counts.min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=stratify)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=8,
                    learning_rate=0.07,
                    max_iter=600,
                    l2_regularization=0.1,
                    random_state=7,
                ),
            ),
        ]
    )

    pipeline.fit(Xtr, ytr)

    auc = None
    precision = recall = f1 = None
    if len(np.unique(yte)) == 2:
        proba = pipeline.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, proba))
        preds = (proba >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            yte, preds, average="binary", zero_division=0
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    dump({"model": pipeline, "features": MODEL_FEATURES}, out_dir / "model.joblib")
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(
    {
        "features": MODEL_FEATURES,
        "version": "hgb-broad-features-v1",
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    },
    indent=2
    ))
    df.to_csv(out_dir / "training_sample.csv", index=False)
    report = {
        "status": "ok",
        "auc": auc,
        "rows": int(len(df)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return report


# --------- MAIN ---------
if __name__ == "__main__":
    print(f"[INIT] Target POS={POS_N} NEG={NEG_N}  checkpoint={CHECKPOINT_EVERY}")
    pos_ids = fetch_koi(POS_N)
    neg_ids = fetch_nonplanets(NEG_N)
    total_target = len(pos_ids) + len(neg_ids)
    print(f"[FETCH] Planned {total_target} targets (pos={len(pos_ids)} neg={len(neg_ids)})")

    removed_neg = refresh_cached_negatives(FEATURES_CSV, set(neg_ids))
    if removed_neg:
        print(f"[RESUME] Removed {removed_neg} cached negatives not in FALSE POSITIVE set")

    processed = load_processed_kepids(FEATURES_CSV)
    print(f"[RESUME] Already have {len(processed)} rows in {FEATURES_CSV.name}")

    combined = [x for pair in zip_longest(pos_ids, neg_ids) for x in pair if x is not None]
    unseen = [k for k in combined if k not in processed]
    seen_pos = set(pos_ids)

    stellar_params = fetch_stellar_params(combined)

    pbar = tqdm(total=len(unseen), desc="Processing NEW targets")
    new_since_ckpt = 0

    for kepid in unseen:
        ts = download_kepler_lc(kepid)
        if ts is None:
            continue
        t, f = ts
        feats = simple_features(t, f, stellar_params.get(int(kepid)))
        if not feats:
            continue

        row = {**feats, "kepid": int(kepid), "label": 1 if kepid in seen_pos else 0}
        append_row(FEATURES_CSV, row)
        processed.add(kepid)
        new_since_ckpt += 1
        pbar.update(1)
        time.sleep(SLEEP_SEC)

        if new_since_ckpt >= CHECKPOINT_EVERY:
            info = train_and_export(FEATURES_CSV, OUT_DIR)
            print(f"[CKPT] csv_rows={info.get('rows')} status={info.get('status')} auc={info.get('auc','n/a')}")
            new_since_ckpt = 0

        if len(processed) >= total_target:
            break

    pbar.close()

    info = train_and_export(FEATURES_CSV, OUT_DIR)
    print(f"[DONE] csv_rows={info.get('rows')} target={total_target} status={info.get('status')} auc={info.get('auc','n/a')}")
    print(f"[ARTIFACTS] {OUT_DIR.resolve()}")
