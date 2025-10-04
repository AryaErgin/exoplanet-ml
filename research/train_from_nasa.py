from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from itertools import zip_longest

# --------- CONFIG ---------
POS_N = int(os.getenv("EV_POS_N", "500"))
NEG_N = int(os.getenv("EV_NEG_N", "500"))
CHECKPOINT_EVERY = int(os.getenv("EV_CKPT_EVERY", "50"))
SLEEP_SEC = float(os.getenv("EV_SLEEP_SEC", "0.02"))

BASE = Path(__file__).resolve().parent                   # research/
WORK_DIR = BASE / "work"
OUT_DIR = BASE.parent / "backend" / "app" / "models" / "current"
FEATURES_CSV = WORK_DIR / "features_incremental.csv"

FEATURE_KEYS = ["median", "std", "min_depth_ppm", "snr", "n"]

CLIP = {
    "min_depth_ppm": (0.0, 1e5),   # 0..100k ppm
    "snr": (0.0, 5.0),             # 0..5  (much lower!)
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
def _transform_frame(df):
    df = df.copy()
    # clips for robustness
    df["min_depth_ppm"] = df["min_depth_ppm"].clip(lower=0, upper=5e5)
    df["snr"] = df["snr"].clip(lower=0, upper=1e3)
    # logs to compress dynamic range
    df["log_depth"] = np.log1p(df["min_depth_ppm"])
    df["log_snr"] = np.log1p(df["snr"])
    # size proxy
    df["log_n"] = np.log1p(df["n"].clip(lower=1))
    return df

# --------- FEATURES (must match backend) ---------
def simple_features(t: np.ndarray, f: np.ndarray) -> Dict:
    med = float(np.median(f))
    # robust std via MAD
    mad = _mad(f)
    rstd = float(1.4826 * mad + 1e-12)

    # smoother baseline
    win = 51 if f.size >= 51 else max(5, (f.size // 20) * 2 + 1)
    kernel = np.ones(win) / win
    sm = np.convolve(f, kernel, mode="same")

    min_depth_ppm = (med - float(np.min(sm))) * 1e6
    snr = (med - float(np.min(sm))) / rstd

    # clip + assemble
    min_depth_ppm = _clip(min_depth_ppm, *CLIP["min_depth_ppm"])
    snr = _clip(snr, *CLIP["snr"])
    n = int(_clip(f.size, *CLIP["n"]))

    feats = {
        "median": med,
        "std": rstd,  # use robust std
        "min_depth_ppm": min_depth_ppm,
        "snr": snr,
        "n": n,
        "log_depth": np.log10(min_depth_ppm + 1.0),
        "log_snr":   np.log10(snr + 1.0),
        "log_n":     np.log10(n + 1.0),
    }
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

def append_row(csv_path: Path, row: Dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists()
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=header, index=False)


# --------- TRAIN & SAVE ---------
def train_and_export(features_csv: Path, out_dir: Path) -> dict:
    df = pd.read_csv(features_csv).dropna(subset=FEATURE_KEYS + ["label"])
    df = _transform_frame(df)
    if df.empty:
        return {"status": "skipped", "reason": "no data", "rows": 0}

    counts = df["label"].value_counts()
    if len(counts) < 2 or counts.min() < 10:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "training_sample.csv", index=False)
        return {"status": "skipped", "reason": "class imbalance", "rows": int(len(df))}

    X = df[["median","std","log_depth","log_snr","log_n"]].values
    y = df["label"].values

    stratify = y if counts.min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=stratify)

    clf = RandomForestClassifier(
    n_estimators=600,
    max_depth=12,
    min_samples_leaf=20,
    max_features=3,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=7,
    )
    clf.fit(Xtr, ytr)

    auc = None
    if len(np.unique(yte)) == 2:
        auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))

    out_dir.mkdir(parents=True, exist_ok=True)
    dump(clf, out_dir / "model.joblib")
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(
    {"features": ["median", "std", "log_depth", "log_snr", "log_n"],
     "version": "sklearn-rf-v2",
     "auc": auc},
    indent=2
    ))
    df.to_csv(out_dir / "training_sample.csv", index=False)
    return {"status": "ok", "auc": auc, "rows": int(len(df))}


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

    pbar = tqdm(total=len(unseen), desc="Processing NEW targets")
    new_since_ckpt = 0

    for kepid in unseen:
        ts = download_kepler_lc(kepid)
        if ts is None:
            continue
        t, f = ts
        feats = simple_features(t, f)
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
