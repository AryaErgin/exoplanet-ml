"""Bootstrap script to cache light curves and assemble feature CSVs.

This script prepares a reproducible local dataset by downloading a
balanced subset of confirmed planets and false positives, extracting the
richer feature vector, and writing them to disk alongside cached light
curves.  Teams can run it before hackathon demos to avoid waiting on
live archive calls."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import sys
import numpy as np
import pandas as pd

# Ensure repository root is importable when executing as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.train_from_nasa import (  # type: ignore  # pylint: disable=wrong-import-position
    FEATURE_COLUMNS,
    fetch_koi,
    fetch_nonplanets,
    fetch_stellar_params,
    download_kepler_lc,
    simple_features,
)


def _save_curve(out_dir: Path, kepid: int, time: np.ndarray, flux: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if np.ma.isMaskedArray(time):
        time = time.filled(np.nan)
    if np.ma.isMaskedArray(flux):
        flux = flux.filled(np.nan)

    data = np.column_stack([np.asarray(time, dtype=float), np.asarray(flux, dtype=float)])
    path = out_dir / f"kic_{kepid}.csv"
    np.savetxt(path, data, delimiter=",", header="time,flux", comments="")
    return path


def _assemble_features(rows: Iterable[dict], csv_path: Path) -> None:
    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURE_COLUMNS]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def bootstrap(out_dir: Path, count: int) -> Path:
    pos_ids = fetch_koi(count)
    neg_ids = fetch_nonplanets(count)
    stellar = fetch_stellar_params(pos_ids + neg_ids)

    curve_dir = out_dir / "lightcurves"
    rows = []
    for kepid in pos_ids + neg_ids:
        label = 1 if kepid in pos_ids else 0
        ts = download_kepler_lc(kepid)
        if ts is None:
            continue
        time, flux = ts
        _save_curve(curve_dir / ("positives" if label else "negatives"), kepid, time, flux)
        feats = simple_features(time, flux, stellar.get(int(kepid)))
        rows.append({**feats, "kepid": int(kepid), "label": label})

    csv_path = out_dir / "features.csv"
    _assemble_features(rows, csv_path)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Bootstrap cached light curves and features")
    parser.add_argument("--out", type=Path, default=Path("research/work/bootstrap"))
    parser.add_argument("--count", type=int, default=50, help="Number of positives/negatives to fetch")
    args = parser.parse_args()

    csv_path = bootstrap(args.out, args.count)
    print(f"[BOOTSTRAP] wrote features -> {csv_path}")


if __name__ == "__main__":
    main()
