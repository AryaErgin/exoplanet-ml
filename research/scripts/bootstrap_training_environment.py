"""Bootstrap script to cache light curves and assemble feature CSVs.

This script prepares a reproducible local dataset by downloading a
balanced subset of confirmed planets and false positives, extracting the
richer feature vector, and writing them to disk alongside cached light
curves.  Teams can run it before hackathon demos to avoid waiting on
live archive calls."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import sys
import numpy as np
import pandas as pd

# Ensure repository root is importable when executing as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.train_from_nasa import (  # type: ignore  # pylint: disable=wrong-import-position
    FEATURE_COLUMNS,
    _ensure_schema,
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


def _load_existing_rows(csv_path: Path) -> Dict[int, dict]:
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    if df.empty or "kepid" not in df.columns:
        return {}

    df = _ensure_schema(df)
    rows = {}
    for entry in df.to_dict(orient="records"):
        try:
            kepid = int(entry.get("kepid"))
        except Exception:
            continue
        entry["kepid"] = kepid
        rows[kepid] = entry
    return rows


def _load_curve_from_disk(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None

    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception:
        return None

    if data.size == 0:
        return None

    if data.ndim == 1:
        data = data.reshape(-1, 2)

    time = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    return time, flux


def bootstrap(out_dir: Path, count: int) -> Path:
    pos_ids = fetch_koi(count)
    neg_ids = fetch_nonplanets(count)
    stellar = fetch_stellar_params(pos_ids + neg_ids)

    curve_dir = out_dir / "lightcurves"
    csv_path = out_dir / "features.csv"
    cached_rows = _load_existing_rows(csv_path)

    reused_pos = reused_neg = 0
    recovered_pos = recovered_neg = 0
    new_pos = new_neg = 0

    rows = []
    for kepid in pos_ids + neg_ids:
        label = 1 if kepid in pos_ids else 0
        curve_path = curve_dir / ("positives" if label else "negatives") / f"kic_{kepid}.csv"

        row: Optional[dict] = None

        if curve_path.exists():
            if kepid in cached_rows:
                row = dict(cached_rows[kepid])
                if label:
                    reused_pos += 1
                else:
                    reused_neg += 1
            else:
                cached_curve = _load_curve_from_disk(curve_path)
                if cached_curve is not None:
                    time, flux = cached_curve
                    feats = simple_features(time, flux, stellar.get(int(kepid)))
                    row = {**feats, "kepid": int(kepid)}
                    cached_rows[kepid] = dict(row)
                    if label:
                        recovered_pos += 1
                    else:
                        recovered_neg += 1

        if row is None:
            ts = download_kepler_lc(kepid)
            if ts is None:
                continue
            time, flux = ts
            _save_curve(curve_dir / ("positives" if label else "negatives"), kepid, time, flux)
            feats = simple_features(time, flux, stellar.get(int(kepid)))
            row = {**feats, "kepid": int(kepid)}
            cached_rows[kepid] = dict(row)
            if label:
                new_pos += 1
            else:
                new_neg += 1

        row["label"] = label
        rows.append(dict(row))

    _assemble_features(rows, csv_path)

    print(
        (
            "[BOOTSTRAP] reused rows pos=%d neg=%d | recovered-from-disk pos=%d neg=%d | "
            "downloaded pos=%d neg=%d"
        )
        % (reused_pos, reused_neg, recovered_pos, recovered_neg, new_pos, new_neg)
    )
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
