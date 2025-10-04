"""Utility script to fetch a small balanced sample of Kepler light curves.

The script downloads 10 confirmed exoplanet hosts and 10 vetted false-positive
targets by default, storing them as CSV files that are ready to be sent to the
backend's `/predict` endpoint. The resulting files can be used to verify the
model's accuracy locally without modifying the training or frontend code.

Example:
    python research/scripts/download_sample_lightcurves.py \
        --out research/data/sample_eval --count 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

DEFAULT_COUNT = 10
MIN_ROWS = 200  # avoid extremely small light curves that break preprocessing


def query_kepids(disposition: str, count: int, seed: int) -> list[int]:
    """Return up to ``count`` random KIC identifiers with the given disposition."""
    NasaExoplanetArchive.ROW_LIMIT = -1
    tbl = NasaExoplanetArchive.query_criteria(
        table="q1_q17_dr25_sup_koi",
        select="kepid,koi_disposition",
        where=f"koi_disposition='{disposition}'",
    )
    df = tbl.to_pandas().dropna(subset=["kepid"]).drop_duplicates("kepid")
    if df.empty:
        return []
    return (
        df.sample(min(count, len(df)), random_state=seed)["kepid"].astype(int).tolist()
    )


def query_false_positives(count: int, seed: int) -> list[int]:
    """Return up to ``count`` random KIC IDs flagged as false positives."""
    NasaExoplanetArchive.ROW_LIMIT = -1
    tbl = NasaExoplanetArchive.query_criteria(
        table="q1_q17_dr25_sup_koi",
        select="kepid,koi_disposition",
        where="koi_disposition='FALSE POSITIVE'",
    )
    df = tbl.to_pandas().dropna(subset=["kepid"]).drop_duplicates("kepid")
    if df.empty:
        return []
    return (
        df.sample(min(count, len(df)), random_state=seed)["kepid"].astype(int).tolist()
    )


def download_lightcurve(kepid: int) -> pd.DataFrame | None:
    """Fetch and preprocess a single long-cadence light curve.

    Returns ``None`` if the download fails or the resulting light curve is too
    short to be useful for evaluation.
    """
    try:
        sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long")
        if len(sr) == 0:
            return None
        lc = sr.download_all()
        if lc is None or len(lc) == 0:
            return None
        stitched = lc.stitch().remove_nans().normalize()
        df = pd.DataFrame({
            "time": stitched.time.value.astype("float64"),
            "flux": stitched.flux.value.astype("float64"),
        })
        df = df.dropna()
        df = df[(df["flux"] > 0) & np.isfinite(df["time"]) & np.isfinite(df["flux"])]
        if len(df) < MIN_ROWS:
            return None
        return df
    except Exception:
        return None


def save_lightcurves(kepids: Iterable[int], out_dir: Path) -> int:
    """Download each KIC to ``out_dir`` and return the number of successes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    for kepid in kepids:
        df = download_lightcurve(kepid)
        if df is None:
            continue
        df.to_csv(out_dir / f"KIC_{kepid}.csv", index=False)
        successes += 1
    return successes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of confirmed planets and false positives to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="research/data/sample_eval",
        help=(
            "Output directory that will contain 'positives' and 'negatives' "
            "(false positives)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used when sampling Kepler targets",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    positives_dir = out_root / "positives"
    negatives_dir = out_root / "negatives"

    print(f"[INFO] Fetching up to {args.count} confirmed planets...", flush=True)
    pos_ids = query_kepids("CONFIRMED", args.count, args.seed)
    pos_ok = save_lightcurves(pos_ids, positives_dir)
    print(f"[DONE] Saved {pos_ok}/{len(pos_ids)} confirmed planet CSVs -> {positives_dir}")

    print(f"[INFO] Fetching up to {args.count} false positives...", flush=True)
    neg_ids = query_false_positives(args.count, args.seed + 1)
    neg_ok = save_lightcurves(neg_ids, negatives_dir)
    print(
        f"[DONE] Saved {neg_ok}/{len(neg_ids)} false-positive CSVs -> {negatives_dir}"
    )

    if pos_ok == 0 or neg_ok == 0:
        print("[WARN] Some downloads failed; try re-running or increasing the count.")


if __name__ == "__main__":
    main()