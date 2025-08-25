from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

def fetch_nonplanets(n: int) -> list[int]:
    NasaExoplanetArchive.ROW_LIMIT = -1
    tbl = NasaExoplanetArchive.query_criteria(
        table="q1_q17_dr25_sup_koi",
        select="kepid,koi_disposition",
        where="koi_disposition!='CONFIRMED'",
    )
    df = tbl.to_pandas().dropna(subset=["kepid"]).drop_duplicates("kepid")
    if df.empty:
        return []
    return df.sample(min(n, len(df)), random_state=123)["kepid"].astype(int).tolist()

def download_kepler_csv(kepid: int, out_csv: Path) -> bool:
    try:
        sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long")
        if len(sr) == 0:
            return False
        lc = sr[0].download()
        if lc is None:
            return False
        lc = lc.normalize().remove_nans()
        df = pd.DataFrame({"time": lc.time.value.astype("float64"),
                           "flux": lc.flux.value.astype("float64")})
        # trim extremes the backend would clip anyway
        df = df[(df["flux"] > 0) & np.isfinite(df["time"]) & np.isfinite(df["flux"])]
        if len(df) < 200:
            return False
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=str, default="research/data/negatives")
    args = ap.parse_args()

    kepids = fetch_nonplanets(args.n)
    out_dir = Path(args.out)
    ok = 0
    for k in kepids:
        if download_kepler_csv(k, out_dir / f"KIC_{k}.csv"):
            ok += 1
    print(f"[DONE] wrote {ok}/{len(kepids)} CSV -> {out_dir.resolve()}")

if __name__ == "__main__":
    main()