from pathlib import Path
import pandas as pd
import lightkurve as lk

TARGETS = {
    "Kepler-10": 11904151,
    "Kepler-22": 10593626,
    "Kepler-90": 11446443,
    "Kepler-8":   6922244,
}

OUTDIR = Path("planet_csv")
OUTDIR.mkdir(exist_ok=True)

def save_stitched_csv(name: str, kepid: int) -> None:
    sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long")
    lcc = sr.download_all()
    if not lcc or len(lcc) == 0:
        print(f"Failed {name}: no products")
        return
    lc = lcc.stitch().remove_nans().normalize()
    # Build DataFrame from attributes to avoid column-name variability
    df = pd.DataFrame({
        "time": lc.time.value,   # days
        "flux": lc.flux.value,   # normalized
    })
    df = df.dropna()
    if df.empty:
        print(f"Failed {name}: empty after cleaning")
        return
    df.to_csv(OUTDIR / f"{name}.csv", index=False)
    print(f"Saved {name}.csv ({len(df)} rows)")

if __name__ == "__main__":
    for name, kepid in TARGETS.items():
        try:
            save_stitched_csv(name, kepid)
        except Exception as e:
            print(f"Failed {name}: {e}")