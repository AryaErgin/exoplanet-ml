import numpy as np
import pandas as pd
from pathlib import Path

csv_path = Path("work/features_incremental.csv")
df = pd.read_csv(csv_path)

changed = False
if "log_depth" not in df.columns:
    if "min_depth_ppm" not in df.columns:
        raise SystemExit("min_depth_ppm missing; nothing to migrate.")
    df["log_depth"] = np.log1p(df["min_depth_ppm"].clip(lower=0.0))
    changed = True

if "log_snr" not in df.columns:
    if "snr" not in df.columns:
        raise SystemExit("snr missing; nothing to migrate.")
    df["log_snr"] = np.log1p(df["snr"].clip(lower=0.0))
    changed = True

if changed:
    df.to_csv(csv_path, index=False)
    print("migrated:", csv_path, "rows:", len(df))
else:
    print("no changes; already migrated")