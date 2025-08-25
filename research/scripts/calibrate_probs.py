import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.isotonic import IsotonicRegression

WORK_DIR = Path("work")
FEATURES_CSV = WORK_DIR / "features_incremental.csv"
MODEL_DIR = Path("../backend/app/models/current")
MODEL_PATH = MODEL_DIR / "model.joblib"
CONFIG_PATH = MODEL_DIR / "config.json"
CAL_PATH = MODEL_DIR / "calibrator.joblib"

def ensure_feature_columns(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    # If model wants log_depth but CSV has min_depth_ppm, build it; and vice‑versa
    if "log_depth" in model_features and "log_depth" not in df and "min_depth_ppm" in df:
        df["log_depth"] = np.log10(df["min_depth_ppm"].clip(lower=0) + 1.0)
    if "min_depth_ppm" in model_features and "min_depth_ppm" not in df and "log_depth" in df:
        df["min_depth_ppm"] = (10.0**df["log_depth"] - 1.0).clip(lower=0)

    if "log_snr" in model_features and "log_snr" not in df and "snr" in df:
        df["log_snr"] = np.log10(df["snr"].clip(lower=0) + 1.0)
    if "snr" in model_features and "snr" not in df and "log_snr" in df:
        df["snr"] = (10.0**df["log_snr"] - 1.0).clip(lower=0)

    if "log_n" in model_features and "log_n" not in df and "n" in df:
        df["log_n"] = np.log10(df["n"].clip(lower=1) + 1.0)
    if "n" in model_features and "n" not in df and "log_n" in df:
        df["n"] = (10.0**df["log_n"] - 1.0).round().clip(lower=1).astype(int)

    return df

def main():
    if not FEATURES_CSV.exists():
        raise SystemExit(f"Missing {FEATURES_CSV}")
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        raise SystemExit("Train a model first (model.joblib + config.json).")

    cfg = json.loads(CONFIG_PATH.read_text())
    model_features: list[str] = cfg.get("features", ["median","std","min_depth_ppm","snr","n"])

    df = pd.read_csv(FEATURES_CSV)
    # Build missing transform columns so that all model_features exist
    df = ensure_feature_columns(df, model_features)

    # Verify we have all required feature columns now
    missing = [c for c in model_features if c not in df.columns]
    if missing:
        raise SystemExit(f"features_incremental.csv is missing required columns for the current model: {missing}")

    # Keep only what we need + label
    if "label" not in df.columns:
        raise SystemExit("features_incremental.csv is missing 'label' column")
    df = df[model_features + ["label"]].dropna()

    X = df[model_features].values
    y = df["label"].astype(int).values

    clf = load(MODEL_PATH)
    p_raw = clf.predict_proba(X)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(iso, CAL_PATH)

    # mark calibrated
    cfg["calibrated"] = True
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

    print(json.dumps({
        "status": "ok",
        "rows": int(len(df)),
        "features_used": model_features,
        "base_pos_rate": float((y == 1).mean()),
        "calibrator": CAL_PATH.name
    }, indent=2))

if __name__ == "__main__":
    main()