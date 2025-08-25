import glob, os, sys, time
import requests
import pandas as pd

API = os.environ.get("EV_API", "http://127.0.0.1:8000/predict")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TIMEOUT = 15  # seconds
MAX_ROWS = 20000  # cap payload size to avoid huge posts

def eval_one(path: str):
    print(f"[START] {os.path.basename(path)}", flush=True)
    df = pd.read_csv(path)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    if not {"time","flux"}.issubset(set(k.lower() for k in df.columns)):
        raise ValueError(f"{os.path.basename(path)} missing columns 'time','flux' (have: {list(df.columns)})")
    tcol = cols[[k for k in cols if k.lower()=="time"][0]]
    fcol = cols[[k for k in cols if k.lower()=="flux"][0]]
    # drop NaNs and cap rows
    df = df[[tcol, fcol]].dropna()
    t = df[tcol].astype(float).tolist()
    f = df[fcol].astype(float).tolist()

    t0 = time.time()
    r = requests.post(API, json={"time": t, "flux": f, "meta": {"filename": os.path.basename(path)}},
                      timeout=TIMEOUT)
    dt = time.time() - t0
    if r.ok:
        j = r.json()
        prob = j.get("probability")
        snr = j.get("snr")
        print(f"[DONE] {os.path.basename(path)} -> prob={prob:.3f} snr={snr} rows={len(t)} in {dt:.2f}s", flush=True)
    else:
        print(f"[HTTP] {os.path.basename(path)} -> {r.status_code} {r.text[:200]}", flush=True)

if __name__ == "__main__":
    print(f"[INFO] API={API}")
    print(f"[INFO] Searching: {DATA_DIR}")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"[INFO] Found {len(files)} CSV", flush=True)
    if not files:
        print("[WARN] No CSV files found. Run download_kepler_csvs.py first.", flush=True)
        sys.exit(0)
    # quick backend health check
    try:
        hr = requests.get(API.replace("/predict","/health"), timeout=5)
        print(f"[HEALTH] {hr.status_code} {hr.text.strip()[:120]}", flush=True)
    except Exception as e:
        print(f"[HEALTH] ERROR {e}. Backend not reachable.", flush=True)
        sys.exit(1)
    for p in files:
        try:
            eval_one(p)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(p)} -> {e}", flush=True)