from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import requests

def score_csv(api: str, csv_path: Path, timeout=20.0):
    df = pd.read_csv(csv_path, usecols=["time","flux"])
    # backend accepts very long series now, but keep memory sane
    if len(df) > 200000:
        df = df.iloc[:200000]
    payload = {
        "time": df["time"].astype(float).tolist(),
        "flux": df["flux"].astype(float).tolist(),
        "meta": {"filename": csv_path.name}
    }
    t0 = time.time()
    r = requests.post(api, json=payload, timeout=timeout)
    dt = time.time() - t0
    return r.status_code, (r.json() if r.headers.get("content-type","").startswith("application/json") else {}), dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--dir", default="research/data/negatives")
    args = ap.parse_args()

    root = Path(args.dir)
    csvs = sorted([p for p in root.glob("*.csv") if p.is_file()])
    if not csvs:
        print(f"[INFO] no CSV in {root.resolve()}")
        return

    # health check
    try:
        h = requests.get(args.api.replace("/predict","/health"), timeout=5)
        print(f"[HEALTH] {h.status_code} {h.text[:100]}")
    except Exception as e:
        print(f"[HEALTH] ERROR {e}. Backend not reachable.")
        return

    hits = 0
    for p in csvs:
        code, body, sec = score_csv(args.api, p)
        if code == 200:
            prob = body.get("probability")
            snr  = body.get("snr")
            print(f"[OK] {p.name:20s}  prob={prob:.3f}  snr={snr:.2f}  rows={len(pd.read_csv(p))}  in {sec:.2f}s")
            if prob is not None and prob >= 0.5:
                hits += 1
        else:
            print(f"[ERR] {p.name:20s}  HTTP {code}  {body}")

    print(f"[SUMMARY] files={len(csvs)} predicted_positive@0.5={hits}")

if __name__ == "__main__":
    import requests
    main()