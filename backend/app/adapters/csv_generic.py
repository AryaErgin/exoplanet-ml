import csv, io
from typing import Dict
from .base import LightCurve

KNOWN_MAPS: Dict[str, Dict[str, str]] = {
    # header-name maps; extend as needed
    "kepler_pdcsap": {"time": "TIME", "flux": "PDCSAP_FLUX", "quality": "SAP_QUALITY"},
    "tess_btjd": {"time": "BTJD", "flux": "DETRENDED_FLUX"},
}

CANDIDATE_TIME = {"time", "t", "bjd", "btjd", "jd", "mjd"}
CANDIDATE_FLUX = {"flux", "pdcsap_flux", "sap_flux", "detrened_flux", "detrended_flux", "flux_norm", "flux_normalized"}

def auto_map_headers(headers):
    h = [str(x).strip() for x in headers]
    # try known maps by exact match
    for _name, m in KNOWN_MAPS.items():
        if all(k in h for k in m.values()):
            return m
    # heuristic fallback
    time_key = next((c for c in h if c.lower() in CANDIDATE_TIME), None)
    flux_key = next((c for c in h if c.lower() in CANDIDATE_FLUX), None)
    if time_key and flux_key:
        return {"time": time_key, "flux": flux_key}
    return None

def load_csv_bytes(blob: bytes, mapping: Dict[str, str], target_id: str = "UNKNOWN") -> LightCurve:
    text = blob.decode("utf-8", errors="ignore")
    rdr = csv.DictReader(io.StringIO(text))
    t, f, q = [], [], []
    for r in rdr:
        try:
            t.append(float(r[mapping["time"]]))
            f.append(float(r[mapping["flux"]]))
            if "quality" in mapping and mapping["quality"] in r:
                q.append(int(r[mapping["quality"]]))
        except Exception:
            # skip unparseable rows
            continue
    return LightCurve(target_id, t, f, q or None, None, {"source": "csv_generic", "mapping": mapping})