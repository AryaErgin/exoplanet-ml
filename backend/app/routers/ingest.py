from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("/csv")
async def ingest_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV supported")
    raw = await file.read()
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV")

    cols_lower = {c.lower(): c for c in df.columns}
    time_key = next((k for k in cols_lower if k in {"time", "bjd", "jd", "t"}), None)
    flux_key = next((k for k in cols_lower if k in {"pdcsap_flux", "flux", "f"}), None)
    if not time_key or not flux_key:
        raise HTTPException(status_code=400, detail="Required columns not found")
    time_col = cols_lower[time_key]
    flux_col = cols_lower[flux_key]

    time_vals = df[time_col].tolist()
    flux_vals = df[flux_col].tolist()

    return {
        "time": time_vals,
        "flux": flux_vals,
        "meta": {
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "mapped": {"time": time_col, "flux": flux_col},
        },
        # keep these extras if you want, tests ignore them
        "rows": int(len(df)),
        "columns": list(df.columns),
        "mapped": {"time": time_col, "flux": flux_col},
    }