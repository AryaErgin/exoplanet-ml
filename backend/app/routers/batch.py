from typing import List, Optional 
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field 
import numpy as np

from ..ml.model import model   # your existing model interface
from ..ml.io import sanitize_series  # new helper

router = APIRouter(prefix="/batch", tags=["batch"])

MAX_ITEMS = 100

class Series(BaseModel):
    id: Optional[str] = None
    time: List[float] = Field(min_items=10)   # was: conlist(float, min_items=10)
    flux: List[float] = Field(min_items=10)   # was: conlist(float, min_items=10)
    meta: Optional[dict] = None

class BatchPredictIn(BaseModel):
    items: List[Series] = Field(min_items=1, max_items=MAX_ITEMS)

@router.post("/predict")
async def batch_predict(body: BatchPredictIn):
    out: List[dict] = []
    for s in body.items:
        t, f = sanitize_series(np.asarray(s.time), np.asarray(s.flux))
        if t.size < 10:
            raise HTTPException(status_code=422, detail="Too few valid points")
        r = model.predict(t, f)
        out.append({
            "id": s.id or r.get("id"),
            "probability": float(r.get("probability", 0.0)),
            "dipsAt": list(map(float, r.get("dipsAt", []))),
            "periodDays": (None if r.get("periodDays") is None else float(r["periodDays"])),
            "t0": (None if r.get("t0") is None else float(r["t0"])),
            "depthPpm": (None if r.get("depthPpm") is None else float(r["depthPpm"])),
            "durationHr": (None if r.get("durationHr") is None else float(r["durationHr"])),
            "snr": (None if r.get("snr") is None else float(r["snr"])),
            "topPeriods": list(map(float, r.get("topPeriods", []))),
            "vetting": r.get("vetting", {}),
            "meta": s.meta or {},
        })
    return {"results": out}