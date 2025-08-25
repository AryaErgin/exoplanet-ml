from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from ..ml.model import predict as ml_predict

router = APIRouter()

class PredictIn(BaseModel):
    time: List[float] = Field(min_length=200, max_length=200000)
    flux: List[float] = Field(min_length=200, max_length=200000)
    meta: Optional[dict] = None

@router.post("/predict")
async def predict(body: PredictIn):
    if len(body.time) != len(body.flux):
        raise HTTPException(status_code=400, detail="time_flux_length_mismatch")
    try:
        out = ml_predict(body.time, body.flux)
        if body.meta:
            out["meta"] = body.meta
        return out
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="internal_error")