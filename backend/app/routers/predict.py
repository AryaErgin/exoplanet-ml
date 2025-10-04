from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from ..ml.model import predict as ml_predict
from ..db import SessionLocal, Run

router = APIRouter()

class PredictIn(BaseModel):
    time: List[float] = Field(min_length=1, max_length=200000)
    flux: List[float] = Field(min_length=1, max_length=200000)
    meta: Optional[dict] = None

@router.post("/predict")
async def predict(body: PredictIn):
    if len(body.time) != len(body.flux):
        raise HTTPException(status_code=400, detail="time_flux_length_mismatch")
    try:
        out = ml_predict(body.time, body.flux, body.meta or {})
        if body.meta:
            out["meta"] = body.meta
        meta = body.meta or {}
        session = SessionLocal()
        try:
            run = Run(
                id=out.get("id"),
                filename=str(meta.get("filename", "")),
                star_id=str(meta.get("star_id") or meta.get("kepid") or ""),
                probability=float(out.get("probability", 0.0)),
                dips_at=",".join(str(x) for x in out.get("dipsAt", [])),
                rows=len(body.time),
                meta=meta,
            )
            session.add(run)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
        return out
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="internal_error")