from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class PredictRequest(BaseModel):
    time: List[float]
    flux: List[float]
    meta: Optional[Dict[str, Any]] = None

class PredictResponse(BaseModel):
    id: str
    probability: float
    dipsAt: List[float]
    periodDays: float
    t0: float
    depthPpm: float
    durationHr: float
    snr: float
    topPeriods: List[float]
    vetting: Dict[str, float]
    meta: Dict[str, Any]

class RunOut(BaseModel):
    id: str
    filename: str
    starId: str
    probability: float
    dipsAt: List[float]
    rows: int
    meta: Dict[str, Any]