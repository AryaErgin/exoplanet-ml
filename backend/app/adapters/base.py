from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class LightCurve:
    target_id: str
    time: List[float]
    flux: List[float]
    quality: Optional[List[int]] = None
    cadence_s: Optional[float] = None
    meta: Dict[str, Any] = None