from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

_ROOT = Path(__file__).resolve().parents[3]
_REPORT_CANDIDATES = [
    _ROOT / "research" / "work" / "eval" / "evaluation_report.json",
    _ROOT / "backend" / "app" / "models" / "current" / "evaluation_report.json",
]


def _load_report() -> Dict[str, Any]:
    for path in _REPORT_CANDIDATES:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"invalid_evaluation_report:{path.name}:{exc.msg}",
            ) from exc
        return {"source": str(path.relative_to(_ROOT)), "report": payload}
    raise HTTPException(status_code=404, detail="evaluation_report_not_found")


@router.get("/report")
def get_evaluation_report() -> Dict[str, Any]:
    """Return the most recent evaluation report produced by the research suite."""
    return _load_report()
