from fastapi import APIRouter, HTTPException, Depends
from typing import List
from ..schemas import RunOut
from ..db import SessionLocal, Run

router = APIRouter(prefix="/runs", tags=["runs"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@router.get("", response_model=List[RunOut])
def list_runs(db=Depends(get_db)):
    rows = db.query(Run).order_by(Run.id.desc()).limit(100).all()
    return [
        RunOut(
            id=r.id,
            filename=r.filename,
            starId=r.star_id,
            probability=r.probability,
            dipsAt=[float(x) for x in r.dips_at.split(",") if x],
            rows=r.rows,
            meta=r.meta,
        )
        for r in rows
    ]

@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: str, db=Depends(get_db)):
    r = db.get(Run, run_id)
    if not r: raise HTTPException(404, "not found")
    return RunOut(
        id=r.id,
        filename=r.filename,
        starId=r.star_id,
        probability=r.probability,
        dipsAt=[float(x) for x in r.dips_at.split(",") if x],
        rows=r.rows,
        meta=r.meta,
    )

@router.delete("/{run_id}")
def delete_run(run_id: str, db=Depends(get_db)):
    r = db.get(Run, run_id)
    if not r: raise HTTPException(404, "not found")
    db.delete(r); db.commit()
    return {"deleted": run_id}