from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()

@router.get("/health")
async def health():
    info = {"status": "ok"}
    config_path = Path("app/models/current/config.json")

    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            info.update({
                "model_version": cfg.get("version", "unknown"),
                "auc": cfg.get("auc", None),
                "features": cfg.get("features", []),
            })
        except Exception:
            info["model_version"] = "corrupt-config"

    return info
