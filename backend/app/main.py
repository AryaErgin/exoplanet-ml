from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, predict, runs
from .routers import ingest  # add
from .db import init_db

app = FastAPI(title="ExoVision API")

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(runs.router)
app.include_router(ingest.router)
