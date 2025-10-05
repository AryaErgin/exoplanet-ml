import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_predict_ok():
    transport = ASGITransport(app=app)
    time = [i*0.02 for i in range(400)]
    flux = [1.0]*len(time)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/predict", json={"time": time, "flux": flux, "meta": {"filename": "flat.csv"}})
        assert r.status_code == 200
        j = r.json()
        assert {"id","probability","dipsAt","periodDays","t0","depthPpm","durationHr","snr","topPeriods","vetting","meta"} <= j.keys()

@pytest.mark.anyio
async def test_predict_bad_lengths():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/predict", json={"time": [0,1,2], "flux": [1,1]})
        assert r.status_code == 400