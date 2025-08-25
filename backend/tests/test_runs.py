import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_runs_after_predict():
    transport = ASGITransport(app=app)
    time = [i*0.02 for i in range(200)]
    flux = [1.0]*len(time)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/predict", json={"time": time, "flux": flux, "meta": {"filename": "flat.csv"}})
        assert r.status_code == 200
        r2 = await ac.get("/runs")
        assert r2.status_code == 200
        rows = r2.json()
        assert isinstance(rows, list) and len(rows) >= 1