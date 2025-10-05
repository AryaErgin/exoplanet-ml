import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_ingest_csv_auto_map():
    transport = ASGITransport(app=app)
    csv_text = "TIME,PDCSAP_FLUX\n0.0,1.0\n0.02,1.0\n0.04,0.999\n"
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"file": ("sample.csv", csv_text, "text/csv")}
        r = await ac.post("/ingest/csv", files=files)
        assert r.status_code == 200
        j = r.json()
        assert "time" in j and "flux" in j and "meta" in j