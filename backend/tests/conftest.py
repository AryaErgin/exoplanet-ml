import os, sys, pytest
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture
def anyio_backend():
    return "asyncio"