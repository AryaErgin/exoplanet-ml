# path: backend/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml poetry.lock* requirements.txt* /app/

# Use pip fallback if you don't use poetry
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi \
 || pip install --no-cache-dir fastapi uvicorn[standard] pydantic==2.* numpy pandas sqlalchemy python-multipart anyio httpx

COPY app /app/app

ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
