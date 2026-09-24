# API (FastAPI) del copiloto. Contexto de build: la raiz del repositorio.
FROM python:3.12-slim

WORKDIR /app
COPY web/backend/requirements.txt web/backend/requirements.txt
RUN pip install --no-cache-dir -r web/backend/requirements.txt

COPY src/ src/
COPY experiments/ experiments/
COPY corpus/ corpus/
COPY data/ data/
COPY web/backend/ web/backend/
RUN mkdir -p results

EXPOSE 8000
CMD ["uvicorn", "web.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
