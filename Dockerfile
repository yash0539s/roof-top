# Roof material classification API with the Production model baked in.
#   docker build -t roof-api .
#   docker run --rm -p 8000:8000 roof-api
# Optional: -v /path/to/orthophotos:/rasters:ro -e RASTER_DIR=/rasters
#           -v roof-logs:/app/logs   (keep the prediction log for monitoring)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models/production \
    PREDICTION_LOG=/app/logs/predictions.jsonl

# libgomp is needed by LightGBM and CatBoost.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY params.yaml .
COPY src/ src/
COPY api/ api/
COPY models/production/ models/production/

RUN useradd --create-home --uid 1000 app && mkdir -p /app/logs && chown -R app /app/logs
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health').status != 200)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
