# Mock-backend image: small, no GPU, suitable for Cloud Run.
# For the real model backend, use Dockerfile.hf instead.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BACKEND=mock

WORKDIR /app

# Install deps first for layer caching. Only the PoC (mock) deps are
# installed here; the HF backend deps live in Dockerfile.hf.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY server/ ./server/
COPY web/ ./web/

# Run as non-root
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Cloud Run injects PORT (default 8080). Bind 0.0.0.0 so the platform can
# reach the container. Shell form so ${PORT} expands at runtime.
EXPOSE 8080
CMD exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8080}
