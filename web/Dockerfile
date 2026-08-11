# Q-Safe IIoT-AD — web dashboard container.
# Build from the REPO ROOT (not from web/), so the trained models/ and
# top-level packages are available:
#
#   docker build -t qsafe-iiot-ad-web -f web/Dockerfile .
#   docker run --rm -p 8000:8000 qsafe-iiot-ad-web
#   open http://localhost:8000

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY web/requirements.txt web/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r web/requirements.txt

COPY scripts/setup_liboqs.sh scripts/setup_liboqs.sh
RUN bash scripts/setup_liboqs.sh

COPY . .

ENV TF_CPP_MIN_LOG_LEVEL=3
# Hosted platforms (Render, Railway, Fly.io, etc.) inject a PORT env var and
# require the app to bind to it; default to 8000 for plain `docker run`.
ENV PORT=8000
EXPOSE 8000

# Shell form (not exec-form JSON array) so $PORT is actually expanded.
CMD uvicorn web.backend.app:app --host 0.0.0.0 --port ${PORT}
