# Q-Safe IIoT-AD — dashboard + live demo container.
# Build from the REPO ROOT (not from web/), so the trained models/ and
# top-level packages are available:
#
#   docker build -t qsafe-iiot-ad-web -f web/Dockerfile .
#   docker run --rm -p 8000:8000 qsafe-iiot-ad-web
#   open http://localhost:8000            <- research dashboard
#   open http://localhost:8000/link/console  <- Q-Safe Field Link demo
#
# The entrypoint is server.py, which mounts both applications in one process
# (see its module docstring for why one service rather than two).

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY web/requirements.txt web/requirements.txt
COPY demo/requirements-runtime.txt demo/requirements-runtime.txt
RUN pip install --no-cache-dir \
      -r requirements.txt \
      -r web/requirements.txt \
      -r demo/requirements-runtime.txt

COPY scripts/setup_liboqs.sh scripts/setup_liboqs.sh
RUN bash scripts/setup_liboqs.sh

COPY . .

ENV TF_CPP_MIN_LOG_LEVEL=3
# Hosted platforms (Render, Railway, Fly.io, etc.) inject a PORT env var and
# require the app to bind to it; default to 8000 for plain `docker run`.
ENV PORT=8000
EXPOSE 8000

# Import the actual entrypoint. A missing runtime dependency is invisible to
# the test suite (CI installs the full developer requirements) and only shows
# up as the container exiting on boot -- this turns that into a build failure
# with a readable traceback instead.
RUN python -c "import server; print('server imports OK')"

# Fail the build rather than shipping an image whose KEM cost ordering is
# inverted: if the baseline profile is not cheaper than the hardened one, every
# CPU-saving figure the system reports flips sign.
RUN python scripts/verify_kem_ordering.py

# Shell form (not exec-form JSON array) so $PORT is actually expanded.
# server:app mounts the dashboard at / and the Field Link demo at /link.
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
