# Q-Safe IIoT-AD — reproducible simulation environment.
#
# Build:  docker build -t qsafe-iiot-ad .
# Run:    docker run --rm -it qsafe-iiot-ad bash
# Tests:  docker run --rm qsafe-iiot-ad pytest tests/ -v
# Benchmark: docker run --rm qsafe-iiot-ad python -m orchestrator.benchmark

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build liboqs (BIKE-L1 + HQC-128 only) into /root/_oqs.
COPY scripts/setup_liboqs.sh scripts/setup_liboqs.sh
RUN bash scripts/setup_liboqs.sh

COPY . .

ENV TF_CPP_MIN_LOG_LEVEL=3

CMD ["bash"]
