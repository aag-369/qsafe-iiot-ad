# Q-Safe Field Link — demo image.
# Build from the repository root:
#   docker build -t qsafe-field-link -f demo/Dockerfile .
#   docker run --rm -p 8000:8000 qsafe-field-link
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OQS_INSTALL_PATH=/opt/_oqs

# Build liboqs with only the two mechanisms this project uses, so the image
# stays small and the build stays fast.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/open-quantum-safe/liboqs /tmp/liboqs \
 && cmake -S /tmp/liboqs -B /tmp/liboqs/build -GNinja \
      -DBUILD_SHARED_LIBS=ON \
      -DOQS_BUILD_ONLY_LIB=ON \
      -DOQS_USE_OPENSSL=OFF \
      -DOQS_MINIMAL_BUILD="KEM_bike_l1;KEM_hqc_1" \
      -DCMAKE_INSTALL_PREFIX=${OQS_INSTALL_PATH} \
 && cmake --build /tmp/liboqs/build --parallel \
 && cmake --build /tmp/liboqs/build --target install \
 && rm -rf /tmp/liboqs

ENV LD_LIBRARY_PATH=${OQS_INSTALL_PATH}/lib:${LD_LIBRARY_PATH}

WORKDIR /app
COPY requirements.txt demo/requirements.txt ./reqs/
RUN pip install -r reqs/requirements.txt -r reqs/demo/requirements.txt liboqs-python

COPY . /app

# Fail the build rather than silently shipping the simulated fallback.
RUN python -c "\
from crypto_agility.kem_backend import get_kem_backend;\
b = get_kem_backend();\
assert type(b).__name__ == 'LiboqsKEMBackend', 'liboqs not active in image';\
print('liboqs active:', type(b).__name__)"

EXPOSE 8000
CMD ["python", "-m", "qsafe_link.run", "--host", "0.0.0.0", "--port", "8000", "--devices", "3"]
