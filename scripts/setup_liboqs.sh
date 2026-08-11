#!/usr/bin/env bash
# Builds and installs the liboqs C library (BIKE-L1 + HQC-128 only, to keep
# the build fast) into $HOME/_oqs, which is where liboqs-python looks for it
# by default. Run this once after `pip install -r requirements.txt`.
#
# Requires: git, a C compiler, cmake, ninja (the last two are also
# installable via `pip install cmake ninja` if you don't have root/apt
# access, e.g. in a restricted CI or sandboxed container).
#
# If your system already has OpenSSL development headers installed, you can
# drop -DOQS_USE_OPENSSL=OFF for hardware-accelerated AES/SHA; it is off
# here so this script works out of the box even without libssl-dev.

set -euo pipefail

INSTALL_PREFIX="${OQS_INSTALL_PATH:-$HOME/_oqs}"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "==> Cloning liboqs into $BUILD_DIR"
git clone --depth 1 https://github.com/open-quantum-safe/liboqs "$BUILD_DIR/liboqs"

echo "==> Configuring (BIKE-L1 + HQC-128 only, no OpenSSL dependency)"
cmake -S "$BUILD_DIR/liboqs" -B "$BUILD_DIR/liboqs/build" -GNinja \
  -DBUILD_SHARED_LIBS=ON \
  -DOQS_BUILD_ONLY_LIB=ON \
  -DOQS_USE_OPENSSL=OFF \
  -DOQS_MINIMAL_BUILD="KEM_bike_l1;KEM_hqc_1" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

echo "==> Building"
cmake --build "$BUILD_DIR/liboqs/build" --parallel "$(nproc)"

echo "==> Installing to $INSTALL_PREFIX"
cmake --build "$BUILD_DIR/liboqs/build" --target install

echo "==> Verifying"
python3 -c "
import oqs
mechs = oqs.get_enabled_kem_mechanisms()
assert 'BIKE-L1' in mechs, 'BIKE-L1 not enabled'
assert 'HQC-1' in mechs, 'HQC-1 (HQC-128) not enabled'
print('liboqs OK. Enabled KEMs:', mechs)
"

echo "==> Done. liboqs installed at $INSTALL_PREFIX"
