#!/bin/bash
set -ex

# Run all tests with AddressSanitizer enabled.
#
# Use cases:
#   ./test_all_asan.sh           # configure + build + test (first time)
#   ./test_all_asan.sh test      # only run tests (skip build)

BUILD_DIR="out/build/pscm-asan"
SRC_DIR="$(pwd)"

# Step 1: Configure with ASan (skip if already configured)
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
  mkdir -p "$BUILD_DIR"
  cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
    -S "$SRC_DIR"
fi

# Step 2: Build
if [ "$1" != "test" ]; then
  ninja -C "$BUILD_DIR" pscm_cc
  ninja -C "$BUILD_DIR"
fi

# Step 3: Run tests with ASan
export ASAN_OPTIONS=detect_leaks=0

ninja -C "$BUILD_DIR" check-base
ninja -C "$BUILD_DIR" check-cont
ninja -C "$BUILD_DIR" check-sicp
ninja -C "$BUILD_DIR" check-macro
ninja -C "$BUILD_DIR" check-core

cd "$BUILD_DIR/test"
ctest --output-on-failure
