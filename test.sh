#!/bin/sh
set -e

# Turnkey test runner for libchaff / libnoise
# Mirrors libcorm test ecosystem standards

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"

echo "=================================================================="
echo "          libchaff / libnoise Test & Metrics Ecosystem            "
echo "=================================================================="

# Ensure all binaries are built
make all

echo ""
echo "--- [1/4] Running Low-Level Core Engine Suite (bin/test_noise) ---"
./bin/test_noise

echo ""
echo "--- [2/4] Running Canonical libchaff Public API Suite (bin/test_chaff) ---"
./bin/test_chaff

echo ""
echo "--- [3/4] Running Shared Library Exported ABI Suite (bin/test_symbols) ---"
./bin/test_symbols

echo ""
echo "--- [4/5] Running GPU Shader Contract & Parity Suite (bin/test_glsl) ---"
./bin/test_glsl

echo ""
echo "--- [5/5] Running Multi-View & Multi-Zoom Consistency Suite (bin/test_chaff_multiview) ---"
./bin/test_chaff_multiview

echo ""
echo "--- Operational Metrics & Throughput Benchmarks (bin/bench_noise) ---"
./bin/bench_noise

echo ""
echo "=================================================================="
echo "          ALL LIBCHAFF / LIBNOISE TESTS & METRICS PASSED          "
echo "=================================================================="
