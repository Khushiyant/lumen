#!/bin/bash

# Exit on any error
set -e

# --- PLATFORM-AWARE CPU DETECTION ---
if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
elif command -v nproc > /dev/null; then
    NUM_CORES=$(nproc)
else
    NUM_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi

# Define build directory
BUILD_DIR="build"

if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf $BUILD_DIR
fi

# Ensure Python requirements are present
if ! python3 -c "import pybind11" &>/dev/null; then
    echo "Installing pybind11 for bindings..."
    pip3 install pybind11 --quiet
fi

# FIX: Use --cmakedir instead of --cmake to find the path
PYBIND11_PATH=$(python3 -m pybind11 --cmakedir)

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure and Build
echo "Configuring and Building Lumen with Python Bindings..."
# Pass the pybind11_DIR to CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -Dpybind11_DIR="${PYBIND11_PATH}" \
         -DCMAKE_PREFIX_PATH="/usr/local"

echo "Building with $NUM_CORES cores..."
cmake --build . --parallel "$NUM_CORES"

echo -e "\n--- Running C++ Tests ---"
ctest --output-on-failure

echo -e "\n Benchmarking Lumen..."
./bin/lumen_bench

echo -e "\nGraph IR example build and run..."
./bin/lumen_graph_example

echo -e "\nRunning Lumen Unit Tests..."
./bin/lumen_unit_tests