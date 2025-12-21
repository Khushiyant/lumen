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
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="${PYBIND11_PATH}"

echo "Building with $NUM_CORES cores..."
cmake --build . --parallel "$NUM_CORES"

echo -e "\n--- Running C++ Tests ---"
ctest --output-on-failure

# Example of how to use the new Python module
echo -e "\n--- Verifying Python Bindings ---"
python3 <<EOF
import sys
import os

# The shared library is in build/lib based on CMake output
lib_path = os.path.join(os.getcwd(), 'lib')
sys.path.append(lib_path)

try:
    import lumen_py
    import numpy as np
    
    rt = lumen_py.Runtime()
    print(f"Lumen initialized successfully!")
    print(f"Active Backend: {rt.current_backend()}")
    
    # Quick test: Alloc and check shape
    buf = rt.alloc([2, 2])
    print(f"Allocated buffer with shape: {buf.shape}")
    
except Exception as e:
    print(f"Python Bindings check failed: {e}")
    sys.exit(1)
EOF

echo -e "\n Benchmarking Lumen..."
./bin/lumen_bench

echo -e "\nGraph IR example build and run..."
./bin/lumen_graph_example