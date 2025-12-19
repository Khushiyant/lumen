#!/bin/bash

# Exit on any error
set -e

# --- PLATFORM-AWARE CPU DETECTION ---
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
elif command -v nproc > /dev/null; then
    # Linux
    NUM_CORES=$(nproc)
else
    # Fallback for other Unix systems
    NUM_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi

# Define build directory
BUILD_DIR="build"

# Optional: Clean flag
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf $BUILD_DIR
fi

# Create build dir if it doesn't exist
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure and Build
echo "Configuring and Building Lumen..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Use the detected number of cores for parallel build
echo "Building with $NUM_CORES cores..."
cmake --build . --parallel "$NUM_CORES"

echo -e "\n Running Unit Tests..."
ctest --output-on-failure

# Run the example
echo -e "\n--- Running Example ---"
./bin/lumen_test

echo -e "\n Benchmarking Lumen..."
./bin/lumen_bench