#!/bin/bash

# Exit on any error
set -e

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
cmake --build . --parallel $(sysctl -n hw.ncpu)

echo -e "\n Running Unit Tests..."
ctest --output-on-failure

# Run the example
echo -e "\n--- Running Example ---"
./bin/lumen_test