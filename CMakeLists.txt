cmake_minimum_required(VERSION 3.16)
project(Lumen VERSION 0.1.0 LANGUAGES CXX)

# Use C++17 or 20 for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define output directories (keeps your build folder clean)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add our source code subdirectory
add_subdirectory(src)

# Add an example project to test our library
add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)