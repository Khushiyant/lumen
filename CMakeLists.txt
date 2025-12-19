cmake_minimum_required(VERSION 3.18)
project(Lumen VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)

# --- PYTHON BINDINGS ---
# Modern discovery method to fix CMP0148 warning
set(PYBIND11_FINDPYTHON ON) 
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

pybind11_add_module(lumen_py src/python/bindings.cpp)
target_link_libraries(lumen_py PRIVATE lumen)