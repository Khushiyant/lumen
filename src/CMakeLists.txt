# lumen/src/CMakeLists.txt

# 1. Base Sources (Core + CPU)
set(LUMEN_SOURCES 
    core/runtime.cpp
    backends/cpu/cpu_backend.cpp
)

# 2. CUDA Support
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Lumen: CUDA detected. Enabling CUDA backend.")
    list(APPEND LUMEN_SOURCES backends/cuda/cuda_backend.cu)
    add_definitions(-DLUMEN_USE_CUDA)
else()
    message(STATUS "Lumen: CUDA NOT detected. Skipping.")
endif()

# 3. Metal Support (Apple)
if(APPLE)
    list(APPEND LUMEN_SOURCES backends/gpu/metal_backend.mm)
    set_source_files_properties(backends/gpu/metal_backend.mm PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
    add_definitions(-DLUMEN_USE_METAL)
endif()

# 4. Create Library
add_library(lumen SHARED ${LUMEN_SOURCES})

target_include_directories(lumen PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 5. Link Libraries
if(APPLE)
    target_link_libraries(lumen PRIVATE "-framework Metal" "-framework Foundation" "-framework Accelerate" "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
endif()

if(CMAKE_CUDA_COMPILER)
    # Link CUDA Runtime and cuBLAS
    target_link_libraries(lumen PRIVATE CUDA::cudart CUDA::cublas)
endif()