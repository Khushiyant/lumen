# Define sources
set(LUMEN_SOURCES 
    core/runtime.cpp
    core/router.cpp
    backends/cpu/cpu_backend.cpp
)

# --- CUDA SUPPORT (NVIDIA GPU) ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "[Lumen] CUDA Detected. Enabling Backend.")
    list(APPEND LUMEN_SOURCES backends/cuda/cuda_backend.cu)
    
    set(LUMEN_HAS_CUDA TRUE)
    
    # Force Architecture 75 (RTX 20xx/30xx/40xx compatible)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
endif()

# --- METAL SUPPORT (Apple GPU) ---
if(APPLE)
    list(APPEND LUMEN_SOURCES backends/gpu/metal_backend.mm)
    set_source_files_properties(backends/gpu/metal_backend.mm PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
endif()

# Create Library
add_library(lumen SHARED ${LUMEN_SOURCES})
target_include_directories(lumen PUBLIC ${CMAKE_SOURCE_DIR}/include)

# --- LINKING & DEFINITIONS ---
if(LUMEN_HAS_CUDA)
    # PUBLIC ensures any code linking against lumen also sees this macro
    target_compile_definitions(lumen PUBLIC LUMEN_USE_CUDA)
    target_link_libraries(lumen PRIVATE CUDA::cudart CUDA::cublas)
endif()

if(APPLE)
    target_compile_definitions(lumen PUBLIC LUMEN_USE_METAL)
    target_compile_definitions(lumen PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    target_link_libraries(lumen PRIVATE "-framework Metal" "-framework Foundation" "-framework Accelerate" "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
else()
    # Linux CPU Fallback
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(OpenBLAS openblas)
    endif()
    
    if(OpenBLAS_FOUND)
        target_link_libraries(lumen PRIVATE ${OpenBLAS_LIBRARIES})
        target_include_directories(lumen PRIVATE ${OpenBLAS_INCLUDE_DIRS})
        target_compile_definitions(lumen PRIVATE LUMEN_USE_OPENBLAS)
    endif()
endif()