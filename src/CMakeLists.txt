# 1. Find Dependencies
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

find_package(Protobuf REQUIRED)
find_package(ONNX REQUIRED)

# 2. Define Sources
set(LUMEN_SOURCES 
    core/runtime.cpp
    core/router.cpp
    core/graph.cpp
    core/op_registry.cpp
    core/onnx_importer.cpp
    backends/cpu/cpu_backend.cpp
)

# --- CUDA SUPPORT ---
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND LUMEN_SOURCES backends/cuda/cuda_backend.cu)
    set(LUMEN_HAS_CUDA TRUE)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
endif()

# --- METAL SUPPORT ---
if(APPLE)
    list(APPEND LUMEN_SOURCES backends/gpu/metal_backend.mm)
endif()

# 3. Create the Library
add_library(lumen SHARED ${LUMEN_SOURCES})

# 4. Configure Include Paths
target_include_directories(lumen PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(APPLE)
    target_include_directories(lumen PRIVATE /opt/homebrew/include)
    target_link_directories(lumen PRIVATE /opt/homebrew/lib)
endif()

target_include_directories(lumen PRIVATE 
    ${Protobuf_INCLUDE_DIRS} 
    ${ONNX_INCLUDE_DIRS}
)

# 5. Define Macros and Link Libraries
target_compile_definitions(lumen PRIVATE ONNX_ML=1)

if(LUMEN_HAS_CUDA)
    target_compile_definitions(lumen PUBLIC LUMEN_USE_CUDA)
    target_link_libraries(lumen PRIVATE CUDA::cudart CUDA::cublas)
endif()

if(APPLE)
    target_compile_definitions(lumen PUBLIC LUMEN_USE_METAL)
    
    # FIXED: Made these PUBLIC so they propagate to examples and suppress warnings
    target_compile_definitions(lumen PUBLIC 
        ACCELERATE_NEW_LAPACK 
        ACCELERATE_LAPACK_ILP64
    )
    
    set_source_files_properties(backends/gpu/metal_backend.mm PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
    
    # FIXED: Link frameworks as PUBLIC so executables can find the symbols
    target_link_libraries(lumen PUBLIC 
        "-framework Metal" 
        "-framework Foundation" 
        "-framework Accelerate" 
        "-framework MetalPerformanceShaders" 
        "-framework MetalPerformanceShadersGraph"
    )
    
    target_link_libraries(lumen PRIVATE 
        protobuf::libprotobuf
        onnx
        onnx_proto
    )
else()
    # Linux / Non-Apple logic
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(OpenBLAS openblas)
    endif()
    
    if(OpenBLAS_FOUND)
        target_link_libraries(lumen PUBLIC ${OpenBLAS_LIBRARIES})
        target_include_directories(lumen PUBLIC ${OpenBLAS_INCLUDE_DIRS})
        target_compile_definitions(lumen PUBLIC LUMEN_USE_OPENBLAS)
    endif()
    
    target_link_libraries(lumen PRIVATE 
        protobuf::libprotobuf 
        onnx 
        onnx_proto
    )
endif()