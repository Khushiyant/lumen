# lumen/src/CMakeLists.txt

set(LUMEN_SOURCES core/runtime.cpp)

if(APPLE)
    list(APPEND LUMEN_SOURCES backends/gpu/metal_backend.mm)
    
    add_definitions(-DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)
    
    # IMPORTANT: Enable ARC (-fobjc-arc) for Metal backend
    set_source_files_properties(backends/gpu/metal_backend.mm PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
endif()

add_library(lumen SHARED ${LUMEN_SOURCES})

target_include_directories(lumen PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(APPLE)
    target_link_libraries(lumen PRIVATE 
        "-framework Metal" 
        "-framework Foundation" 
        "-framework Accelerate"
        "-framework MetalPerformanceShaders"
        "-framework MetalPerformanceShadersGraph")
endif()