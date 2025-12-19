add_library(lumen SHARED 
    core/runtime.mm 
)

target_include_directories(lumen PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Add MetalPerformanceShaders to the list
target_link_libraries(lumen PRIVATE 
    "-framework Metal" 
    "-framework Foundation" 
    "-framework Accelerate"
    "-framework MetalPerformanceShaders")

# lumen/src/CMakeLists.txt
# lumen/src/CMakeLists.txt
if(APPLE)
    # Add these two flags to handle the new Accelerate API
    add_definitions(-DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)
    
    set_source_files_properties(core/runtime.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
    target_link_libraries(lumen PRIVATE 
        "-framework Metal" 
        "-framework Foundation" 
        "-framework Accelerate"
        "-framework MetalPerformanceShaders"
        "-framework MetalPerformanceShadersGraph")
endif()