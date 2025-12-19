add_library(lumen SHARED 
    core/runtime.mm 
)

target_include_directories(lumen PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(APPLE)
    set_source_files_properties(core/runtime.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")


    target_link_libraries(lumen PRIVATE "-framework Metal" "-framework Foundation" "-framework Accelerate")
endif()