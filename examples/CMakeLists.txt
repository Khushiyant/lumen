add_executable(lumen_bench benchmark.cpp)
target_link_libraries(lumen_bench PRIVATE lumen)

add_executable(lumen_test main.cpp)
target_link_libraries(lumen_test PRIVATE lumen)

# NEW: Graph IR example
add_executable(lumen_graph_example graph.cpp)
target_link_libraries(lumen_graph_example PRIVATE lumen)