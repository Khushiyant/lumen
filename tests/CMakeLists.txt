add_executable(lumen_unit_tests unit_tests.cpp onnx_tests.cpp)
target_link_libraries(lumen_unit_tests PRIVATE lumen)

add_test(NAME AllTests COMMAND lumen_unit_tests)