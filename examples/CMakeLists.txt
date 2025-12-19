# Create an executable named 'lumen_test' from main.cpp
add_executable(lumen_test main.cpp)

# Link the executable to our 'lumen' library
# This automatically handles include paths for us
target_link_libraries(lumen_test PRIVATE lumen)