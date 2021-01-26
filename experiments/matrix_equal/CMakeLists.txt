
add_executable(matrix_equal matrix_equal.cpp)

target_include_directories(matrix_equal PUBLIC ${besthea_INCLUDE_DIRS})
target_link_libraries(matrix_equal PUBLIC besthea)
