
add_executable(matrix_dim_print matrix_dim_print.cpp)

target_include_directories(matrix_dim_print PUBLIC ${besthea_INCLUDE_DIRS})
target_link_libraries(matrix_dim_print PUBLIC besthea)
