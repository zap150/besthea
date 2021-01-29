
add_executable(matrix_solve matrix_solve.cpp)

target_include_directories(matrix_solve PUBLIC ${besthea_INCLUDE_DIRS})
target_link_libraries(matrix_solve PUBLIC besthea)
