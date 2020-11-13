add_executable(test_parallel_double_layer_operator run.cpp)
target_link_libraries(test_parallel_double_layer_operator PUBLIC besthea)
target_link_libraries(test_parallel_double_layer_operator PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
