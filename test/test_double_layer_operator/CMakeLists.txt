add_executable(test_double_layer_operator run.cpp)
target_link_libraries(test_double_layer_operator PUBLIC besthea)
target_link_libraries(test_double_layer_operator PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
