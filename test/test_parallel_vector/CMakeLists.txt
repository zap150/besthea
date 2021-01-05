add_executable(test_parallel_vector run.cpp)
target_link_libraries(test_parallel_vector PUBLIC besthea)
target_link_libraries(test_parallel_vector PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
