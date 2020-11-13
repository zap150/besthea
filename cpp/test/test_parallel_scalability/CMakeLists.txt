add_executable(test_parallel_scalability run.cpp)
target_link_libraries(test_parallel_scalability PUBLIC besthea)
target_link_libraries(test_parallel_scalability PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
