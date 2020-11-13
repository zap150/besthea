add_executable(fast_tensor_neumann run.cpp)
target_link_libraries(fast_tensor_neumann PUBLIC besthea)
target_link_libraries(fast_tensor_neumann PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
