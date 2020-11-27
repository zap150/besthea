add_executable(parallel_dirichlet_indirect run.cpp)
target_link_libraries(parallel_dirichlet_indirect PUBLIC besthea)
target_link_libraries(parallel_dirichlet_indirect PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
