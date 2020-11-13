add_executable(dirichlet_indirect run.cpp)
target_link_libraries(dirichlet_indirect PUBLIC besthea)
target_link_libraries(dirichlet_indirect PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
