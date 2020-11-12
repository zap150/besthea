add_executable(uniform_tensor_dirichlet run.cpp)
target_link_libraries(uniform_tensor_dirichlet PUBLIC besthea)
target_link_libraries(uniform_tensor_dirichlet PUBLIC ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
