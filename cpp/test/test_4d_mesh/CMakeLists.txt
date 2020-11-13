add_executable(test_4d_mesh run.cpp)
target_link_libraries(test_4d_mesh PUBLIC besthea)
target_link_libraries(test_4d_mesh PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
