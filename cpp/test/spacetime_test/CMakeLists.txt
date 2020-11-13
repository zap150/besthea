add_executable(spacetime_test run.cpp)
target_link_libraries(spacetime_test PUBLIC besthea)
target_link_libraries(spacetime_test PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
