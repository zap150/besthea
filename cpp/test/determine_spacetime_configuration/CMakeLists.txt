add_executable(determine_spacetime_configuration run.cpp)
target_link_libraries(determine_spacetime_configuration PUBLIC besthea)
target_link_libraries(determine_spacetime_configuration PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
