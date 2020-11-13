add_executable(routines run.cpp)
target_link_libraries(routines PUBLIC besthea)
target_link_libraries(routines PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
