add_executable(test_hypersingular_operator run.cpp)
target_link_libraries(test_hypersingular_operator PUBLIC besthea)
target_link_libraries(test_hypersingular_operator PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
