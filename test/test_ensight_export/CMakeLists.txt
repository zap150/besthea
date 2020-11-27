add_executable(test_ensight_export test_ensight_export.cpp)
target_link_libraries(test_ensight_export PUBLIC besthea)
target_link_libraries(test_ensight_export PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
