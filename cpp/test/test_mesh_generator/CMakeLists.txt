add_executable(test_mesh_generator run.cpp)
target_link_libraries(test_mesh_generator PUBLIC besthea)
target_link_libraries(test_mesh_generator PUBLIC 
  ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
