add_executable(distributed_tensor_dirichlet run.cpp)
target_link_libraries(distributed_tensor_dirichlet PUBLIC besthea)
install(TARGETS distributed_tensor_dirichlet
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
