add_executable(uniform_tensor_dirichlet_onthefly_cpu run.cpp)
target_include_directories(uniform_tensor_dirichlet_onthefly_cpu PUBLIC
  ${besthea_INCLUDE_DIRS})
target_link_libraries(uniform_tensor_dirichlet_onthefly_cpu PUBLIC besthea)
install(TARGETS uniform_tensor_dirichlet_onthefly_cpu
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
