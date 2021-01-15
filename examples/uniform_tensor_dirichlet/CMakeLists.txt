add_executable(uniform_tensor_dirichlet run.cpp)
target_include_directories(uniform_tensor_dirichlet PUBLIC
  ${besthea_INCLUDE_DIRS})
target_link_libraries(uniform_tensor_dirichlet PUBLIC besthea)
install(TARGETS uniform_tensor_dirichlet
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
