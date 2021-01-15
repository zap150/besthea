add_executable(uniform_tensor_neumann run.cpp)
target_include_directories(uniform_tensor_neumann PUBLIC
  ${besthea_INCLUDE_DIRS})
target_link_libraries(uniform_tensor_neumann PUBLIC besthea)
install(TARGETS uniform_tensor_neumann
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
