add_executable(uniform_tensor_neumann run.cpp)
target_link_libraries(uniform_tensor_neumann PUBLIC besthea)
install(TARGETS uniform_tensor_neumann
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
