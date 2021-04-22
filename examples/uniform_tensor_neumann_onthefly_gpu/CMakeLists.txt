add_executable(uniform_tensor_neumann_onthefly_gpu run.cpp)
target_include_directories(uniform_tensor_neumann_onthefly_gpu PUBLIC
  ${besthea_INCLUDE_DIRS})
target_link_libraries(uniform_tensor_neumann_onthefly_gpu PUBLIC besthea)
install(TARGETS uniform_tensor_neumann_onthefly_gpu
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
