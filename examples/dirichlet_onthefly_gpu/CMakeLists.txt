add_executable(dirichlet_onthefly_gpu dirichlet_onthefly_gpu.cpp)
target_include_directories(dirichlet_onthefly_gpu PUBLIC
  ${besthea_INCLUDE_DIRS})
target_link_libraries(dirichlet_onthefly_gpu PUBLIC besthea)
install(TARGETS dirichlet_onthefly_gpu
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
