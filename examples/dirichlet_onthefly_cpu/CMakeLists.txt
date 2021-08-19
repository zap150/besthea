add_executable(dirichlet_onthefly_cpu dirichlet_onthefly_cpu.cpp)
target_include_directories(dirichlet_onthefly_cpu PUBLIC
  ${besthea_INCLUDE_DIRS})
target_link_libraries(dirichlet_onthefly_cpu PUBLIC besthea)
install(TARGETS dirichlet_onthefly_cpu
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
