add_executable(neumann_onthefly_cpu neumann_onthefly_cpu.cpp)
target_include_directories(neumann_onthefly_cpu PUBLIC
  ${besthea_INCLUDE_DIRS})
target_link_libraries(neumann_onthefly_cpu PUBLIC besthea)
install(TARGETS neumann_onthefly_cpu
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
