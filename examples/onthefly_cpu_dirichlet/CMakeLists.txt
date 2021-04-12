add_executable(onthefly_cpu_dirichlet run.cpp)
target_include_directories(onthefly_cpu_dirichlet PUBLIC
  ${besthea_INCLUDE_DIRS})
target_include_directories(onthefly_cpu_dirichlet SYSTEM PUBLIC
  ${Lyra_INCLUDE_DIRS})
target_link_libraries(onthefly_cpu_dirichlet PUBLIC besthea)
install(TARGETS onthefly_cpu_dirichlet
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
