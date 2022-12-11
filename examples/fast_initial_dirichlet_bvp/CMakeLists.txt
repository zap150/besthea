add_executable(fast_initial_dirichlet_bvp run.cpp)
target_link_libraries(fast_initial_dirichlet_bvp PUBLIC besthea)
install(TARGETS fast_initial_dirichlet_bvp
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
