add_executable(fast_dirichlet_ibvp run.cpp)
target_link_libraries(fast_dirichlet_ibvp PUBLIC besthea)
install(TARGETS fast_dirichlet_ibvp
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
