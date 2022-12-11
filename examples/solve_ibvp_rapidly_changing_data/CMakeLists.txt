add_executable(solve_ibvp_rapidly_changing_data run.cpp)
target_link_libraries(solve_ibvp_rapidly_changing_data PUBLIC besthea)
install(TARGETS solve_ibvp_rapidly_changing_data
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
