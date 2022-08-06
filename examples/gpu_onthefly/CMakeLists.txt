add_executable(onthefly_multiply multiply.cpp)
add_executable(onthefly_solve solve.cpp)
target_link_libraries(onthefly_multiply PUBLIC besthea besthea_cuda)
target_link_libraries(onthefly_solve PUBLIC besthea besthea_cuda)
install(TARGETS onthefly_multiply onthefly_solve
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
install(PROGRAMS run_onthefly_multigpu.sh run_onthefly_components.sh
  run_onthefly_comparison_mult.sh run_onthefly_comparison_solve.sh
  run_onthefly_convergence.sh
  DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
