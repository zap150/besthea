add_executable(onthefly_multiply multiply.cpp)
add_executable(onthefly_solve solve.cpp)
target_link_libraries(onthefly_multiply PUBLIC besthea besthea_cuda)
target_link_libraries(onthefly_solve PUBLIC besthea besthea_cuda)
install(TARGETS onthefly_multiply onthefly_solve
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
install(PROGRAMS run_components.sh run_convergence.sh run_gpu_algorithm.sh
  run_multigpu.sh run_multiply.sh run_solve.sh
  DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
