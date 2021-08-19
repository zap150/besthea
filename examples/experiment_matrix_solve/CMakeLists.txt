add_executable(experiment_matrix_solve experiment_matrix_solve.cpp)
target_include_directories(experiment_matrix_solve PUBLIC
  ${besthea_INCLUDE_DIRS})
target_include_directories(experiment_matrix_solve SYSTEM PUBLIC
  ${Lyra_INCLUDE_DIRS})
target_link_libraries(experiment_matrix_solve PUBLIC besthea)
install(TARGETS experiment_matrix_solve
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
