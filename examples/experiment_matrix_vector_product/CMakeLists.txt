add_executable(experiment_matrix_vector_product experiment_matrix_vector_product.cpp)
target_include_directories(experiment_matrix_vector_product PUBLIC
  ${besthea_INCLUDE_DIRS})
target_include_directories(experiment_matrix_vector_product SYSTEM PUBLIC
  ${Lyra_INCLUDE_DIRS})
target_link_libraries(experiment_matrix_vector_product PUBLIC besthea)
install(TARGETS experiment_matrix_vector_product
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
