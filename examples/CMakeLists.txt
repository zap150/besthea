add_subdirectory(distributed_tensor_dirichlet)
add_subdirectory(uniform_tensor_dirichlet)
add_subdirectory(uniform_tensor_neumann)

install(FILES
  mesh_files/cube_192.txt
  mesh_files/grid_cube_xy.txt
  mesh_files/cube_24_half_scale.txt
  mesh_files/cube_12_half_scale.txt
  mesh_files/scaled_crankshaft_11k.txt
  DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
