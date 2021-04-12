add_subdirectory(uniform_tensor_dirichlet)
add_subdirectory(uniform_tensor_neumann)
add_subdirectory(onthefly_cpu_dirichlet)

install(FILES 
  mesh_files/cube_192.txt mesh_files/grid_cube_xy.txt 
  DESTINATION "${CMAKE_BINARY_DIR}/bin")
