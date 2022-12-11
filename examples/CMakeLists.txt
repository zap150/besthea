add_subdirectory(distributed_tensor_dirichlet)
add_subdirectory(fast_initial_dirichlet_bvp)
add_subdirectory(solve_ibvp_rapidly_changing_data)
add_subdirectory(uniform_tensor_dirichlet)
add_subdirectory(uniform_tensor_neumann)

install(FILES
  mesh_files/cube_192.txt
  mesh_files/grid_cube_xy.txt
  mesh_files/cube_24_half_scale.txt
  mesh_files/cube_12_half_scale.txt
  mesh_files/cube_12_vol_half_scale.txt
  mesh_files/scaled_crankshaft_11k.txt
  mesh_files/time_adaptive_mesh_exp_decay.txt
  mesh_files/time_adaptive_mesh_rapidly_changing_boundary_data.txt
  DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
