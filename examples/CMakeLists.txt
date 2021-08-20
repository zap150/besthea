add_subdirectory(dirichlet)
add_subdirectory(distributed_tensor_dirichlet)
add_subdirectory(uniform_tensor_dirichlet)
add_subdirectory(uniform_tensor_neumann)
add_subdirectory(dirichlet_onthefly_cpu)
add_subdirectory(neumann_onthefly_cpu)

if(USE_CUDA)
  add_subdirectory(dirichlet_onthefly_gpu)
  add_subdirectory(neumann_onthefly_gpu)
endif()

install(FILES 
  mesh_files/cube_192.txt mesh_files/grid_cube_xy.txt mesh_files/cube_24_half_scale.txt mesh_files/cube_12_half_scale.txt mesh_files/scaled_crankshaft_11k.txt
  DESTINATION "${CMAKE_BINARY_DIR}/bin")
