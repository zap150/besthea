add_subdirectory(dirichlet)
add_subdirectory(uniform_tensor_dirichlet)
add_subdirectory(uniform_tensor_neumann)
add_subdirectory(dirichlet_onthefly_cpu)
add_subdirectory(neumann_onthefly_cpu)

if(USE_CUDA)
  add_subdirectory(dirichlet_onthefly_gpu)
  add_subdirectory(neumann_onthefly_gpu)
endif()

install(FILES 
  mesh_files/cube_192.txt mesh_files/grid_cube_xy.txt 
  DESTINATION "${CMAKE_BINARY_DIR}/bin")
