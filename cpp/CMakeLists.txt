cmake_minimum_required(VERSION 3.5)
project(besthea)

include(cmake/macros.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

check_insource()
setup_compiler()

enable_MKL()
enable_MPI()
enable_OpenMP()
enable_Boost()
enable_Eigen()

file(GLOB besthea_SOURCES src/*.cpp)

add_library(besthea STATIC ${besthea_SOURCES})


add_subdirectory(test/determine_spacetime_configuration)
add_subdirectory(test/dirichlet_indirect)
add_subdirectory(test/fast_tensor_dirichlet)
add_subdirectory(test/fast_tensor_neumann)
add_subdirectory(test/parallel_dirichlet_indirect)
add_subdirectory(test/parallel_fast_tensor_dirichlet)
add_subdirectory(test/routines)
add_subdirectory(test/spacetime_test)
add_subdirectory(test/test_4d_mesh)
add_subdirectory(test/test_double_layer_operator)
add_subdirectory(test/test_ensight_export)
add_subdirectory(test/test_hypersingular_operator)
add_subdirectory(test/test_mesh_generator)
add_subdirectory(test/test_parallel_double_layer_operator)
add_subdirectory(test/test_parallel_scalability)
add_subdirectory(test/uniform_tensor_dirichlet)
add_subdirectory(test/uniform_tensor_neumann)
