cmake_minimum_required(VERSION 3.5)
project(besthea)

include(cmake/macros.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

check_insource()
setup_compiler()

enable_filesystem()
enable_MKL()
enable_MPI()
enable_OpenMP()
enable_Boost()
enable_Eigen()
enable_catch()
enable_Lyra()

set(besthea_SOURCES
include/besthea/basis_function.h
include/besthea/basis_tetra_p1.h src/basis_tetra_p1.cpp
include/besthea/basis_tri_p0.h src/basis_tri_p0.cpp
include/besthea/basis_tri_p1.h src/basis_tri_p1.cpp
include/besthea/bem.h
include/besthea/besthea.h
include/besthea/block_iterative_inverse.h
include/besthea/block_linear_operator.h src/block_linear_operator.cpp
include/besthea/block_lower_triangular_toeplitz_matrix.h src/block_lower_triangular_toeplitz_matrix.cpp
include/besthea/block_matrix.h
include/besthea/block_mkl_cg_inverse.h src/block_mkl_cg_inverse.cpp
include/besthea/block_mkl_fgmres_inverse.h src/block_mkl_fgmres_inverse.cpp
include/besthea/block_row_linear_operator.h
include/besthea/block_row_matrix.h src/block_row_matrix.cpp
include/besthea/block_vector.h src/block_vector.cpp
include/besthea/chebyshev_evaluator.h
include/besthea/compound_block_linear_operator.h src/compound_block_linear_operator.cpp
include/besthea/compound_linear_operator.h src/compound_linear_operator.cpp
include/besthea/coordinates.h src/coordinates.cpp
include/besthea/distributed_fast_spacetime_be_assembler.h src/distributed_fast_spacetime_be_assembler.cpp
include/besthea/distributed_fast_spacetime_be_space.h src/distributed_fast_spacetime_be_space.cpp
include/besthea/distributed_pFMM_matrix.h src/distributed_pFMM_matrix.cpp
include/besthea/distributed_spacetime_cluster_tree.h src/distributed_spacetime_cluster_tree.cpp
include/besthea/distributed_spacetime_tensor_mesh.h src/distributed_spacetime_tensor_mesh.cpp
include/besthea/fast_spacetime_be_assembler.h src/fast_spacetime_be_assembler.cpp
include/besthea/fast_spacetime_be_space.h src/fast_spacetime_be_space.cpp
include/besthea/fe_identity.h src/fe_identity.cpp
include/besthea/fe_space.h src/fe_space.cpp
include/besthea/full_matrix.h src/full_matrix.cpp
include/besthea/general_spacetime_cluster.h
include/besthea/indices.h src/indices.cpp
include/besthea/io_routines.h src/io_routines.cpp
include/besthea/iterative_inverse.h
include/besthea/lagrange_interpolant.h
include/besthea/linear_algebra.h
include/besthea/linear_operator.h src/linear_operator.cpp
include/besthea/local_vector_routines.h
include/besthea/mainpage.h
include/besthea/matrix.h
include/besthea/mesh.h
include/besthea/mesh_structures.h
include/besthea/mkl_cg_inverse.h src/mkl_cg_inverse.cpp
include/besthea/mkl_fgmres_inverse.h src/mkl_fgmres_inverse.cpp
include/besthea/pFMM_matrix.h src/pFMM_matrix.cpp
include/besthea/quadrature.h src/quadrature.cpp
include/besthea/scheduling_time_cluster.h
include/besthea/settings.h
include/besthea/space_cluster.h
include/besthea/space_cluster_tree.h src/space_cluster_tree.cpp
include/besthea/spacetime_be_identity.h src/spacetime_be_identity.cpp
include/besthea/spacetime_be_space.h src/spacetime_be_space.cpp
include/besthea/spacetime_cluster.h
include/besthea/spacetime_cluster_tree.h src/spacetime_cluster_tree.cpp
include/besthea/spacetime_heat_adl_kernel_antiderivative.h
include/besthea/spacetime_heat_dl_kernel_antiderivative.h
include/besthea/spacetime_heat_hs_kernel_antiderivative.h
include/besthea/spacetime_heat_initial_kernel_antiderivative.h
include/besthea/spacetime_heat_initial_m0_kernel_antiderivative.h
include/besthea/spacetime_heat_initial_m1_kernel_antiderivative.h
include/besthea/spacetime_heat_kernel.h
include/besthea/spacetime_heat_kernel_antiderivative.h
include/besthea/spacetime_heat_sl_kernel_antiderivative.h
include/besthea/spacetime_initial_kernel_antiderivative.h
include/besthea/spacetime_kernel.h
include/besthea/spacetime_kernel_antiderivative.h
include/besthea/spacetime_mesh_generator.h src/spacetime_mesh_generator.cpp
include/besthea/spacetime_slice.h
include/besthea/spacetime_tensor_mesh.h src/spacetime_tensor_mesh.cpp
include/besthea/sparse_matrix.h src/sparse_matrix.cpp
include/besthea/temporal_mesh.h src/temporal_mesh.cpp
include/besthea/tetrahedral_spacetime_mesh.h src/tetrahedral_spacetime_mesh.cpp
include/besthea/tetrahedral_volume_mesh.h src/tetrahedral_volume_mesh.cpp
include/besthea/time_cluster.h 
include/besthea/time_cluster_tree.h src/time_cluster_tree.cpp
include/besthea/timer.h
include/besthea/tools.h
include/besthea/tree_structure.h src/tree_structure.cpp
include/besthea/triangular_surface_mesh.h src/triangular_surface_mesh.cpp
include/besthea/uniform_spacetime_be_assembler.h src/uniform_spacetime_be_assembler.cpp
include/besthea/uniform_spacetime_be_evaluator.h src/uniform_spacetime_be_evaluator.cpp
include/besthea/uniform_spacetime_be_identity.h src/uniform_spacetime_be_identity.cpp
include/besthea/uniform_spacetime_be_solver.h src/uniform_spacetime_be_solver.cpp
include/besthea/uniform_spacetime_be_space.h src/uniform_spacetime_be_space.cpp
include/besthea/uniform_spacetime_initial_assembler.h src/uniform_spacetime_initial_assembler.cpp
include/besthea/uniform_spacetime_initial_evaluator.h src/uniform_spacetime_initial_evaluator.cpp
include/besthea/uniform_spacetime_tensor_mesh.h src/uniform_spacetime_tensor_mesh.cpp
include/besthea/vector.h src/vector.cpp
include/besthea/volume_basis_function.h
)
include_directories(include)
add_library(besthea STATIC ${besthea_SOURCES})

install (TARGETS besthea ARCHIVE DESTINATION "${CMAKE_BINARY_DIR}/lib")

option(BUILD_EXAMPLES "Build examples using the library" OFF)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
