cmake_minimum_required(VERSION 3.8)
project(besthea)

include(cmake/macros.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

check_insource()
setup_compiler()
if(USE_CUDA)
  setup_CUDA()
endif()

enable_filesystem()
enable_MKL()
enable_MPI()
enable_OpenMP()
enable_Boost()
enable_Eigen()
if(BUILD_EXAMPLES)
  enable_Lyra()
endif()

set(besthea_INCLUDES
  include/besthea/basis_function.h
  include/besthea/basis_tetra_p1.h
  include/besthea/basis_tri_p0.h
  include/besthea/basis_tri_p1.h
  include/besthea/bem.h
  include/besthea/besthea.h
  include/besthea/block_iterative_inverse.h
  include/besthea/block_linear_operator.h
  include/besthea/block_lower_triangular_toeplitz_matrix.h
  include/besthea/block_matrix.h
  include/besthea/block_mkl_cg_inverse.h
  include/besthea/block_mkl_fgmres_inverse.h
  include/besthea/block_row_linear_operator.h
  include/besthea/block_row_matrix.h
  include/besthea/block_vector.h
  include/besthea/chebyshev_evaluator.h
  include/besthea/compound_block_linear_operator.h
  include/besthea/compound_linear_operator.h
  include/besthea/coordinates.h
  include/besthea/distributed_block_vector.h
  include/besthea/distributed_fast_spacetime_be_assembler.h
  include/besthea/distributed_fast_spacetime_be_space.h
  include/besthea/distributed_pFMM_matrix.h
  include/besthea/distributed_spacetime_cluster_tree.h
  include/besthea/distributed_spacetime_tensor_mesh.h
  include/besthea/fast_spacetime_be_assembler.h
  include/besthea/fast_spacetime_be_space.h
  include/besthea/fe_identity.h
  include/besthea/fe_space.h
  include/besthea/full_matrix.h
  include/besthea/general_spacetime_cluster.h
  include/besthea/indices.h
  include/besthea/io_routines.h
  include/besthea/iterative_inverse.h
  include/besthea/lagrange_interpolant.h
  include/besthea/linear_algebra.h
  include/besthea/linear_operator.h
  include/besthea/local_vector_routines.h
  include/besthea/mainpage.h
  include/besthea/matrix.h
  include/besthea/mesh.h
  include/besthea/mesh_structures.h
  include/besthea/mkl_cg_inverse.h
  include/besthea/mkl_fgmres_inverse.h
  include/besthea/pFMM_matrix.h
  include/besthea/quadrature.h
  include/besthea/scheduling_time_cluster.h
  include/besthea/settings.h
  include/besthea/space_cluster.h
  include/besthea/space_cluster_tree.h
  include/besthea/spacetime_be_identity.h
  include/besthea/spacetime_be_space.h
  include/besthea/spacetime_cluster.h
  include/besthea/spacetime_cluster_tree.h
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
  include/besthea/spacetime_mesh_generator.h
  include/besthea/spacetime_slice.h
  include/besthea/spacetime_tensor_mesh.h
  include/besthea/sparse_matrix.h
  include/besthea/temporal_mesh.h
  include/besthea/tetrahedral_spacetime_mesh.h
  include/besthea/tetrahedral_volume_mesh.h
  include/besthea/time_cluster.h
  include/besthea/time_cluster_tree.h
  include/besthea/time_measurer.h
  include/besthea/timer.h
  include/besthea/tools.h
  include/besthea/tree_structure.h
  include/besthea/triangular_surface_mesh.h
  include/besthea/uniform_spacetime_be_assembler.h
  include/besthea/uniform_spacetime_be_evaluator.h
  include/besthea/uniform_spacetime_be_identity.h
  include/besthea/uniform_spacetime_be_matrix_onthefly_cpu.h
  include/besthea/uniform_spacetime_be_solver.h
  include/besthea/uniform_spacetime_be_space.h
  include/besthea/uniform_spacetime_initial_assembler.h
  include/besthea/uniform_spacetime_initial_evaluator.h
  include/besthea/uniform_spacetime_tensor_mesh.h
  include/besthea/vector.h
  include/besthea/volume_basis_function.h
)

if(USE_CUDA)
  set(besthea_INCLUDES
    ${besthea_INCLUDES}
    include/besthea/gpu_onthefly_helpers.h
    include/besthea/time_measurer_cuda.h
    include/besthea/uniform_spacetime_be_matrix_onthefly_gpu.h
    include/besthea/uniform_spacetime_tensor_mesh_gpu.h
  )
endif()

set(besthea_SOURCES
  src/basis_tetra_p1.cpp
  src/basis_tri_p0.cpp
  src/basis_tri_p1.cpp
  src/block_linear_operator.cpp
  src/block_lower_triangular_toeplitz_matrix.cpp
  src/block_mkl_cg_inverse.cpp
  src/block_mkl_fgmres_inverse.cpp
  src/block_row_matrix.cpp
  src/block_vector.cpp
  src/compound_block_linear_operator.cpp
  src/compound_linear_operator.cpp
  src/coordinates.cpp
  src/distributed_block_vector.cpp
  src/distributed_fast_spacetime_be_assembler.cpp
  src/distributed_fast_spacetime_be_space.cpp
  src/distributed_pFMM_matrix.cpp
  src/distributed_spacetime_cluster_tree.cpp
  src/distributed_spacetime_tensor_mesh.cpp
  src/fast_spacetime_be_assembler.cpp
  src/fast_spacetime_be_space.cpp
  src/fe_identity.cpp
  src/fe_space.cpp
  src/full_matrix.cpp
  src/indices.cpp
  src/io_routines.cpp
  src/linear_operator.cpp
  src/mkl_cg_inverse.cpp
  src/mkl_fgmres_inverse.cpp
  src/pFMM_matrix.cpp
  src/quadrature.cpp
  src/settings.cpp
  src/space_cluster_tree.cpp
  src/spacetime_be_identity.cpp
  src/spacetime_be_space.cpp
  src/spacetime_cluster_tree.cpp
  src/spacetime_mesh_generator.cpp
  src/spacetime_tensor_mesh.cpp
  src/sparse_matrix.cpp
  src/temporal_mesh.cpp
  src/tetrahedral_spacetime_mesh.cpp
  src/tetrahedral_volume_mesh.cpp
  src/time_cluster_tree.cpp
  src/tree_structure.cpp
  src/triangular_surface_mesh.cpp
  src/uniform_spacetime_be_assembler.cpp
  src/uniform_spacetime_be_evaluator.cpp
  src/uniform_spacetime_be_identity.cpp
  src/uniform_spacetime_be_matrix_onthefly_cpu.cpp
  src/uniform_spacetime_be_solver.cpp
  src/uniform_spacetime_be_space.cpp
  src/uniform_spacetime_initial_assembler.cpp
  src/uniform_spacetime_initial_evaluator.cpp
  src/uniform_spacetime_tensor_mesh.cpp
  src/vector.cpp
)

if(USE_CUDA)
  set(besthea_SOURCES
    ${besthea_SOURCES}
    src/gpu_onthefly_helpers.cpp
    src/uniform_spacetime_be_matrix_onthefly_gpu.cu
    src/uniform_spacetime_tensor_mesh_gpu.cpp
  )
endif()

if (NOT DEFINED DATA_WIDTH)
set(DATA_WIDTH 8)
endif()

message(STATUS "Setting DATA_WIDTH to ${DATA_WIDTH}")

set(besthea_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(besthea STATIC ${besthea_SOURCES})
target_include_directories(besthea PUBLIC ${besthea_INCLUDE_DIRS})
target_include_directories(besthea SYSTEM PUBLIC
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${MKL_INCLUDE_DIRS}
  ${MPI_INCLUDE_DIRS}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(besthea PUBLIC
  ${STDFS_LIBRARIES}
  ${MKL_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  ${CUDA_LIBRARIES})
target_compile_definitions(besthea PUBLIC
  MKL_INT=long
  DATA_WIDTH=${DATA_WIDTH})

if(USE_CUDA)
  target_include_directories(besthea SYSTEM PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(besthea PUBLIC
    ${CUDA_LIBRARIES})
  target_compile_definitions(besthea PUBLIC
    BESTHEA_USE_CUDA)
endif()


install (TARGETS besthea ARCHIVE DESTINATION "${CMAKE_BINARY_DIR}/lib")
install(FILES ${besthea_INCLUDES}
  DESTINATION "${CMAKE_BINARY_DIR}/include/besthea")

option(BUILD_EXAMPLES "Build examples using the library" OFF)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
