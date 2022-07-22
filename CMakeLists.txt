cmake_minimum_required(VERSION 3.13)
project(besthea)

include(GNUInstallDirs)

include(cmake/macros.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

# like option but works for more than two values
set(BESTHEA_CUDA_OPTION_DESCRIPTION "Also build besthea_cuda to enable "
  "GPU-accelerated functionality of besthea using CUDA")
set(BESTHEA_CUDA "AUTO" CACHE STRING "${BESTHEA_CUDA_OPTION_DESCRIPTION}")
set_property(CACHE BESTHEA_CUDA PROPERTY STRINGS "AUTO" "ENABLE" "DISABLE")

check_insource()
setup_compiler()
setup_CUDA()

enable_filesystem()
enable_MKL()
enable_MPI()
enable_OpenMP()
enable_Boost()
enable_Eigen()
enable_Lyra()

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
  include/besthea/distributed_diagonal_matrix.h
  include/besthea/distributed_fast_spacetime_be_assembler.h
  include/besthea/distributed_fast_spacetime_initial_be_assembler.h
  include/besthea/distributed_fast_spacetime_be_space.h
  include/besthea/distributed_initial_pFMM_matrix.h
  include/besthea/distributed_pFMM_matrix.h
  include/besthea/distributed_spacetime_be_identity.h
  include/besthea/distributed_spacetime_cluster_tree.h
  include/besthea/distributed_spacetime_tensor_mesh.h
  include/besthea/fe_identity.h
  include/besthea/fe_space.h
  include/besthea/fmm_routines.h
  include/besthea/full_matrix.h
  include/besthea/general_spacetime_cluster.h
  include/besthea/indices.h
  include/besthea/io_routines.h
  include/besthea/iterative_inverse.h
  include/besthea/lagrange_interpolant.h
  include/besthea/linear_algebra.h
  include/besthea/linear_operator.h
  include/besthea/mainpage.h
  include/besthea/matrix.h
  include/besthea/mesh.h
  include/besthea/mesh_structures.h
  include/besthea/mkl_cg_inverse.h
  include/besthea/mkl_fgmres_inverse.h
  include/besthea/quadrature.h
  include/besthea/scheduling_time_cluster.h
  include/besthea/settings.h
  include/besthea/volume_space_cluster.h
  include/besthea/volume_space_cluster_tree.h
  include/besthea/spacetime_be_identity.h
  include/besthea/spacetime_be_space.h
  include/besthea/spacetime_heat_adl_kernel_antiderivative.h
  include/besthea/spacetime_heat_dl_kernel_antiderivative.h
  include/besthea/spacetime_heat_hs_kernel_antiderivative.h
  include/besthea/spacetime_heat_initial_kernel_antiderivative.h
  include/besthea/spacetime_heat_initial_m0_kernel_antiderivative.h
  include/besthea/spacetime_heat_initial_m1_kernel_antiderivative.h
  include/besthea/spacetime_heat_kernel.h
  include/besthea/spacetime_heat_kernel_normal_derivative.h
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
  include/besthea/spacetime_basis_function.h
  include/besthea/spacetime_basis_tetra_p0.h
  include/besthea/spacetime_basis_tetra_p1.h
  include/besthea/tetrahedral_spacetime_be_space.h
  include/besthea/tetrahedral_spacetime_be_identity.h
  include/besthea/tetrahedral_spacetime_be_assembler.h
  include/besthea/spacetime_constant_kernel.h
  include/besthea/tetrahedral_spacetime_be_evaluator.h
)

set(besthea_cuda_INCLUDES
  include/besthea/besthea_cuda.h
  include/besthea/gpu_apply_load_distribution.h
  include/besthea/gpu_apply_timer_collection.h
  include/besthea/gpu_apply_vectors_data.h
  include/besthea/gpu_onthefly_helpers.h
  include/besthea/timer_cuda.h
  include/besthea/uniform_spacetime_be_matrix_onthefly_gpu.h
  include/besthea/uniform_spacetime_tensor_mesh_gpu.h
)

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
  src/distributed_diagonal_matrix.cpp
  src/distributed_fast_spacetime_be_assembler.cpp
  src/distributed_fast_spacetime_initial_be_assembler.cpp
  src/distributed_fast_spacetime_be_space.cpp
  src/distributed_initial_pFMM_matrix.cpp
  src/distributed_pFMM_matrix.cpp
  src/distributed_spacetime_be_identity.cpp
  src/distributed_spacetime_cluster_tree.cpp
  src/distributed_spacetime_tensor_mesh.cpp
  src/fe_identity.cpp
  src/fe_space.cpp
  src/fmm_routines.cpp
  src/full_matrix.cpp
  src/general_spacetime_cluster.cpp
  src/indices.cpp
  src/io_routines.cpp
  src/linear_operator.cpp
  src/mkl_cg_inverse.cpp
  src/mkl_fgmres_inverse.cpp
  src/quadrature.cpp
  src/settings.cpp
  src/volume_space_cluster_tree.cpp
  src/spacetime_be_identity.cpp
  src/spacetime_be_space.cpp
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
  src/spacetime_basis_tetra_p0.cpp
  src/spacetime_basis_tetra_p1.cpp
  src/tetrahedral_spacetime_be_space.cpp
  src/tetrahedral_spacetime_be_identity.cpp
  src/tetrahedral_spacetime_be_assembler.cpp
  src/tetrahedral_spacetime_be_evaluator.cpp
)

set(besthea_cuda_SOURCES
  src/gpu_apply_load_distribution.cpp
  src/gpu_apply_timer_collection.cpp
  src/gpu_apply_vectors_data.cpp
  src/gpu_onthefly_helpers.cpp
  src/uniform_spacetime_be_matrix_onthefly_gpu.cu
  src/uniform_spacetime_tensor_mesh_gpu.cpp
)

if (NOT DEFINED BESTHEA_SIMD_WIDTH)
  set(BESTHEA_SIMD_WIDTH 8)
endif()

message(STATUS "Setting BESTHEA_SIMD_WIDTH to ${BESTHEA_SIMD_WIDTH}")

set(besthea_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(besthea STATIC ${besthea_SOURCES})
target_include_directories(besthea PUBLIC ${besthea_INCLUDE_DIRS})
target_include_directories(besthea SYSTEM PUBLIC
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${Lyra_INCLUDE_DIRS}
  ${MKL_INCLUDE_DIRS}
  ${MPI_INCLUDE_DIRS})
target_link_libraries(besthea PUBLIC
  ${STDFS_LIBRARIES}
  ${MKL_LIBRARIES}
  ${MPI_CXX_LIBRARIES})
target_compile_definitions(besthea PUBLIC
  MKL_INT=long
  BESTHEA_SIMD_WIDTH=${BESTHEA_SIMD_WIDTH})
  
if(BESTHEA_USE_CUDA)
  set(besthea_cuda_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_library(besthea_cuda STATIC ${besthea_cuda_SOURCES})
  target_include_directories(besthea_cuda PUBLIC
    ${besthea_cuda_INCLUDE_DIRS}
    ${besthea_INCLUDE_DIRS})
  target_include_directories(besthea_cuda SYSTEM PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${Lyra_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIRS}
    ${MPI_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(besthea_cuda PUBLIC
    ${STDFS_LIBRARIES}
    ${MKL_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${CUDA_LIBRARIES}
    besthea)
  target_compile_definitions(besthea_cuda PUBLIC
    MKL_INT=long
    BESTHEA_SIMD_WIDTH=${BESTHEA_SIMD_WIDTH}
    BESTHEA_USE_CUDA)
endif()

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
  )  
install(FILES ${besthea_INCLUDES}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

if(BESTHEA_USE_CUDA)
  install(TARGETS besthea_cuda
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
    )
  install(FILES ${besthea_cuda_INCLUDES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
endif()

option(BUILD_EXAMPLES "Build examples using the library" ON)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
