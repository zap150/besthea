/*
 * Copyright 2019, VSB - Technical University of Ostrava and Graz University of
 * Technology All rights reserved. Redistribution and use in source and binary
 * forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided
 * with the distribution. Neither the name of VSB - Technical University of
 * Ostrava and Graz University of Technology nor the names of its contributors
 * may be used to endorse or promote products  derived from this software
 * without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL VSB - TECHNICAL UNIVERSITY OF OSTRAVA AND
 * GRAZ UNIVERSITY OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "besthea/besthea.h"
#include "catch.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mpi.h>

using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem;
using namespace besthea::tools;

// structure defining Cauchy data
struct cauchy_data {
  static sc dirichlet( sc x1, sc x2, sc x3, const coordinates< 3 > & n, sc t ) {
    sc norm2 = ( x1 - _y[ 0 ] ) * ( x1 - _y[ 0 ] )
      + ( x2 - _y[ 1 ] ) * ( x2 - _y[ 1 ] )
      + ( x3 - _y[ 2 ] ) * ( x3 - _y[ 2 ] );
    sc value = std::pow( 4.0 * M_PI * _alpha * t, -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * t ) );
    return value;
  }

  static constexpr sc _alpha{ 4 };
  static constexpr std::array< sc, 3 > _y{ 0.0, 0.0, 1.5 };
};

TEST_CASE( "System matrices are assembled", "[matrices]" ) {
  // number of time-slices and initial time-tree parameters
  sc end_time = 1.0;
  lo n_timeslices = GENERATE( 10, 16 );
  lo initial_time_levels = 6;
  lo initial_n_min_time_elems = 2;

  // space-time tree parameters
  lo max_n_levels = GENERATE( 6, 8 );
  lo n_min_elems = GENERATE( 10, 20 );
  sc st_coeff = 4.0;
  slou spatial_nearfield_limit = GENERATE( 3, 4 );

  // space-time mesh refinement
  lo refine = GENERATE( 1 );
  lo temp_refine_factor = GENERATE( 1, 2 );

  // pFMM parameters
  lo temp_order = GENERATE( 6 );
  lo spat_order = GENERATE( 6 );
  lo order_sing = 4;
  lo order_reg = 4;

  // MPI info
  int my_rank, n_processes;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( comm, &my_rank );
  MPI_Comm_size( comm, &n_processes );

  // setup mesh parameters
  lo space_init_refine = 0;
  lo time_refinement = temp_refine_factor * refine;
  lo space_refinement = refine;
  lo process_assignment_strategy = 1;

  // define paths to files
  std::string spatial_mesh_file = "./mesh_files/cube_12.txt";
  std::string geometry_dir = "./catch/decomposition/";
  std::string tree_vector_file = geometry_dir + "tree_structure.bin";
  std::string cluster_bounds_file = geometry_dir + "cluster_bounds.bin";
  std::string process_assignment_file = geometry_dir + "process_assignment.bin";
  std::filesystem::create_directory( geometry_dir );

  // process #0 creates time slices, initial temporal tree, and its
  // decomposition
  if ( my_rank == 0 ) {
    // load time mesh defining slices and create temporal tree
    temporal_mesh time_slices_mesh( 0, end_time, n_timeslices );

    time_cluster_tree initial_time_tree(
      time_slices_mesh, initial_time_levels, initial_n_min_time_elems );

    // write tree structure to file
    initial_time_tree.print_tree_structure( tree_vector_file );

    // write cluster bounds to file
    initial_time_tree.print_cluster_bounds( cluster_bounds_file );

    // write assignment of clusters to MPI processes to a file
    initial_time_tree.print_process_assignments(
      n_processes, process_assignment_strategy, process_assignment_file );

    // load the spatial mesh and combine it with initial temporal mesh
    triangular_surface_mesh space_mesh( spatial_mesh_file );
    if ( space_init_refine > 0 ) {
      space_mesh.refine( space_init_refine );
    }
    spacetime_mesh_generator generator(
      space_mesh, end_time, n_timeslices, time_refinement, space_refinement );
    generator.generate( geometry_dir, "test_mesh", "txt" );
  }
  MPI_Barrier( comm );

  // all processes load the created distributed mesh
  distributed_spacetime_tensor_mesh distributed_mesh(
    geometry_dir + "test_mesh_d.txt", tree_vector_file, cluster_bounds_file,
    process_assignment_file, &comm );

  // number of blocks in a blockvector corresponds to global number of timesteps
  lo n_blocks = distributed_mesh.get_n_temporal_elements( );

  distributed_spacetime_cluster_tree distributed_st_tree( distributed_mesh,
    max_n_levels, n_min_elems, st_coeff, spatial_nearfield_limit, &comm );
  MPI_Barrier( comm );

  // create distributed spaces
  distributed_fast_spacetime_be_space< basis_tri_p0 > distributed_space_p0(
    distributed_st_tree );
  distributed_fast_spacetime_be_space< basis_tri_p1 > distributed_space_p1(
    distributed_st_tree );

  // data for toeplitz reference matrices
  triangular_surface_mesh space_mesh_full( spatial_mesh_file );
  if ( space_init_refine > 0 ) {
    space_mesh_full.refine( space_init_refine );
  }
  uniform_spacetime_tensor_mesh spacetime_mesh_full(
    space_mesh_full, end_time, n_timeslices );
  spacetime_mesh_full.refine( refine, temp_refine_factor );

  // create spaces for full assembly
  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh_full );
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh_full );

  // section specific for single layer matrix
  SECTION( "Testing single layer matrix" ) {
    // size of a block in blockvector corresponds to number of spatial
    // elements
    lo size_of_block = distributed_mesh.get_n_elements( ) / n_blocks;

    // kernel for the single layer operator
    spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );

    // create empty matrix
    distributed_pFMM_matrix_heat_sl_p0p0 * V_dist_pFMM
      = new distributed_pFMM_matrix_heat_sl_p0p0;

    // create matrix assembler
    distributed_fast_spacetime_be_assembler distributed_assembler_v( kernel_v,
      distributed_space_p0, distributed_space_p0, &comm, order_sing, order_reg,
      temp_order, spat_order, cauchy_data::_alpha );

    // assemble pFMM matrix
    distributed_assembler_v.assemble( *V_dist_pFMM, false );

    // assemble Toeplitz reference matrix
    block_lower_triangular_toeplitz_matrix * V_full
      = new block_lower_triangular_toeplitz_matrix( );
    uniform_spacetime_be_assembler assembler_v_full(
      kernel_v, space_p0, space_p0, order_sing, order_reg );
    assembler_v_full.assemble( *V_full );

    SECTION( "Matrix-vector multiplication" ) {
      // setup the testing vector
      block_vector x_fmm( n_blocks, size_of_block, true );
      block_vector x_full( n_blocks, size_of_block, true );
      block_vector y_fmm( n_blocks, size_of_block, true );
      block_vector y_full( n_blocks, size_of_block, true );
      x_fmm.fill( 1.0 );
      x_full.fill( 1.0 );
      y_fmm.fill( 1.0 );
      y_full.fill( 1.0 );

      // multiply with approximated and full matrices
      V_dist_pFMM->apply( x_fmm, y_fmm, false, -1.0, 1.0 );
      V_full->apply( x_full, y_full, false, -1.0, 1.0 );

      // copy results to non-blocked vectors
      vector diff, x_fmm_vector;

      // compare results
      y_fmm.add( y_full, -1.0 );

      y_fmm.copy_to_vector( diff );
      x_fmm.copy_to_vector( x_fmm_vector );

      // check norm of the difference
      REQUIRE( diff.norm( ) / x_fmm_vector.norm( ) < 1e-5 );

      // check individual values in the vector
      std::vector< sc > std_diff, std_zeros;
      for ( lo i = 0; i < y_fmm.size( ); ++i ) {
        std_diff.push_back( std::abs( diff[ i ] ) );
        std_zeros.push_back( 0.0 );
      }
      CHECK_THAT( std_diff, Catch::Approx( std_zeros ).margin( 1e-3 ) );

      // delete matrix
      delete V_dist_pFMM;
    }

    SECTION( "Iterative solution" ) {
      // iterative solver settings
      sc gmres_prec = 1e-8;
      lo gmres_iter = 500;

      // allocate the RHS and solution vectors
      block_vector dir_proj;
      block_vector M_dir_proj( n_blocks, size_of_block, true );
      block_vector density_full( n_blocks, size_of_block );
      block_vector density_fmm( n_blocks, size_of_block, true );

      // setup the RHS
      space_p0.L2_projection( cauchy_data::dirichlet, dir_proj );
      uniform_spacetime_be_identity M( space_p0, space_p0 );
      M.assemble( );
      M.apply( dir_proj, M_dir_proj );

      // solve with full matrix
      sc full_prec = gmres_prec;
      lo full_iter = gmres_iter;
      V_full->mkl_fgmres_solve(
        M_dir_proj, density_full, full_prec, full_iter, full_iter );

      sc fmm_prec = gmres_prec;
      lo fmm_iter = gmres_iter;
      // solve with pFMM matrix
      V_dist_pFMM->mkl_fgmres_solve(
        M_dir_proj, density_fmm, fmm_prec, fmm_iter, fmm_iter );

      CHECK( fmm_iter == Approx( full_iter ).epsilon( 0.05 ) );

      // compare results
      density_fmm.add( density_full, -1.0 );

      // copy results to non-blocked vectors
      vector diff, density_full_vector;
      density_fmm.copy_to_vector( diff );
      density_full.copy_to_vector( density_full_vector );

      REQUIRE( diff.norm( ) / density_full_vector.norm( ) < 1e-3 );

      // delete matrix
      delete V_dist_pFMM;
    }
  }

  SECTION( "Testing double layer matrix" ) {
    // kernel for the double layer operator
    spacetime_heat_dl_kernel_antiderivative kernel_k( cauchy_data::_alpha );

    // create empty matrix
    distributed_pFMM_matrix_heat_dl_p0p1 * K_dist_pFMM
      = new distributed_pFMM_matrix_heat_dl_p0p1;

    // create matrix assembler
    distributed_fast_spacetime_be_assembler distributed_assembler_k( kernel_k,
      distributed_space_p0, distributed_space_p1, &comm, order_sing, order_reg,
      temp_order, spat_order, cauchy_data::_alpha );

    // assemble pFMM matrix
    distributed_assembler_k.assemble( *K_dist_pFMM );

    // get numbers of rows/columns of individual blocks
    lo rows_of_block = K_dist_pFMM->get_dim_domain( );
    lo cols_of_block = K_dist_pFMM->get_dim_range( );

    // assemble Toeplitz reference matrix
    block_lower_triangular_toeplitz_matrix * K_full
      = new block_lower_triangular_toeplitz_matrix( );
    uniform_spacetime_be_assembler assembler_k_full(
      kernel_k, space_p0, space_p1, order_sing, order_reg );
    assembler_k_full.assemble( *K_full );

    SECTION( "Matrix-vector multiplication" ) {
      // setup the testing vector
      block_vector x_fmm( n_blocks, cols_of_block, true );
      block_vector x_full( n_blocks, cols_of_block, true );
      block_vector y_fmm( n_blocks, rows_of_block, true );
      block_vector y_full( n_blocks, rows_of_block, true );
      x_fmm.fill( 1.0 );
      x_full.fill( 1.0 );
      y_fmm.fill( 1.0 );
      y_full.fill( 1.0 );

      // multiply with approximated and full matrices
      K_dist_pFMM->apply( x_fmm, y_fmm );
      K_full->apply( x_full, y_full );

      // copy results to non-blocked vectors
      vector diff, x_fmm_vector;

      // compare results
      y_fmm.add( y_full, -1.0 );

      y_fmm.copy_to_vector( diff );
      x_fmm.copy_to_vector( x_fmm_vector );

      // check norm of the difference
      REQUIRE( diff.norm( ) / x_fmm_vector.norm( ) < 1e-5 );

      // check individual values in the vector
      std::vector< sc > std_diff, std_zeros;
      for ( lo i = 0; i < y_fmm.size( ); ++i ) {
        std_diff.push_back( std::abs( diff[ i ] ) );
        std_zeros.push_back( 0.0 );
      }
      CHECK_THAT( std_diff, Catch::Approx( std_zeros ).margin( 1e-3 ) );

      // delete matrix
      delete K_dist_pFMM;
    }
  }

  SECTION( "Testing adjoint double layer matrix" ) {
    // kernel for the adjoint double layer operator
    spacetime_heat_adl_kernel_antiderivative kernel_ak( cauchy_data::_alpha );

    // create empty matrix
    distributed_pFMM_matrix_heat_adl_p1p0 * K_adj_dist_pFMM
      = new distributed_pFMM_matrix_heat_adl_p1p0;

    // create matrix assembler
    distributed_fast_spacetime_be_assembler distributed_assembler_ak( kernel_ak,
      distributed_space_p1, distributed_space_p0, &comm, order_sing, order_reg,
      temp_order, spat_order, cauchy_data::_alpha );

    // assemble pFMM matrix
    distributed_assembler_ak.assemble( *K_adj_dist_pFMM );

    // get numbers of rows/columns of individual blocks
    lo rows_of_block = K_adj_dist_pFMM->get_dim_domain( );
    lo cols_of_block = K_adj_dist_pFMM->get_dim_range( );

    // assemble Toeplitz reference matrix
    block_lower_triangular_toeplitz_matrix * K_adj_full
      = new block_lower_triangular_toeplitz_matrix( );

    uniform_spacetime_be_assembler assembler_k_full(
      kernel_ak, space_p1, space_p0, order_sing, order_reg );
    assembler_k_full.assemble( *K_adj_full );

    SECTION( "Matrix-vector multiplication" ) {
      // setup the testing vector
      block_vector x_fmm( n_blocks, cols_of_block, true );
      block_vector x_full( n_blocks, cols_of_block, true );
      block_vector y_fmm( n_blocks, rows_of_block, true );
      block_vector y_full( n_blocks, rows_of_block, true );
      x_fmm.fill( 1.0 );
      x_full.fill( 1.0 );
      y_fmm.fill( 1.0 );
      y_full.fill( 1.0 );

      // multiply with approximated and full matrices
      K_adj_dist_pFMM->apply( x_fmm, y_fmm );
      K_adj_full->apply( x_full, y_full );

      // copy results to non-blocked vectors
      vector diff, x_fmm_vector;

      // compare results
      y_fmm.add( y_full, -1.0 );

      y_fmm.copy_to_vector( diff );
      x_fmm.copy_to_vector( x_fmm_vector );

      // check norm of the difference
      REQUIRE( diff.norm( ) / x_fmm_vector.norm( ) < 1e-5 );

      // check individual values in the vector
      std::vector< sc > std_diff, std_zeros;
      for ( lo i = 0; i < y_fmm.size( ); ++i ) {
        std_diff.push_back( std::abs( diff[ i ] ) );
        std_zeros.push_back( 0.0 );
      }
      CHECK_THAT( std_diff, Catch::Approx( std_zeros ).margin( 1e-3 ) );

      // delete matrix
      delete K_adj_dist_pFMM;
    }
  }
}
