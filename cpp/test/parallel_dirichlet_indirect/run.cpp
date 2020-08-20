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

#define USE_P0_BASIS

#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem;
using namespace besthea::tools;

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

int main( int argc, char * argv[] ) {
  // initialize MPI related parameters
  int provided;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  int myRank, n_processes;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( MPI_COMM_WORLD, &myRank );
  MPI_Comm_size( comm, &n_processes );

  // declaration of required parameters.
  lo order_sing, order_reg, temp_order, spat_order;
  sc st_coeff;
  lo n_blocks, size_of_block;

  st_coeff = 4.0;
  temp_order = 6;
  spat_order = 6;
  order_sing = 4;
  order_reg = 4;

  lo test_case = 2;

  // mesh data to construct equivalent standard spacetime mesh and distributed
  // spacetime mesh for the test.
  std::string spatial_mesh_file = "./mesh_files/cube_12.txt";
  // parameters for standard spacetime mesh
  int refine = 1;
  int temp_refine_factor = 2;
  lo n_timesteps = 8;
  sc end_time = 1.0;
  // parameters for distributed spacetime mesh
  std::string time_file = "./testfile.txt";  // file defining temporal slices
  lo time_refinement = 2;                    // defining mesh within slices
  lo space_refinement = 1;
  //
  lo process_assignment_strategy = 1;

  // kernel for the single layer operator
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );

  // V approximated by pFMM
  // spacetime_cluster_tree tree( spacetime_mesh, 5, 2, 10, st_coeff );
  // tree.print( );

  // fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
  // pFMM_matrix_heat_sl_p0p0 * V_pFMM = new pFMM_matrix_heat_sl_p0p0;
  // fast_spacetime_be_assembler fast_assembler_v( kernel_v, space_p0_pFMM,
  //   space_p0_pFMM, order_sing, order_reg, temp_order, spat_order,
  //   cauchy_data::_alpha, 1.5, false );

  // fast_assembler_v.assemble( *V_pFMM );

  // V approximated by distributed pFMM


  MPI_Barrier( comm );

  // generation of distributed mesh
  std::string tree_vector_file
    = "./parallel_dirichlet_indirect/test_case_1/tree_structure.bin";
  std::string cluster_bounds_file
    = "./parallel_dirichlet_indirect/test_case_1/cluster_bounds.bin";
  std::string process_assignment_file
    = "./parallel_dirichlet_indirect/test_case_1/process_assignment.bin";
  if ( myRank == 0 ) {
    std::cout << "### start mesh generation ###" << std::endl;

    // load time mesh defining slices and create temporal tree
    temporal_mesh time_mesh( time_file );
    lo time_levels = 6;
    lo n_min_time_elems = 2;
    time_cluster_tree time_tree( time_mesh, time_levels, n_min_time_elems );

    // write tree structure to file
    time_tree.print_tree_structure( tree_vector_file );

    // write cluster bounds to file
    time_tree.print_cluster_bounds( cluster_bounds_file );

    // compute process assignment and write it to file

    std::cout << "n_processes: " << n_processes << ", strategy: "
              << process_assignment_strategy << std::endl;

    time_tree.print_process_assignments(
      n_processes, process_assignment_strategy, process_assignment_file );

    spacetime_mesh_generator generator(
      spatial_mesh_file, time_file, time_refinement, space_refinement );

    generator.generate( "./parallel_dirichlet_indirect/test_case_1/",
                        "test_mesh", "txt" );
    std::cout << "### end mesh generation ###" << std::endl;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  if ( myRank == 0 ) {
    std::cout << "assembling distributed pFMM matrix" << std::endl;
  }

  distributed_spacetime_tensor_mesh distributed_mesh(
    "./parallel_dirichlet_indirect/test_case_1/test_mesh_d.txt",
    tree_vector_file, cluster_bounds_file, process_assignment_file, &comm );

  // number of blocks in a blockvector corresponds to global number of timesteps
  n_blocks = distributed_mesh.get_n_temporal_elements( );
  // size of a block in blockvector corresponds to number of spatial elements
  size_of_block = distributed_mesh.get_n_elements( ) / n_blocks;

  distributed_spacetime_cluster_tree distributed_st_tree(
    distributed_mesh, 6, 40, st_coeff, 3, &comm );
  MPI_Barrier( comm );
  if ( myRank == 0 ) {
    distributed_st_tree.print( );
    std::cout << std::endl << std::endl << std::endl;
    // distributed_st_tree.get_distribution_tree( )->print( );
  }

  distributed_fast_spacetime_be_space< basis_tri_p0 > distributed_space_p0(
    distributed_st_tree );
  distributed_pFMM_matrix_heat_sl_p0p0 * V_dist_pFMM
    = new distributed_pFMM_matrix_heat_sl_p0p0;

  distributed_fast_spacetime_be_assembler distributed_assembler_v( kernel_v,
    distributed_space_p0, distributed_space_p0, &comm, order_sing, order_reg,
    temp_order, spat_order, cauchy_data::_alpha);

  distributed_assembler_v.assemble( *V_dist_pFMM );

  MPI_Barrier( comm );
  if ( myRank == 0 ) {
    std::cout << "finished assembling distributed pFMM matrix" << std::endl;
  }

  if ( myRank == 0 ) {
    distributed_mesh.get_distribution_tree( )->print_tree_human_readable( 2, true );
  }
  MPI_Barrier( comm );
  if ( myRank == 1 ) {
    distributed_mesh.get_distribution_tree( )->print_tree_human_readable( 2, true );
  }
  MPI_Barrier( comm );
  if ( myRank == 2 ) {
    distributed_mesh.get_distribution_tree( )->print_tree_human_readable( 2, true );
  }

  if ( test_case == 1 ) {
    //##########################################################################
    //##########################################################################
    // lo block_id = n_blocks - 1 - 2;
    lo block_id = 0;
    lo block_evaluation_id = 16;
    lo toeplitz_id = block_evaluation_id - block_id;
    //   lo block_evaluation_id = n_blocks - 1;

    // let process 0 compute the result for a subblock with the directly
    // generated matrix V
    vector x_loc_0( size_of_block );
    // x_loc_0( 9 ) = 1.0;
    x_loc_0.fill( 1.0 );
    vector y_loc( size_of_block );

    block_vector full_block_vector( n_blocks, size_of_block, true );
    full_block_vector.get_block( block_id ) = x_loc_0;
    block_vector applied_pFMM( n_blocks, size_of_block, true );

    if ( myRank == 0 ) {
      // construct matrix V directly
      triangular_surface_mesh space_mesh( spatial_mesh_file );
      uniform_spacetime_tensor_mesh spacetime_mesh(
        space_mesh, end_time, n_timesteps );
      spacetime_mesh.refine( refine, temp_refine_factor );
      // V without approximation
      block_lower_triangular_toeplitz_matrix * V
        = new block_lower_triangular_toeplitz_matrix( );

      uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
      uniform_spacetime_be_assembler assembler_v(
        kernel_v, space_p0, space_p0, order_sing, order_reg );
      assembler_v.assemble( *V );

      // spacetime_cluster_tree tree( spacetime_mesh, 5, 5, 10, st_coeff );
      // tree.print( );
      // fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
      // pFMM_matrix_heat_sl_p0p0 * V_pFMM = new pFMM_matrix_heat_sl_p0p0;
      // fast_spacetime_be_assembler fast_assembler_v( kernel_v, space_p0_pFMM,
      //   space_p0_pFMM, order_sing, order_reg, temp_order, spat_order,
      //   cauchy_data::_alpha, 1.5, false );
      // fast_assembler_v.assemble( *V_pFMM );

      std::cout << "applying V" << std::endl;
      std::cout << "block_id = " << block_id << ", block_evaluation_id "
                << block_evaluation_id << std::endl;
      full_matrix & V_block_loc = V->get_block( toeplitz_id );
      V_block_loc.apply( x_loc_0, y_loc );
      // V_pFMM->apply( full_block_vector, applied_pFMM );
    }
    // compute the corresponding result with the distributed matrix V_dist_pFMM
    if ( myRank == 0 ) {
      std::cout << "applying distributed pFMM matrix" << std::endl;
    }

    MPI_Barrier( comm );
    block_vector applied_dist_pFMM( n_blocks, size_of_block, true );

    V_dist_pFMM->apply( full_block_vector, applied_dist_pFMM );

    // ############################################################
    // ### print the results
    if ( myRank == 0 ) {
      std::cout << "resulting subblock distributed pFMM multiplication"
              << std::endl;
      std::cout << "block id " << block_id << std::endl;
      vector & subvec_dist_pFMM
        = applied_dist_pFMM.get_block( block_evaluation_id );
      subvec_dist_pFMM.print_h( );

      std::cout << "exact result block" << std::endl;
      y_loc.print_h( );

      std::cout << "error timewise" << std::endl;
      subvec_dist_pFMM.add( y_loc, -1.0 );
      std::cout << subvec_dist_pFMM.norm( ) << ", rel. "
                << subvec_dist_pFMM.norm( ) / y_loc.norm( ) << std::endl;
    }

    // // direct comparison of pFMM and dist pFMM
    // std::cout << "difference between pFMM results:" << std::endl;
    // subvec_dist_pFMM.add( subvec_pFMM, -1.0 );
    // std::cout << subvec_dist_pFMM.norm( ) << ", rel. "
    //           << subvec_dist_pFMM.norm( ) / subvec_pFMM.norm( ) << std::endl;
  } else if ( test_case == 2 ) {
    //##########################################################################
    //##########################################################################
    block_vector rand_vec( n_blocks, size_of_block );
    // let process 0 fill the random vector
    if ( myRank == 0 ) {
      std::cout << "comparison of multiplication with random vector"
                << std::endl;
      for ( lo i = 0; i < n_blocks; ++i ) {
        rand_vec.get_block( i ).random_fill( 0.5, 1.5 );
      }
    }
    // broadcast the random vector to all processes
    for ( lo i = 0; i < n_blocks; ++i ) {
      MPI_Bcast( rand_vec.get_block( i ).data( ), size_of_block,
      get_scalar_type< sc >::MPI_SC( ), 0, comm );
    }

    // let process 0 compute the result with the directly generated matrix V
    block_vector applied_std( n_blocks, size_of_block, true );
    if ( myRank == 0 ) {
      // construct matrix V directly
      triangular_surface_mesh space_mesh( spatial_mesh_file );
      uniform_spacetime_tensor_mesh spacetime_mesh(
        space_mesh, end_time, n_timesteps );
      spacetime_mesh.refine( refine, temp_refine_factor );
      // V without approximation
      block_lower_triangular_toeplitz_matrix * V
        = new block_lower_triangular_toeplitz_matrix( );

      uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
      uniform_spacetime_be_assembler assembler_v(
        kernel_v, space_p0, space_p0, order_sing, order_reg );
      assembler_v.assemble( *V );

      std::cout << "applying V" << std::endl;

      block_vector temp_vec( n_blocks, size_of_block, true );
      V->apply( rand_vec, applied_std );
      // V->apply( rand_vec, applied_std );
    }

    // compute the result with the distributed matrix V_dist_pFMM
    if ( myRank == 0 ) {
      std::cout << "applying V_dist_pFMM" << std::endl;
    }
    MPI_Barrier( comm );
    block_vector temp_vec( n_blocks, size_of_block, true );
    block_vector applied_dist_pFMM( n_blocks, size_of_block, true );
    V_dist_pFMM->apply( rand_vec, applied_dist_pFMM );

    if ( myRank == 0 ) {
      std::cout << "error timewise distributed pFMM" << std::endl;
      block_vector applied_diff( applied_dist_pFMM );
      for ( lo i = 0; i < n_blocks; ++i ) {
        applied_diff.get_block( i ).add( applied_std.get_block( i ), -1.0 );
        std::cout << applied_diff.get_block( i ).norm( ) << ", rel. "
                  << applied_diff.get_block( i ).norm( )
            / applied_std.get_block( i ).norm( )
                  << std::endl;
      }
    }
    MPI_Barrier( comm );
  }
  else if ( test_case == 3 ) {
    //##########################################################################
    //##########################################################################
    sc gmres_prec = 1e-8;
    lo gmres_iter = 500;

    block_vector M_dir_proj( n_blocks, size_of_block, true );
    block_vector direct_density( n_blocks, size_of_block );
    // let process 0 compute the result with the directly generated matrix V
    if ( myRank == 0 ) {
      // construct matrix V directly
      triangular_surface_mesh space_mesh( spatial_mesh_file );
      uniform_spacetime_tensor_mesh spacetime_mesh(
        space_mesh, end_time, n_timesteps );
      spacetime_mesh.refine( refine, temp_refine_factor );
      // V without approximation
      block_lower_triangular_toeplitz_matrix * V
        = new block_lower_triangular_toeplitz_matrix( );

      uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
      uniform_spacetime_be_assembler assembler_v(
        kernel_v, space_p0, space_p0, order_sing, order_reg );
      assembler_v.assemble( *V );

      std::cout << "applying V" << std::endl;

      block_vector dir_proj;
      space_p0.L2_projection( cauchy_data::dirichlet, dir_proj );
      uniform_spacetime_be_identity M( space_p0, space_p0 );
      M.assemble( );
      M.apply( dir_proj, M_dir_proj );

      V->mkl_fgmres_solve(
        M_dir_proj, direct_density, gmres_prec, gmres_iter, gmres_iter );
      std::cout << "iterations standard: " << gmres_iter << std::endl;
    }
    // broadcast M_dir_proj to all processes
    for ( lo i = 0; i < n_blocks; ++i ) {
      MPI_Bcast( M_dir_proj.get_block( i ).data( ), size_of_block,
      get_scalar_type< sc >::MPI_SC( ), 0, comm );
    }
    MPI_Barrier( comm );
    // compute the result with the distributed matrix V_dist_pFMM
    gmres_prec = 1e-8;
    gmres_iter = 500;
    block_vector density_pFMM( n_blocks, size_of_block, true );
    V_dist_pFMM->mkl_fgmres_solve(
      M_dir_proj, density_pFMM, gmres_prec, gmres_iter, gmres_iter );
    if ( myRank == 0 ) {
      std::cout << "iterations pFMM: " << gmres_iter << std::endl;

      std::cout << "error of densities timewise" << std::endl;
      block_vector dens_diff( density_pFMM );
      for ( lo i = 0; i < n_blocks; ++i ) {
        dens_diff.get_block( i ).add( direct_density.get_block( i ), -1.0 );
        std::cout << dens_diff.get_block( i ).norm( ) << ", rel. "
                  << dens_diff.get_block( i ).norm( )
                      / direct_density.get_block( i ).norm( )
                  << std::endl;
      }
    }
  }

  MPI_Finalize( );

}
