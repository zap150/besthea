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
    sc value = std::pow( 4.0 * M_PI * _alpha * ( t + _shift ), -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * ( t + _shift ) ) );

    return value;
  }

  static sc neumann( sc x1, sc x2, sc x3, const coordinates< 3 > & n, sc t ) {
    sc dot = ( x1 - _y[ 0 ] ) * n[ 0 ] + ( x2 - _y[ 1 ] ) * n[ 1 ]
      + ( x3 - _y[ 2 ] ) * n[ 2 ];
    sc value = ( -1.0 / ( 2.0 * ( t + _shift ) ) ) * dot
      * dirichlet( x1, x2, x3, n, t );

    return value;
  }

  static sc initial( sc x1, sc x2, sc x3 ) {
    sc norm2 = ( x1 - _y[ 0 ] ) * ( x1 - _y[ 0 ] )
      + ( x2 - _y[ 1 ] ) * ( x2 - _y[ 1 ] )
      + ( x3 - _y[ 2 ] ) * ( x3 - _y[ 2 ] );
    sc value = std::pow( 4.0 * M_PI * _alpha * _shift, -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * _shift ) );

    return value;
  }

  static constexpr sc _alpha{ 4.0 };
  static constexpr std::array< sc, 3 > _y{ 0.0, 0.0, 1.5 };
  static constexpr sc _shift{ 0.0 };
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
  lo max_n_levels;
  lo n_blocks;

  st_coeff = 4.0;
  max_n_levels = 20;
  temp_order = 6;
  spat_order = 6;
  order_sing = 4;
  order_reg = 4;

  lo test_case = 3;
  lo geometry_case = 2;
  if ( argc == 3 ) {
    geometry_case = strtol( argv[ 1 ], NULL, 10 );
    test_case = strtol( argv[ 2 ], NULL, 10 );
  }

  // mesh data to construct equivalent standard spacetime mesh and distributed
  // spacetime mesh for the test.
  std::string spatial_mesh_file;
  std::string time_file;
  // WARNING: only uniform timesteps are allowed for the non-approximated single
  // layer operator. choose a time file and set the parameters for the standard
  // spacetime mesh accordingly
  // parameters for standard spacetime mesh
  lo n_timesteps = 8;
  int refine = 1;
  int temp_refine_factor = 2;
  sc end_time = 1.0;

  std::string geometry_dir
    = "./test_parallel_double_layer_operator/geometry_case_"
    + std::to_string( geometry_case ) + "/";
  std::filesystem::create_directory( geometry_dir );
  if ( geometry_case == 1 ) {
    spatial_mesh_file = "./mesh_files/cube_12.txt";
    time_file = "./mesh_files/time_1_8_uniform.txt";
  } else if ( geometry_case == 2 ) {
    spatial_mesh_file = "./mesh_files/icosahedron.txt";
    time_file = "./mesh_files/time_1_8_uniform.txt";
  } else if ( geometry_case == 3 ) {
    spatial_mesh_file = "./mesh_files/icosahedron.txt";
    time_file = "./mesh_files/time_1_10.txt";
    n_timesteps = 10;
  }

  // parameters for distributed spacetime mesh
  // refinement of mesh within slices
  lo time_refinement = temp_refine_factor * refine;
  lo space_refinement = refine;
  lo process_assignment_strategy = 1;

  // kernel for the double layer and adjoint double layer operator
  spacetime_heat_dl_kernel_antiderivative kernel_k( cauchy_data::_alpha );
  spacetime_heat_adl_kernel_antiderivative kernel_ak( cauchy_data::_alpha );

  MPI_Barrier( comm );
  timer t;

  // generation of distributed mesh
  std::string tree_vector_file = geometry_dir + "tree_structure.bin";
  std::string cluster_bounds_file = geometry_dir + "cluster_bounds.bin";
  std::string process_assignment_file = geometry_dir + "process_assignment.bin";
  if ( myRank == 0 ) {
    std::cout << "### geometry case: " << geometry_case << std::endl;
    std::cout << "### test case: " << test_case << std::endl;
    std::cout << "n_processes: " << n_processes
              << ", strategy: " << process_assignment_strategy << std::endl;
    std::cout << "###############################################" << std::endl;
    std::cout << "WARNING: when comparing standard and distributed pFMM"
              << " matrices the results match only if the trees coincide!"
              << std::endl;
    std::cout << "in case of bad agreement try modifying the expansion orders"
              << std::endl;
    std::cout << "###############################################" << std::endl;

    t.reset( "mesh generation" );

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
    time_tree.print_process_assignments(
      n_processes, process_assignment_strategy, process_assignment_file );

    spacetime_mesh_generator generator(
      spatial_mesh_file, time_file, time_refinement, space_refinement );

    generator.generate( geometry_dir, "test_mesh", "txt" );
    t.measure( );
  }
  MPI_Barrier( comm );

  if ( myRank == 0 ) {
    t.reset( "assembly of distributed mesh and tree " );
  }

  distributed_spacetime_tensor_mesh distributed_mesh(
    geometry_dir + "test_mesh_d.txt", tree_vector_file, cluster_bounds_file,
    process_assignment_file, &comm );

  // number of blocks in a blockvector corresponds to global number of timesteps
  n_blocks = distributed_mesh.get_n_temporal_elements( );

  distributed_spacetime_cluster_tree distributed_st_tree(
    distributed_mesh, max_n_levels, 40, st_coeff, 3, &comm );
  MPI_Barrier( comm );
  if ( myRank == 0 ) {
    t.measure( );
    // distributed_st_tree.print( );
    // std::cout << std::endl << std::endl << std::endl;
    // distributed_st_tree.get_distribution_tree( )->print( );
  }

  distributed_fast_spacetime_be_space< basis_tri_p0 > distributed_space_p0(
    distributed_st_tree );
  distributed_fast_spacetime_be_space< basis_tri_p1 > distributed_space_p1(
    distributed_st_tree );

  MPI_Barrier( comm );

  if ( myRank == 0 ) {
    std::cout << "################" << std::endl;
    std::cout << "distribution tree for process 0" << std::endl;
    distributed_mesh.get_distribution_tree( )->print_tree_human_readable(
      2, true );
    std::cout << "################" << std::endl;
  }
  MPI_Barrier( comm );
  if ( myRank == 1 ) {
    std::cout << "################" << std::endl;
    std::cout << "distribution tree for process 1" << std::endl;
    distributed_mesh.get_distribution_tree( )->print_tree_human_readable(
      2, true );
    std::cout << "################" << std::endl;
    // distributed_mesh.get_distribution_tree( )->print( );
  }

  if ( test_case == 1 ) {
    if ( myRank == 0 ) {
      t.reset( "assembling distributed pFMM matrix" );
    }
    distributed_pFMM_matrix_heat_dl_p0p1 * K_dist_pFMM
      = new distributed_pFMM_matrix_heat_dl_p0p1;
    distributed_fast_spacetime_be_assembler distributed_assembler_k( kernel_k,
      distributed_space_p0, distributed_space_p1, &comm, order_sing, order_reg,
      temp_order, spat_order, cauchy_data::_alpha );
    distributed_assembler_k.assemble( *K_dist_pFMM );

    if ( myRank == 0 ) {
      t.measure( );
    }

    lo rows_of_block = K_dist_pFMM->get_dim_domain( );
    lo cols_of_block = K_dist_pFMM->get_dim_range( );
    if ( myRank == 0 ) {
      std::cout << "cols of block = " << cols_of_block
                << ", rows of block = " << rows_of_block << std::endl;
    }
    MPI_Barrier( comm );

    lo entry_id = 0;
    lo block_id = 0;
    lo block_evaluation_id = 8;
    vector x_loc_0( cols_of_block );
    x_loc_0( entry_id ) = 1.0;
    block_vector x_block_vec( n_blocks, cols_of_block, true );
    x_block_vec.get_block( block_id ) = x_loc_0;
    // multiplicate x_block_vec with Toeplitz matrix K
    block_vector applied_std( n_blocks, rows_of_block, true );
    if ( myRank == 0 ) {
      t.reset( "assembly and application of toeplitz matrix K (from scratch)" );
      triangular_surface_mesh space_mesh( spatial_mesh_file );
      uniform_spacetime_tensor_mesh spacetime_mesh(
        space_mesh, end_time, n_timesteps );
      spacetime_mesh.refine( refine, temp_refine_factor );
      uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
      uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );

      // toeplitz matrix
      block_lower_triangular_toeplitz_matrix * K
        = new block_lower_triangular_toeplitz_matrix( );
      uniform_spacetime_be_assembler assembler_k(
        kernel_k, space_p0, space_p1, order_sing, order_reg );
      assembler_k.assemble( *K );

      // standard pFMM matrix
      // besthea::mesh::spacetime_cluster_tree tree(
      //   spacetime_mesh, max_n_levels, 2, 10, st_coeff );
      // tree.print( );
      // fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
      // fast_spacetime_be_space< basis_tri_p1 > space_p1_pFMM( tree );
      // pFMM_matrix_heat_dl_p0p1 * K_pFMM = new pFMM_matrix_heat_dl_p0p1;
      // fast_spacetime_be_assembler fast_assembler_k( kernel_k,
      //   space_p0_pFMM, space_p1_pFMM, order_sing, order_reg, temp_order,
      //   spat_order, cauchy_data::_alpha, 1.5, false );
      // fast_assembler_k.assemble( *K_pFMM );

      // apply either standard pFMM matrix or toeplitz matrix
      // K_pFMM->apply( x_block_vec, applied_std );
      K->apply( x_block_vec, applied_std );
      t.measure( );
      delete K;
      // delete K_pFMM;
    }
    MPI_Barrier( comm );

    // multiplicate x_block_vec with pFMM matrix K
    block_vector applied_dist_pFMM( n_blocks, rows_of_block, true );
    if ( myRank == 0 ) {
      t.reset( "applying distributed pFMM matrix K" );
    }
    K_dist_pFMM->apply( x_block_vec, applied_dist_pFMM );

    if ( myRank == 0 ) {
      t.measure( );
      std::cout << "resulting subblock pFMM multiplication" << std::endl;
      std::cout << "source id " << block_id << std::endl;
      std::cout << "target id " << block_evaluation_id << std::endl;
      vector & subvec_pFMM = applied_dist_pFMM.get_block( block_evaluation_id );
      vector & subvec_toeplitz = applied_std.get_block( block_evaluation_id );
      lo eval_entry_id = 8;
      std::cout << "id: " << eval_entry_id << std::endl;
      std::cout << "entry is " << subvec_pFMM[ eval_entry_id ] << std::endl;
      std::cout << "should be " << subvec_toeplitz[ eval_entry_id ]
                << std::endl;
      subvec_pFMM.print_h( );
      std::cout << "exact result block" << std::endl;
      subvec_toeplitz.print_h( );
      std::cout << "error timewise" << std::endl;
      subvec_pFMM.add( subvec_toeplitz, -1.0 );
      std::cout << subvec_pFMM.norm( ) << ", rel. "
                << subvec_pFMM.norm( ) / subvec_toeplitz.norm( ) << std::endl;
    }
    delete K_dist_pFMM;
  } else if ( test_case == 2 ) {
    if ( myRank == 0 ) {
      t.reset( "assembling distributed pFMM matrix" );
    }
    distributed_pFMM_matrix_heat_dl_p0p1 * K_dist_pFMM
      = new distributed_pFMM_matrix_heat_dl_p0p1;
    distributed_fast_spacetime_be_assembler distributed_assembler_k( kernel_k,
      distributed_space_p0, distributed_space_p1, &comm, order_sing, order_reg,
      temp_order, spat_order, cauchy_data::_alpha );
    distributed_assembler_k.assemble( *K_dist_pFMM );
    if ( myRank == 0 ) {
      t.measure( );
    }

    lo rows_of_block = K_dist_pFMM->get_dim_domain( );
    lo cols_of_block = K_dist_pFMM->get_dim_range( );
    if ( myRank == 0 ) {
      std::cout << "cols of block = " << cols_of_block
                << ", rows of block = " << rows_of_block << std::endl;
    }
    MPI_Barrier( comm );

    block_vector dir_proj( n_blocks, cols_of_block, true );
    block_vector applied_std( n_blocks, rows_of_block, true );
    if ( myRank == 0 ) {
      t.reset( "assembly and application of toeplitz matrix K (from scratch)" );
      triangular_surface_mesh space_mesh( spatial_mesh_file );
      uniform_spacetime_tensor_mesh spacetime_mesh(
        space_mesh, end_time, n_timesteps );
      spacetime_mesh.refine( refine, temp_refine_factor );
      uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
      uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );
      // compute l2 projection of dirichlet datum
      space_p1.L2_projection( cauchy_data::dirichlet, dir_proj );

      // toeplitz matrix
      block_lower_triangular_toeplitz_matrix * K
        = new block_lower_triangular_toeplitz_matrix( );
      uniform_spacetime_be_assembler assembler_k(
        kernel_k, space_p0, space_p1, order_sing, order_reg );
      assembler_k.assemble( *K );

      // standard pFMM matrix
      // besthea::mesh::spacetime_cluster_tree tree(
      //   spacetime_mesh, max_n_levels, 2, 10, st_coeff );
      // tree.print( );
      // fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
      // fast_spacetime_be_space< basis_tri_p1 > space_p1_pFMM( tree );
      // pFMM_matrix_heat_dl_p0p1 * K_pFMM = new pFMM_matrix_heat_dl_p0p1;
      // fast_spacetime_be_assembler fast_assembler_k( kernel_k,
      //   space_p0_pFMM, space_p1_pFMM, order_sing, order_reg, temp_order,
      //   spat_order, cauchy_data::_alpha, 1.5, false );
      // fast_assembler_k.assemble( *K_pFMM );

      // apply either standard pFMM matrix or toeplitz matrix
      // K_pFMM->apply( dir_proj, applied_std );
      K->apply( dir_proj, applied_std );
      t.measure( );

      delete K;
      // delete K_pFMM;
    }
    MPI_Barrier( comm );
    // broadcast dir_proj to all processes
    for ( lo i = 0; i < n_blocks; ++i ) {
      MPI_Bcast( dir_proj.get_block( i ).data( ), cols_of_block,
        get_scalar_type< sc >::MPI_SC( ), 0, comm );
    }

    // multiplicate dir_proj with pFMM matrix K
    block_vector applied_dist_pFMM( n_blocks, rows_of_block, true );
    if ( myRank == 0 ) {
      t.reset( "applying K_dist_pFMM" );
    }
    K_dist_pFMM->apply( dir_proj, applied_dist_pFMM );
    MPI_Barrier( comm );
    if ( myRank == 0 ) {
      t.measure( );
      std::cout << "error timewise" << std::endl;
      for ( lo i = 0; i < applied_std.get_block_size( ); ++i ) {
        applied_dist_pFMM.get_block( i ).add(
          applied_std.get_block( i ), -1.0 );
        std::cout << applied_dist_pFMM.get_block( i ).norm( ) << ", rel. "
                  << applied_dist_pFMM.get_block( i ).norm( )
            / applied_std.get_block( i ).norm( )
                  << std::endl;
      }
    }
    delete K_dist_pFMM;
  } else if ( test_case == 3 ) {
    if ( myRank == 0 ) {
      t.reset( "assembling distributed pFMM matrix K_adj" );
    }
    distributed_pFMM_matrix_heat_adl_p1p0 * K_adj_dist_pFMM
      = new distributed_pFMM_matrix_heat_adl_p1p0;

    distributed_fast_spacetime_be_assembler distributed_assembler_k_adj(
      kernel_ak, distributed_space_p1, distributed_space_p0, &comm, order_sing,
      order_reg, temp_order, spat_order, cauchy_data::_alpha );
    distributed_assembler_k_adj.assemble( *K_adj_dist_pFMM );

    if ( myRank == 0 ) {
      t.measure( );
    }

    lo rows_of_block = K_adj_dist_pFMM->get_dim_domain( );
    lo cols_of_block = K_adj_dist_pFMM->get_dim_range( );
    if ( myRank == 0 ) {
      std::cout << "cols of block = " << rows_of_block
                << ", rows of block = " << rows_of_block << std::endl;
    }

    lo block_id = 0;
    lo block_evaluation_id = 8;
    lo entry_id = 0;
    vector x_loc_0( cols_of_block );
    // x_loc_0.fill( 1.0 );
    x_loc_0( entry_id ) = 1.0;
    block_vector x_block_vec( n_blocks, cols_of_block, true );
    x_block_vec.get_block( block_id ) = x_loc_0;
    // multiplicate x_block_vec with Toeplitz matrix K
    block_vector applied_std( n_blocks, rows_of_block, true );
    if ( myRank == 0 ) {
      // t.reset(
      //   "assembly and application of pFMM matrix K_adj (from scratch)" );
      t.reset( "assembly and application of adjoint of K (from scratch)" );
      triangular_surface_mesh space_mesh( spatial_mesh_file );
      uniform_spacetime_tensor_mesh spacetime_mesh(
        space_mesh, end_time, n_timesteps );
      spacetime_mesh.refine( refine, temp_refine_factor );
      uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
      uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );

      // toeplitz matrix
      block_lower_triangular_toeplitz_matrix * K
        = new block_lower_triangular_toeplitz_matrix( );
      uniform_spacetime_be_assembler assembler_k(
        kernel_k, space_p0, space_p1, order_sing, order_reg );
      assembler_k.assemble( *K );

      // // standard pFMM matrix
      // besthea::mesh::spacetime_cluster_tree tree(
      //   spacetime_mesh, max_n_levels, 2, 10, st_coeff );
      // // tree.print( );
      // fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
      // fast_spacetime_be_space< basis_tri_p1 > space_p1_pFMM( tree );
      // pFMM_matrix_heat_adl_p1p0 * K_adj_pFMM = new pFMM_matrix_heat_adl_p1p0;
      // fast_spacetime_be_assembler fast_assembler_k_adj( kernel_ak,
      //   space_p1_pFMM, space_p0_pFMM, order_sing, order_reg, temp_order,
      //   spat_order, cauchy_data::_alpha, 1.5, false );
      // fast_assembler_k_adj.assemble( *K_adj_pFMM );

      K->apply( x_block_vec, applied_std, true );
      // K_adj_pFMM->apply( x_block_vec, applied_std );
      t.measure( );
      delete K;
      // delete K_adj_pFMM;
    }
    MPI_Barrier( comm );
    // multiplicate x_block_vec with pFMM matrix K
    if ( myRank == 0 ) {
      t.reset( "applying distributed pFMM matrix K_adj" );
    }
    block_vector applied_dist_pFMM( n_blocks, rows_of_block, true );
    K_adj_dist_pFMM->apply( x_block_vec, applied_dist_pFMM );
    if ( myRank == 0 ) {
      t.measure( );
      std::cout << "resulting subblock pFMM multiplication" << std::endl;
      std::cout << "source id " << block_id << std::endl;
      std::cout << "target id " << block_evaluation_id << std::endl;
      vector & subvec_pFMM = applied_dist_pFMM.get_block( block_evaluation_id );
      vector & subvec_toeplitz = applied_std.get_block( block_evaluation_id );
      lo eval_entry_id = 0;
      std::cout << "id: " << eval_entry_id << std::endl;
      std::cout << "entry is " << subvec_pFMM[ eval_entry_id ] << std::endl;
      std::cout << "should be " << subvec_toeplitz[ eval_entry_id ]
                << std::endl;
      subvec_pFMM.print_h( );
      std::cout << "exact result block" << std::endl;
      subvec_toeplitz.print_h( );
      std::cout << "error timewise" << std::endl;
      subvec_pFMM.add( subvec_toeplitz, -1.0 );
      std::cout << subvec_pFMM.norm( ) << ", rel. "
                << subvec_pFMM.norm( ) / subvec_toeplitz.norm( ) << std::endl;
    }
    delete K_adj_dist_pFMM;
  } else if ( test_case == 4 ) {
    if ( myRank == 0 ) {
      t.reset( "assembling distributed pFMM matrix K_adj" );
    }
    distributed_pFMM_matrix_heat_adl_p1p0 * K_adj_dist_pFMM
      = new distributed_pFMM_matrix_heat_adl_p1p0;
    distributed_fast_spacetime_be_assembler distributed_assembler_k_adj(
      kernel_ak, distributed_space_p1, distributed_space_p0, &comm, order_sing,
      order_reg, temp_order, spat_order, cauchy_data::_alpha );
    distributed_assembler_k_adj.assemble( *K_adj_dist_pFMM );

    if ( myRank == 0 ) {
      t.measure( );
    }

    lo rows_of_block = K_adj_dist_pFMM->get_dim_domain( );
    lo cols_of_block = K_adj_dist_pFMM->get_dim_range( );

    block_vector neu_proj( n_blocks, cols_of_block, true );
    // for ( lo i = 0; i < n_blocks; ++i ) {
    //   neu_proj.get_block( i ).fill( 1.0 );
    // }
    block_vector applied_std( n_blocks, rows_of_block, true );

    if ( myRank == 0 ) {
      // t.reset(
      //   "assembly and application of pFMM matrix K_adj (from scratch)" );
      t.reset( "assembly and application of adjoint of K (from scratch)" );
      triangular_surface_mesh space_mesh( spatial_mesh_file );
      uniform_spacetime_tensor_mesh spacetime_mesh(
        space_mesh, end_time, n_timesteps );
      spacetime_mesh.refine( refine, temp_refine_factor );
      uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
      uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );
      space_p0.L2_projection( cauchy_data::neumann, neu_proj );

      // toeplitz matrix
      block_lower_triangular_toeplitz_matrix * K
        = new block_lower_triangular_toeplitz_matrix( );
      uniform_spacetime_be_assembler assembler_k(
        kernel_k, space_p0, space_p1, order_sing, order_reg );
      assembler_k.assemble( *K );

      // standard pFMM matrix
      // besthea::mesh::spacetime_cluster_tree tree(
      //   spacetime_mesh, max_n_levels, 2, 10, st_coeff );
      // // tree.print( );
      // fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
      // fast_spacetime_be_space< basis_tri_p1 > space_p1_pFMM( tree );
      // pFMM_matrix_heat_adl_p1p0 * K_adj_pFMM = new pFMM_matrix_heat_adl_p1p0;
      // fast_spacetime_be_assembler fast_assembler_k_adj( kernel_ak,
      //   space_p1_pFMM, space_p0_pFMM, order_sing, order_reg, temp_order,
      //   spat_order, cauchy_data::_alpha, 1.5, false );
      // fast_assembler_k_adj.assemble( *K_adj_pFMM );

      K->apply( neu_proj, applied_std, true );
      // K_adj_pFMM->apply( neu_proj, applied_std );
      t.measure( );

      // delete K_adj_pFMM;
      delete K;
    }
    // broadcast neu_proj to all processes
    for ( lo i = 0; i < n_blocks; ++i ) {
      MPI_Bcast( neu_proj.get_block( i ).data( ), cols_of_block,
        get_scalar_type< sc >::MPI_SC( ), 0, comm );
    }
    // multiplicate neu_proj with spatially adjoint pFMM matrix K
    block_vector applied_dist_pFMM( n_blocks, rows_of_block, true );
    if ( myRank == 0 ) {
      t.reset( "applying distributed pFMM matrix K_adj" );
    }
    K_adj_dist_pFMM->apply( neu_proj, applied_dist_pFMM );
    if ( myRank == 0 ) {
      t.measure( );
      std::cout << "error timewise" << std::endl;
      for ( lo i = 0; i < applied_std.get_block_size( ); ++i ) {
        applied_dist_pFMM.get_block( i ).add(
          applied_std.get_block( i ), -1.0 );
        std::cout << applied_dist_pFMM.get_block( i ).norm( ) << ", rel. "
                  << applied_dist_pFMM.get_block( i ).norm( )
            / applied_std.get_block( i ).norm( )
                  << std::endl;
      }
    }
    delete K_adj_dist_pFMM;
  }
  MPI_Finalize( );
}
