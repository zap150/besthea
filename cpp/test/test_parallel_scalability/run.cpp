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
#include "besthea/tools.h"

#define USE_P0_BASIS

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem;
using namespace besthea::tools;

struct cauchy_data {
  static sc dirichlet( sc x1, sc x2, sc x3, const coordinates< 3 > &, sc t ) {
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
  int provided, my_rank, n_processes;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
  MPI_Comm_size( comm, &n_processes );

  // declaration of required parameters.
  lo order_sing, order_reg, temp_order, spat_order;
  sc st_coeff;
  lo n_blocks, size_of_block;

  st_coeff = 4.0;
  temp_order = 5;
  spat_order = 5;
  order_sing = 4;
  order_reg = 4;

  std::string spatial_mesh_file = "./mesh_files/cube_12.txt";
  int refine = 2;
  int temp_refine_factor = 2;
  lo n_time_slices = 256;
  lo spacetime_levels = 8;
  lo time_levels = 9;
  lo n_min_time_elems = 2;
  lo n_min_st_elems = 50;

  if ( argc > 1 ) {
    spatial_mesh_file.assign( argv[ 1 ] );
  }
  if ( argc > 2 ) {
    n_time_slices = std::atoi( argv[ 2 ] );
  }
  if ( argc > 3 ) {
    refine = std::atoi( argv[ 3 ] );
  }
  if ( argc > 4 ) {
    temp_refine_factor = std::atoi( argv[ 4 ] );
  }
  if ( argc > 5 ) {
    time_levels = std::atoi( argv[ 5 ] );
  }
  if ( argc > 6 ) {
    spacetime_levels = std::atoi( argv[ 6 ] );
  }
  if ( argc > 7 ) {
    n_min_st_elems = std::atoi( argv[ 7 ] );
  }
  if ( argc > 8 ) {
    n_min_time_elems = std::atoi( argv[ 8 ] );
  }

  lo time_refinement = temp_refine_factor * refine;
  lo space_refinement = refine;

  lo process_assignment_strategy = 1;

  // kernel for the single layer operator
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );

  // V approximated by distributed pFMM

  MPI_Barrier( comm );

  // generation of distributed mesh
  std::string tree_vector_file
    = "./test_parallel_scalability/test_case_1/tree_structure.bin";
  std::string cluster_bounds_file
    = "./test_parallel_scalability/test_case_1/cluster_bounds.bin";
  std::string process_assignment_file
    = "./test_parallel_scalability/test_case_1/process_assignment.bin";
  if ( my_rank == 0 ) {
    std::cout << "### start mesh generation ###" << std::endl;

    // load time mesh defining slices and create temporal tree
    // temporal_mesh time_mesh( time_file );
    temporal_mesh time_mesh( 0.0, 1.0, n_time_slices );
    time_cluster_tree time_tree( time_mesh, time_levels, n_min_time_elems );

    // write tree structure to file
    time_tree.print_tree_structure( tree_vector_file );

    // write cluster bounds to file
    time_tree.print_cluster_bounds( cluster_bounds_file );

    // compute process assignment and write it to file

    std::cout << "n_processes: " << n_processes
              << ", strategy: " << process_assignment_strategy << std::endl;

    time_tree.print_process_assignments(
      n_processes, process_assignment_strategy, process_assignment_file );

    // spacetime_mesh_generator generator(
    //  spatial_mesh_file, time_file, time_refinement, space_refinement );

    triangular_surface_mesh tri_mesh( spatial_mesh_file );
    spacetime_mesh_generator generator(
      tri_mesh, 1.0, n_time_slices, time_refinement, space_refinement );

    generator.generate(
      "./parallel_dirichlet_indirect/test_case_1/", "test_mesh", "txt" );
    std::cout << "### end mesh generation ###" << std::endl;
    std::cout << "Number of temporal slices: " << n_time_slices << std::endl;
    std::cout << "Number of temporal elements: "
              << n_time_slices * std::pow( 2.0, time_refinement ) << std::endl;
    std::cout << "Number of spatial elements: " << tri_mesh.get_n_elements( )
              << std::endl;
  }
  MPI_Barrier( comm );

  if ( my_rank == 0 ) {
    std::cout << "assembling distributed pFMM matrix" << std::endl;
  }

  distributed_spacetime_tensor_mesh distributed_mesh(
    "./parallel_dirichlet_indirect/test_case_1/test_mesh_d.txt",
    tree_vector_file, cluster_bounds_file, process_assignment_file, &comm );
  if ( my_rank == 0 ) {
    std::cout << "Number of spacetime elements: "
              << distributed_mesh.get_n_elements( ) << std::endl;
  }
  // number of blocks in a blockvector corresponds to global number of timesteps
  n_blocks = distributed_mesh.get_n_temporal_elements( );
  // size of a block in blockvector corresponds to number of spatial elements
  size_of_block = distributed_mesh.get_n_elements( ) / n_blocks;

  distributed_spacetime_cluster_tree distributed_st_tree(
    distributed_mesh, spacetime_levels, n_min_st_elems, st_coeff, 3, &comm );
  MPI_Barrier( comm );
  if ( my_rank == 0 ) {
    // distributed_st_tree.print( );
    // std::cout << std::endl << std::endl << std::endl;
    // distributed_st_tree.get_distribution_tree( )->print( );
  }

  distributed_fast_spacetime_be_space< basis_tri_p0 > distributed_space_p0(
    distributed_st_tree );
  distributed_pFMM_matrix_heat_sl_p0p0 * V_dist_pFMM
    = new distributed_pFMM_matrix_heat_sl_p0p0;

  distributed_fast_spacetime_be_assembler distributed_assembler_v( kernel_v,
    distributed_space_p0, distributed_space_p0, &comm, order_sing, order_reg,
    temp_order, spat_order, cauchy_data::_alpha );

  timer t;
  t.reset( "Assembling distributed FMM V" );
  distributed_assembler_v.assemble( *V_dist_pFMM );
  t.measure( );

  MPI_Barrier( comm );
  if ( my_rank == 0 ) {
    std::cout << "finished assembling distributed pFMM matrix" << std::endl;
  }

  if ( my_rank == 0 ) {
    distributed_mesh.get_distribution_tree( )->print_tree_human_readable(
      2, true );
  }
  MPI_Barrier( comm );
  if ( my_rank == 1 ) {
    distributed_mesh.get_distribution_tree( )->print_tree_human_readable(
      2, true );
  }
  MPI_Barrier( comm );
  if ( my_rank == 2 ) {
    distributed_mesh.get_distribution_tree( )->print_tree_human_readable(
      2, true );
  }

  block_vector x( n_blocks, size_of_block, true );
  block_vector y( n_blocks, size_of_block, true );
  t.reset( "Application" );
  V_dist_pFMM->apply( x, y );
  t.measure( );
  delete V_dist_pFMM;
  MPI_Finalize( );
}
