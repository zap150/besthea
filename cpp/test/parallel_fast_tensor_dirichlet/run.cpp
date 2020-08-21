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
using namespace besthea::bem;
using namespace besthea::linear_algebra;
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
      * dirichlet( x1, x2, x3, n, ( t + _shift ) );

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
  lo n_blocks, n_space_elements, n_space_nodes;

  st_coeff = 4.0;
  temp_order = 6;
  spat_order = 6;
  order_sing = 4;
  order_reg = 4;

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
  // feine refinement for mesh inside of slices.
  lo time_refinement = temp_refine_factor * refine;
  lo space_refinement = refine;
  //
  lo process_assignment_strategy = 1;

  // kernel for the single layer and double layer operator
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );
  spacetime_heat_dl_kernel_antiderivative kernel_k( cauchy_data::_alpha );


  MPI_Barrier( comm );

  // generation of distributed mesh
  std::string tree_vector_file
    = "./parallel_fast_tensor_dirichlet/test_case_1/tree_structure.bin";
  std::string cluster_bounds_file
    = "./parallel_fast_tensor_dirichlet/test_case_1/cluster_bounds.bin";
  std::string process_assignment_file
    = "./parallel_fast_tensor_dirichlet/test_case_1/process_assignment.bin";
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

    generator.generate( "./parallel_fast_tensor_dirichlet/test_case_1/",
                        "test_mesh", "txt" );
    std::cout << "### end mesh generation ###" << std::endl;
  }
  MPI_Barrier( comm );

  if ( myRank == 0 ) {
    std::cout << "assembling distributed mesh and tree" << std::endl;
  }

  distributed_spacetime_tensor_mesh distributed_mesh(
    "./parallel_fast_tensor_dirichlet/test_case_1/test_mesh_d.txt",
    tree_vector_file, cluster_bounds_file, process_assignment_file, &comm );

  // number of blocks in a blockvector corresponds to global number of timesteps
  n_blocks = distributed_mesh.get_n_temporal_elements( );
  n_space_elements
    = distributed_mesh.get_local_mesh( )->get_n_spatial_elements( );
  n_space_nodes
    = distributed_mesh.get_local_mesh( )->get_n_spatial_nodes( );

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
  distributed_fast_spacetime_be_space< basis_tri_p1 > distributed_space_p1(
    distributed_st_tree );

  MPI_Barrier( comm );

  if ( myRank == 0 ) {
    std::cout << "assembling distributed pFMM matrix V" << std::endl;
  }


  distributed_pFMM_matrix_heat_sl_p0p0 * V
    = new distributed_pFMM_matrix_heat_sl_p0p0;

  distributed_fast_spacetime_be_assembler distributed_assembler_v( kernel_v,
    distributed_space_p0, distributed_space_p0, &comm, order_sing, order_reg,
    temp_order, spat_order, cauchy_data::_alpha);

  distributed_assembler_v.assemble( *V );

  MPI_Barrier( comm );
  if ( myRank == 0 ) {
    std::cout << "finished assembling distributed pFMM matrix V" << std::endl;
  }

  if ( myRank == 0 ) {
    std::cout << "assembling distributed pFMM matrix K" << std::endl;
  }

  distributed_pFMM_matrix_heat_dl_p0p1 * K
    = new distributed_pFMM_matrix_heat_dl_p0p1;
  distributed_fast_spacetime_be_assembler distributed_assembler_k( kernel_k,
    distributed_space_p0, distributed_space_p1, &comm, order_sing, order_reg,
    temp_order, spat_order, cauchy_data::_alpha);
  distributed_assembler_k.assemble( *K );

  MPI_Barrier( comm );
  if ( myRank == 0 ) {
    std::cout << "finished assembling distributed pFMM matrix K" << std::endl;
  }


  block_vector dir_proj( n_blocks, n_space_nodes, true );
  block_vector neu_block( n_blocks, n_space_elements, true );
  uniform_spacetime_be_space< basis_tri_p0 > * space_p0 = nullptr;
  uniform_spacetime_be_space< basis_tri_p1 > * space_p1 = nullptr;
  triangular_surface_mesh * space_mesh = nullptr;
  uniform_spacetime_tensor_mesh * spacetime_mesh = nullptr;
  // todo: replace this by a suitable distributed version:
  if ( myRank == 0 ) {
    space_mesh = new triangular_surface_mesh( spatial_mesh_file );
    spacetime_mesh = new uniform_spacetime_tensor_mesh(
      *space_mesh, end_time, n_timesteps );
    spacetime_mesh->refine( refine, temp_refine_factor );
    std::cout << "number of elements is "
              << spacetime_mesh->get_n_spatial_elements( )
              << ", should be " << n_space_elements << std::endl;
    std::cout << "number of nodes is "
              << spacetime_mesh->get_n_spatial_nodes( )
              << ", should be " << n_space_nodes << std::endl;
    space_p0
      = new uniform_spacetime_be_space< basis_tri_p0 >( *spacetime_mesh );
    space_p1
      = new uniform_spacetime_be_space< basis_tri_p1 >( *spacetime_mesh );
    // todo: parallel version of M?
    besthea::bem::spacetime_be_identity M( *space_p0, *space_p1, 1 );
    M.assemble( );

    block_vector neu_proj;
    // todo: parallel versions of projections in distributed fast spacetime
    // be spaces.
    space_p1->L2_projection( cauchy_data::dirichlet, dir_proj );
    space_p0->L2_projection( cauchy_data::neumann, neu_proj );
    std::cout << "Dirichlet L2 projection relative error: "
              << space_p1->L2_relative_error( cauchy_data::dirichlet, dir_proj )
              << std::endl;
    std::cout << "Neumann L2 projection relative error: "
              << space_p0->L2_relative_error( cauchy_data::neumann, neu_proj )
              << std::endl;
    std::cout << "applying M" << std::endl;
    M.apply( dir_proj, neu_block, false, 0.5, 0.0 );
    std::cout << "applied M" << std::endl;
    delete space_p1;
  }
  // broadcast dir_proj to all processes
  for ( lo i = 0; i < n_blocks; ++i ) {
    MPI_Bcast( dir_proj.get_block( i ).data( ), n_space_nodes,
    get_scalar_type< sc >::MPI_SC( ), 0, comm );
  }
  MPI_Barrier( comm );
  // broadcast intermediate value of neu_block( after application of M )
  for ( lo i = 0; i < n_blocks; ++i ) {
    MPI_Bcast( neu_block.get_block( i ).data( ), n_space_elements,
    get_scalar_type< sc >::MPI_SC( ), 0, comm );
  }
  MPI_Barrier( comm );
  // apply K to dir_proj and add result to neu_block
  K->apply( dir_proj, neu_block, false, 1.0, 1.0 );

  if ( myRank == 0 ) {
    std::cout << "applied K" << std::endl;
    std::cout << "solving for neumann datum" << std::endl;
  }

  block_vector rhs( neu_block );
  sc gmres_prec = 1e-8;
  lo gmres_iter = 500;
  V->mkl_fgmres_solve( rhs, neu_block, gmres_prec, gmres_iter, gmres_iter );

  if ( myRank == 0 ) {
    std::cout << "finished, required number of iterations: "
              << gmres_iter << std::endl;
    std::cout << "Neumann L2 relative error: "
            << space_p0->L2_relative_error( cauchy_data::neumann, neu_block )
            << std::endl;
    delete space_p0;
    delete spacetime_mesh;
    delete space_mesh;
  }
  delete V;
  delete K;
  MPI_Finalize( );
}
