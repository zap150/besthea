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
  std::string file = "./mesh_files/cube_12.txt";
  lo test_case = 1;
  //   int refine = 1;
  int refine = 1;
  lo n_timesteps = 8;
  sc end_time = 1.0;
  std::string grid_file = "./mesh_files/grid_xy.txt";
  // int grid_refine = 2;
  // if ( argc > 6 ) {
  //  grid_refine = std::atoi( argv[ 6 ] );
  //}
  triangular_surface_mesh space_mesh( file );
  uniform_spacetime_tensor_mesh spacetime_mesh(
    space_mesh, end_time, n_timesteps );
  spacetime_mesh.refine( refine, 1 );

  spacetime_mesh.print_info( );

  timer t;

  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );

  lo order_sing = 5;
  lo order_reg = 5;

  block_lower_triangular_toeplitz_matrix * K
    = new block_lower_triangular_toeplitz_matrix( );
  spacetime_heat_dl_kernel_antiderivative kernel_k( cauchy_data::_alpha );
  uniform_spacetime_be_assembler assembler_k(
    kernel_k, space_p0, space_p1, order_sing, order_reg );
  t.reset( "K" );
  assembler_k.assemble( *K );
  t.measure( );

  lo n_blocks = K->get_block_dim( );
  lo rows_of_block = K->get_n_rows( );
  lo cols_of_block = K->get_n_columns( );

  //   sc st_coeff = 4.0;
  //   spacetime_cluster_tree tree( spacetime_mesh, 5, 2, 10, st_coeff );
  sc st_coeff = 1.0;
  spacetime_cluster_tree tree( spacetime_mesh, 5, 2, 10, st_coeff );

  fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
  fast_spacetime_be_space< basis_tri_p1 > space_p1_pFMM( tree );

  lo temp_order = 2;
  lo spat_order = 2;

  if ( argc > 1 ) {
    temp_order = std::atoi( argv[ 1 ] );
  }
  if ( argc > 2 ) {
    spat_order = std::atoi( argv[ 2 ] );
  }
  if ( argc > 3 ) {
    test_case = std::atoi( argv[ 3 ] );
  }

  pFMM_matrix_heat_dl_p0p1 * K_pFMM = new pFMM_matrix_heat_dl_p0p1;
  //   tree.print( );

  fast_spacetime_be_assembler fast_assembler_k( kernel_k, space_p0_pFMM,
    space_p1_pFMM, order_sing, order_reg, temp_order, spat_order,
    cauchy_data::_alpha, 1.5, false );
  t.reset( "K_pFMM" );
  fast_assembler_k.assemble( *K_pFMM );
  t.measure( );

  //   block_vector applied_pFMM ( n_blocks, rows_of_block, true );
  //
  //   K_pFMM->apply( dir_proj, applied_pFMM );
  //   std::cout << "applied K_pFMM" << std::endl;
  //
  //   std::cout << "error: "
  //             << space_p0.l2_relative_error( applied_std, applied_pFMM )
  //             << std::endl;
  //
  //   std::cout << "standard " << std::endl;
  //   applied_std.get_block( 0 ).print_h( );
  //   std::cout << "pFMM " << std::endl;
  //   applied_pFMM.get_block( 0 ).print_h( );

  if ( test_case == 1 ) {
    lo entry_id = 0;
    lo block_id = 0;
    lo block_evaluation_id = 2;
    vector x_loc_0( cols_of_block );
    x_loc_0( entry_id ) = 1.0;
    block_vector x_block_vec( n_blocks, cols_of_block, true );
    x_block_vec.get_block( block_id ) = x_loc_0;
    // multiplicate x_block_vec with Toeplitz matrix K
    block_vector applied_toeplitz( n_blocks, rows_of_block, true );
    K->apply( x_block_vec, applied_toeplitz );
    // multiplicate x_block_vec with pFMM matrix K
    block_vector applied_pFMM( n_blocks, rows_of_block, true );
    K_pFMM->apply( x_block_vec, applied_pFMM );

    std::cout << "resulting subblock pFMM multiplication" << std::endl;
    std::cout << "source id " << block_id << std::endl;
    std::cout << "target id " << block_evaluation_id << std::endl;
    vector & subvec_pFMM = applied_pFMM.get_block( block_evaluation_id );
    vector & subvec_toeplitz
      = applied_toeplitz.get_block( block_evaluation_id );
    lo eval_entry_id = 8;
    std::cout << "id: " << eval_entry_id << std::endl;
    std::cout << "entry is " << subvec_pFMM[ eval_entry_id ] << std::endl;
    std::cout << "should be " << subvec_toeplitz[ eval_entry_id ] << std::endl;
    subvec_pFMM.print_h( );
    std::cout << "exact result block" << std::endl;
    subvec_toeplitz.print_h( );
    std::cout << "error timewise" << std::endl;
    subvec_pFMM.add( subvec_toeplitz, -1.0 );
    std::cout << subvec_pFMM.norm( ) << ", rel. "
              << subvec_pFMM.norm( ) / subvec_toeplitz.norm( ) << std::endl;
  } else if ( test_case == 2 ) {
    block_vector dir_proj;
    space_p1.L2_projection( cauchy_data::dirichlet, dir_proj );
    // multiplicate dir_proj with Toeplitz matrix K
    block_vector applied_toeplitz( n_blocks, rows_of_block, true );
    K->apply( dir_proj, applied_toeplitz );
    // multiplicate dir_proj with pFMM matrix K
    block_vector applied_pFMM( n_blocks, rows_of_block, true );
    K_pFMM->apply( dir_proj, applied_pFMM );
    std::cout << "error timewise" << std::endl;
    for ( lo i = 0; i < applied_toeplitz.get_block_size( ); ++i ) {
      applied_pFMM.get_block( i ).add( applied_toeplitz.get_block( i ), -1.0 );
      std::cout << applied_pFMM.get_block( i ).norm( ) << ", rel. "
                << applied_pFMM.get_block( i ).norm( )
          / applied_toeplitz.get_block( i ).norm( )
                << std::endl;
    }
  } else if ( test_case == 3 ) {
    lo block_id = 0;
    lo block_evaluation_id = 2;
    lo entry_id = 0;
    vector x_loc_0( rows_of_block );
    x_loc_0( entry_id ) = 1.0;
    block_vector x_block_vec( n_blocks, rows_of_block, true );
    x_block_vec.get_block( block_id ) = x_loc_0;
    // multiplicate x_block_vec with Toeplitz matrix K
    block_vector applied_toeplitz( n_blocks, cols_of_block, true );
    K->apply( x_block_vec, applied_toeplitz, true );
    // multiplicate x_block_vec with pFMM matrix K
    block_vector applied_pFMM( n_blocks, cols_of_block, true );
    K_pFMM->apply( x_block_vec, applied_pFMM, true );

    std::cout << "resulting subblock pFMM multiplication" << std::endl;
    std::cout << "source id " << block_id << std::endl;
    std::cout << "target id " << block_evaluation_id << std::endl;
    vector & subvec_pFMM = applied_pFMM.get_block( block_evaluation_id );
    vector & subvec_toeplitz
      = applied_toeplitz.get_block( block_evaluation_id );
    lo eval_entry_id = 0;
    std::cout << "id: " << eval_entry_id << std::endl;
    std::cout << "entry is " << subvec_pFMM[ eval_entry_id ] << std::endl;
    std::cout << "should be " << subvec_toeplitz[ eval_entry_id ] << std::endl;
    subvec_pFMM.print_h( );
    std::cout << "exact result block" << std::endl;
    subvec_toeplitz.print_h( );
    std::cout << "error timewise" << std::endl;
    subvec_pFMM.add( subvec_toeplitz, -1.0 );
    std::cout << subvec_pFMM.norm( ) << ", rel. "
              << subvec_pFMM.norm( ) / subvec_toeplitz.norm( ) << std::endl;
  } else if ( test_case == 4 ) {
    block_vector neu_proj;
    space_p0.L2_projection( cauchy_data::neumann, neu_proj );
    // multiplicate neu_proj with spatially adjoint Toeplitz matrix K
    block_vector applied_toeplitz( n_blocks, cols_of_block, true );
    K->apply( neu_proj, applied_toeplitz, true );
    // multiplicate neu_proj with spatially adjoint pFMM matrix K
    block_vector applied_pFMM( n_blocks, cols_of_block, true );
    K_pFMM->apply( neu_proj, applied_pFMM, true );
    std::cout << "error timewise" << std::endl;
    for ( lo i = 0; i < applied_toeplitz.get_block_size( ); ++i ) {
      applied_pFMM.get_block( i ).add( applied_toeplitz.get_block( i ), -1.0 );
      std::cout << applied_pFMM.get_block( i ).norm( ) << ", rel. "
                << applied_pFMM.get_block( i ).norm( )
          / applied_toeplitz.get_block( i ).norm( )
                << std::endl;
    }
  }
  delete K;

  delete K_pFMM;
}
