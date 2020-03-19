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
    sc value = std::pow( 4.0 * M_PI * _alpha * t, -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * t ) );
    return value;
  }

  static constexpr sc _alpha{ 4 };
  static constexpr std::array< sc, 3 > _y{ 0.0, 0.0, 1.5 };
};

int main( int argc, char * argv[] ) {
  lo test_case = 3;
  std::string file = "./mesh_files/cube_12.txt";
  int refine = 2;
  lo n_timesteps = 8;
  sc end_time = 1.0;
  std::string grid_file = "./mesh_files/grid_xy.txt";
  int grid_refine = 2;

  if ( argc > 1 ) {
    file.assign( argv[ 1 ] );
  }
  if ( argc > 2 ) {
    n_timesteps = std::atoi( argv[ 2 ] );
  }
  if ( argc > 3 ) {
    end_time = std::atof( argv[ 3 ] );
  }
  if ( argc > 4 ) {
    refine = std::atoi( argv[ 4 ] );
  }
  if ( argc > 5 ) {
    grid_file.assign( argv[ 5 ] );
  }
  if ( argc > 6 ) {
    grid_refine = std::atoi( argv[ 6 ] );
  }

  triangular_surface_mesh space_mesh( file );
  uniform_spacetime_tensor_mesh spacetime_mesh(
    space_mesh, end_time, n_timesteps );

  spacetime_mesh.refine( refine, 1 );
  spacetime_mesh.print_info( );

  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );

  lo order_sing = 4;
  lo order_reg = 4;

  // V without approximation
  block_lower_triangular_toeplitz_matrix * V
    = new block_lower_triangular_toeplitz_matrix( );
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );
  besthea::bem::uniform_spacetime_be_assembler assembler_v(
    kernel_v, space_p0, space_p0, order_sing, order_reg );

  assembler_v.assemble( *V );

  block_vector dir_proj;
  //   uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );
  space_p0.L2_projection( cauchy_data::dirichlet, dir_proj );

  besthea::bem::uniform_spacetime_be_identity M( space_p0, space_p0 );
  M.assemble( );
  block_vector M_dir_proj;

  lo n_blocks = V->get_block_dim( );
  lo size_of_block = V->get_n_rows( );

  M_dir_proj.resize( n_blocks );
  M_dir_proj.resize_blocks( size_of_block );
  M.apply( dir_proj, M_dir_proj );

  // V approximated by pFMM
  // sc st_coeff = 4.0;
  sc st_coeff = 4.0;
  lo temp_order = 8;
  lo spat_order = 8;
  spacetime_cluster_tree tree( spacetime_mesh, 5, 2, 10, st_coeff );
  //   tree.print( );
  fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
  pFMM_matrix_heat_sl_p0p0 * V_pFMM = new pFMM_matrix_heat_sl_p0p0;

  fast_spacetime_be_assembler fast_assembler_v( kernel_v, space_p0_pFMM,
    space_p0_pFMM, order_sing, order_reg, temp_order, spat_order,
    cauchy_data::_alpha, 1.5, false );
  // t.reset( "V" );
  fast_assembler_v.assemble( *V_pFMM );
  // t.measure( );

  // t.reset( "Solving the system" );

  if ( test_case == 1 ) {
    //   lo block_id = n_blocks - 1 - 2;
    lo block_id = 0;
    lo block_evaluation_id = 4;
    lo toeplitz_id = block_evaluation_id - block_id;
    //   lo block_evaluation_id = n_blocks - 1;

    full_matrix & V_block_loc = V->get_block( toeplitz_id );
    vector y_loc( size_of_block );
    vector x_loc_0( size_of_block );
    x_loc_0( 0 ) = 1.0;
    //   x_loc_0.fill( 1.0 );
    V_block_loc.apply( x_loc_0, y_loc );

    block_vector block_ones( n_blocks, size_of_block, true );
    block_ones.get_block( block_id ) = x_loc_0;
    block_vector applied_pFMM( n_blocks, size_of_block, true );
    V_pFMM->apply( block_ones, applied_pFMM );

    std::cout << "resulting subblock pFMM multiplication" << std::endl;
    std::cout << "block id " << block_id << std::endl;
    vector & subvec_pFMM = applied_pFMM.get_block( block_evaluation_id );
    subvec_pFMM.print_h( );
    std::cout << "exact result block" << std::endl;
    y_loc.print_h( );
    std::cout << "error timewise" << std::endl;
    subvec_pFMM.add( y_loc, -1.0 );
    std::cout << subvec_pFMM.norm( ) << ", rel. "
              << subvec_pFMM.norm( ) / y_loc.norm( ) << std::endl;
  } else if ( test_case == 2 ) {
    std::cout << "comparison of multiplication with random vector" << std::endl;
    for ( lo i = 0; i < n_blocks; ++i ) {
      M_dir_proj.get_block( i ).random_fill( 0.5, 1.5 );
    }
    std::cout << "applying V" << std::endl;
    block_vector applied_std( n_blocks, size_of_block, true );
    V->apply( M_dir_proj, applied_std );
    std::cout << "applying V_pFMM" << std::endl;
    block_vector applied_pFMM( n_blocks, size_of_block, true );
    V_pFMM->apply( M_dir_proj, applied_pFMM );
    std::cout << "error timewise" << std::endl;
    for ( lo i = 0; i < n_blocks; ++i ) {
      applied_pFMM.get_block( i ).add( applied_std.get_block( i ), -1.0 );
      std::cout << applied_pFMM.get_block( i ).norm( ) << ", rel. "
                << applied_pFMM.get_block( i ).norm( )
          / applied_std.get_block( i ).norm( )
                << std::endl;
    }
  } else if ( test_case == 3 ) {
    sc gmres_prec = 1e-8;
    lo gmres_iter = 500;
    block_vector dens( n_blocks, size_of_block );
    V->mkl_fgmres_solve( M_dir_proj, dens, gmres_prec, gmres_iter, gmres_iter );
    std::cout << "iterations standard: " << gmres_iter << std::endl;
    gmres_prec = 1e-8;
    gmres_iter = 500;
    block_vector dens_pFMM( n_blocks, size_of_block, true );
    V_pFMM->mkl_fgmres_solve(
      M_dir_proj, dens_pFMM, gmres_prec, gmres_iter, gmres_iter );
    std::cout << "iterations pFMM: " << gmres_iter << std::endl;

    std::cout << "error timewise" << std::endl;
    block_vector dens_diff( dens_pFMM );
    for ( lo i = 0; i < n_blocks; ++i ) {
      dens_diff.get_block( i ).add( dens.get_block( i ), -1.0 );
      std::cout << dens_diff.get_block( i ).norm( ) << ", rel. "
                << dens_diff.get_block( i ).norm( )
          / dens.get_block( i ).norm( )
                << std::endl;
    }

    if ( !grid_file.empty( ) ) {
      triangular_surface_mesh grid_space_mesh( grid_file );
      grid_space_mesh.scale( 0.95 );
      grid_space_mesh.refine( grid_refine );
      uniform_spacetime_tensor_mesh grid_spacetime_mesh(
        grid_space_mesh, end_time, spacetime_mesh.get_n_temporal_elements( ) );

      block_vector repr;
      block_vector repr_pFMM;
      besthea::bem::uniform_spacetime_be_evaluator evaluator_v(
        kernel_v, space_p0 );
      evaluator_v.evaluate( grid_space_mesh.get_nodes( ), dens, repr );
      evaluator_v.evaluate(
        grid_space_mesh.get_nodes( ), dens_pFMM, repr_pFMM );

      block_vector sol_interp;
      uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > grid_space_p1(
        grid_spacetime_mesh );
      grid_space_p1.interpolation( cauchy_data::dirichlet, sol_interp );
      std::cout << "Solution l2 relative error: "
                << grid_space_p1.l2_relative_error( sol_interp, repr )
                << std::endl;
      std::cout << "Solution l2 relative error pFMM: "
                << grid_space_p1.l2_relative_error( sol_interp, repr_pFMM )
                << std::endl;

      // // print the result in the Ensight format
      // std::vector< std::string > grid_node_labels{
      // "Temperature_interpolation",
      //   "Temperature_result" };
      // std::vector< block_vector * > grid_node_data{ &sol_interp, &repr };
      // std::string ensight_grid_dir = "ensight_grid";
      // std::filesystem::create_directory( ensight_grid_dir );
      // grid_spacetime_mesh.print_ensight_case(
      //   ensight_grid_dir, &grid_node_labels );
      // grid_spacetime_mesh.print_ensight_geometry( ensight_grid_dir );
      // grid_spacetime_mesh.print_ensight_datafiles(
      //   ensight_grid_dir, &grid_node_labels, &grid_node_data, nullptr,
      //   nullptr );
      //   }
    }
  }
}
