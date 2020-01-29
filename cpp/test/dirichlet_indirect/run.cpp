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

  static constexpr sc _alpha{ 0.5 };
  static constexpr std::array< sc, 3 > _y{ 0.0, 0.0, 1.5 };
};

int main( int argc, char * argv[] ) {
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

  block_lower_triangular_toeplitz_matrix * V
    = new block_lower_triangular_toeplitz_matrix( );
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );
  besthea::bem::uniform_spacetime_be_assembler assembler_v(
    kernel_v, space_p0, space_p0, order_sing, order_reg );

  assembler_v.assemble( *V );

  block_vector dir_proj;
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );
  space_p0.L2_projection( cauchy_data::dirichlet, dir_proj );

  besthea::bem::uniform_spacetime_be_identity M( space_p0, space_p0 );
  M.assemble( );
  block_vector M_dir_proj;
  M_dir_proj.resize( V->get_block_dim( ) );
  M_dir_proj.resize_blocks( V->get_n_rows( ) );
  M.apply( dir_proj, M_dir_proj );

  sc gmres_prec = 1e-8;
  lo gmres_iter = 500;
  block_vector dens(
    dir_proj.get_block_size( ), dir_proj.get_size_of_block( ) );
  V->mkl_fgmres_solve( M_dir_proj, dens, gmres_prec, gmres_iter, gmres_iter );

  if ( !grid_file.empty( ) ) {
    triangular_surface_mesh grid_space_mesh( grid_file );
    grid_space_mesh.scale( 0.95 );
    grid_space_mesh.refine( grid_refine );
    uniform_spacetime_tensor_mesh grid_spacetime_mesh(
      grid_space_mesh, end_time, spacetime_mesh.get_n_temporal_elements( ) );

    block_vector repr;
    besthea::bem::uniform_spacetime_be_evaluator evaluator_v(
      kernel_v, space_p0 );
    evaluator_v.evaluate( grid_space_mesh.get_nodes( ), dens, repr );

    block_vector sol_interp;
    uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > grid_space_p1(
      grid_spacetime_mesh );
    grid_space_p1.interpolation( cauchy_data::dirichlet, sol_interp );
    std::cout << "Solution l2 relative error: "
              << space_p1.l2_relative_error( sol_interp, repr ) << std::endl;

    // print the result in the Ensight format
    std::vector< std::string > grid_node_labels{ "Temperature_interpolation",
      "Temperature_result" };
    std::vector< block_vector * > grid_node_data{ &sol_interp, &repr };
    std::string ensight_grid_dir = "ensight_grid";
    std::filesystem::create_directory( ensight_grid_dir );
    grid_spacetime_mesh.print_ensight_case(
      ensight_grid_dir, &grid_node_labels );
    grid_spacetime_mesh.print_ensight_geometry( ensight_grid_dir );
    grid_spacetime_mesh.print_ensight_datafiles(
      ensight_grid_dir, &grid_node_labels, &grid_node_data, nullptr, nullptr );
  }
}
