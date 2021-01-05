/*
Copyright (c) 2020, VSB - Technical University of Ostrava and Graz University of
Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the names of VSB - Technical University of  Ostrava and Graz
  University of Technology nor the names of its contributors may be used to
  endorse or promote products derived from this software without specific prior
  written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL VSB - TECHNICAL UNIVERSITY OF OSTRAVA AND
GRAZ UNIVERSITY OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
  static sc dirichlet( sc x1, sc x2, sc x3, const coordinates< 3 > &, sc t ) {
    sc norm2 = ( x1 - _y[ 0 ] ) * ( x1 - _y[ 0 ] )
      + ( x2 - _y[ 1 ] ) * ( x2 - _y[ 1 ] )
      + ( x3 - _y[ 2 ] ) * ( x3 - _y[ 2 ] );
    sc value = std::pow( 4.0 * M_PI * _alpha * t, -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * t ) );

    return value;
  }

  static sc neumann( sc x1, sc x2, sc x3, const coordinates< 3 > & n, sc t ) {
    sc dot = ( x1 - _y[ 0 ] ) * n[ 0 ] + ( x2 - _y[ 1 ] ) * n[ 1 ]
      + ( x3 - _y[ 2 ] ) * n[ 2 ];
    sc value = ( -1.0 / ( 2.0 * t ) ) * dot * dirichlet( x1, x2, x3, n, t );

    return value;
  }

  static constexpr sc _alpha{ 4.0 };
  static constexpr std::array< sc, 3 > _y{ 0.0, 0.0, 1.5 };
};

int main( int argc, char * argv[] ) {
  std::string file = "./mesh_files/cube_12.txt";
  //   int refine = 1;
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
  //   if ( argc > 6 ) {
  //     grid_refine = std::atoi( argv[ 6 ] );
  //   }
  triangular_surface_mesh space_mesh( file );
  uniform_spacetime_tensor_mesh spacetime_mesh(
    space_mesh, end_time, n_timesteps );
  spacetime_mesh.refine( refine, 1 );

  spacetime_mesh.print_info( );

  timer t;

  sc st_coeff = 4.0;
  spacetime_cluster_tree tree( spacetime_mesh, 5, 2, 10, st_coeff );

  fast_spacetime_be_space< basis_tri_p0 > space_p0( tree );
  fast_spacetime_be_space< basis_tri_p1 > space_p1( tree );

  lo order_sing = 4;
  lo order_reg = 4;
  lo temp_order = 6;
  lo spat_order = 6;

  pFMM_matrix_heat_adl_p1p0 * K_adj = new pFMM_matrix_heat_adl_p1p0;
  //   tree.print( );

  spacetime_heat_adl_kernel_antiderivative kernel_ak( cauchy_data::_alpha );
  fast_spacetime_be_assembler fast_assembler_adj_k( kernel_ak, space_p1,
    space_p0, order_sing, order_reg, temp_order, spat_order,
    cauchy_data::_alpha, 1.5, false );
  t.reset( "K_adj" );
  fast_assembler_adj_k.assemble( *K_adj );
  t.measure( );

  spacetime_be_identity M( space_p0, space_p1, 1 );
  t.reset( "M" );
  M.assemble( );
  t.measure( );

  block_vector dir_proj, neu_proj;
  space_p1.L2_projection( cauchy_data::dirichlet, dir_proj );
  space_p0.L2_projection( cauchy_data::neumann, neu_proj );
  std::cout << "Dirichlet L2 projection relative error: "
            << space_p1.L2_relative_error( cauchy_data::dirichlet, dir_proj )
            << std::endl;
  std::cout << "Neumann L2 projection relative error: "
            << space_p0.L2_relative_error( cauchy_data::neumann, neu_proj )
            << std::endl;

  t.reset( "Setting up RHS" );
  block_vector dir;
  dir.resize( K_adj->get_block_dim( ) );
  dir.resize_blocks( K_adj->get_dim_domain( ), true );
  M.apply( neu_proj, dir, true, 0.5, 0.0 );
  K_adj->apply( neu_proj, dir, false, -1.0, 1.0 );
  t.measure( );

  delete K_adj;

  pFMM_matrix_heat_hs_p1p1 * D = new pFMM_matrix_heat_hs_p1p1;

  spacetime_heat_hs_kernel_antiderivative kernel_d( cauchy_data::_alpha );
  fast_spacetime_be_assembler fast_assembler_d( kernel_d, space_p1, space_p1,
    order_sing, order_reg, temp_order, spat_order, cauchy_data::_alpha, 1.5,
    false );
  t.reset( "D" );
  fast_assembler_d.assemble( *D );
  t.measure( );

  pFMM_matrix_heat_sl_p1p1 * V11 = new pFMM_matrix_heat_sl_p1p1( );
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );
  fast_spacetime_be_assembler fast_assembler_v( kernel_v, space_p1, space_p1,
    order_sing, order_reg, temp_order, spat_order, cauchy_data::_alpha, 1.5,
    false );
  t.reset( "V11" );
  fast_assembler_v.assemble( *V11 );
  t.measure( );
  spacetime_be_identity M11( space_p1, space_p1, 2 );
  t.reset( "M11" );
  M11.assemble( );
  t.measure( );

  block_mkl_cg_inverse M11_inv( M11, 1e-8, 100 );
  compound_block_linear_operator preconditioner;
  preconditioner.push_back( M11_inv );
  preconditioner.push_back( *V11 );
  preconditioner.push_back( M11_inv );

  t.reset( "Solving the system" );
  block_vector rhs( dir );
  sc gmres_prec = 1e-8;
  lo gmres_iter = 500;
  // D->mkl_fgmres_solve( rhs, dir, gmres_prec, gmres_iter, gmres_iter );
  // std::cout << "  iterations: " << gmres_iter << ", residual: " << gmres_prec
  //           << std::endl;
  // gmres_prec = 1e-8;
  // gmres_iter = 500;
  D->mkl_fgmres_solve(
    preconditioner, rhs, dir, gmres_prec, gmres_iter, gmres_iter );
  std::cout << "  iterations: " << gmres_iter << ", residual: " << gmres_prec
            << std::endl;
  t.measure( );

  delete D;
  delete V11;

  std::cout << "Dirichlet L2 relative error: "
            << space_p1.L2_relative_error( cauchy_data::dirichlet, dir )
            << std::endl;

  if ( !grid_file.empty( ) ) {
    triangular_surface_mesh grid_space_mesh( grid_file );
    grid_space_mesh.scale( 0.95 );
    grid_space_mesh.refine( grid_refine );
    uniform_spacetime_tensor_mesh grid_spacetime_mesh(
      grid_space_mesh, end_time, spacetime_mesh.get_n_temporal_elements( ) );
    grid_spacetime_mesh.print_info( );

    block_vector slp;
    uniform_spacetime_be_evaluator evaluator_v( kernel_v, space_p0, order_reg );
    t.reset( "SLP" );
    evaluator_v.evaluate( grid_space_mesh.get_nodes( ), neu_proj, slp );
    t.measure( );

    block_vector dlp;
    spacetime_heat_dl_kernel_antiderivative kernel_k( cauchy_data::_alpha );
    uniform_spacetime_be_evaluator evaluator_k( kernel_k, space_p1, order_reg );
    t.reset( "DLP" );
    evaluator_k.evaluate( grid_space_mesh.get_nodes( ), dir, dlp );
    t.measure( );

    slp.add( dlp, -1.0 );

    block_vector sol_interp;
    uniform_spacetime_be_space< basis_tri_p1 > grid_space_p1(
      grid_spacetime_mesh );
    grid_space_p1.interpolation( cauchy_data::dirichlet, sol_interp );
    std::cout << "Solution l2 relative error: "
              << space_p1.l2_relative_error( sol_interp, slp ) << std::endl;

    /*
    t.reset( "Printing Ensight grid" );
    std::vector< std::string > grid_node_labels{
    "Temperature_interpolation",
      "Temperature_result" };
    std::vector< block_vector * > grid_node_data{ &sol_interp, &slp };
    std::string ensight_grid_dir = "ensight_grid";
    std::filesystem::create_directory( ensight_grid_dir );
    grid_spacetime_mesh.print_ensight_case(
      ensight_grid_dir, &grid_node_labels );
    grid_spacetime_mesh.print_ensight_geometry( ensight_grid_dir );
    grid_spacetime_mesh.print_ensight_datafiles(
      ensight_grid_dir, &grid_node_labels, &grid_node_data, nullptr,
      nullptr );
    t.measure( );
    */
  }

  /*
  t.reset( "Printing Ensight surface" );
  std::vector< std::string > node_labels{ "Dirichlet_projection",
    "Dirichlet_result" };
  std::vector< std::string > elem_labels{ "Neumann_projection" };
  std::vector< block_vector * > node_data{ &dir_proj, &dir };
  std::vector< block_vector * > elem_data{ &neu_proj };
  std::string ensight_dir = "ensight_surface";
  std::filesystem::create_directory( ensight_dir );
  spacetime_mesh.print_ensight_case( ensight_dir, &node_labels,
  &elem_labels ); spacetime_mesh.print_ensight_geometry( ensight_dir );
  spacetime_mesh.print_ensight_datafiles(
    ensight_dir, &node_labels, &node_data, &elem_labels, &elem_data );
  t.measure( );
  */
}