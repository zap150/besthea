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

/** @file run.cpp
 * @brief Solve the Dirichlet problem via an indirect boundary element method.
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

/**
 * Provides the Dirichlet datum for the function
 * \f$ (x,t) \mapsto  G_\alpha(x-y, t) \f$,
 * where \f$ \alpha=4.0 \f$ and \f$ y = (0, 0, 1.5)^\top \f$
 */

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

/**
 * Computes the solution of the Dirichlet problem for the heat equation
 * using space-time boundary element methods (indirect approach via the
 * single layer boundary integral operator) and evaluates it in a provided set
 * of grid points.
 * \param[in] argc An integer argument count of the command line arguments.
 * \param  argv An argument vector of the command line arguments. Between 0
 *              and 6 input arguments are considered (additional ones are
 *              ignored):
 *              - argv[1]: Relative path of the spatial mesh file used for
 *                         computations. This has to be a triangular mesh.
 *                         (default file is test/mesh_files/cube_12.txt).
 *              - argv[2]: Number of uniform timesteps in the temporal mesh
 *                         (default is 8).
 *              - argv[3]: End time of the time interval (default is 1).
 *              - argv[4]: Number of refinements of the mesh. The spatial and
 *                         temporal mesh are refined equally (default is 1).
 *              - argv[5]: Relative path of the spatial grid file used to
 *                         evaluate the solution. This is assumed to be a
 *                         triangular mesh but does not have to describe a
 *                         closed surface. The mesh is scaled by a factor 0.95.
 *                         (default file is test/mesh_files/grid_xy.txt).
 *              - argv[6]: Number of refinements of the spatial grid.  (default
 *                         is 2)
 * \note The grid in which the solution is evaluated consists of the nodes of
 * the loaded and refined spatial grid file and the temporal nodes of the mesh
 * on which the density is computed.
 */
int main( int argc, char * argv[] ) {
  // triangular mesh on which the density is computed.
  std::string file = "./../test/mesh_files/cube_12.txt";
  // number of refinements in space and time.
  int refine = 1;
  // number of timesteps of the initial temporal mesh.
  lo n_timesteps = 8;
  // time horizon (temporal computation domain is (0, end_time ).
  sc end_time = 1.0;
  // triangular mesh which is used to form a grid in which the solution is
  // evaluated.
  std::string grid_file = "./../test/mesh_files/grid_xy.txt";
  // number of refinements of the spatial grid mesh.
  int grid_refine = 2;

  // if provided, substitute default parameters by user input
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

  // create the appropriate space time mesh
  triangular_surface_mesh space_mesh( file );
  uniform_spacetime_tensor_mesh spacetime_mesh(
    space_mesh, end_time, n_timesteps );

  spacetime_mesh.refine( refine, 1 );
  // spacetime_mesh.print_info( );

  // quadrature order
  lo order_sing = 4;
  lo order_reg = 4;

  block_vector dir_proj;

  // single layer operator V without approximation
  block_lower_triangular_toeplitz_matrix * V
    = new block_lower_triangular_toeplitz_matrix( );
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );

  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
// compute L2 projection of the Dirichlet datum and declare the mass matrix and
// matrix assembler depending on the choice of basis functions
#ifdef USE_P0_BASIS
  space_p0.L2_projection( cauchy_data::dirichlet, dir_proj );
  besthea::bem::uniform_spacetime_be_identity M( space_p0, space_p0 );
  besthea::bem::uniform_spacetime_be_assembler assembler_v(
    kernel_v, space_p0, space_p0, order_sing, order_reg );
#else
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );
  space_p1.L2_projection( cauchy_data::dirichlet, dir_proj );
  besthea::bem::uniform_spacetime_be_identity M( space_p1, space_p1 );
  besthea::bem::uniform_spacetime_be_assembler assembler_v(
    kernel_v, space_p1, space_p1, order_sing, order_reg );
#endif
  assembler_v.assemble( *V );
  M.assemble( );

  // update the number of time steps and get the number of spatial dofs
  // (elements in case of p0 basis, nodes in case of p1 basis)
  n_timesteps = V->get_block_dim( );
  lo spatial_dofs = V->get_n_rows( );

  // compute the rhs of the linear system by multiplying the projection
  // coefficients with the mass matrix.
  block_vector M_dir_proj;
  M_dir_proj.resize( n_timesteps );
  M_dir_proj.resize_blocks( spatial_dofs );
  M.apply( dir_proj, M_dir_proj );

  // V approximated by pFMM
  // space time coupling coefficient for clustering in FMM
  sc st_coeff = 4.0;
  // temporal and spatial expansion orders for FMM
  lo temp_order = 6;
  lo spat_order = 6;
  spacetime_cluster_tree tree( spacetime_mesh, 5, 2, 10, st_coeff );

  // tree.print( );
  // assemble the structure of the pFMM matrix corresponding to the single
  // layer boundary integral operator V.
#ifdef USE_P0_BASIS
  fast_spacetime_be_space< basis_tri_p0 > space_p0_pFMM( tree );
  pFMM_matrix_heat_sl_p0p0 * V_pFMM = new pFMM_matrix_heat_sl_p0p0;
  fast_spacetime_be_assembler fast_assembler_v( kernel_v, space_p0_pFMM,
    space_p0_pFMM, order_sing, order_reg, temp_order, spat_order,
    cauchy_data::_alpha, 1.5, false );
#else
  fast_spacetime_be_space< basis_tri_p1 > space_p1_pFMM( tree );
  pFMM_matrix_heat_sl_p1p1 * V_pFMM = new pFMM_matrix_heat_sl_p1p1;
  fast_spacetime_be_assembler fast_assembler_v( kernel_v, space_p1_pFMM,
    space_p1_pFMM, order_sing, order_reg, temp_order, spat_order,
    cauchy_data::_alpha, 1.5, false );
#endif
  fast_assembler_v.assemble( *V_pFMM );

  timer t;
  t.reset( "Solving the system using the standard matrix V" );
  sc gmres_prec = 1e-8;
  lo gmres_iter = 500;
  // Solve for the unknown density using GMRES and the standard version of the
  // single layer matrix.
  block_vector dens( n_timesteps, spatial_dofs );
  V->mkl_fgmres_solve( M_dir_proj, dens, gmres_prec, gmres_iter, gmres_iter );
  t.measure( );
  std::cout << "iterations standard: " << gmres_iter << std::endl;
  gmres_prec = 1e-8;
  gmres_iter = 500;
  // Solve for the unknown density using GMRES and the pFMM version of the
  // single layer matrix.
  t.reset( "Solving the system using the pFMM matrix V" );
  block_vector dens_pFMM( n_timesteps, spatial_dofs, true );
  V_pFMM->mkl_fgmres_solve(
    M_dir_proj, dens_pFMM, gmres_prec, gmres_iter, gmres_iter );
  t.measure( );
  std::cout << "iterations pFMM: " << gmres_iter << std::endl;

  // compute for each timestep the difference (absolute and relative) of the
  // two densities
  std::cout << "error timewise" << std::endl;
  block_vector dens_diff( dens_pFMM );
  for ( lo i = 0; i < n_timesteps; ++i ) {
    dens_diff.get_block( i ).add( dens.get_block( i ), -1.0 );
    std::cout << dens_diff.get_block( i ).norm( ) << ", rel. "
              << dens_diff.get_block( i ).norm( ) / dens.get_block( i ).norm( )
              << std::endl;
  }

  // evaluate the solution using the provided grid file.
  if ( !grid_file.empty( ) ) {
    // load the grid as triangular surface mesh, scale it and refine it.
    triangular_surface_mesh grid_space_mesh( grid_file );
    grid_space_mesh.scale( 0.95 );
    grid_space_mesh.refine( grid_refine );
    // use the same temporal mesh as for the mesh for which the density was
    // computed.
    uniform_spacetime_tensor_mesh grid_spacetime_mesh(
      grid_space_mesh, end_time, spacetime_mesh.get_n_temporal_elements( ) );

    block_vector repr;
    block_vector repr_pFMM;
#ifdef USE_P0_BASIS
    besthea::bem::uniform_spacetime_be_evaluator evaluator_v(
      kernel_v, space_p0 );
#else
    besthea::bem::uniform_spacetime_be_evaluator evaluator_v(
      kernel_v, space_p1 );
#endif

    // evaluate the solution for the two densities
    evaluator_v.evaluate( grid_space_mesh.get_nodes( ), dens, repr );
    evaluator_v.evaluate( grid_space_mesh.get_nodes( ), dens_pFMM, repr_pFMM );

    // compute the l2 relative error between the approximated solutions and the
    // exact solutions in the nodes of the grid_spacetime_mesh
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

    // print the solution in the grid points (interpolation, and the
    // approximation using the standard matrix) to file in the Ensight format.
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
