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

#include "besthea/bem.h"
#include "besthea/linear_algebra.h"
#include "besthea/settings.h"
#include "besthea/tools.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"

#include <cstdlib>
#include <iostream>

using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem;
using namespace besthea::tools;

struct cauchy_data {
  static sc dirichlet( sc x1, sc x2, sc x3, sc * n, sc t ) {
    sc norm2 = ( x1 - y[ 0 ] ) * ( x1 - y[ 0 ] )
      + ( x2 - y[ 1 ] ) * ( x2 - y[ 1 ] ) + ( x3 - y[ 2 ] ) * ( x3 - y[ 2 ] );
    sc value = std::pow( 4.0 * M_PI * alpha * t, -1.5 )
      * std::exp( -norm2 / ( 4.0 * alpha * t ) );

    return value;
  }

  static sc neumann( sc x1, sc x2, sc x3, sc * n, sc t ) {
    sc dot = ( x1 - y[ 0 ] ) * n[ 0 ] + ( x2 - y[ 1 ] ) * n[ 1 ]
      + ( x3 - y[ 2 ] ) * n[ 2 ];
    sc value = ( -1.0 / ( 2.0 * t ) ) * dot * dirichlet( x1, x2, x3, n, t );

    return value;
  }

  static constexpr sc alpha{ 0.5 };
  static constexpr std::array< sc, 3 > y{ 0.0, 0.0, 1.5 };
};

int main( int argc, char * argv[] ) {
  std::string file = "../mesh_files/cube_192.txt";
  int refine = 0;
  lo n_timesteps = 8;
  sc end_time = 1.0;

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
  triangular_surface_mesh space_mesh( file );
  uniform_spacetime_tensor_mesh spacetime_mesh(
    space_mesh, end_time, n_timesteps );
  spacetime_mesh.refine( refine, 1 );

  spacetime_mesh.print_info( );

  timer t;

  uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > space_p0(
    spacetime_mesh );
  uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > space_p1(
    spacetime_mesh );

  //*
  lo order_sing = 4;
  lo order_reg = 4;
  block_lower_triangular_toeplitz_matrix D;
  uniform_spacetime_heat_hs_kernel_antiderivative kernel_d(
    spacetime_mesh.get_timestep( ), cauchy_data::alpha );
  uniform_spacetime_be_assembler assembler_d(
    kernel_d, space_p1, space_p1, order_sing, order_reg );
  t.reset( "D" );
  assembler_d.assemble( D );
  t.measure( );
  // D.print( );

  block_lower_triangular_toeplitz_matrix K;
  uniform_spacetime_heat_dl_kernel_antiderivative kernel_k(
    spacetime_mesh.get_timestep( ), cauchy_data::alpha );
  uniform_spacetime_be_assembler assembler_k(
    kernel_k, space_p0, space_p1, order_sing, order_reg );
  t.reset( "K" );
  assembler_k.assemble( K );
  t.measure( );
  // K.print( );

  sparse_matrix M;
  uniform_spacetime_be_identity identity( space_p0, space_p1, 1 );
  t.reset( "M" );
  identity.assemble( M );
  t.measure( );
  // M.print( );
  //*/

  block_vector bv_dir_proj, bv_neu_proj, bv_dir;
  space_p1.l2_projection( cauchy_data::dirichlet, bv_dir_proj );
  space_p0.l2_projection( cauchy_data::neumann, bv_neu_proj );
  std::cout << "Dirichlet L2 projection relative error: "
            << space_p1.l2_relative_error( cauchy_data::dirichlet, bv_dir_proj )
            << std::endl;
  std::cout << "Neumann L2 projection relative error: "
            << space_p0.l2_relative_error( cauchy_data::neumann, bv_neu_proj )
            << std::endl;

  t.reset( "Solving the system" );
  uniform_spacetime_be_solver::time_marching_neumann(
    D, K, M, bv_neu_proj, bv_dir );
  t.measure( );
  std::cout << "Dirichlet L2 relative error: "
            << space_p1.l2_relative_error( cauchy_data::dirichlet, bv_dir )
            << std::endl;

  /*
  std::vector< std::string > node_labels{ "Dirichlet_projection",
    "Dirichlet_result" };
  std::vector< std::string > elem_labels{ "Neumann_projection" };
  std::stringstream ss;
  for ( lo d = 0; d < spacetime_mesh.get_n_temporal_elements( ); ++d ) {
    ss.str( "" );
    ss.clear( );
    ss << "output.vtu." << d;
    std::vector< sc * > node_data{ bv_dir_proj.get_block( d ).data( ),
      bv_dir.get_block( d ).data( ) };
    std::vector< sc * > elem_data{ bv_neu_proj.get_block( d ).data( ) };
    spacetime_mesh.print_vtu(
      ss.str( ), &node_labels, &node_data, &elem_labels, &elem_data );
  }
  */
}
