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
#include "besthea/uniform_spacetime_be_onthefly_matrix.h"

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

  static constexpr sc _alpha{ 1.0 };
  static constexpr std::array< sc, 3 > _y{ 1.5, 1.5, 1.5 };
  static constexpr sc _shift{ 0.0 };
};

int main( int argc, char * argv[] ) {
  std::string file;

  srand(time(nullptr));

  // default values
  file = "../examples/mesh_files/cube_12.txt";
  int refine = 0;
  lo n_timesteps = 8;
  sc end_time = 1.0;

  // read from commandl line
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

  // load spatial mesh from file and refine it
  triangular_surface_mesh space_mesh;
  space_mesh.load( file );
  space_mesh.refine( refine );

  // refine number of timesteps
  n_timesteps *= std::exp2( 2 * refine );

  // create spacetime mesh as a tensor product of spatial and temporal meshes
  uniform_spacetime_tensor_mesh spacetime_mesh(
    space_mesh, end_time, n_timesteps );

  space_mesh.print_info( );
  spacetime_mesh.print_info( );

  timer t;

  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );

  // numerical quadrature orders
  lo order_sing
    = 4;  // for singular integrals (adjacent or identical spatial elements)
  lo order_reg = 4;  // disjoint spatial elements

  // create matrix assembler
  block_lower_triangular_toeplitz_matrix * V
    = new block_lower_triangular_toeplitz_matrix( );
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );
  uniform_spacetime_be_assembler assembler_v(
    kernel_v, space_p0, space_p0, order_sing, order_reg );



  std::cout << "T" << spacetime_mesh.get_n_temporal_nodes() << " S" << spacetime_mesh.get_n_nodes() << "\n";

  block_vector x (spacetime_mesh.get_n_temporal_nodes(), spacetime_mesh.get_n_nodes(), false);
  block_vector y1(spacetime_mesh.get_n_temporal_nodes(), spacetime_mesh.get_n_nodes(), false);
  block_vector y2(spacetime_mesh.get_n_temporal_nodes(), spacetime_mesh.get_n_nodes(), false);
  for (lo b = 0; b < x.get_block_size(); b++) {
    for (lo i = 0; i < x.get_size_of_block(); i++) {
      x.set(b, i, rand() / RAND_MAX);
      y1.set(b, i, 3.14);
      y2.set(b, i, 3.14);
    }    
  }
  sc alpha = 2;
  sc beta = 3;

  t.reset( "InMemory" );
  assembler_v.assemble( *V );
  V->apply(x, y1, false, alpha, beta);
  t.measure( );

  t.reset( "OnTheFly" );
  besthea::uniform_spacetime_be_onthefly_matrix onthefly(kernel_v, space_p0, space_p0, order_sing, order_reg);
  onthefly.my_apply(x, y2, false, alpha, beta);
  t.measure();

  bool areEqual = true;
  for (lo b = 0; b < x.get_block_size(); b++) {
    for (lo i = 0; i < x.get_size_of_block(); i++) {
      sc v1 = y1.get(b, i);
      sc v2 = y2.get(b, i);
      if( std::abs(v1 - v2) / v1 > 1e-6 ) {
        std::cout << "Vectors dont match: B" << b << " I" << i << " " << v1 << " " << v2 << "\n";
        areEqual = false;
      }
    }    
  }
  if(areEqual)
    std::cout << "Vectors are equal!\n";

  delete V;
}
