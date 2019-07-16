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

#include "besthea/full_matrix.h"
#include "besthea/settings.h"
#include "besthea/sparse_matrix.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"
#include "besthea/vector.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main( int argc, char * argv[] ) {
  using b_mesh = besthea::mesh::triangular_surface_mesh;
  using b_vector = besthea::linear_algebra::vector;
  using b_matrix = besthea::linear_algebra::full_matrix;
  using b_sparse_matrix = besthea::linear_algebra::sparse_matrix;

  std::string file = "../mesh_files/cube_12.txt";

  if ( argc > 1 ) {
    file.assign( argv[ 1 ] );
  }

  b_mesh mesh( file );
  mesh.print_info( );

  // mesh.refine( 6 );
  // mesh.map_to_unit_sphere( );
  // mesh.print_info( );
  // mesh.print_vtu( "output.vtu" );

  lo size = 10;
  b_vector x( size );
  x.random_fill( -1, 1 );

  b_matrix A( size, size );
  A.random_fill_diag( 1.0, 2.0 );

  b_vector b( size );
  A.apply_symmetric( x, b );

  A.choleski_decompose_solve( b, 1 );
  b.add( x, -1.0 );
  std::cout << b.norm( ) << std::endl;

  b_sparse_matrix B( );
}
