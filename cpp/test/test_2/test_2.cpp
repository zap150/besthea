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
#include "besthea/triangular_surface_mesh.h"
#include "besthea/uniform_spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"

#include <iostream>

using SMesh = besthea::mesh::triangular_surface_mesh;
using STMesh = besthea::mesh::uniform_spacetime_tensor_mesh;
using Vector = besthea::linear_algebra::vector;
using Kernel = besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative;

int main( int argc, char * argv[] ) {
  /*
  std::string file = "../mesh_files/cube_12.txt";
  if ( argc > 1 ) {
    file.assign( argv[ 1 ] );
  }
  SMesh space_mesh( file );
  STMesh spacetime_mesh( space_mesh, 1.0, 8 );
  */

  sc alpha = 2.5;
  sc ht = 0.125;
  Kernel kernel( ht, alpha );

  lo size = 100000;
  Vector xy1( size, false );
  Vector xy2( size, false );
  Vector xy3( size, false );
  xy1.random_fill( 0.1, 2.0 );
  xy2.random_fill( 0.1, 2.0 );
  xy3.random_fill( 0.1, 2.0 );
  Vector value( size, false );

  const sc * xy1_data = xy1.data( );
  const sc * xy2_data = xy2.data( );
  const sc * xy3_data = xy3.data( );
  sc * value_data = value.data( );

  lo delta = 1;

#pragma omp simd simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    value_data[ i ] = kernel.anti_tau_anti_t(
      xy1_data[ i ], xy2_data[ i ], xy3_data[ i ], nullptr, nullptr, delta );
  }

  // value.print( );
}
