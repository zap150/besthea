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
#include "besthea/uniform_spacetime_tensor_mesh.h"

#include <iostream>

using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem;

int main( int argc, char * argv[] ) {
  std::string file = "../mesh_files/cube_12.txt";
  if ( argc > 1 ) {
    file.assign( argv[ 1 ] );
  }
  triangular_surface_mesh space_mesh( file );
  uniform_spacetime_tensor_mesh spacetime_mesh( space_mesh, 1.0, 8 );

  uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > test_space(
    spacetime_mesh );
  uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > trial_space(
    spacetime_mesh );
  sc alpha = 2.5;
  uniform_spacetime_heat_sl_kernel_antiderivative kernel(
    spacetime_mesh.get_timestep( ), alpha );
  lo order_sing = 4;
  lo order_reg = 4;
  uniform_spacetime_be_assembler assembler(
    kernel, test_space, trial_space, order_sing, order_reg );

  block_lower_triangular_toeplitz_matrix matrix;
  assembler.assemble( matrix );

  matrix.print( );
}
