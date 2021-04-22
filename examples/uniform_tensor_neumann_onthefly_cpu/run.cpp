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
using namespace besthea::bem::onthefly;

struct cauchy_data {
  static sc dirichlet( sc x1, sc x2, sc x3, const coordinates< 3 > &, sc t ) {
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
    int dummy = 0.0;
    sc value = std::pow( 4.0 * M_PI * _alpha * _shift, -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * _shift + dummy ) );

    return value;
  }

  static constexpr sc _alpha{ 1.0 };
  static constexpr std::array< sc, 3 > _y{ 1.5, 1.5, 1.5 };
  static constexpr sc _shift{ 0.0 };
};  // struct cauchy_data

int main() {

  std::string space_mesh_file = "bin/cube_192.txt";
  int refine = 0;
  lo n_timesteps = 8;
  sc end_time = 1.0;

  // load surface mesh, refine, create spacetime tensor mesh
  triangular_surface_mesh space_mesh;
  space_mesh.load( space_mesh_file );
  space_mesh.refine( refine );
  n_timesteps *= std::exp2( 2 * refine );
  uniform_spacetime_tensor_mesh spacetime_mesh( space_mesh, end_time, n_timesteps );

  // create BE spaces
  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );

  // create heat kernel antiderivatives
  spacetime_heat_adl_kernel_antiderivative kernel_kt( cauchy_data::_alpha );
  spacetime_heat_hs_kernel_antiderivative  kernel_d ( cauchy_data::_alpha );

  // project boundary condition onto the BE space
  block_vector bc_neu;
  space_p0.L2_projection( cauchy_data::neumann, bc_neu );

  // create the CPU on-the-fly matrices. no assembly needed.
  uniform_spacetime_be_matrix_onthefly_cpu Kt(kernel_kt, space_p1, space_p0);
  uniform_spacetime_be_matrix_onthefly_cpu D (kernel_d,  space_p1, space_p1);

  // create and assemble the mass matrix
  uniform_spacetime_be_identity M( space_p0, space_p1 );
  M.assemble( );

  // create and assemble right hand side vector
  block_vector rhs( D.get_block_dim(), D.get_n_rows() );
  M.apply ( bc_neu, rhs, true,   0.5, 0.0 );
  Kt.apply( bc_neu, rhs, false, -1.0, 1.0 );

  // solve the system
  block_vector sol_dir( D.get_block_dim(), D.get_n_columns() );
  sc rel_error = 1e-6;
  lo n_iters = 1000;
  D.mkl_fgmres_solve( rhs, sol_dir, rel_error, n_iters );

  // do something with the solution on boundary ...

  return 0;
}
