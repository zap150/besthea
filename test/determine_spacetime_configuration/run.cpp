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
using namespace besthea::bem;
using namespace besthea::linear_algebra;
using namespace besthea::tools;

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
      * dirichlet( x1, x2, x3, n, ( t + _shift ) );

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

  static constexpr sc _alpha{ 4.0 };
  static constexpr std::array< sc, 3 > _y{ 1.5, 1.5, 1.5 };
  static constexpr sc _shift{ 0.0 };
};

int main( int, char ** ) {
  // read the space mesh from file and refine it twice (to get 192 elements)
  std::string spatial_mesh_file = "./mesh_files/cube_12.txt";
  triangular_surface_mesh space_mesh( spatial_mesh_file );
  // for a sequence of timesteps determine the L2 error of the piecewise
  // constant L2 projection of the Neumann data.

  for ( lo space_refine = 0; space_refine < 7; ++space_refine ) {
    for ( lo n_timesteps = 4; n_timesteps < ( 1 << 15 ); n_timesteps *= 2 ) {
      uniform_spacetime_tensor_mesh spacetime_mesh(
        space_mesh, 1.0, n_timesteps );
      block_vector neu_proj(
        n_timesteps, spacetime_mesh.get_n_spatial_elements( ) );
      uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
      space_p0.L2_projection( cauchy_data::neumann, neu_proj );
      std::cout << space_p0.L2_relative_error( cauchy_data::neumann, neu_proj )
                << ", ";
    }
    std::cout << std::endl;
    space_mesh.refine( 1 );
  }

  // // different test: check best approximation error for the examples of
  // // Messner, Schanz, Tausch (2014), Section 5.1, Table 3
  // lo n_timesteps = 32;
  // sc end_time = 0.5;
  // for ( lo n_refinements = 0; n_refinements < 2; ++n_refinements ) {
  //   uniform_spacetime_tensor_mesh spacetime_mesh(
  //     space_mesh, end_time, n_timesteps );
  //   spacetime_mesh.refine( n_refinements, 2 );
  //   block_vector neu_proj(
  //     n_timesteps, spacetime_mesh.get_n_spatial_elements( ) );
  //   uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
  //   space_p0.L2_projection( cauchy_data::neumann, neu_proj );
  //   std::cout << "n_refinements: " << n_refinements;
  //   std::cout << ", n_space_elements: "
  //             << spacetime_mesh.get_n_spatial_elements( );
  //   std::cout << ", n_timesteps: " << n_timesteps << ", rel. error: "
  //             << space_p0.L2_relative_error( cauchy_data::neumann, neu_proj )
  //             << std::endl;
  //   n_timesteps *= 4;
  // }
}

