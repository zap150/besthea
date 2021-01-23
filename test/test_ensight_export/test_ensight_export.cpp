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

#include "besthea/block_vector.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

/**
 *  @todo delete?
 */
int main( int argc, char * argv[] ) {
  using b_mesh = besthea::mesh::triangular_surface_mesh;
  using b_vector = besthea::linear_algebra::vector;
  using b_block_vector = besthea::linear_algebra::block_vector;

  std::string file = "../mesh_files/icosahedron.txt";

  if ( argc > 1 ) {
    file.assign( argv[ 1 ] );
  }

  b_mesh mesh( file );
  mesh.refine( 5 );
  mesh.print_info( );

  const lo n_timesteps = 1000;
  besthea::mesh::uniform_spacetime_tensor_mesh space_time_mesh(
    mesh, 5.0, n_timesteps );

  std::vector< std::string > node_labels = { "Pressure", "Temperature" };
  std::vector< std::string > elem_labels = { "Elem" };

  b_block_vector pressure( n_timesteps, mesh.get_n_nodes( ) );
  b_block_vector temperature( n_timesteps, mesh.get_n_nodes( ) );
  b_block_vector something( n_timesteps, mesh.get_n_elements( ) );

  for ( lo ts = 0; ts < n_timesteps; ++ts ) {
    b_vector & pressure_data = pressure.get_block( ts );
    b_vector & temperature_data = temperature.get_block( ts );
    b_vector & something_data = something.get_block( ts );

    for ( lo i = 0; i < mesh.get_n_nodes( ); ++i ) {
      pressure_data[ i ] = cos( ts / 100.0 ) + sin( i / 2.0 );
      temperature_data[ i ] = sin( ts / 100.0 ) + cos( i * 1.4 );
    }

    for ( lo i = 0; i < mesh.get_n_elements( ); ++i ) {
      something_data[ i ] = sin( (double) i ) * cos( (double) ts );
    }
  }

  std::filesystem::create_directory( "ensight" );

  std::vector< b_block_vector * > node_data{ &pressure, &temperature };
  std::vector< b_block_vector * > elem_data{ &something };

  const lo time_stride = 10;

  space_time_mesh.print_ensight_case(
    "ensight", &node_labels, &elem_labels, time_stride );
  space_time_mesh.print_ensight_geometry( "ensight" );
  space_time_mesh.print_ensight_datafiles( "ensight", &node_labels, &node_data,
    &elem_labels, &elem_data, time_stride );
}
