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

#include <besthea/space_cluster_tree.h>

#include "besthea/settings.h"
#include "besthea/spacetime_slice.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/temporal_mesh.h"
#include "besthea/time_cluster_tree.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"
#include "besthea/vector.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main( int argc, char * argv[] ) {
  using b_t_mesh = besthea::mesh::temporal_mesh;
  using b_s_mesh = besthea::mesh::triangular_surface_mesh;
  using b_st_mesh = besthea::mesh::spacetime_tensor_mesh;
  using b_ust_mesh = besthea::mesh::uniform_spacetime_tensor_mesh;
  using b_st_slice = besthea::mesh::spacetime_slice;
  using space_cluster_tree = besthea::mesh::space_cluster_tree;
  using time_cluster_tree = besthea::mesh::time_cluster_tree;

  std::string file = "./test/mesh_files/time_1_10.txt";

  if ( argc > 1 ) {
    file.assign( argv[ 1 ] );
  }

  b_t_mesh time_mesh( file );
  time_mesh.print_info( );

  // time_mesh.refine( 1 );
  // time_mesh.print_info( );

  // std::string file_spatial = "./test/mesh_files/icosahedron.txt";
  std::string file_spatial = "./test/mesh_files/nuniform.txt";
  b_s_mesh space_mesh( file_spatial );

  b_st_mesh tensor_mesh( space_mesh, time_mesh );

  tensor_mesh.print_info( );

  // tensor_mesh.refine( 1 );

  // tensor_mesh.print_info( );

  b_ust_mesh uniform_time_mesh( space_mesh, 1, 10 );
  uniform_time_mesh.print_info( );
  // uniform_time_mesh.refine( 2 );
  // uniform_time_mesh.print_info( );

  b_st_slice slice( file_spatial, file );

  space_mesh.refine( 1 );
  space_mesh.print_info( );
  space_mesh.print_vtu( "test" );

  space_cluster_tree ct( space_mesh, 4, 8 );
  time_cluster_tree tt( time_mesh, 2 );
  ct.print_tree_separately( "test", false );

  //  lo elem[ 6 ];
  //  sc node[ 4 ];
  //  std::cout << tensor_mesh.get_n_elements( ) << std::endl;
  //  for ( int i = 0; i < tensor_mesh.get_n_elements( ); ++i ) {
  //	  std::cout<< i << std::endl;
  //    tensor_mesh.get_element( i, elem );
  //    for ( int j = 0; j < 6; ++j ) {
  //      std::cout << elem[ j ] << " ";
  //    }
  //    std::cout << std::endl;
  //    for ( int j = 0; j < 6; ++j ) {
  //      tensor_mesh.get_node( elem[ j ], node );
  //      std::cout << node[ 0 ] << ", " << node[ 1 ] << ", " << node[ 2 ] << ",
  //      "
  //                << node[ 3 ] << std::endl;
  //    }
  //  }
}
