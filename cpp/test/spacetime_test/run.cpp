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

#include <cmath>
#include <cstdlib>
#include <iostream>

int main( int argc, char * argv[] ) {
  using b_t_mesh = besthea::mesh::temporal_mesh;
  using b_s_mesh = besthea::mesh::triangular_surface_mesh;
  using b_st_mesh = besthea::mesh::spacetime_tensor_mesh;
  using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;
  // using b_ust_mesh = besthea::mesh::uniform_spacetime_tensor_mesh;
  // using b_st_slice = besthea::mesh::spacetime_slice;
  // using space_cluster_tree = besthea::mesh::space_cluster_tree;
  using tree_structure = besthea::mesh::tree_structure< scheduling_time_cluster >;
  using time_cluster_tree = besthea::mesh::time_cluster_tree;
  using space_time_cluster_tree = besthea::mesh::spacetime_cluster_tree;
  // using full_matrix = besthea::linear_algebra::full_matrix;

  std::string file = "./mesh_files/time_nuniform.txt";
  // std::string file = "./mesh_files/time_1_10.txt";

  if ( argc > 1 ) {
    file.assign( argv[ 1 ] );
  }

  b_t_mesh time_mesh( file );
  time_mesh.print_info( );

  // time_mesh.refine( 1 );
  // time_mesh.print_info( );

  // std::string file_spatial = "./test/mesh_files/icosahedron.txt";
  // std::string file_spatial = "./mesh_files/nuniform.txt";
  std::string file_spatial =  "./mesh_files/cube_12.txt";
  b_s_mesh space_mesh( file_spatial );

  // tensor_mesh.print_info( );

  // tensor_mesh.refine( 1 );

  // tensor_mesh.print_info( );

  // b_ust_mesh uniform_time_mesh( space_mesh, 1, 10 );
  // uniform_time_mesh.print_info( );
  // uniform_time_mesh.refine( 2 );
  // uniform_time_mesh.print_info( );

  // b_st_slice slice( file_spatial, file );

  space_mesh.refine( 1 );
  space_mesh.print_info( );
  space_mesh.print_vtu( "spacetime_test" );

  // space_cluster_tree ct( space_mesh, 4, 8 );
  // time_cluster_tree tt( time_mesh, 2, 4 );
  // ct.print_tree_separately( "test", false );

  // time_mesh.refine( 1 );

  b_st_mesh tensor_mesh( space_mesh, time_mesh );

  // coefficient to determine coupling of spatial and temoral levels
  sc st_coeff = 4.0;  // corresponds to \rho_L = 8 in Messner's paper
// const spacetime_tensor_mesh & spacetime_mesh,
//     lo time_levels, lo n_min_time_elems, lo n_min_space_elems, sc st_coeff,
//     lo spatial_nearfield_limit = 3 
  space_time_cluster_tree spt( tensor_mesh, 7, 3, 10, st_coeff );

  time_cluster_tree* time_tree = spt.get_time_tree( );
  std::string tree_vector_file = "./spacetime_test/tree_structure.bin";
  time_tree->print_tree_structure( tree_vector_file );
  time_tree->print( );

  std::vector< char > tree_vector = time_tree->load_tree_structure( 
    tree_vector_file );

  std::cout << "read tree structure from file: " << std::endl;
  for ( lou i = 0; i < tree_vector.size( ); ++i ) {
    std::cout << ( int ) tree_vector[ i ] << std::endl;
  }
  std::cout << "RECONSTRUCTED TREE" << std::endl;
  tree_structure skeleton( tree_vector_file );
  skeleton.print( );
}
