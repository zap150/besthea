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
  // using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;
  using tree_structure = besthea::mesh::tree_structure;
  using time_cluster_tree = besthea::mesh::time_cluster_tree;
  using spacetime_cluster_tree = besthea::mesh::spacetime_cluster_tree;

  //############################################################################
  // load the geometry, create the trees and tree structure and write the 
  // tree structure and process assignments to file
  //############################################################################

  // choose the parameters for the process assignments, and a process id for
  // reconstruction 
  lo strategy = 1;
  lo n_processes = 6;
  lo proc_id = 4;
  std::cout << "n_processes: " << n_processes << ", strategy: "
            << strategy << ", proc_id: " << proc_id << std::endl;

  std::string file_temporal, file_spatial, output_basis;
  // choose the appropriate meshes (one in each block)
  {
    file_temporal = "time_nuniform";
    // file_temporal = "time_1_10";
  }
  {
    // file_spatial = "icosahedron";
    // file_spatial = "nuniform"; 
    file_spatial = "cube_12";
  }

  // uncomment to load mesh from file
  std::string path_temporal = "./mesh_files/" + file_temporal + ".txt";
  b_t_mesh time_mesh( path_temporal );
  time_mesh.refine( 1 );

  // uncomment to generate regular mesh
  // lo levels = 3;
  // file_temporal = "time_uniform_" + std::to_string( levels );
  // b_t_mesh time_mesh( 0, 1, 1 << ( levels + 1 ) );

  output_basis = "./associate_space_time/" + file_temporal + "_" + file_spatial;

  std::string path_spatial = "./mesh_files/" + file_spatial + ".txt";
  b_s_mesh space_mesh( path_spatial );
  space_mesh.refine( 2 );
  b_st_mesh tensor_mesh( space_mesh, time_mesh );

  // coefficient to determine coupling of spatial and temoral levels
  sc st_coeff = 1.0;  // corresponds to \rho_L = 8 in Messner's paper
  spacetime_cluster_tree st_tree( tensor_mesh, 20, 3, 10, st_coeff );
  // st_tree.print( );
  std::cout << "time tree " << std::endl;
  std::cout << "space tree " << std::endl;

  time_cluster_tree* time_tree = st_tree.get_time_tree( );

  // write tree structure to file
  std::string tree_vector_file = output_basis + ".tree_vec";
  time_tree->print_tree_structure( tree_vector_file );
  // compute process assignment and write it to file
  std::string process_assignment_file = output_basis + ".proc_assign";
  time_tree->print_process_assignments( n_processes, strategy, 
    process_assignment_file );

  //############################################################################
  // reconstruct the tree structure and reduce it to the locally essential tree
  //############################################################################

  tree_structure time_structure( 
    tree_vector_file, time_mesh.get_start( ), time_mesh.get_end( ) );
  time_structure.load_process_assignments( process_assignment_file );

  // print process ids at cluster positions
  // lo digits = ( lo ) ( ceil( log10( n_processes + 1 ) ) + 1 );
  // bool print_process_ids = true;

  // print global cluster ids at cluster positions
  lo digits = 4;
  bool print_process_ids = false;

  // print original tree structure
  std::cout << "printing full tree structure:" << std::endl;
  time_structure.print_tree_human_readable( digits, print_process_ids );

  // print locally essential tree structure
  std::cout << "printing reduced tree structure:" << std::endl;
  time_structure.reduce_2_essential( proc_id );
  time_structure.print_tree_human_readable( digits, print_process_ids );

  //############################################################################
  // refine the spacetime mesh, create a new spacetime tree, expand the 
  // locally essential tree and find the associated spacetime clusters for all 
  // its clusters
  //############################################################################
  
  tensor_mesh.refine( 1, 2 );
  spacetime_cluster_tree st_tree_refined( tensor_mesh, 20, 3, 10, st_coeff );

  std::cout << "refined time tree" << std::endl;
  st_tree_refined.get_time_cluster_tree( )->print( );
  std::cout << "refined space tree" << std::endl;
  st_tree_refined.get_space_cluster_tree( )->print( );

  time_structure.expand_tree_structure_essentially( &st_tree_refined  );
  std::cout << "expanded tree structure" << std::endl; 

  // time_structure.print_tree_human_readable( 2, print_process_ids );
  time_structure.print_tree_human_readable( 4, false );
  time_structure.print_tree_human_readable( 4, true );

  // deactivate to construct refined tree structure for comparison
  std::string refined_tree_vector_file = output_basis + "_refined.tree_vec";
  st_tree_refined.get_time_cluster_tree( )->print_tree_structure(
    refined_tree_vector_file );
  tree_structure refined_time_structure( 
    refined_tree_vector_file, 0.0, 1.0 );
  std::cout << "complete refined tree structure for comparison " 
            << std::endl;
  refined_time_structure.print_tree_human_readable( 4, false );
  
  std::cout << "find associated space time clusters" << std::endl;
  time_structure.find_associated_space_time_clusters( &st_tree_refined );
  time_structure.print( );
}
