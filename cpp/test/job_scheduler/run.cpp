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
#include <list>

int main( int argc, char * argv[] ) {
  using b_t_mesh = besthea::mesh::temporal_mesh;
  using tree_structure = besthea::mesh::tree_structure;
  using time_cluster_tree = besthea::mesh::time_cluster_tree;
  using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;

  // uncomment appropriate mesh
  std::string file_temporal = "./mesh_files/time_nuniform.txt";
  // std::string file_temporal = "./mesh_files/time_1_10.txt";

  // uncomment to load mesh from file
  // b_t_mesh time_mesh( file_temporal );
  // time_mesh.refine( 1 );

  // uncomment to generate regular mesh
  lo levels = 5;
  b_t_mesh time_mesh( 0, 1, 1 << ( levels + 1 ) );

  lo time_levels = 20;
  lo n_min_time_elems = 3;
  time_cluster_tree time_tree( time_mesh, time_levels, n_min_time_elems );

  // TEST OF TREE STRUCTURE AND PROCESS ASSIGNMENT (INCLUDING IO)

  // write tree structure to file
  std::string tree_vector_file = "./job_scheduler/tree_structure.bin";
  time_tree.print_tree_structure( tree_vector_file );

  // compute process assignment and write it to file
  lo strategy = 1;
  lo n_processes = 1;
  std::cout << "n_processes: " << n_processes << ", strategy: "
            << strategy << std::endl;
  std::string process_assignment_file = 
    "./job_scheduler/process_assignment.bin";
  time_tree.print_process_assignments( n_processes, strategy, 
    process_assignment_file );

  // generate tree_structure from file, load process assignments and print
  tree_structure skeleton( tree_vector_file, time_mesh.get_start( ),
    time_mesh.get_end( ) );
  skeleton.load_process_assignments( process_assignment_file );
  lo digits = ( lo ) ( ceil( log10( n_processes + 1 ) ) + 1 );
  bool print_process_ids = true;

  // lo digits = 3;
  // bool print_process_ids = false;

  skeleton.print_tree_human_readable( digits, true );
  lou n_leaves = skeleton.get_leaves( ).size( );

  // std::vector< scheduling_time_cluster* > leaves = skeleton.get_leaves( );
  // for ( auto it = leaves.begin( ); it != leaves.end( ); ++it ) {
  //   std::cout << "cluster is: " << std::endl;
  //   ( *it )->print( );
  //   std::vector< scheduling_time_cluster* > * nearfield
  //     = ( *it )->get_nearfield( );
  //   std::cout << "nearfield clusters are " << std::endl;
  //   for ( auto it_nf = nearfield->begin( ); it_nf != nearfield->end( ); 
  //         ++it_nf ) {
  //     ( *it_nf )->print( );
  //   }
  //   std::cout << "###############################" << std::endl;
  // }

  // reduce to locally essential tree
  lo my_process_id = 0;
  std::cout << "reducing to locally essential tree for process "
            << my_process_id << std::endl;
  skeleton.reduce_2_essential( my_process_id );
  skeleton.print_tree_human_readable( digits, print_process_ids );
  // skeleton.print( );

  std::cout << "preparing and executing pseudo-fmm" << std::endl;
  std::list< scheduling_time_cluster* > m_list, m2l_list, l_list, n_list;
  std::vector< std::pair< scheduling_time_cluster*, lo > > receive_vector;
  lou n_moments_to_receive;
  skeleton.prepare_fmm( m_list, m2l_list, l_list, n_list, receive_vector,
                        n_moments_to_receive );
  std::vector< sc > input_vector( n_leaves, 1.0 );
  std::vector< sc > output_vector( input_vector );

  apply_fmm( my_process_id, receive_vector, n_moments_to_receive, m_list,
    m2l_list, l_list, n_list, input_vector, output_vector );
  std::cout << "output vector: " << std::endl;
  for ( lou i = 0; i < output_vector.size( ); ++i ) {
    std::cout << output_vector[ i ] << std::endl;
  }
}
