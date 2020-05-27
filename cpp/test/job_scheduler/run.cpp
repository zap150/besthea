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
  using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;
  using tree_structure 
    = besthea::mesh::tree_structure< scheduling_time_cluster >;
  using time_cluster_tree = besthea::mesh::time_cluster_tree;

  // uncomment appropriate mesh
  std::string file_temporal = "./mesh_files/time_nuniform.txt";
  // std::string file_temporal = "./mesh_files/time_1_10.txt";

  // uncomment to load mesh from file
  b_t_mesh time_mesh( file_temporal );
  time_mesh.refine( 1 );

  // uncomment to generate regular mesh
  // lo levels = 5;
  // b_t_mesh time_mesh( 0, 1, 1 << ( levels + 1 ) );

  lo time_levels = 20;
  lo n_min_time_elems = 3;
  time_cluster_tree time_tree( time_mesh, time_levels, n_min_time_elems );

  // TEST OF TREE STRUCTURE AND PROCESS ASSIGNMENT (INCLUDING IO)

  // write tree structure to file
  std::string tree_vector_file = "./job_scheduler/tree_structure.bin";
  time_tree.print_tree_structure( tree_vector_file );

  // compute process assignment and write it to file
  lo strategy = 1;
  lo n_processes = 7;
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
  skeleton.print_processes_human_readable( digits );

  // reduce to locally essential tree
  lo my_process_id = 6;
  std::cout << "reducing to locally essential tree for process "
            << my_process_id << std::endl;
  skeleton.reduce_2_essential( my_process_id );
  // skeleton.print( );
  skeleton.print_processes_human_readable( digits );
}
