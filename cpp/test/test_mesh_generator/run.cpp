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
#include "besthea/distributed_spacetime_cluster_tree.h"
#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/settings.h"
#include "besthea/spacetime_mesh_generator.h"
#include "besthea/temporal_mesh.h"
#include "besthea/time_cluster_tree.h"
#include "besthea/tree_structure.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

using besthea::mesh::distributed_spacetime_tensor_mesh;
using besthea::mesh::spacetime_mesh_generator;
using besthea::mesh::tree_structure;

int main( int argc, char * argv[] ) {
  using b_t_mesh = besthea::mesh::temporal_mesh;
  using time_cluster_tree = besthea::mesh::time_cluster_tree;

  int provided;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );

  int myRank, n_processes;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( MPI_COMM_WORLD, &myRank );
  MPI_Comm_size( comm, &n_processes );

  // output/input files for temporal tree and distribution of clusters among
  // processes
  std::string tree_vector_file = "./job_scheduler/tree_structure.bin";
  std::string process_assignment_file
    = "./job_scheduler/process_assignment.bin";
  std::string cluster_bounds_file 
    = "./job_scheduler/cluster_bounds.bin";

  if ( myRank == 0 ) {
    std::cout << "### start mesh generation ###" << std::endl;
    std::string space_file = "./mesh_files/cube_12.txt";  // spatial mesh
    std::string time_file = "./testfile.txt";  // file defining temporal slices
    lo time_refinement = 3;                    // defining mesh within slices
    lo space_refinement = 2;

    // load time mesh defining slices and create temporal tree
    b_t_mesh time_mesh( time_file );
    lo time_levels = 20;
    lo n_min_time_elems = 2;
    time_cluster_tree time_tree( time_mesh, time_levels, n_min_time_elems );

    // write tree structure to file
    time_tree.print_tree_structure( tree_vector_file );

    // write cluster bounds to file
    time_tree.print_cluster_bounds( cluster_bounds_file );

    // std::vector< sc > cluster_bounds 
    //   = read_vector_from_bin_file< sc >( cluster_bounds_file );

    // for ( lou i = 0; i < cluster_bounds.size( ); i += 2 ) {
    //   std::cout << "[" << cluster_bounds[ i ] << ", " 
    //             << cluster_bounds[ i + 1 ] << "]" << std::endl;
    // }
    
    // tree_structure temp_struct( tree_vector_file, cluster_bounds_file );

    // temp_struct.print( );

    // compute process assignment and write it to file

    lo strategy = 1;
    std::cout << "n_processes: " << n_processes << ", strategy: " << strategy
              << std::endl;

    time_tree.print_process_assignments(
      n_processes, strategy, process_assignment_file );

    spacetime_mesh_generator generator(
      space_file, time_file, time_refinement, space_refinement );

    generator.generate( "", "test_mesh", "txt" );
    std::cout << "### end mesh generation ###" << std::endl;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  distributed_spacetime_tensor_mesh mesh(
    "test_mesh_d.txt", tree_vector_file, cluster_bounds_file, 
    process_assignment_file, &comm );

  // std::cout << mesh.get_n_elements( ) << std::endl;

  besthea::mesh::distributed_spacetime_cluster_tree tree(
    mesh, 20, 10, 1.0, 1, &comm );
  
  for ( int output_id = 0; output_id < n_processes; ++output_id ) {
    lo digits = 3;
    if ( output_id == 0 && myRank == 0 ) {
      std::cout << "myRank is " << myRank << std::endl;
      tree_structure time_structure( 
        tree_vector_file, mesh.get_start( ), mesh.get_end( ), myRank );
      time_structure.load_process_assignments( process_assignment_file );
      std::cout << "process ids in initial global distribution tree:" 
                << std::endl;
      time_structure.print_tree_human_readable( digits, true );
      std::cout << "global number of elements is " << mesh.get_n_elements( ) 
                << std::endl;
      std::cout << "printing local part of distributed cluster tree: " 
                << std::endl;
      tree.print( );
    }
    if ( output_id == myRank ) {
      std::cout << "process ids in " << myRank << "'s locally essential "
                << "distribution tree:" << std::endl;
      mesh.get_distribution_tree( )->print_tree_human_readable( digits, true );
    }
    MPI_Barrier( comm );
  }
  MPI_Finalize( );
}
