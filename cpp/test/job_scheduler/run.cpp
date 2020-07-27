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

#include <chrono>         // std::chrono::seconds
#include <cmath>
#include <iostream>
#include <list>
#include <string>
#include <thread>         // std::this_thread::sleep_for


int main( int argc, char * argv[] ) {
  using b_t_mesh = besthea::mesh::temporal_mesh;
  using tree_structure = besthea::mesh::tree_structure;
  using time_cluster_tree = besthea::mesh::time_cluster_tree;
  using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;

  // Choose if the tree structure should be generated.
  // If yes, some choices of geometries can be set below.
  bool generate_structure_and_process_assignment = true;
  // choose if the pseudo FMM should be executed.
  bool execute_pseudo_fmm = true;
  // choose verbose mode or not
  bool verbose = true;
  std::string verbose_dir = "./job_scheduler/output_verbose";

  // Choose the number of MPI processes used for computation and the id of the
  // cluster which is responsible for the output.
  lo n_processes = 4;
  lo output_id = 3;

  MPI_Init(&argc,&argv);
  MPI_Comm communicator = MPI_COMM_WORLD;
  int n_mpi_processes, process_id;
  MPI_Comm_size( communicator, &n_mpi_processes );
  MPI_Comm_rank( communicator, &process_id );

  if ( generate_structure_and_process_assignment ) {
    if ( process_id == 0 ) {
      // uncomment appropriate mesh
      std::string file_temporal = "./mesh_files/time_nuniform.txt";
      // std::string file_temporal = "./mesh_files/time_1_10.txt";

      // uncomment to load mesh from file
      // b_t_mesh time_mesh( file_temporal );
      // time_mesh.refine( 2 );

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
      std::cout << "Generating assignment for n_processes: " << n_processes 
                << ", strategy: " << strategy << std::endl;
      std::string process_assignment_file = 
        "./job_scheduler/process_assignment.bin";
      time_tree.print_process_assignments( n_processes, strategy, 
        process_assignment_file );
    }
  } 
  MPI_Barrier( communicator );
  if ( execute_pseudo_fmm ) {
    if ( n_mpi_processes == n_processes ) {
      // generate tree_structure from file, load process assignments and print
      sc mesh_start = 0.0;
      sc mesh_end = 1.0;
      std::string tree_vector_file = "./job_scheduler/tree_structure.bin";
      std::string process_assignment_file = 
        "./job_scheduler/process_assignment.bin";
      tree_structure time_structure( tree_vector_file, mesh_start, mesh_end );
      time_structure.load_process_assignments( process_assignment_file );
      
      // help variables to print the process ids in human readable format
      // lo digits = ( lo ) ( ceil( log10( n_processes + 1 ) ) + 1 );
      // bool print_process_ids = true;
      lo digits = 3;
      bool print_process_ids = true;

      if ( process_id == output_id ) {
        std::cout << "process ids:" << std::endl;
        time_structure.print_tree_human_readable( digits, true );
        std::cout << "global cluster ids: " << std::endl;
        time_structure.print_tree_human_readable( digits, false );
      }

      lou n_leaves = time_structure.get_leaves( ).size( );

      // reduce to locally essential tree
      
      time_structure.reduce_2_essential( process_id );

      if ( process_id == output_id ) {
        std::cout << "original number of leaves is " << n_leaves << std::endl;
        std::cout << "reducing to locally essential tree for process "
                  << process_id << std::endl;
        time_structure.print_tree_human_readable( digits, print_process_ids );
        // time_structure.print( );
        std::cout << "preparing and executing pseudo-fmm" << std::endl;
      }
      // let the processes start computations at the same time
      MPI_Barrier( communicator );

      // prepare the structures needed for the fmm
      std::list< scheduling_time_cluster* > m_list, m2l_list, l_list, n_list;
      std::vector< std::pair< scheduling_time_cluster*, lo > > receive_vector;
      lou n_moments_upward;
      lou n_moments_m2l;
      time_structure.prepare_fmm( m_list, m2l_list, l_list, n_list, 
                                  receive_vector, n_moments_upward, 
                                  n_moments_m2l );
      std::vector< sc > input_vector( n_leaves, 1.0 );
      std::vector< sc > output_vector( input_vector );
      // apply the distributed pseudo fmm
      apply_fmm( communicator, receive_vector, n_moments_upward, n_moments_m2l,
        m_list, m2l_list, l_list, n_list, input_vector, output_vector, verbose,
        verbose_dir );
      
      // collect the results using a reduce operation and output result.
      if ( process_id == output_id ) {
        sc reduced_output_vector[ output_vector.size( ) ];
        for ( lou i = 0; i < output_vector.size( ); ++i ) {
          reduced_output_vector[ i ] = 0.0;
        }
        MPI_Reduce( output_vector.data( ), reduced_output_vector, 
          output_vector.size( ), MPI_DOUBLE, MPI_MIN, output_id, communicator );
        std::cout << "output vector is: " << std::endl;
        for ( lou i = 0; i < output_vector.size( ); ++i ) {
          std::cout << reduced_output_vector[ i ] << std::endl;
        }
      } else {
        MPI_Reduce( output_vector.data( ), nullptr, 
          output_vector.size( ), MPI_DOUBLE, MPI_MIN, output_id, communicator );
      }
    }
    // the number of mpi processes has to match the number of processes for 
    // which the assignment to clusters is created!
    else if ( process_id == 0 ) {
      std::cout << "n_mpi_processes = " << n_mpi_processes 
                << ", should be " << n_processes << std::endl;
    }
  }
  MPI_Finalize( );
}
