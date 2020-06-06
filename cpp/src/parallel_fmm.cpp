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

#include "besthea/parallel_fmm.h"

#include <iostream>
#include <chrono>         // std::chrono::seconds
#include <thread>         // std::this_thread::sleep_for

using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;

void apply_fmm( const MPI_Comm communicator,
  const std::vector< std::pair< scheduling_time_cluster*, lo > > &
    receive_vector, const lou n_moments_upward, const lou n_moments_m2l,
  std::list< scheduling_time_cluster* > & m_list,
  std::list< scheduling_time_cluster* > & m2l_list,
  std::list< scheduling_time_cluster* > & l_list,
  std::list< scheduling_time_cluster* > & n_list,
  const std::vector< sc > & input_vector, std::vector< sc > & output_vector ) {
  
  int my_process_id;
  MPI_Comm_rank( communicator, &my_process_id );
  
  // start the receive operations
  MPI_Request array_of_requests[ receive_vector.size( ) ];
  start_receive_operations( 
    receive_vector, n_moments_upward, n_moments_m2l, array_of_requests );
  
  // if ( my_process_id == 0 ) {
  //   std::cout << "reached this point" << std::endl;
  // }
  // MPI_Barrier( communicator );
  // if ( my_process_id != 0 ) {
  //   std::this_thread::sleep_for( std::chrono::milliseconds(500) );
  // }
  
  // initialize data which is used to check for received data.
  int outcount = 0;
  int array_of_indices[ receive_vector.size( ) ];
  for ( lou i = 0; i < receive_vector.size( ); ++i ) {
    array_of_indices[ i ] = 0;
  }

  while ( !m_list.empty( ) || !m2l_list.empty( ) || !l_list.empty( ) 
          || !n_list.empty( ) ) {
    if ( outcount != MPI_UNDEFINED ) {
      check_for_received_data( communicator, receive_vector, n_moments_upward, 
        n_moments_m2l, array_of_requests, array_of_indices, outcount );
    }
    // #################
    char status = 0;
    std::list< scheduling_time_cluster* >::iterator it_current_cluster;
    find_cluster_in_m_list( m_list, it_current_cluster, status );
    if ( status == 0 ) {
      // search in l list if no cluster was found yet
      find_cluster_in_l_list ( l_list, it_current_cluster, status );
      if ( status == 0 ) {
        // search in m2l list if no cluster was found yet
        find_cluster_in_m2l_list ( m2l_list, it_current_cluster, status );
        if ( status == 0 ) {
          // take the first cluster from the n-list if it is not empty
          if ( n_list.size( ) != 0 ) {
            it_current_cluster = n_list.begin( );
            status = 4;
          }
        }
      }
    }
    // start the appropriate fmm operations according to status
    // TODO: the switch routine could be avoided by using several else 
    // statements above. Switch is used for better clarity.
    switch ( status ) {
      case 1: {
        // apply S2M operations for leaf clusters
        call_s2m_operations( input_vector, *it_current_cluster );
        // update dependencies of targets in send list if they are local or send
        // data for interactions 
        provide_moments_for_m2l( communicator, *it_current_cluster );
        // apply M2M operations 
        call_m2m_operations( *it_current_cluster );       
        // update dependencies of parent if it is handled by the same process or
        // send data to other process if not
        provide_moments_to_parents( communicator, *it_current_cluster );
        // remove cluster from the list
        m_list.erase( it_current_cluster );
        break;
      }
      case 2: {
        // apply L2L operations
        call_l2l_operations( *it_current_cluster );
        if ( ( *it_current_cluster )->get_interaction_list( ) == nullptr ||
             ( *it_current_cluster )->get_m2l_counter( ) 
              == ( *it_current_cluster )->get_interaction_list( )->size( ) ) {
          // set status of parent's local contributions to completed
          ( *it_current_cluster )->set_downward_path_status( 2 );
          // apply L2T operations
          call_l2t_operations( *it_current_cluster, output_vector );
          // send data to all other processes which handle a child
          provide_local_contributions_to_children( 
            communicator, *it_current_cluster );
        } else {
          ( *it_current_cluster )->set_downward_path_status( 1 );
        }
        l_list.erase( it_current_cluster );
        break; 
      }
      case 3: {
        // std::cout << "applying m2l-list operation" << std::endl;
        std::vector< scheduling_time_cluster* > * ready_interaction_list
          = ( *it_current_cluster )->get_ready_interaction_list( );
        for ( lou i = ( *it_current_cluster )->get_m2l_counter( ); 
              i < ready_interaction_list->size( ); ++i ) {
          call_m2l_operations( ( 
            *ready_interaction_list )[ i ], *it_current_cluster );
          ( *it_current_cluster )->increase_m2l_counter( );
        }
        if ( ( *it_current_cluster )->get_m2l_counter( ) ==
            ( *it_current_cluster )->get_interaction_list( )->size( ) ) {
          if ( ( *it_current_cluster )->get_downward_path_status( ) == 1 ) {
            // set status of parent's local contributions to completed
            ( *it_current_cluster )->set_downward_path_status( 2 );
            // apply L2T operations
            call_l2t_operations( *it_current_cluster, output_vector );
            // send data to all other processes which handle a child
            provide_local_contributions_to_children( 
              communicator, *it_current_cluster );
          }
          m2l_list.erase( it_current_cluster );
        }
        break;
      }
      case 4: {
        // std::cout << "applying n-list operation" << std::endl;
        std::vector< scheduling_time_cluster* > * nearfield 
          = ( *it_current_cluster )->get_nearfield( );
        for ( auto it = nearfield->begin( ); it != nearfield->end( ); ++it ) {
          call_nearfield_operations(
            input_vector, *it, *it_current_cluster, output_vector );
        }
        n_list.erase( it_current_cluster );
        break;
      }
    }
  }
}

void call_s2m_operations( const std::vector< sc > & sources,
  scheduling_time_cluster* time_cluster ) {
  // execute only for leaf clusters
  if ( time_cluster->get_n_children( ) == 0 ) {
    // copy source value to moment
    lo cluster_index = time_cluster->get_leaf_index( );
    time_cluster->set_moment( sources[ cluster_index ] );
  }
}

void call_m2m_operations( scheduling_time_cluster* time_cluster ) {
  // add child moment to parent moment
  sc parent_moment = time_cluster->get_parent( )->get_moment( );
  time_cluster->get_parent( )->set_moment( 
    parent_moment + time_cluster->get_moment( ) );
}

void call_m2l_operations( scheduling_time_cluster* src_cluster,
  scheduling_time_cluster* tar_cluster ) {
  // add moment of src_cluster to local contribution of tar_cluster
  sc tar_local = tar_cluster->get_local_contribution( );
  tar_cluster->set_local_contribution( tar_local + src_cluster->get_moment( ) );
}

void call_l2l_operations( scheduling_time_cluster* time_cluster ) {
  // add local contribution of the parent of the time cluster to its own
  sc child_local = time_cluster->get_local_contribution( );
  time_cluster->set_local_contribution( 
    child_local + time_cluster->get_parent( )->get_local_contribution( ) );
}

void call_l2t_operations( scheduling_time_cluster* time_cluster, 
    std::vector< sc > & output_vector ) {
  // execute only for leaf clusters
  if ( time_cluster->get_n_children( ) == 0 ) {
    // subtract local contribution from correct position of output vector
    lo cluster_index = time_cluster->get_leaf_index( );
    output_vector[ cluster_index ] -= time_cluster->get_local_contribution( );
  }
}

void call_nearfield_operations( const std::vector< sc > & sources,
  scheduling_time_cluster* src_cluster, scheduling_time_cluster* tar_cluster, 
  std::vector< sc > & output_vector ) {
  lo tar_ind = tar_cluster->get_leaf_index( );
  lo src_ind = src_cluster->get_leaf_index( );
  output_vector[ tar_ind ] -= sources[ src_ind ];
}

void check_for_received_data( const MPI_Comm communicator,
  const std::vector< std::pair< besthea::mesh::scheduling_time_cluster*, lo > > 
    & receive_vector, const lou n_moments_upward, const lou n_moments_m2l, 
  MPI_Request * array_of_requests, int array_of_indices[ ], int & outcount ) {

  MPI_Testsome( receive_vector.size( ), array_of_requests, &outcount, 
    array_of_indices, MPI_STATUSES_IGNORE );
  if ( outcount != MPI_UNDEFINED && outcount > 0 ) {
    for ( lou i = 0; i < (lou) outcount; ++i ) {
      lou current_index = array_of_indices[ i ];
      scheduling_time_cluster* current_cluster 
        = receive_vector[ current_index ].first;
      // distinguish which data has been received
      if ( current_index < n_moments_upward ) {
        // received data are moments in the upward path. add up 
        // moments and update dependencies.
        lo source_id = receive_vector[ current_index ].second;
        sc old_moment = current_cluster->get_moment( );
        sc received_moment 
          = *( current_cluster->get_extraneous_moment_pointer( source_id ) );
        current_cluster->set_moment( old_moment + received_moment );
        current_cluster->reduce_upward_path_counter( );
      }
      else if ( current_index < n_moments_upward + n_moments_m2l ) {
        // received data are moments for m2l. update dependencies.
        int my_process_id;
        MPI_Comm_rank( communicator, &my_process_id );
        std::vector< scheduling_time_cluster* > * send_list 
          = current_cluster->get_send_list( );
        if ( send_list != nullptr ) {
          for ( auto it = send_list->begin( ); it != send_list->end( ); ++it ) {
            lo tar_process_id = ( *it )->get_process_id( );
            if ( tar_process_id == my_process_id ) {
              ( *it )->add_to_ready_interaction_list( current_cluster );
            }
          }
        }
      } else {
        // received data are local contributions. update dependencies.
        current_cluster->set_downward_path_status( 2 );
      }
    }
  }
}

void find_cluster_in_l_list( std::list< scheduling_time_cluster* > & l_list, 
  std::list< scheduling_time_cluster* >::iterator & it_next_cluster,
  char & status ) {
  it_next_cluster = l_list.begin( );
  // a cluster is ready if its parents downward path status equals 2
  while ( status != 2 && it_next_cluster != l_list.end( ) ) {
    if ( ( *it_next_cluster )->get_parent( )->get_downward_path_status( ) == 2 )
      status = 2;
    else
      ++ it_next_cluster;
  }
}

void find_cluster_in_m_list( std::list< scheduling_time_cluster* > & m_list, 
  std::list< scheduling_time_cluster* >::iterator & it_next_cluster,
  char & status ) {
  it_next_cluster = m_list.begin( );
  // a cluster is ready if its upward path counter equals 0
  while ( status != 1 && it_next_cluster != m_list.end( ) ) {
    if ( ( *it_next_cluster )->get_upward_path_counter( ) == 0 )
      status = 1;
    else
      ++ it_next_cluster;
  }
}

void find_cluster_in_m2l_list( std::list< scheduling_time_cluster* > & m2l_list,
  std::list< scheduling_time_cluster* >::iterator & it_next_cluster,
  char & status ) {
  it_next_cluster = m2l_list.begin( );
  // a cluster is ready if there is a non-completed interaction ready, i.e.
  // if the size of ready_interaction_list is greater than m2l_counter.
  while ( status != 3 && it_next_cluster != m2l_list.end( ) ) {
    std::vector< scheduling_time_cluster* > * ready_interaction_list
      = ( *it_next_cluster )->get_ready_interaction_list( );
    if ( ready_interaction_list != nullptr  
          && ( ready_interaction_list->size( ) 
                > ( *it_next_cluster )->get_m2l_counter( ) ) )
      status = 3;
    else
      ++ it_next_cluster;
  }
}

void provide_moments_for_m2l( MPI_Comm communicator, 
  scheduling_time_cluster* src_cluster ) {
  int my_process_id;
  MPI_Comm_rank( communicator, &my_process_id );
  std::vector< scheduling_time_cluster* > * send_list 
    = src_cluster->get_send_list( );
  if ( send_list != nullptr ) {
    for ( auto it = send_list->begin( ); it != send_list->end( ); ++it ) {
      lo tar_process_id = ( *it )->get_process_id( );
      if ( tar_process_id == my_process_id ) {
        ( *it )->add_to_ready_interaction_list( src_cluster );
      } else {
        lo tag = 2 * src_cluster->get_global_index( );
        sc* moment_buffer = src_cluster->get_moment_pointer( );
        MPI_Request req;
        MPI_Isend( moment_buffer, 1, MPI_DOUBLE, tar_process_id, tag, 
          communicator, &req );
        MPI_Request_free( &req ); 
      } 
    } 
  }
}

void provide_moments_to_parents( const MPI_Comm communicator, 
  scheduling_time_cluster* child_cluster ) {
  int my_process_id;
  MPI_Comm_rank( communicator, &my_process_id );
  scheduling_time_cluster* parent_cluster = child_cluster->get_parent( );
  lo parent_process_id = parent_cluster->get_process_id( );
  if ( parent_process_id == my_process_id ) {
    parent_cluster->reduce_upward_path_counter( );
  } 
  else if ( parent_process_id != -1 )  {
    lo tag = 2 * parent_cluster->get_global_index( );
    sc* moment_buffer = parent_cluster->get_moment_pointer( );
    MPI_Request req;
    MPI_Isend( moment_buffer, 1, MPI_DOUBLE, parent_process_id, tag, 
      communicator, &req );
    MPI_Request_free( &req );
  }
} 

void provide_local_contributions_to_children( const MPI_Comm communicator, 
  scheduling_time_cluster* parent_cluster ) {
  int my_process_id;
  MPI_Comm_rank( communicator, &my_process_id );
  std::vector< scheduling_time_cluster* > * children 
    = parent_cluster->get_children( );
  if ( children != nullptr ) {
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      lo child_process_id = ( *it )->get_process_id( );
      if ( child_process_id != my_process_id ) {
        lo tag = 2 * parent_cluster->get_global_index( ) + 1;
        sc* local_contribution_buffer 
          = parent_cluster->get_local_contribution_pointer( );
        MPI_Request req;
        MPI_Isend( local_contribution_buffer, 1, MPI_DOUBLE, child_process_id, 
          tag, communicator, &req );
        MPI_Request_free( &req );
      }
    }
  }
}

void start_receive_operations( 
  const std::vector< std::pair< scheduling_time_cluster*, lo > > 
  & receive_vector, const lou n_moments_upward, const lou n_moments_m2l,
  MPI_Request array_of_requests[ ] ) {
  // start the receive operations for the moments in the upward path
  int process_id;
  MPI_Comm_rank( MPI_COMM_WORLD, &process_id );

  // std::cout << "call receive operations for moments in upward path: " 
  //           << std::endl;
  for ( lou i = 0; i < n_moments_upward; ++i ) {
    lo source_id = receive_vector[ i ].second;
    lo tag = 2 * receive_vector[ i ].first->get_global_index( );
    sc* moment_buffer 
      = receive_vector[ i ].first->get_extraneous_moment_pointer( source_id );
    MPI_Irecv( moment_buffer, 1, MPI_DOUBLE, source_id, tag, MPI_COMM_WORLD, 
               &array_of_requests[ i ] );
    // std::cout << "source: " << source_id << ", tag: " << tag << std::endl;
  }
  
  // start the receive operations for the moments needed for m2l
  // std::cout << "call receive operations for moments needed for m2l: " 
  //           << std::endl;
  for ( lou i = n_moments_upward; i < n_moments_upward + n_moments_m2l; ++i ) {
    lo source_id = receive_vector[ i ].second;
    lo tag = 2 * receive_vector[ i ].first->get_global_index( );
    sc* moment_buffer = receive_vector[ i ].first->get_moment_pointer( );
    MPI_Irecv( moment_buffer, 1, MPI_DOUBLE, source_id, tag, MPI_COMM_WORLD, 
               &array_of_requests[ i ] );
    // std::cout << "source: " << source_id << ", tag: " << tag << std::endl;
  }

  // start the receive operations for the local contributions
  // std::cout << "receive operations for local contributions: " << std::endl;
  for ( lou i =  n_moments_upward + n_moments_m2l; i < receive_vector.size( ); 
        ++i ) {
    lo source_id = receive_vector[ i ].second;
    lo tag = 2 * receive_vector[ i ].first->get_global_index( ) + 1;
    sc* local_contribution_buffer 
      = receive_vector[ i ].first->get_local_contribution_pointer( );
    MPI_Irecv( local_contribution_buffer, 1, MPI_DOUBLE, source_id, tag, 
               MPI_COMM_WORLD, &array_of_requests[ i ] );

    // std::cout << "source: " << source_id << ", tag: " << tag << std::endl;
  }
}