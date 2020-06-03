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

using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;

void apply_fmm( const lo my_process_id,
  const std::vector< std::pair< scheduling_time_cluster*, lo > > &
    receive_vector, const lou n_moments_to_receive,
  std::list< scheduling_time_cluster* > & m_list,
  std::list< scheduling_time_cluster* > & m2l_list,
  std::list< scheduling_time_cluster* > & l_list,
  std::list< scheduling_time_cluster* > & n_list ) {
  
  // start the receive operations
  // TODO: currently this is just a dummy function.
  start_receive_operations( receive_vector, n_moments_to_receive );

  while ( !m_list.empty( ) || !m2l_list.empty( ) || !l_list.empty( ) 
          || !n_list.empty( ) ) {
    // TODO: check for received data and process it appropriately
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
        std::cout << "applying m-list operation" << std::endl;
        //################################
        // TODO: add m-list FMM operations
        //################################

        //update dependencies for interactions
        std::vector< scheduling_time_cluster* > * send_list
          = ( *it_current_cluster )->get_send_list( );
        if ( send_list != nullptr ) {
          for ( auto it = send_list->begin( ); it != send_list->end( ); ++it ) {
            if ( ( *it )->get_process_id( ) == my_process_id ) {
              ( *it )->add_to_ready_interaction_list( *it_current_cluster );
            }
            // TODO: else: start sending of moments for interaction
          }
        }
        //################################
        // TODO: add m-list FMM operations
        //################################
        
        // update dependencies in upward path
        if ( ( *it_current_cluster )->get_parent( )->get_process_id( ) 
              == my_process_id ) {
          ( *it_current_cluster )->get_parent( )->reduce_upward_path_counter( );
        }
        // TODO: else: start sending of moments by parent
        m_list.erase( it_current_cluster );
        break;
      }
      case 2: {
        std::cout << "applying l-list operation" << std::endl;
        //################################
        // TODO: add l-list FMM operations
        //################################
        if ( ( *it_current_cluster )->get_m2l_counter( ) 
              == ( *it_current_cluster )->get_interaction_list( )->size( ) ) {
          ( *it_current_cluster )->set_downward_path_status( 2 );
          //####################################################################
          // TODO: add downward path FMM operations and start sending if a child
          // is not handled by the same process
          //####################################################################
        } else {
          ( *it_current_cluster )->set_downward_path_status( 1 );
        }
        l_list.erase( it_current_cluster );
        break; 
      }
      case 3: {
        std::cout << "applying m2l-list operation" << std::endl;
        std::vector< scheduling_time_cluster* > * ready_interaction_list
          = ( *it_current_cluster )->get_ready_interaction_list( );
        for ( lou i = ( *it_current_cluster )->get_m2l_counter( ); 
              i < ready_interaction_list->size( ); ++i ) {
          //##################################
          // TODO: add m2l-list FMM operations
          //##################################
          ( *it_current_cluster )->increase_m2l_counter( );
        }
        if ( ( *it_current_cluster )->get_m2l_counter( ) ==
            ( *it_current_cluster )->get_interaction_list( )->size( ) ) {
          if ( ( *it_current_cluster )->get_downward_path_status( ) == 1 ) {
            ( *it_current_cluster )->set_downward_path_status( 2 );
            //##################################################################
            // TODO: add downward path FMM operations and start sending if a 
            // child is not handled by the same process
            //##################################################################
          }
          m2l_list.erase( it_current_cluster );
        }
        break;
      }
      case 4: {
        std::cout << "applying n-list operation" << std::endl;
        //################################
        // TODO: add n-list FMM operations
        //################################
        n_list.erase( it_current_cluster );
        break;
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

void start_receive_operations( 
  const std::vector< std::pair< scheduling_time_cluster*, lo > > 
  & receive_vector, const lou n_moments_to_receive ) {
  // currently, this function is just used to print a message to console for
  // every receive operation. Later on, this should be exchanged by appropriate
  // calls of MPI_IRecv
  std::cout << "receive operations for moments: " << std::endl;
  for ( lou i = 0; i < n_moments_to_receive; ++i ) {
    std::cout << "source: " << receive_vector[ i ].second
              << ", tag: " << receive_vector[ i ].first->get_global_index( )
              << std::endl;
  }
  std::cout << "receive operations for local contributions: " << std::endl;
  for ( lou i = n_moments_to_receive; i < receive_vector.size( ); ++i ) {
    std::cout << "source: " << receive_vector[ i ].second
              << ", tag: " << -receive_vector[ i ].first->get_global_index( )
              << std::endl;
  }
}