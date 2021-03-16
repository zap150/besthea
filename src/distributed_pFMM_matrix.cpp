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

#include "besthea/distributed_pFMM_matrix.h"

#include "besthea/quadrature.h"
#include "besthea/timer.h"

#include <filesystem>
#include <mkl_rci.h>
#include <set>
#include <sstream>

using besthea::linear_algebra::full_matrix;
using besthea::mesh::distributed_spacetime_cluster_tree;
using besthea::mesh::general_spacetime_cluster;
using besthea::mesh::scheduling_time_cluster;

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply( const block_vector & /*x*/,
  block_vector & /*y*/, bool /*trans*/, sc /*alpha*/, sc /*beta*/ ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply( const block_vector & x,
  block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

//! template specialization for double layer p0p1 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::apply( const block_vector & x,
  block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

//! template specialization for double layer p1p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply( const block_vector & x,
  block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_sl_dl( const block_vector & x,
  block_vector & y, bool trans, sc alpha, sc beta ) const {
  // Specialization for the single and double layer operators

  //############################################################################
  //#### multiply the global result vector by beta ####
  // @todo discuss: should the multiplication be done like this or only in a
  // local part of the result vector?
#pragma omp parallel for schedule( static )
  for ( lo i = 0; i < y.get_block_size( ); ++i ) {
    for ( lo j = 0; j < y.get_size_of_block( ); ++j ) {
      y.set( i, j, y.get( i, j ) * beta );
    }
  }

  //############################################################################
  //#### setup phase ####
  // reset the contributions of all clusters to zero
  _scheduling_tree_structure->clear_moment_contributions(
    *_scheduling_tree_structure->get_root( ) );
  _scheduling_tree_structure->clear_local_contributions(
    *_scheduling_tree_structure->get_root( ) );

  // reset the dependency data of all the clusters in the 4 lists.
  reset_scheduling_clusters_dependency_data( );

  // @todo allocate buffers which are reused in computations to avoid
  // reallocation
  // std::vector< full_matrix > buffer_matrices;
  // buffer_matrices.resize( 8 );
  // for ( auto it = buffer_matrices.begin( ); it != buffer_matrices.end( );
  //       ++it ) {
  // ( *it ).resize( _temp_order + 1, _spat_contribution_size );
  // }
  // vector_type buffer_for_gaussians( ( _m2l_integration_order + 1 )
  //     * ( _m2l_integration_order + 1 ) * ( _temp_order + 1 )
  //     * ( _temp_order + 1 ), false );
  // vector_type buffer_for_coeffs( ( _spat_order + 1 ) * ( _spat_order + 1 )
  //     * ( _temp_order + 1 ) * ( _temp_order + 1 ), false );
  // full_matrix aux_buffer_0( ( _temp_order + 1 ) * ( _temp_order + 1 ),
  //   _spat_contribution_size, false );
  // full_matrix aux_buffer_1( ( _temp_order + 1 ) * ( _temp_order + 1 ),
  //   _spat_contribution_size, false );

  // initialize data which is used to check for received data.
  int outcount = 0;
  std::vector< int > array_of_indices( _receive_data_information.size( ) );
  for ( lou i = 0; i < _receive_data_information.size( ); ++i ) {
    array_of_indices[ i ] = 0;
  }
  // copy the 4 FMM lists to avoid reallocating them in each application
  std::list< mesh::scheduling_time_cluster * > m_list = _m_list;
  std::list< mesh::scheduling_time_cluster * > m2l_list = _m2l_list;
  std::list< mesh::scheduling_time_cluster * > l_list = _l_list;
  std::list< mesh::scheduling_time_cluster * > n_list = _n_list;

  // @todo: add appropriate verbose mode if desired
  bool verbose = false;
  // std::string verbose_file = verbose_dir + "/process_";
  std::string verbose_file = "verbose/process_";
  verbose_file += std::to_string( _my_rank );
  if ( verbose ) {
    std::filesystem::create_directory( "./verbose/" );
    // remove existing verbose file and write to new one
    remove( verbose_file.c_str( ) );
  }

  //############################################################################
  //#### distributed pFMM ####
  // allocate a global result vector. Only the entries corresponding to clusters
  // assigned to the current process are computed.
  block_vector y_pFMM( y.get_block_size( ), y.get_size_of_block( ), true );

  // auxiliary arrays for OpenMP dependencis (in OpenMP 4.5 must not be members)
  auto first = m2l_list.begin( );
  auto last = m2l_list.end( );
  char * aux_dep_m2l = new char[ std::distance( first, last ) ];
  std::fill( aux_dep_m2l, aux_dep_m2l + sizeof( aux_dep_m2l ), 0 );

  char * aux_dep_m2l_send = new char[ std::distance( first, last ) ];
  std::fill(
    aux_dep_m2l_send, aux_dep_m2l_send + sizeof( aux_dep_m2l_send ), 0 );

  first = l_list.begin( );
  last = l_list.end( );
  char * aux_dep_l = new char[ std::distance( first, last ) ];
  std::fill( aux_dep_l, aux_dep_l + sizeof( aux_dep_l ), 0 );

  first = m_list.begin( );
  last = m_list.end( );
  char * aux_dep_m = new char[ std::distance( first, last ) ];
  std::fill( aux_dep_m, aux_dep_m + sizeof( aux_dep_m ), 0 );

  // allocate buffers for m2l computation
  _aux_buffer_0.resize( omp_get_max_threads( ) );
  _aux_buffer_1.resize( omp_get_max_threads( ) );
  // start the main "job scheduling" algorithm
  // the "master" thread checks for new available data, spawns tasks, and
  // removes clusters from lists
#pragma omp parallel
  {
    _aux_buffer_0[ omp_get_thread_num( ) ].resize(
      ( _temp_order + 1 ) * ( _temp_order + 1 ), _spat_contribution_size );
    _aux_buffer_1[ omp_get_thread_num( ) ].resize(
      ( _temp_order + 1 ) * ( _temp_order + 1 ), _spat_contribution_size );

//    besthea::tools::timer t;
//    t.reset();
#pragma omp single
    {
      // start the receive operationss
      std::vector< MPI_Request > array_of_requests(
        _receive_data_information.size( ) );
      start_receive_operations( array_of_requests );
      while ( true ) {
        if ( m_list.empty( ) && m2l_list.empty( ) && l_list.empty( )
          && n_list.empty( ) ) {
          break;
        }
        // check if data has been received since the last iteration
        if ( outcount != MPI_UNDEFINED ) {
          check_for_received_data(
            array_of_requests, array_of_indices, outcount );
        }

        // we have to do this here to spawn tasks with correct dependencies
        if ( outcount != MPI_UNDEFINED && outcount > 0 ) {
          for ( lo i = 0; i < outcount; ++i ) {
            lou current_index = array_of_indices[ i ];
            scheduling_time_cluster * current_cluster
              = _receive_data_information[ current_index ].first;
            if ( verbose ) {
              std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
              if ( outfile.is_open( ) ) {
                outfile << "received data of cluster "
                        << current_cluster->get_global_index( )
                        << " from process "
                        << _receive_data_information[ current_index ].second
                        << std::endl;
                outfile.close( );
              }
            }
            // distinguish which data has been received
            if ( current_index < _n_moments_to_receive_upward ) {
              // received data are moments in the upward path. add up
              // moments and update dependencies.
              lo idx = current_cluster->get_pos_in_m_list( );

              // task depends on previously generated M-list tasks to prevent
              // collision in m2m operations
#pragma omp task depend( inout : aux_dep_m [idx:1] ) priority( 1000 )
              upward_path_task( current_index, current_cluster );
            } else if ( current_index
              < _n_moments_to_receive_upward + _n_moments_to_receive_m2l ) {
              // received data are moments for m2l. update dependencies.
              std::vector< scheduling_time_cluster * > * send_list
                = current_cluster->get_send_list( );
              if ( send_list != nullptr ) {
                for ( auto it = send_list->begin( ); it != send_list->end( );
                      ++it ) {
                  lo tar_process_id = ( *it )->get_process_id( );
                  if ( tar_process_id == _my_rank ) {
                    lo idx_receiver = ( *it )->get_pos_in_m2l_list( );
                    // task depends on previously generated m2l-list task to
                    // avoid collision when adding to ready interaction list
#pragma omp task depend( inout                                 \
                         : aux_dep_m2l_send [idx_receiver:1] ) \
  priority( 1000 )
                    ( *it )->update_ready_interaction_size( );
                  }
                }
              }
            } else {
              // received data are local contributions. update dependencies.
#pragma omp task priority( 1000 )
              current_cluster->set_downward_path_status( 2 );
            }
          }
        }

        // check if there is a cluster in one of the 4 lists whose operations
        // are ready to be executed.
        char status = 0;
        std::list< scheduling_time_cluster * >::iterator it_current_cluster;
        find_cluster_in_m_list( m_list, it_current_cluster, status );
        if ( status == 0 ) {
          // search in l list if no cluster was found yet
          find_cluster_in_l_list( l_list, it_current_cluster, status );
          if ( status == 0 ) {
            // search in m2l list if no cluster was found yet
            find_cluster_in_m2l_list( m2l_list, it_current_cluster, status );
            if ( status == 0 ) {
              // take the first cluster from the n-list if it is not empty
              if ( n_list.size( ) != 0 ) {
                it_current_cluster = n_list.begin( );
                status = 4;
              }
            } else {
              add_nn_operations( );
            }
          } else {
            add_nn_operations( );
          }
        } else {
          add_nn_operations( );
        }

        // if verbose mode is chosen, write info about next operation to file
        if ( verbose && status != 0 ) {
          std::stringstream outss;
          outss << "executing ";
          switch ( status ) {
            case 1:
              outss << "m-list operations ";
              break;
            case 2:
              outss << "l-list operations ";
              break;
            case 3:
              outss << "m2l-list operations ";
              break;
            case 4:
              outss << "n-list operations ";
              break;
          }
          outss << "for cluster "
                << ( *it_current_cluster )->get_global_index( );
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << outss.str( ) << std::endl;
            outfile.close( );
          }
        }

        scheduling_time_cluster * current_cluster = *it_current_cluster;

        // start the appropriate fmm operations according to status
        switch ( status ) {
          case 1: {
            // M-list task
            m_list.erase( it_current_cluster );

            lo idx_m_parent
              = current_cluster->get_parent( )->get_pos_in_m_list( );
            lo idx_m = current_cluster->get_pos_in_m_list( );

            std::vector< scheduling_time_cluster * > * send_list
              = current_cluster->get_send_list( );
            lo send_list_size = 0;
            if ( send_list != nullptr ) {
              send_list_size = send_list->size( );
            }
            if ( idx_m_parent != -1 ) {
              // m-list task depends on previously generated tasks
              // with the same parent (so the parent's m2m operations do not
              // collide)
              switch ( send_list_size ) {
                // m-list task depends on previously generated tasks with the
                // same cluster in the send-list (so the receiving does not
                // collide)
                case 0: {
#pragma omp task depend( inout                                               \
                         : aux_dep_m [idx_m_parent:1], aux_dep_m [idx_m:1] ) \
  priority( 500 )
                  m_list_task( x, current_cluster, verbose, verbose_file );
                  break;
                }
                case 1: {
                  lo idx_receiver_1
                    = send_list->at( 0 )->get_pos_in_m2l_list( );
#pragma omp task depend( inout                                               \
                         : aux_dep_m [idx_m_parent:1], aux_dep_m [idx_m:1] ) \
  depend( inout                                                              \
          : aux_dep_m2l_send [idx_receiver_1:1] ) priority( 500 )
                  m_list_task( x, current_cluster, verbose, verbose_file );
                  break;
                }
                case 2: {
                  lo idx_receiver_1
                    = send_list->at( 0 )->get_pos_in_m2l_list( );
                  lo idx_receiver_2
                    = send_list->at( 1 )->get_pos_in_m2l_list( );

#pragma omp task depend( inout                                               \
                         : aux_dep_m [idx_m_parent:1], aux_dep_m [idx_m:1] ) \
  depend( inout                                                              \
          : aux_dep_m2l_send [idx_receiver_1:1] )                            \
    depend( inout                                                            \
            : aux_dep_m2l_send[ idx_receiver_2 ] ) priority( 500 )
                  m_list_task( x, current_cluster, verbose, verbose_file );
                  break;
                }
              }
            } else {
              // parent is not in the m list, no dependency needed
              switch ( send_list->size( ) ) {
                // m-list task depends on previously generated tasks with the
                // same cluster in the send-list (so the receiver's m2l
                // operations do not collide)
                case 0: {
#pragma omp task depend( inout : aux_dep_m [idx_m:1] ) priority( 500 )
                  m_list_task( x, current_cluster, verbose, verbose_file );
                  break;
                }
                case 1: {
                  lo idx_receiver_1
                    = send_list->at( 0 )->get_pos_in_m2l_list( );
#pragma omp task depend( \
  inout                  \
  : aux_dep_m2l_send [idx_receiver_1:1], aux_dep_m [idx_m:1] ) priority( 500 )
                  m_list_task( x, current_cluster, verbose, verbose_file );
                  break;
                }
                case 2: {
                  lo idx_receiver_1
                    = send_list->at( 0 )->get_pos_in_m2l_list( );
                  lo idx_receiver_2
                    = send_list->at( 1 )->get_pos_in_m2l_list( );

#pragma omp task depend(                                       \
  inout                                                        \
  : aux_dep_m2l_send [idx_receiver_1:1], aux_dep_m [idx_m:1] ) \
  depend( inout                                                \
          : aux_dep_m2l_send [idx_receiver_2:1] ) priority( 500 )
                  m_list_task( x, current_cluster, verbose, verbose_file );
                  break;
                }
              }
            }
            break;
          }

          case 2: {
            // L-list task
            lo idx_l = current_cluster->get_pos_in_l_list( );
            l_list.erase( it_current_cluster );
            // l-list task depends on previously generated m2l tasks processing
            // the same cluster in the l-list (to prevent collision with m2l
            // tasks)
#pragma omp task depend( inout : aux_dep_l [idx_l:1] ) priority( 400 )
            l_list_task( y_pFMM, current_cluster, verbose, verbose_file );
            break;
          }
          case 3: {
            // M2l-list task

            lo idx_l = current_cluster->get_pos_in_l_list( );
            lou ready_int_list_size
              = current_cluster->get_ready_interaction_list_size( );
            if ( ready_int_list_size
              == current_cluster->get_interaction_list( )->size( ) ) {
              m2l_list.erase( it_current_cluster );
            }
            if ( idx_l == -1 ) {
              // cluster is not in the L-list, have to depend only on the
              // previously generated tasks processing the same cluster in the
              // m2l list
#pragma omp task priority( 300 )
              m2l_list_task( y_pFMM, current_cluster, verbose, verbose_file );
            } else {
              // cluster depends additionally on the previously generated task
              // with the same position in the L-list
#pragma omp task depend( inout : aux_dep_l [idx_l:1] ) priority( 300 )
              m2l_list_task( y_pFMM, current_cluster, verbose, verbose_file );
            }
            break;
          }
          case 4: {
            // nearfiel task
            n_list.erase( it_current_cluster );
            // no dependencies, possible collisions are treated by atomic
            // operations
#pragma omp task priority( 200 )
            apply_nearfield_operations(
              current_cluster, x, trans, y_pFMM, verbose, verbose_file );
            break;
          }
        }
        if ( verbose ) {
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << std::endl;
            outfile.close( );
          }
        }
      }
    }
  }

  //############################################################################
  //### communicate the result with an Allreduce operation for each timestep ###
  // @todo: Can we do this in a less cumbersome way?! Is a global reduction even
  // necessary?
  for ( lo block_idx = 0; block_idx < y_pFMM.get_block_size( ); ++block_idx ) {
    MPI_Allreduce( MPI_IN_PLACE, y_pFMM.get_block( block_idx ).data( ),
      y_pFMM.get_size_of_block( ), get_scalar_type< sc >::MPI_SC( ), MPI_SUM,
      *_comm );
  }
  // Scale the global update y_pFMM by alpha and add it to the global vector y.
  y.add( y_pFMM, alpha );
  // if ( _my_rank == 0 ) {
  //   std::cout << "application executed" << std::endl;
  // }

  delete[] aux_dep_m;
  delete[] aux_dep_l;
  delete[] aux_dep_m2l;
  delete[] aux_dep_m2l_send;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::set_trees( distributed_spacetime_cluster_tree *
    distributed_spacetime_tree ) {
  _distributed_spacetime_tree = distributed_spacetime_tree;
  _scheduling_tree_structure
    = _distributed_spacetime_tree->get_distribution_tree( );
  const std::vector< general_spacetime_cluster * > & local_leaves
    = distributed_spacetime_tree->get_local_leaves( );
  for ( auto leaf : local_leaves ) {
    _clusterwise_nearfield_matrices.insert(
      { leaf, std::vector< full_matrix * >( ) } );
    _clusterwise_nearfield_matrices[ leaf ].resize(
      leaf->get_nearfield_list( )->size( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::prepare_fmm( ) {
  _scheduling_tree_structure->init_fmm_lists(
    *_scheduling_tree_structure->get_root( ), _m_list, _m2l_list, _l_list,
    _n_list );
  // sort the m_list from bottom up, right to left
  _m_list.sort(
    _scheduling_tree_structure->compare_clusters_bottom_up_right_2_left );
  _m2l_list.sort(
    _scheduling_tree_structure->compare_clusters_top_down_right_2_left );
  _l_list.sort(
    _scheduling_tree_structure->compare_clusters_top_down_right_2_left );

  // allocate the timers
  for ( auto & it : _m_task_times ) {
    it.reserve( 2 * _m_list.size( ) );
  }
  for ( auto & it : _m2l_task_times ) {
    it.reserve( 2 * _m2l_list.size( ) );
  }
  for ( auto & it : _l_task_times ) {
    it.reserve( 2 * _l_list.size( ) );
  }
  for ( auto & it : _n_task_times ) {
    it.reserve( 2 * _n_list.size( ) );
  }
  for ( auto & it : _m_subtask_times ) {
    it.reserve( 4 * _m_list.size( ) );
  }
  for ( auto & it : _m2l_subtask_times ) {
    it.reserve( 4 * _m2l_list.size( ) );
  }
  for ( auto & it : _l_subtask_times ) {
    it.reserve( 4 * _l_list.size( ) );
  }
  for ( auto & it : _n_subtask_times ) {
    it.reserve( 4 * _n_list.size( ) );
  }
  for ( auto & it : _mpi_send_m2l ) {
    it.reserve( _m2l_list.size( ) );
  }
  for ( auto & it : _mpi_send_m_parent ) {
    it.reserve( _m_list.size( ) );
  }
  for ( auto & it : _mpi_send_l_children ) {
    it.reserve( _l_list.size( ) );
  }
  for ( auto & it : _mpi_recv_m2l ) {
    it.reserve( _m2l_list.size( ) );
  }
  for ( auto & it : _mpi_recv_m_parent ) {
    it.reserve( _m_list.size( ) );
  }
  for ( auto & it : _mpi_recv_l_children ) {
    it.reserve( _l_list.size( ) );
  }

  // set the positions of clusters in the lists
  lo counter = 0;
  for ( auto clst : _m_list ) {
    clst->set_pos_in_m_list( counter );
    counter++;
  }
  counter = 0;
  for ( auto clst : _l_list ) {
    clst->set_pos_in_l_list( counter );
    counter++;
  }
  counter = 0;
  for ( auto clst : _m2l_list ) {
    clst->set_pos_in_m2l_list( counter );
    counter++;
  }

  // fill the receive list by determining all incoming data.
  // check for receive operations in the upward path
  for ( auto it = _m_list.begin( ); it != _m_list.end( ); ++it ) {
    std::vector< scheduling_time_cluster * > * children
      = ( *it )->get_children( );
    if ( children != nullptr ) {
      for ( auto it_child = children->begin( ); it_child != children->end( );
            ++it_child ) {
        // if the current cluster has a child which is handled by a different
        // process p the current process has to receive the processed moments
        // from p.
        if ( ( *it_child )->get_process_id( ) != _my_rank ) {
          _receive_data_information.push_back(
            { *it, ( *it_child )->get_process_id( ) } );
          ( *it )->add_receive_buffer( ( *it_child )->get_process_id( ) );
        }
      }
    }
  }
  _n_moments_to_receive_upward = _receive_data_information.size( );
  // check for receive operations in the interaction phase
  for ( auto it = _m2l_list.begin( ); it != _m2l_list.end( ); ++it ) {
    std::vector< scheduling_time_cluster * > * interaction_list
      = ( *it )->get_interaction_list( );
    // interaction list is never empty for clusters in the m2l_list, so it is
    // not checked, whether the pointer is nullptr
    for ( auto it_src = interaction_list->begin( );
          it_src != interaction_list->end( ); ++it_src ) {
      // if the source cluster is handled by a different process p the current
      // process has to receive its moments from p.
      if ( ( *it_src )->get_process_id( ) != _my_rank ) {
        _receive_data_information.push_back(
          { *it_src, ( *it_src )->get_process_id( ) } );
      }
    }
  }
  // if two clusters have the same source cluster in their interaction list
  // its moments have to be received only once -> find and eliminate double
  // entries in the second part of the receive vector
  std::sort( _receive_data_information.begin( ) + _n_moments_to_receive_upward,
    _receive_data_information.end( ),
    [ & ]( const std::pair< scheduling_time_cluster *, lo > pair_one,
      const std::pair< scheduling_time_cluster *, lo > pair_two ) {
      return _scheduling_tree_structure->compare_clusters_top_down_right_2_left(
        pair_one.first, pair_two.first );
    } );
  auto new_end = std::unique(
    _receive_data_information.begin( ) + _n_moments_to_receive_upward,
    _receive_data_information.end( ),
    []( const std::pair< scheduling_time_cluster *, lo > pair_one,
      const std::pair< scheduling_time_cluster *, lo > pair_two ) {
      return pair_one.first == pair_two.first;
    } );
  _receive_data_information.resize(
    std::distance( _receive_data_information.begin( ), new_end ) );
  _n_moments_to_receive_m2l
    = _receive_data_information.size( ) - _n_moments_to_receive_upward;
  // check for receive operations in the downward path
  for ( auto it = _l_list.begin( ); it != _l_list.end( ); ++it ) {
    scheduling_time_cluster * parent = ( *it )->get_parent( );
    // if the parent cluster is handled by a different process p the current
    // process has to receive its local contributions from p.
    if ( parent->get_process_id( ) != _my_rank
      && parent->get_process_id( ) != -1 ) {
      _receive_data_information.push_back(
        { parent, parent->get_process_id( ) } );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
besthea::linear_algebra::full_matrix *
besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, target_space,
  source_space >::create_nearfield_matrix( lou leaf_index, lou source_index ) {
  general_spacetime_cluster * target_cluster
    = _distributed_spacetime_tree->get_local_leaves( )[ leaf_index ];
  general_spacetime_cluster * source_cluster
    = ( *( target_cluster->get_nearfield_list( ) ) )[ source_index ];
  lo n_dofs_source = source_cluster->get_n_dofs< source_space >( );
  lo n_dofs_target = target_cluster->get_n_dofs< target_space >( );
  full_matrix * local_matrix = new full_matrix( n_dofs_target, n_dofs_source );

  _clusterwise_nearfield_matrices[ target_cluster ][ source_index ]
    = local_matrix;

  return local_matrix;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_spatial_m2m_coeffs( ) {
  lo max_space_level
    = _distributed_spacetime_tree->get_local_max_space_level( );
  // Declare the structures containing coefficients of appropriate size.
  // NOTE: The M2M coefficients are computed for all levels except the last
  // one,
  //       even in case they are not needed for the first few levels.
  _m2m_coeffs_s_dim_0_left.resize( max_space_level );
  _m2m_coeffs_s_dim_0_right.resize( max_space_level );
  _m2m_coeffs_s_dim_1_left.resize( max_space_level );
  _m2m_coeffs_s_dim_1_right.resize( max_space_level );
  _m2m_coeffs_s_dim_2_left.resize( max_space_level );
  _m2m_coeffs_s_dim_2_right.resize( max_space_level );
  auto it1 = _m2m_coeffs_s_dim_0_left.begin( );
  auto it2 = _m2m_coeffs_s_dim_0_right.begin( );
  auto it3 = _m2m_coeffs_s_dim_1_left.begin( );
  auto it4 = _m2m_coeffs_s_dim_1_right.begin( );
  auto it5 = _m2m_coeffs_s_dim_2_left.begin( );
  auto it6 = _m2m_coeffs_s_dim_2_right.begin( );

  for ( ; it1 != _m2m_coeffs_s_dim_0_left.end( );
        ++it1, ++it2, ++it3, ++it4, ++it5, ++it6 ) {
    ( *it1 ).resize( ( _spat_order + 1 ) * ( _spat_order + 1 ) );
    ( *it2 ).resize( ( _spat_order + 1 ) * ( _spat_order + 1 ) );
    ( *it3 ).resize( ( _spat_order + 1 ) * ( _spat_order + 1 ) );
    ( *it4 ).resize( ( _spat_order + 1 ) * ( _spat_order + 1 ) );
    ( *it5 ).resize( ( _spat_order + 1 ) * ( _spat_order + 1 ) );
    ( *it6 ).resize( ( _spat_order + 1 ) * ( _spat_order + 1 ) );
  }

  std::vector< sc > paddings_refinementwise( max_space_level + 1, 0.0 );
  const std::vector< sc > & paddings_levelwise
    = _distributed_spacetime_tree->get_spatial_paddings( );
  // paddings_levelwise contains the padding levelwise with respect to the
  // clusters levels. we need the padding with respect to the number of
  // refinements in space.
  lo initial_space_refinement
    = _distributed_spacetime_tree->get_initial_space_refinement( );
  if ( initial_space_refinement > 0 ) {
    // padding is only computed starting from the spatial refinement level
    // initial_space_refinement. set it to this value for all lower levels
    for ( lo i = 0; i < initial_space_refinement; ++i ) {
      paddings_refinementwise[ i ] = paddings_levelwise[ 0 ];
    }
    // get the correct padding from paddings_levelwise (spatial refinement
    // every second step)
    lo current_idx = 0;
    for ( lo i = initial_space_refinement; i <= max_space_level; ++i ) {
      // note: by construction current_idx should never be out of bound for
      // paddings_levelwise
      paddings_refinementwise[ i ] = paddings_levelwise.at( current_idx );
      current_idx += 2;
    }
  } else {
    paddings_refinementwise[ 0 ] = paddings_levelwise[ 0 ];
    // the level of the first spatial refinement is known
    lo current_idx = _distributed_spacetime_tree->get_start_space_refinement( );
    for ( lo i = 1; i <= max_space_level; ++i ) {
      paddings_refinementwise[ i ] = paddings_levelwise.at( current_idx );
      current_idx += 2;
    }
  }

  // declare half box side lengths of parent and child cluster + initialize
  vector_type h_par_no_pad( 3, false ), h_child_no_pad( 3, false );
  sc dummy_val;
  _distributed_spacetime_tree->get_root( )->get_half_size(
    h_par_no_pad, dummy_val );

  vector_type nodes( _spat_order + 1, false );
  for ( lo i = 0; i <= _spat_order; ++i )
    nodes[ i ] = cos( ( M_PI * ( 2 * i + 1 ) ) / ( 2 * ( _spat_order + 1 ) ) );
  // evaluate Chebyshev polynomials at the nodes (needed for coefficients)
  vector_type all_values_cheb_std_intrvl(
    ( _spat_order + 1 ) * ( _spat_order + 1 ), false );
  _chebyshev.evaluate( nodes, all_values_cheb_std_intrvl );
  // vector to store values of Chebyshev polynomials for transformed intervals
  vector_type all_values_cheb_trf_intrvl(
    ( _spat_order + 1 ) * ( _spat_order + 1 ), false );
  // initialize vectors to store transformed nodes
  vector_type nodes_l_child_dim_0( _spat_order + 1, false );
  vector_type nodes_r_child_dim_0( _spat_order + 1, false );
  vector_type nodes_l_child_dim_1( _spat_order + 1, false );
  vector_type nodes_r_child_dim_1( _spat_order + 1, false );
  vector_type nodes_l_child_dim_2( _spat_order + 1, false );
  vector_type nodes_r_child_dim_2( _spat_order + 1, false );

  for ( lo curr_level = 0; curr_level < max_space_level; ++curr_level ) {
    h_child_no_pad[ 0 ] = h_par_no_pad[ 0 ] / 2.0;
    h_child_no_pad[ 1 ] = h_par_no_pad[ 1 ] / 2.0;
    h_child_no_pad[ 2 ] = h_par_no_pad[ 2 ] / 2.0;
    sc padding_par = paddings_refinementwise[ curr_level ];
    sc padding_child = paddings_refinementwise[ curr_level + 1 ];
    // transform the nodes from [-1, 1] to the child interval and then back to
    // [-1, 1] with the transformation of the parent interval:
    for ( lo j = 0; j <= _spat_order; ++j ) {
      nodes_l_child_dim_0[ j ] = 1.0 / ( h_par_no_pad[ 0 ] + padding_par )
        * ( -h_child_no_pad[ 0 ]
          + ( h_child_no_pad[ 0 ] + padding_child ) * nodes[ j ] );
      nodes_r_child_dim_0[ j ] = 1.0 / ( h_par_no_pad[ 0 ] + padding_par )
        * ( h_child_no_pad[ 0 ]
          + ( h_child_no_pad[ 0 ] + padding_child ) * nodes[ j ] );
      nodes_l_child_dim_1[ j ] = 1.0 / ( h_par_no_pad[ 1 ] + padding_par )
        * ( -h_child_no_pad[ 1 ]
          + ( h_child_no_pad[ 1 ] + padding_child ) * nodes[ j ] );
      nodes_r_child_dim_1[ j ] = 1.0 / ( h_par_no_pad[ 1 ] + padding_par )
        * ( h_child_no_pad[ 1 ]
          + ( h_child_no_pad[ 1 ] + padding_child ) * nodes[ j ] );
      nodes_l_child_dim_2[ j ] = 1.0 / ( h_par_no_pad[ 2 ] + padding_par )
        * ( -h_child_no_pad[ 2 ]
          + ( h_child_no_pad[ 2 ] + padding_child ) * nodes[ j ] );
      nodes_r_child_dim_2[ j ] = 1.0 / ( h_par_no_pad[ 2 ] + padding_par )
        * ( h_child_no_pad[ 2 ]
          + ( h_child_no_pad[ 2 ] + padding_child ) * nodes[ j ] );
    }
    // compute m2m coefficients at current level along all dimensions
    // for i1 > i0 the coefficients are known to be zero
    _chebyshev.evaluate( nodes_l_child_dim_0, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = 0; i1 <= i0; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i1 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i0 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1.0 );
        if ( i1 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_0_left[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    _chebyshev.evaluate( nodes_r_child_dim_0, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = 0; i1 <= i0; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i1 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i0 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i1 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_0_right[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    _chebyshev.evaluate( nodes_l_child_dim_1, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = 0; i1 <= i0; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i1 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i0 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i1 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_1_left[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    _chebyshev.evaluate( nodes_r_child_dim_1, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = 0; i1 <= i0; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i1 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i0 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i1 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_1_right[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    _chebyshev.evaluate( nodes_l_child_dim_2, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = 0; i1 <= i0; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i1 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i0 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i1 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_2_left[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    _chebyshev.evaluate( nodes_r_child_dim_2, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = 0; i1 <= i0; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i1 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i0 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i1 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_2_right[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }
    // update for next iteration
    h_par_no_pad[ 0 ] = h_child_no_pad[ 0 ];
    h_par_no_pad[ 1 ] = h_child_no_pad[ 1 ];
    h_par_no_pad[ 2 ] = h_child_no_pad[ 2 ];
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_chebyshev( ) {
  // initialize Chebyshev nodes for numerical integration
  vector_type cheb_nodes( _m2l_integration_order + 1,
    false );  //!< Chebyshev nodes for numerical integration
  vector_type all_poly_vals(
    ( _m2l_integration_order + 1 ) * ( _spat_order + 1 ),
    false );  //!< evaluation of Chebyshev polynomials

  for ( lo i = 0; i <= _m2l_integration_order; ++i ) {
    cheb_nodes[ i ] = std::cos(
      M_PI * ( 2 * i + 1 ) / ( 2 * ( _m2l_integration_order + 1 ) ) );
  }

  // evaluate Chebyshev polynomials for all degrees <= _spat_order for
  // integrals

  _chebyshev.evaluate( cheb_nodes, all_poly_vals );

  if ( _cheb_nodes_sum_coll.size( )
    != (lou) cheb_nodes.size( ) * cheb_nodes.size( ) ) {
    _cheb_nodes_sum_coll.resize( cheb_nodes.size( ) * cheb_nodes.size( ) );
  }
  lo counter = 0;

  for ( lo mu = 0; mu < cheb_nodes.size( ); ++mu ) {
    for ( lo nu = 0; nu < cheb_nodes.size( ); ++nu ) {
      _cheb_nodes_sum_coll[ counter ] = cheb_nodes[ mu ] - cheb_nodes[ nu ];
      ++counter;
    }
  }

  if ( _all_poly_vals_mult_coll.size( )
    != ( lou )( _spat_order + 1 ) * ( _spat_order + 1 ) * cheb_nodes.size( )
      * cheb_nodes.size( ) ) {
    _all_poly_vals_mult_coll.resize( ( _spat_order + 1 ) * ( _spat_order + 1 )
      * cheb_nodes.size( ) * cheb_nodes.size( ) );
  }

  counter = 0;

  for ( lo alpha = 0; alpha <= _spat_order; ++alpha ) {
    for ( lo beta = 0; beta <= _spat_order; ++beta ) {
      for ( lo mu = 0; mu < cheb_nodes.size( ); ++mu ) {
        for ( lo nu = 0; nu < cheb_nodes.size( ); ++nu ) {
          _all_poly_vals_mult_coll[ counter ]
            = all_poly_vals[ alpha * cheb_nodes.size( ) + mu ]
            * all_poly_vals[ beta * cheb_nodes.size( ) + nu ];
          ++counter;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
bool besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::mkl_fgmres_solve_parallel( const block_vector &
                                                             rhs,
  block_vector & solution, sc & relative_residual_error, lo & n_iterations,
  lo n_iterations_until_restart, bool trans, int root_id ) const {
  lo size = _dim_domain * _block_dim;

  if ( _dim_domain != _dim_range || size != rhs.size( )
    || size != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  if ( n_iterations_until_restart == 0 ) {
    n_iterations_until_restart = n_iterations;
  }

  lo rci;  // indicates current status in FGMRES
  // the following data is only relevant for process root_id, but declared for
  // all for simplicity
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp;
  vector_type * rhs_contiguous = nullptr;
  vector_type * solution_contiguous = nullptr;
  sc * tmp_data = nullptr;
  if ( _my_rank == root_id ) {
    // allocate a buffer for the mkl routines
    tmp.resize( ( 2 * n_iterations_until_restart + 1 ) * size
      + n_iterations_until_restart * ( n_iterations_until_restart + 9 ) / 2
      + 1 );
    tmp_data = tmp.data( );

    rhs_contiguous = new vector_type( size );
    solution_contiguous = new vector_type( size );
    rhs.copy_to_vector( *rhs_contiguous );
    solution.copy_to_vector( *solution_contiguous );

    dfgmres_init( &size, solution_contiguous->data( ), rhs_contiguous->data( ),
      &rci, ipar, dpar, tmp_data );
    if ( rci ) {
      std::cout << "Failed to initialize MKL FGMRES." << std::endl;
      delete rhs_contiguous;
      delete solution_contiguous;
      return false;
    }

    ipar[ 0 ] = size;          // size of the problem
    ipar[ 4 ] = n_iterations;  // maximum number of iterations
    ipar[ 7 ] = 1;             // perform the iteration stopping test
    ipar[ 8 ] = 1;             // do the residual stopping test
    ipar[ 9 ] = 0;             // do not request user stopping test
    ipar[ 10 ] = 0;            // non-preconditioned
    ipar[ 11 ] = 1;  // perform test for zero norm of generated direction
    ipar[ 14 ]
      = n_iterations_until_restart;  // number of iterations before restart

    dpar[ 0 ] = relative_residual_error;  // relative tolerance

    dfgmres_check( &size, solution_contiguous->data( ), rhs_contiguous->data( ),
      &rci, ipar, dpar, tmp_data );
    if ( rci ) {
      std::cout << "MKL parameters incorrect." << std::endl;
      delete rhs_contiguous;
      delete solution_contiguous;
      return false;
    }
  }

  block_vector tmp_1( _block_dim, _dim_domain );
  block_vector tmp_2( _block_dim, _dim_domain );

  while ( true ) {
    if ( _my_rank == root_id ) {
      dfgmres( &size, solution_contiguous->data( ), rhs_contiguous->data( ),
        &rci, ipar, dpar, tmp_data );
    }
    // broadcast rci to decide whether to apply the operator or not
    MPI_Bcast( &rci, 1, get_index_type< lo >::MPI_LO( ), root_id, *_comm );
    if ( rci == 1 ) {  // apply operator
      if ( _my_rank == root_id ) {
        tmp_1.copy_from_raw(
          _block_dim, _dim_domain, tmp_data + ipar[ 21 ] - 1 );
      }
      // broadcast block vector tmp_1 to all processes.
      for ( lo i = 0; i < _block_dim; ++i ) {
        MPI_Bcast( tmp_1.get_block( i ).data( ), _dim_domain,
          get_scalar_type< sc >::MPI_SC( ), root_id, *_comm );
      }
      apply( tmp_1, tmp_2, trans, 1.0, 0.0 );
      if ( _my_rank == root_id ) {
        tmp_2.copy_to_raw( tmp_data + ipar[ 22 ] - 1 );
      }
      continue;
    } else if ( rci == 0 ) {  // success, no further applications needed
      if ( _my_rank == root_id ) {
        dfgmres_get( &size, solution_contiguous->data( ),
          rhs_contiguous->data( ), &rci, ipar, dpar, tmp_data, &iter );
        solution.copy_from_vector(
          _block_dim, _dim_domain, *solution_contiguous );
        n_iterations = iter;
        relative_residual_error = dpar[ 4 ] / dpar[ 2 ];
      }
      // broadcast solution, n_iterations and relative_residual_error to all
      // processes (for the sake of completeness)
      for ( lo i = 0; i < _block_dim; ++i ) {
        MPI_Bcast( solution.get_block( i ).data( ), _dim_domain,
          get_scalar_type< sc >::MPI_SC( ), root_id, *_comm );
      }
      MPI_Bcast( &relative_residual_error, 1, get_scalar_type< sc >::MPI_SC( ),
        root_id, *_comm );
      MPI_Bcast(
        &n_iterations, 1, get_index_type< lo >::MPI_LO( ), root_id, *_comm );
      break;
    } else {
      std::cout << "Only RCI codes 0,1 supported." << std::endl;
      if ( _my_rank == root_id ) {
        delete rhs_contiguous;
        delete solution_contiguous;
      }
      return false;
    }
  }
  if ( _my_rank == root_id ) {
    delete rhs_contiguous;
    delete solution_contiguous;
  }
  return true;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::print_information( const int root_process ) {
  // first print some information of the underlying distributed space time
  // tree
  _distributed_spacetime_tree->print_information( root_process );

  // print rough nearfield percentage
  // compute the nearfield ratios (two versions) on each process.
  sc local_nearfield_ratio = compute_nearfield_ratio( );
  sc local_nonzero_nearfield_ratio = compute_nonzero_nearfield_ratio( );
  sc global_nearfield_ratio( 0 ), global_nonzero_nearfield_ratio( 0 );
  // gather the nearfield ratios at the root process
  int n_processes;
  MPI_Comm_size( *_comm, &n_processes );
  sc * all_local_nonzero_nearfield_ratios = nullptr;
  sc * all_local_nearfield_ratios = nullptr;
  if ( _my_rank == root_process ) {
    all_local_nonzero_nearfield_ratios = new sc[ n_processes ];
    all_local_nearfield_ratios = new sc[ n_processes ];
  }

  MPI_Gather( &local_nearfield_ratio, 1, get_scalar_type< sc >::MPI_SC( ),
    all_local_nearfield_ratios, 1, get_scalar_type< sc >::MPI_SC( ),
    root_process, *_comm );
  MPI_Gather( &local_nonzero_nearfield_ratio, 1,
    get_scalar_type< sc >::MPI_SC( ), all_local_nonzero_nearfield_ratios, 1,
    get_scalar_type< sc >::MPI_SC( ), root_process, *_comm );
  if ( _my_rank == root_process ) {
    for ( lo i = 0; i < n_processes; ++i ) {
      global_nearfield_ratio += all_local_nearfield_ratios[ i ];
      global_nonzero_nearfield_ratio += all_local_nonzero_nearfield_ratios[ i ];
    }
    std::cout << "nearfield ratio (including zeros) = "
              << global_nearfield_ratio << std::endl;
    std::cout << "nearfield ratio (counting non-zero entries only) = "
              << global_nonzero_nearfield_ratio << std::endl;
  }
  // count the fmm operations levelwise
  std::vector< lou > n_s2m_operations, n_m2m_operations, n_m2l_operations,
    n_l2l_operations, n_l2t_operations;
  count_fmm_operations_levelwise( n_s2m_operations, n_m2m_operations,
    n_m2l_operations, n_l2l_operations, n_l2t_operations );
  // collect the numbers of operations at the root process via reduce
  // operations
  lo n_max_levels = _distributed_spacetime_tree->get_max_levels( );
  if ( _my_rank == root_process ) {
    MPI_Reduce( MPI_IN_PLACE, n_s2m_operations.data( ), n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_m2m_operations.data( ), n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_m2l_operations.data( ), n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_l2l_operations.data( ), n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_l2t_operations.data( ), n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
  } else {
    MPI_Reduce( n_s2m_operations.data( ), nullptr, n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_m2m_operations.data( ), nullptr, n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_m2l_operations.data( ), nullptr, n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_l2l_operations.data( ), nullptr, n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_l2t_operations.data( ), nullptr, n_max_levels,
      get_index_type< lou >::MPI_LO( ), MPI_SUM, root_process, *_comm );
  }
  // count the number of allocated moments (own and received) and
  // local contributions
  lo local_n_moments( 0 ), local_n_moments_receive( 0 ),
    local_n_local_contributions( 0 );
  _scheduling_tree_structure->count_number_of_contributions(
    _scheduling_tree_structure->get_root( ), local_n_moments,
    local_n_moments_receive, local_n_local_contributions );
  lo * all_n_moments = nullptr;
  lo * all_n_moments_receive = nullptr;
  lo * all_n_local_contributions = nullptr;
  if ( _my_rank == root_process ) {
    all_n_moments = new lo[ n_processes ];
    all_n_moments_receive = new lo[ n_processes ];
    all_n_local_contributions = new lo[ n_processes ];
  }
  // gather the computed numbers at the root process
  MPI_Gather( &local_n_moments, 1, get_index_type< lo >::MPI_LO( ),
    all_n_moments, 1, get_index_type< lo >::MPI_LO( ), root_process, *_comm );
  MPI_Gather( &local_n_moments_receive, 1, get_index_type< lo >::MPI_LO( ),
    all_n_moments_receive, 1, get_index_type< lo >::MPI_LO( ), root_process,
    *_comm );
  MPI_Gather( &local_n_local_contributions, 1, get_index_type< lo >::MPI_LO( ),
    all_n_local_contributions, 1, get_index_type< lo >::MPI_LO( ), root_process,
    *_comm );
  if ( _my_rank == root_process ) {
    lo start_space_refinement
      = _distributed_spacetime_tree->get_start_space_refinement( );
    std::cout << "number of s2m operations: " << std::endl;
    for ( lo i = 2; i < n_max_levels; ++i ) {
      std::cout << "level " << i << ": " << n_s2m_operations[ i ] << std::endl;
    }
    std::cout << "number of m2m operations: " << std::endl;
    for ( lo i = 2; i < n_max_levels; ++i ) {
      std::cout << "level " << i << ": " << n_m2m_operations[ i ];
      if ( i >= start_space_refinement
        && ( ( i - start_space_refinement ) ) % 2 == 0 ) {
        std::cout << " spacetime m2m" << std::endl;
      } else {
        std::cout << " temporal m2m" << std::endl;
      }
    }
    std::cout << "number of m2l operations: " << std::endl;
    for ( lo i = 2; i < n_max_levels; ++i ) {
      std::cout << "level " << i << ": " << n_m2l_operations[ i ] << std::endl;
    }
    std::cout << "number of l2l operations: " << std::endl;
    for ( lo i = 2; i < n_max_levels; ++i ) {
      std::cout << "level " << i << ": " << n_l2l_operations[ i ];
      if ( i >= start_space_refinement
        && ( ( i - start_space_refinement ) ) % 2 == 0 ) {
        std::cout << " spacetime l2l" << std::endl;
      } else {
        std::cout << " temporal l2l" << std::endl;
      }
    }
    std::cout << "number of l2t operations: " << std::endl;
    for ( lo i = 2; i < n_max_levels; ++i ) {
      std::cout << "level " << i << ": " << n_l2t_operations[ i ] << std::endl;
    }
    std::cout << "#############################################################"
              << "###########################" << std::endl;
    std::cout << "rough memory estimates per process: " << std::endl;
    lo n_global_elements
      = _distributed_spacetime_tree->get_mesh( ).get_n_elements( );
    sc total_storage_nearfield = 0.0;
    sc total_storage_contributions = 0.0;
    for ( int i = 0; i < n_processes; ++i ) {
      sc local_storage_nearfield = n_global_elements * n_global_elements
        * all_local_nearfield_ratios[ i ];
      local_storage_nearfield
        *= 8. / 1024. / 1024. / 1024.;  // get memory for double entries in GiB.
      sc local_storage_contributions
        = ( all_n_moments[ i ] + all_n_moments_receive[ i ]
            + all_n_local_contributions[ i ] )
        * 8. * _contribution_size / 1024. / 1024. / 1024.;
      total_storage_nearfield += local_storage_nearfield;
      total_storage_contributions += local_storage_contributions;
      std::cout << "process " << i
                << ": nearfield_matrices: " << local_storage_nearfield
                << " GiB, moment and local contributions: "
                << local_storage_contributions << " GiB." << std::endl;
    }
    std::cout << "total storage: nearfield matrices: "
              << total_storage_nearfield
              << " GiB, moment and local contributions: "
              << total_storage_contributions << " GiB." << std::endl;
    std::cout << "storage per allocated vector: "
              << n_global_elements * 8. / 1024. / 1024. / 1024. << " GiB."
              << std::endl;
    delete[] all_local_nearfield_ratios;
    delete[] all_local_nonzero_nearfield_ratios;
    delete[] all_n_moments;
    delete[] all_n_moments_receive;
    delete[] all_n_local_contributions;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::call_s2m_operations( const block_vector &
                                                       sources,
  besthea::mesh::scheduling_time_cluster * time_cluster, bool verbose,
  const std::string & verbose_file ) const {
  // execute only for associated spacetime leaves
  if ( time_cluster->get_n_associated_leaves( ) > 0 ) {
    if ( verbose ) {
      std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "call S2M for cluster " << time_cluster->get_global_index( )
                << std::endl;
        outfile.close( );
      }
    }
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = time_cluster->get_associated_spacetime_clusters( );
    // there is an implicit taskgroup after this taskloop
#pragma omp taskloop shared( sources )
    for ( lou i = 0; i < time_cluster->get_n_associated_leaves( ); ++i ) {
      general_spacetime_cluster * current_cluster
        = ( *associated_spacetime_clusters )[ i ];

      apply_s2m_operation( sources, current_cluster );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_s2m_operation( const block_vector &
  /*source_vector*/,
  general_spacetime_cluster * /*source_cluster*/ ) const {
  std::cout << "S2M operation not implemented " << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply_s2m_operation( const block_vector &
                                                           source_vector,
  general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p0( source_vector, source_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::apply_s2m_operation( const block_vector &
                                                           source_vector,
  general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operations_p1_normal_drv( source_vector, source_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply_s2m_operation( const block_vector &
                                                           source_vector,
  general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p0( source_vector, source_cluster );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_s2m_operation_p0( const block_vector &
                                                          source_vector,
  general_spacetime_cluster * source_cluster ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  lo n_space_elements = source_cluster->get_n_space_elements( );
  full_matrix sources( n_time_elements, n_space_elements, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references of current moment and all required matrices
  sc * moment = source_cluster->get_pointer_to_moment( );

  full_matrix T;
  compute_chebyshev_quadrature_p0( source_cluster, T );
  full_matrix L;
  compute_lagrange_quadrature( source_cluster, L );

  // get the relevant entries of the block vector x and store them in sources
  const std::vector< lo > & spacetime_elements
    = source_cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = source_cluster->get_mesh( );
  // a cluster for which an S2M operation is executed is always local!
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );
  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component to
    // determine the local time index only once
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      lo global_space_index = local_mesh->get_space_element(
        distributed_mesh->global_2_local( local_start_idx,
          spacetime_elements[ i_time * n_space_elements + i_space ] ) );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      sources( i_time, i_space )
        = source_vector.get( distributed_mesh->local_2_global_time(
                               local_start_idx, local_time_index ),
          global_space_index );
    }
  }
  // compute D = Q * T and then the moment mu = L * D
  aux_matrix.multiply( sources, T );
  // mu = L * D with explicit cblas routine call
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_cols_aux_matrix = aux_matrix.get_n_columns( );
  lo n_rows_aux_matrix = aux_matrix.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_aux_matrix;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange,
    n_cols_aux_matrix, n_rows_aux_matrix, alpha, L.data( ), lda,
    aux_matrix.data( ), ldb, beta, moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2m_operations_p1_normal_drv( const block_vector &
                                                        source_vector,
  general_spacetime_cluster * source_cluster ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  lo n_space_elements = source_cluster->get_n_space_elements( );
  lo n_space_nodes = source_cluster->get_n_space_nodes( );
  full_matrix sources( n_time_elements, n_space_nodes, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references of current moment and all required matrices
  sc * moment = source_cluster->get_pointer_to_moment( );
  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1( source_cluster, T_drv );
  full_matrix L;
  compute_lagrange_quadrature( source_cluster, L );

  // get the relevant entries of the block vector x and store them in sources
  const std::vector< lo > & spacetime_elements
    = source_cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = source_cluster->get_mesh( );
  // a cluster for which an S2M operation is executed is always local!
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  const std::vector< lo > & local_2_global_nodes
    = source_cluster->get_local_2_global_nodes( );
  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component to
    // determine the local time index only once
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      // local_2_global_nodes gives the indices of the spacetime nodes. take
      // the rest from division by the number of global spatial nodes to get
      // the spatial node index
      lo global_space_index
        = local_2_global_nodes[ i_space ] % local_mesh->get_n_spatial_nodes( );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      sources( i_time, i_space )
        = source_vector.get( distributed_mesh->local_2_global_time(
                               local_start_idx, local_time_index ),
          global_space_index );
    }
  }
  // compute D = Q * T_drv and then the moment mu = L * D
  aux_matrix.multiply( sources, T_drv );
  // mu = L * D with explicit cblas routine call
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_cols_aux_matrix = aux_matrix.get_n_columns( );
  lo n_rows_aux_matrix = aux_matrix.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_aux_matrix;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange,
    n_cols_aux_matrix, n_rows_aux_matrix, alpha, L.data( ), lda,
    aux_matrix.data( ), ldb, beta, moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::call_m2m_operations( scheduling_time_cluster *
                                                       time_cluster,
  bool verbose, const std::string & verbose_file ) const {
  scheduling_time_cluster * parent_cluster = time_cluster->get_parent( );
  // m2m operations are only executed if the parent is active in the upward
  // path
  if ( parent_cluster->is_active_in_upward_path( ) ) {
    if ( verbose ) {
#pragma omp critical( verbose )
      {
        std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "call M2M for cluster "
                  << time_cluster->get_global_index( ) << " at level "
                  << time_cluster->get_level( ) << std::endl;
          outfile.close( );
        }
      }
    }
    slou configuration = time_cluster->get_configuration( );
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = parent_cluster->get_associated_spacetime_clusters( );
    lou n_associated_leaves = parent_cluster->get_n_associated_leaves( );

    // call the m2m operations for all non-leaf spacetime clusters which are
    // associated with the parent scheduling time cluster

    // there is an implicit taskgroup after this taskloop
#pragma omp taskloop
    for ( lou i = n_associated_leaves;
          i < associated_spacetime_clusters->size( ); ++i ) {
      if ( _measure_tasks ) {
        _m_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      general_spacetime_cluster * current_cluster
        = ( *associated_spacetime_clusters )[ i ];

      apply_grouped_m2m_operation( current_cluster, configuration );
      if ( _measure_tasks ) {
        _m_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_grouped_m2m_operation( mesh::general_spacetime_cluster *
                                                 parent_cluster,
  slou child_configuration ) const {
  sc * parent_moment = parent_cluster->get_pointer_to_moment( );

  std::vector< general_spacetime_cluster * > * children
    = parent_cluster->get_children( );
  // ###############################################
  // ### compute the matrix for the temporal m2m ###
  full_matrix temporal_m2m_matrix( _temp_order + 1, _temp_order + 1 );
  lo child_idx = 0;
  // find a child with the right temporal configuration. Note: such a child
  // exists, otherwise the routine is not executed
  while ( ( *children )[ child_idx ]->get_temporal_configuration( )
    != child_configuration ) {
    ++child_idx;
  }
  // get the temporal data of the parent and child cluster.
  sc parent_time_center = parent_cluster->get_time_center( );
  sc parent_time_half_size = parent_cluster->get_time_half_size( );
  sc child_time_center = ( *children )[ child_idx ]->get_time_center( );
  sc child_time_half_size = ( *children )[ child_idx ]->get_time_half_size( );

  const vector_type & nodes = _lagrange.get_nodes( );
  vector_type nodes_child( _temp_order + 1, false );
  vector_type values_lagrange( _temp_order + 1, false );
  // transform the nodes from [-1, 1] to the child interval and then back to
  // [-1, 1] with the transformation of the parent interval:
  for ( lo j = 0; j <= _temp_order; ++j ) {
    nodes_child[ j ] = ( child_time_center + (child_time_half_size) *nodes[ j ]
                         - parent_time_center )
      / parent_time_half_size;
  }
  // compute entries of the m2m matrix (by evaluating lagrange polynomials)
  for ( lo j = 0; j <= _temp_order; ++j ) {
    _lagrange.evaluate( j, nodes_child, values_lagrange );
    for ( lo k = 0; k <= _temp_order; ++k )
      temporal_m2m_matrix.set( j, k, values_lagrange[ k ] );
  }
  // ###############################################
  // determine whether a space-time m2m operation is necessary or only a
  // temporal m2m operation
  lo n_space_div_parent, n_space_div_children, dummy;
  parent_cluster->get_n_divs( n_space_div_parent, dummy );
  ( *children )[ child_idx ]->get_n_divs( n_space_div_children, dummy );
  bool temporal_only = ( n_space_div_parent == n_space_div_children );
  if ( temporal_only ) {
    // execute only temporal m2m operation for children with correct
    // configuration
    for ( auto child : *children ) {
      if ( child->get_temporal_configuration( ) == child_configuration ) {
        const sc * child_moment = child->get_pointer_to_moment( );
        apply_temporal_m2m_operation(
          child_moment, temporal_m2m_matrix, parent_moment );
      }
    }
  } else {
    // apply the spatial m2m operation to all child moments and store the
    // results in a buffer
    // @todo use buffer as input argument to avoid reallocation.
    std::vector< sc > buffer_array(
      ( _temp_order + 1 ) * _spat_contribution_size );
    for ( int i = 0; i < ( _temp_order + 1 ) * _spat_contribution_size; ++i ) {
      buffer_array[ i ] = 0.0;
    }
    for ( auto child : *children ) {
      short child_octant, current_configuration;
      child->get_position( child_octant, current_configuration );
      if ( current_configuration == child_configuration ) {
        const sc * child_moment = child->get_pointer_to_moment( );
        apply_spatial_m2m_operation(
          child_moment, n_space_div_parent, child_octant, buffer_array );
      }
    }
    // apply the temporal m2m operation to the buffer and add the result to
    // the parent moment
    apply_temporal_m2m_operation(
      buffer_array.data( ), temporal_m2m_matrix, parent_moment );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_temporal_m2m_operation( const sc *
                                                                child_moment,
  const full_matrix & temporal_m2m_matrix, sc * parent_moment ) const {
  // call the appropriate cblas routine for matrix matrix multiplication.
  lo n_rows_m2m_matrix = temporal_m2m_matrix.get_n_rows( );
  lo n_cols_moment = _spat_contribution_size;
  lo n_cols_m2m_matrix = n_rows_m2m_matrix;
  lo lda = n_rows_m2m_matrix;
  lo ldb = n_cols_m2m_matrix;
  sc alpha = 1.0;
  sc beta = 1.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_m2m_matrix,
    n_cols_moment, n_cols_m2m_matrix, alpha, temporal_m2m_matrix.data( ), lda,
    child_moment, ldb, beta, parent_moment, n_rows_m2m_matrix );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_spatial_m2m_operation( const sc *
                                                               child_moment,
  const lo n_space_div_parent, const slou octant,
  std::vector< sc > & output_array ) const {
  const vector_type * m2m_coeffs_s_dim_0;
  const vector_type * m2m_coeffs_s_dim_1;
  const vector_type * m2m_coeffs_s_dim_2;
  switch ( octant ) {
    case 0:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 1:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 2:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 3:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 4:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 5:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 6:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 7:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    default:  // default case should never be used, program will crash!
      m2m_coeffs_s_dim_0 = nullptr;
      m2m_coeffs_s_dim_1 = nullptr;
      m2m_coeffs_s_dim_2 = nullptr;
  }

  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  // initialize auxiliary matrices lambda_1/2 for intermediate results with 0
  // @todo use buffers to avoid allocations
  full_matrix lambda_1( _temp_order + 1, n_coeffs_s, true );
  full_matrix lambda_2( _temp_order + 1, n_coeffs_s, true );

  for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
    lo child_index = 0;
    for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
      for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0; ++alpha1 ) {
        lo alpha2;
        for ( alpha2 = 0; alpha2 <= beta2; ++alpha2 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            lambda_1( b,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 )
              += ( *m2m_coeffs_s_dim_2 )[ beta2 * ( _spat_order + 1 ) + alpha2 ]
              * child_moment[ b + child_index * ( _temp_order + 1 ) ];
          }
          ++child_index;
        }
        // correction needed for skipped entries of child_moment
        child_index += _spat_order + 1 - alpha0 - alpha1 - alpha2;
      }
      // correction for current index; necessary since alpha1 does not run
      // until _spat_order - alpha0 as it does in stored child_moment
      child_index += ( ( beta2 + 1 ) * beta2 ) / 2;
    }
  }

  // compute intermediate result lambda_1 ignoring zero entries for the sake
  // of better readability
  for ( lo beta1 = 0; beta1 <= _spat_order; ++beta1 ) {
    for ( lo beta2 = 0; beta2 <= _spat_order - beta1; ++beta2 ) {
      for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
        for ( lo alpha1 = 0; alpha1 <= beta1; ++alpha1 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            lambda_2( b,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                + ( _spat_order + 1 ) * beta2 + alpha0 )
              += ( *m2m_coeffs_s_dim_1 )[ beta1 * ( _spat_order + 1 ) + alpha1 ]
              * lambda_1( b,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                  + ( _spat_order + 1 ) * alpha0 + alpha1 );
          }
        }
      }
    }
  }

  lo output_array_index = 0;
  for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
    for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
        for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            output_array[ b + output_array_index * ( _temp_order + 1 ) ]
              += ( *m2m_coeffs_s_dim_0 )[ beta0 * ( _spat_order + 1 ) + alpha0 ]
              * lambda_2( b,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                  + ( _spat_order + 1 ) * beta2 + alpha0 );
          }
        }
        ++output_array_index;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::call_m2l_operations( scheduling_time_cluster *
                                                       src_cluster,
  scheduling_time_cluster * tar_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( verbose ) {
#pragma omp critical( verbose )
    {
      std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "call M2L for source " << src_cluster->get_global_index( )
                << " and target " << tar_cluster->get_global_index( )
                << " at level " << src_cluster->get_level( ) << std::endl;
        outfile.close( );
      }
    }
  }

  // execute an m2l operation for each spacetime cluster associated with
  // tar_cluster and each source in its interaction list, whose temporal
  // component is src_cluster (global_time_index coincides with global index
  // of src_cluster)

  std::vector< general_spacetime_cluster * > * associated_spacetime_targets
    = tar_cluster->get_associated_spacetime_clusters( );
  // there is an implicit taskgroup after this taskloop
#pragma omp taskloop
  for ( lou i = 0; i < associated_spacetime_targets->size( ); ++i ) {
    if ( _measure_tasks ) {
      _m2l_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
    //      for ( auto spacetime_tar : *associated_spacetime_targets ) {
    std::vector< general_spacetime_cluster * > * spacetime_interaction_list
      = ( *associated_spacetime_targets )[ i ]->get_interaction_list( );

    for ( auto spacetime_src : *spacetime_interaction_list ) {
      if ( spacetime_src->get_global_time_index( )
        == src_cluster->get_global_index( ) ) {
        apply_m2l_operation(
          spacetime_src, ( *associated_spacetime_targets )[ i ] );
      }
    }
    if ( _measure_tasks ) {
      _m2l_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_m2l_operation( const mesh::general_spacetime_cluster *
                                         src_cluster,
  mesh::general_spacetime_cluster * tar_cluster ) const {
  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2l coefficients
  vector_type buffer_for_gaussians( ( _spat_order + 1 ) * ( _spat_order + 1 )
    * ( _temp_order + 1 ) * ( _temp_order + 1 ) );
  // buffer to store m2l coefficients.
  vector_type buffer_for_coeffs( ( _spat_order + 1 ) * ( _spat_order + 1 )
    * ( _temp_order + 1 ) * ( _temp_order + 1 ) );
  // buffer matrices to store intermediate m2l results.
  lo thread_num = omp_get_thread_num( );

  _aux_buffer_0[ thread_num ].fill( 0.0 );
  _aux_buffer_1[ thread_num ].fill( 0.0 );

  // get geometrical data of the clusters
  sc src_half_size_time;
  vector_type half_size_space( 3, false );
  src_cluster->get_half_size( half_size_space, src_half_size_time );
  sc tar_half_size_time = tar_cluster->get_time_half_size( );
  sc src_center_time, tar_center_time;
  vector_type src_center_space( 3, false );
  vector_type tar_center_space( 3, false );
  src_cluster->get_center( src_center_space, src_center_time );
  tar_cluster->get_center( tar_center_space, tar_center_time );

  // initialize temporal interpolation nodes in source and target cluster
  vector_type src_time_nodes( _temp_order + 1, false );
  vector_type tar_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    tar_time_nodes[ i ]
      = tar_center_time + tar_half_size_time * time_nodes[ i ];
    src_time_nodes[ i ]
      = src_center_time + src_half_size_time * time_nodes[ i ];
  }

  // get spatial properties ( difference of cluster, half length )
  vector_type center_diff_space( tar_center_space );
  for ( lo i = 0; i < 3; ++i ) {
    center_diff_space[ i ] -= src_center_space[ i ];
  }

  sc padding_space = _distributed_spacetime_tree
                       ->get_spatial_paddings( )[ src_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // compute coupling coefficients for dimension 2
  compute_coupling_coeffs( src_time_nodes, tar_time_nodes, half_size_space[ 2 ],
    center_diff_space[ 2 ], buffer_for_gaussians, buffer_for_coeffs );

  const sc * src_moment = src_cluster->get_pointer_to_moment( );
  sc * tar_local = tar_cluster->get_pointer_to_local_contribution( );
  // efficient m2l operation similar to Tausch, 2009, p. 3558
  // help variables for accessing right values in coefficient buffer
  lo hlp_acs_alpha
    = ( _spat_order + 1 ) * ( _temp_order + 1 ) * ( _temp_order + 1 );
  lo hlp_acs_beta = ( _temp_order + 1 ) * ( _temp_order + 1 );
  lo hlp_acs_a = ( _temp_order + 1 );

  sc * aux_buffer_0_data = _aux_buffer_0[ thread_num ].data( );
  sc * buffer_for_coeffs_data = buffer_for_coeffs.data( );

  // compute first intermediate product and store it in aux_buffer_0
  lo buffer_0_index = 0;
  for ( lo alpha2 = 0; alpha2 <= _spat_order; ++alpha2 ) {
    lo moment_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order - alpha2; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - alpha2 - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          // no need for reduction, in a single inner cycle data are written on
          // unique positions
          for ( lo a = 0; a <= _temp_order; ++a ) {
#pragma omp simd aligned(                               \
  aux_buffer_0_data, buffer_for_coeffs_data, src_moment \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo b = 0; b <= _temp_order; ++b ) {
              aux_buffer_0_data[ buffer_0_index * hlp_acs_beta + hlp_acs_a * a
                + b ]
                += buffer_for_coeffs_data[ alpha2 * hlp_acs_alpha
                     + beta2 * hlp_acs_beta + a * hlp_acs_a + b ]
                * src_moment[ b + moment_index * ( _temp_order + 1 ) ];
            }
          }
          ++moment_index;
        }
        ++buffer_0_index;
      }
      // correction for moment index; this is necessary since beta1 does not
      // run until _spat_order - beta0 as it does in src_moment;
      moment_index += ( ( alpha2 + 1 ) * alpha2 ) / 2;
    }
  }
  // update coefficients and compute 2nd intermediate product in aux_buffer_1
  compute_coupling_coeffs( src_time_nodes, tar_time_nodes, half_size_space[ 1 ],
    center_diff_space[ 1 ], buffer_for_gaussians, buffer_for_coeffs );

  sc * aux_buffer_1_data = _aux_buffer_1[ thread_num ].data( );

  lo buffer_1_index = 0;
  for ( lo alpha1 = 0; alpha1 <= _spat_order; ++alpha1 ) {
    buffer_0_index = 0;
    for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha1; ++alpha2 ) {
      for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
        for ( lo beta1 = 0; beta1 <= _spat_order - beta0 - alpha2; ++beta1 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
#pragma omp simd aligned(                                      \
  aux_buffer_1_data, buffer_for_coeffs_data, aux_buffer_0_data \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo b = 0; b <= _temp_order; ++b ) {
              aux_buffer_1_data[ buffer_1_index * hlp_acs_beta + hlp_acs_a * a
                + b ]
                += buffer_for_coeffs_data[ alpha1 * hlp_acs_alpha
                     + beta1 * hlp_acs_beta + a * hlp_acs_a + b ]
                * aux_buffer_0_data[ buffer_0_index * hlp_acs_beta
                  + hlp_acs_a * a + b ];
            }
          }
          ++buffer_0_index;
        }
        ++buffer_1_index;
      }
      // correction for buffer_0 index; this is necessary since beta0 does not
      // run until _spat_order - alpha2 as it does in aux_buffer_0;
      buffer_0_index += ( ( alpha1 + 1 ) * alpha1 ) / 2;
    }
  }

  // update coefficients and update targets local contribution with m2l result
  compute_coupling_coeffs( src_time_nodes, tar_time_nodes, half_size_space[ 0 ],
    center_diff_space[ 0 ], buffer_for_gaussians, buffer_for_coeffs );
  sc val = 0;
  int local_index = 0;
  for ( lo alpha0 = 0; alpha0 <= _spat_order; ++alpha0 ) {
    buffer_1_index = 0;
    for ( lo alpha1 = 0; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
      for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha0 - alpha1; ++alpha2 ) {
        for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            val = 0;
#pragma omp simd aligned( buffer_for_coeffs_data, aux_buffer_1_data, tar_local : DATA_ALIGN ) simdlen( DATA_WIDTH) reduction( + : val)
            for ( lo b = 0; b <= _temp_order; ++b ) {
              val += buffer_for_coeffs_data[ alpha0 * hlp_acs_alpha
                       + beta0 * hlp_acs_beta + a * hlp_acs_a + b ]
                * aux_buffer_1_data[ buffer_1_index * hlp_acs_beta
                  + hlp_acs_a * a + b ];
            }
            tar_local[ a + local_index * ( _temp_order + 1 ) ] += val;
          }
          ++buffer_1_index;
        }
        ++local_index;
      }
      // correction for buffer_1 index; this is necessary since alpha0 does
      // not run until _spat_order - alpha1 as it does in aux_buffer_1;
      buffer_1_index += ( ( alpha0 + 1 ) * alpha0 ) / 2;
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::call_l2l_operations( scheduling_time_cluster *
                                                       time_cluster,
  bool verbose, const std::string & verbose_file ) const {
  scheduling_time_cluster * parent_cluster = time_cluster->get_parent( );
  // m2m operations are only executed if the parent is active in the upward
  // path
  if ( verbose ) {
#pragma omp critical( verbose )
    {
      if ( _measure_tasks ) {
        _l_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "call L2L for cluster " << time_cluster->get_global_index( )
                << " at level " << time_cluster->get_level( ) << std::endl;
        outfile.close( );
      }
      if ( _measure_tasks ) {
        _l_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }
  slou configuration = time_cluster->get_configuration( );
  std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
    = parent_cluster->get_associated_spacetime_clusters( );
  lou n_associated_leaves = parent_cluster->get_n_associated_leaves( );
  // call the l2l operations for all non-leaf spacetime clusters which are
  // associated with the parent scheduling time cluster
  // there is an implicit taskgroup after this taskloop
#pragma omp taskloop
  for ( lou i = n_associated_leaves; i < associated_spacetime_clusters->size( );
        ++i ) {
    apply_grouped_l2l_operation(
      ( *associated_spacetime_clusters )[ i ], configuration );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_grouped_l2l_operation( mesh::general_spacetime_cluster *
                                                 parent_cluster,
  slou child_configuration ) const {
  sc * parent_local_contribution
    = parent_cluster->get_pointer_to_local_contribution( );
  std::vector< general_spacetime_cluster * > * children
    = parent_cluster->get_children( );
  // ###############################################
  // ### compute the matrix for the temporal l2l ###
  full_matrix temporal_l2l_matrix( _temp_order + 1, _temp_order + 1 );
  lo child_idx = 0;
  // find a child with the right temporal configuration. Note: such a child
  // exists, otherwise the routine is not executed
  while ( ( *children )[ child_idx ]->get_temporal_configuration( )
    != child_configuration ) {
    ++child_idx;
  }
  // get the temporal data of the parent and child cluster.
  sc parent_time_center = parent_cluster->get_time_center( );
  sc parent_time_half_size = parent_cluster->get_time_half_size( );
  sc child_time_center = ( *children )[ child_idx ]->get_time_center( );
  sc child_time_half_size = ( *children )[ child_idx ]->get_time_half_size( );

  const vector_type & nodes = _lagrange.get_nodes( );
  vector_type nodes_child( _temp_order + 1, false );
  vector_type values_lagrange( _temp_order + 1, false );
  // transform the nodes from [-1, 1] to the child interval and then back to
  // [-1, 1] with the transformation of the parent interval:
  for ( lo j = 0; j <= _temp_order; ++j ) {
    nodes_child[ j ] = ( child_time_center + (child_time_half_size) *nodes[ j ]
                         - parent_time_center )
      / parent_time_half_size;
  }
  // compute entries of the l2l matrix (by evaluating lagrange polynomials)
  for ( lo j = 0; j <= _temp_order; ++j ) {
    _lagrange.evaluate( j, nodes_child, values_lagrange );
    for ( lo k = 0; k <= _temp_order; ++k )
      temporal_l2l_matrix.set( k, j, values_lagrange[ k ] );
  }
  // ###############################################
  // determine whether a space-time l2l operation is necessary or only a
  // temporal l2l operation
  lo n_space_div_parent, n_space_div_children, dummy;
  parent_cluster->get_n_divs( n_space_div_parent, dummy );
  ( *children )[ 0 ]->get_n_divs( n_space_div_children, dummy );
  bool temporal_only = ( n_space_div_parent == n_space_div_children );
  if ( temporal_only ) {
    // execute only temporal l2l operation
    for ( auto child : *children ) {
      if ( child->get_temporal_configuration( ) == child_configuration ) {
        sc * child_local_contribution
          = child->get_pointer_to_local_contribution( );
        apply_temporal_l2l_operation( parent_local_contribution,
          temporal_l2l_matrix, child_local_contribution );
      }
    }
  } else {
    // apply the temporal l2l operation to the parent's local contribution and
    // store the results in a buffer
    // @todo use buffer as input argument to avoid reallocation.
    std::vector< sc > buffer_array(
      ( _temp_order + 1 ) * _spat_contribution_size );
    for ( int i = 0; i < ( _temp_order + 1 ) * _spat_contribution_size; ++i ) {
      buffer_array[ i ] = 0.0;
    }
    apply_temporal_m2m_operation(
      parent_local_contribution, temporal_l2l_matrix, buffer_array.data( ) );

    for ( auto child : *children ) {
      short child_octant, current_configuration;
      child->get_position( child_octant, current_configuration );
      if ( current_configuration == child_configuration ) {
        sc * child_local_contribution
          = child->get_pointer_to_local_contribution( );
        apply_spatial_l2l_operation( buffer_array.data( ), n_space_div_parent,
          child_octant, child_local_contribution );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_temporal_l2l_operation( const sc *
                                                  parent_local_contribution,
  const full_matrix & temporal_l2l_matrix,
  sc * child_local_contribution ) const {
  // call the appropriate cblas routine for matrix matrix multiplication.
  lo n_rows_l2l_matrix = temporal_l2l_matrix.get_n_rows( );
  lo n_cols_moment = _spat_contribution_size;
  lo n_cols_l2l_matrix = n_rows_l2l_matrix;
  lo lda = n_rows_l2l_matrix;
  lo ldb = n_cols_l2l_matrix;
  sc alpha = 1.0;
  sc beta = 1.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_l2l_matrix,
    n_cols_moment, n_cols_l2l_matrix, alpha, temporal_l2l_matrix.data( ), lda,
    parent_local_contribution, ldb, beta, child_local_contribution,
    n_rows_l2l_matrix );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_spatial_l2l_operation( const sc *
                                                               parent_local,
  const lo n_space_div_parent, const slou octant, sc * child_local ) const {
  const vector_type * m2m_coeffs_s_dim_0;
  const vector_type * m2m_coeffs_s_dim_1;
  const vector_type * m2m_coeffs_s_dim_2;
  switch ( octant ) {
    case 0:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 1:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 2:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 3:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 4:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 5:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 6:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 7:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    default:  // default case should never be used, program will crash!
      m2m_coeffs_s_dim_0 = nullptr;
      m2m_coeffs_s_dim_1 = nullptr;
      m2m_coeffs_s_dim_2 = nullptr;
  }

  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  // initialize auxiliary matrices lambda_1/2 for intermediate results with 0
  // TODO instead of allocating every time use buffers
  full_matrix lambda_1( _temp_order + 1, n_coeffs_s, true );
  full_matrix lambda_2( _temp_order + 1, n_coeffs_s, true );

  for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
    lou parent_index = 0;
    for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
      for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0; ++alpha1 ) {
        // correction for skipped entries of parent_local due to starting
        // point alpha2 = beta2 in the next loop
        parent_index += beta2;
        for ( lo alpha2 = beta2; alpha2 <= _spat_order - alpha0 - alpha1;
              ++alpha2 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            lambda_1( a,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 )
              += ( *m2m_coeffs_s_dim_2 )[ alpha2 * ( _spat_order + 1 ) + beta2 ]
              * parent_local[ a + parent_index * ( _temp_order + 1 ) ];
          }
          ++parent_index;
        }
      }
      // correction for current index; this is necessary since alpha1 does not
      // run until _spat_order - alpha0 as it does in parent_local;
      parent_index += ( ( beta2 + 1 ) * beta2 ) / 2;
    }
  }

  for ( lo beta1 = 0; beta1 <= _spat_order; ++beta1 ) {
    for ( lo beta2 = 0; beta2 <= _spat_order - beta1; ++beta2 ) {
      for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
        for ( lo alpha1 = beta1; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
          for ( lo a = 0; a <= _temp_order; ++a )
            lambda_2( a,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                + ( _spat_order + 1 ) * beta2 + alpha0 )
              += ( *m2m_coeffs_s_dim_1 )[ alpha1 * ( _spat_order + 1 ) + beta1 ]
              * lambda_1( a,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                  + ( _spat_order + 1 ) * alpha0 + alpha1 );
        }
      }
    }
  }

  lou child_index = 0;
  for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
    for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
        for ( lo alpha0 = beta0; alpha0 <= _spat_order - beta1 - beta2;
              ++alpha0 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            child_local[ a + child_index * ( _temp_order + 1 ) ]
              += ( *m2m_coeffs_s_dim_0 )[ alpha0 * ( _spat_order + 1 ) + beta0 ]
              * lambda_2( a,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                  + ( _spat_order + 1 ) * beta2 + alpha0 );
          }
        }
        ++child_index;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_l2t_operations( mesh::scheduling_time_cluster *
                                         time_cluster,
  block_vector & output_vector, bool verbose,
  const std::string & verbose_file ) const {
  // execute only for associated spacetime leaves
  if ( time_cluster->get_n_associated_leaves( ) > 0 ) {
    if ( verbose ) {
      std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "call L2T for cluster " << time_cluster->get_global_index( )
                << std::endl;
        outfile.close( );
      }
    }
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = time_cluster->get_associated_spacetime_clusters( );
    lou i = 0;
    lou n = time_cluster->get_n_associated_leaves( );
    // there is an implicit taskgroup after this taskloop
#pragma omp taskloop shared( output_vector, associated_spacetime_clusters )
    for ( i = 0; i < n; ++i ) {
      apply_l2t_operation(
        ( *associated_spacetime_clusters )[ i ], output_vector );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_l2t_operations( mesh::scheduling_time_cluster *
                                         time_cluster,
  distributed_block_vector & output_vector, bool verbose,
  const std::string & verbose_file ) const {
  // execute only for associated spacetime leaves
  if ( time_cluster->get_n_associated_leaves( ) > 0 ) {
    if ( verbose ) {
#pragma omp critical( verbose )
      {
        std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "call L2T for cluster "
                  << time_cluster->get_global_index( ) << " at level "
                  << time_cluster->get_level( ) << std::endl;
          outfile.close( );
        }
      }
    }
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = time_cluster->get_associated_spacetime_clusters( );
    lou i = 0;
    lou n = time_cluster->get_n_associated_leaves( );
    // there is an implicit taskgroup after this taskloop
#pragma omp taskloop shared( output_vector, associated_spacetime_clusters )
    for ( i = 0; i < n; ++i ) {
      if ( _measure_tasks ) {
        _l_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      apply_l2t_operation(
        ( *associated_spacetime_clusters )[ i ], output_vector );
      if ( _measure_tasks ) {
        _l_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_l2t_operation( const mesh::general_spacetime_cluster *
  /*cluster*/,
  block_vector & /*output_vector*/ ) const {
  std::cout << "L2T operation not implemented!" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_l2t_operation( const mesh::general_spacetime_cluster * cluster,
    block_vector & output_vector ) const {
  apply_l2t_operation_p0( cluster, output_vector );
}

//! template specialization for double layer p0p1 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation( const mesh::general_spacetime_cluster * cluster,
    block_vector & output_vector ) const {
  apply_l2t_operation_p0( cluster, output_vector );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_l2t_operation( const mesh::general_spacetime_cluster * cluster,
    block_vector & output_vector ) const {
  apply_l2t_operation_p1_normal_drv( cluster, output_vector );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p0( const mesh::general_spacetime_cluster * cluster,
    block_vector & output_vector ) const {
  lo n_time_elements = cluster->get_n_time_elements( );
  lo n_space_elements = cluster->get_n_space_elements( );
  full_matrix targets( n_time_elements, n_space_elements, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution = cluster->get_pointer_to_local_contribution( );

  full_matrix T;
  compute_chebyshev_quadrature_p0( cluster, T );
  full_matrix L;
  compute_lagrange_quadrature( cluster, L );

  // compute D = trans(L) * lambda and then the result Y = D * trans(T)
  //  D = trans(L) * lambda with explicit cblas routine call:
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo n_rows_lagrange = L.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_lagrange,
    n_cols_local, n_rows_lagrange, alpha, L.data( ), lda, local_contribution,
    ldb, beta, aux_matrix.data( ), n_cols_lagrange );
  // compute Y = D * trans(T)
  targets.multiply( aux_matrix, T, false, true );

  // add the results to the correct positions of the output vector
  const std::vector< lo > & spacetime_elements = cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = cluster->get_mesh( );
  // a cluster for which an S2M operation is executed is always local!
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      lo global_space_index = local_mesh->get_space_element(
        distributed_mesh->global_2_local( local_start_idx,
          spacetime_elements[ i_time * n_space_elements + i_space ] ) );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      output_vector.add_atomic( distributed_mesh->local_2_global_time(
                                  local_start_idx, local_time_index ),
        global_space_index, targets( i_time, i_space ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * cluster,
    block_vector & output_vector ) const {
  lo n_time_elements = cluster->get_n_time_elements( );
  lo n_space_elements = cluster->get_n_space_elements( );
  lo n_space_nodes = cluster->get_n_space_nodes( );
  full_matrix targets( n_time_elements, n_space_nodes, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution = cluster->get_pointer_to_local_contribution( );
  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1( cluster, T_drv );
  full_matrix L;
  compute_lagrange_quadrature( cluster, L );

  // compute D = trans(L) * lambda and then the result Y = D * trans(T)
  //  D = trans(L) * lambda with explicit cblas routine call:
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo n_rows_lagrange = L.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_lagrange,
    n_cols_local, n_rows_lagrange, alpha, L.data( ), lda, local_contribution,
    ldb, beta, aux_matrix.data( ), n_cols_lagrange );
  // compute Y = D * trans(T)
  targets.multiply( aux_matrix, T_drv, false, true );

  // add the results to the correct positions of the output vector
  const std::vector< lo > & spacetime_elements = cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = cluster->get_mesh( );
  // a cluster for which an S2M operation is executed is always local!
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  const std::vector< lo > & local_2_global_nodes
    = cluster->get_local_2_global_nodes( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      // local_2_global_nodes gives the indices of the spacetime nodes. take
      // the rest from division by the number of global spatial nodes to get
      // the spatial node index
      lo global_space_index
        = local_2_global_nodes[ i_space ] % local_mesh->get_n_spatial_nodes( );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      output_vector.add_atomic( distributed_mesh->local_2_global_time(
                                  local_start_idx, local_time_index ),
        global_space_index, targets( i_time, i_space ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_nearfield_operations( const mesh::scheduling_time_cluster * cluster,
    const block_vector & sources, bool trans, block_vector & output_vector,
    bool verbose, const std::string & verbose_file ) const {
  if ( verbose ) {
    std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
    if ( outfile.is_open( ) ) {
      outfile << "apply NF for cluster " << cluster->get_global_index( );
      outfile.close( );
    }
  }
  vector_type local_sources;
  const std::vector< general_spacetime_cluster * > *
    associated_spacetime_targets
    = cluster->get_associated_spacetime_clusters( );
  lou n_associated_leaves = cluster->get_n_associated_leaves( );
  // there is an implicit taskgroup after this taskloop
#pragma omp taskloop shared( output_vector, _clusterwise_nearfield_matrices )
  for ( lou i = 0; i < n_associated_leaves; ++i ) {
    general_spacetime_cluster * current_spacetime_target
      = ( *associated_spacetime_targets )[ i ];
    // construct a local result_vector
    vector_type local_result(
      current_spacetime_target->get_n_dofs< target_space >( ), true );
    // get the nearfield list of the current spacetime target cluster and
    // apply the nearfield operations for all the clusters in this list.
    std::vector< general_spacetime_cluster * > * spacetime_nearfield_list
      = current_spacetime_target->get_nearfield_list( );
    for ( lou src_index = 0; src_index < spacetime_nearfield_list->size( );
          ++src_index ) {
      general_spacetime_cluster * current_spacetime_source
        = ( *spacetime_nearfield_list )[ src_index ];
      local_sources.resize(
        current_spacetime_source->get_n_dofs< source_space >( ) );
      // get the sources corresponding to the current spacetime source cluster
      sources.get_local_part< source_space >(
        current_spacetime_source, local_sources );

      full_matrix * current_block = _clusterwise_nearfield_matrices.at(
        current_spacetime_target )[ src_index ];
      // apply the nearfield matrix and add the result to local_result
      current_block->apply( local_sources, local_result, trans, 1.0, 1.0 );
    }
    // add the local result to the output vector
    output_vector.add_local_part< target_space >(
      current_spacetime_target, local_result );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::check_for_received_data( std::vector< MPI_Request > &
                                             array_of_requests,
  std::vector< int > & array_of_indices, int & outcount ) const {
  MPI_Testsome( _receive_data_information.size( ), array_of_requests.data( ),
    &outcount, array_of_indices.data( ), MPI_STATUSES_IGNORE );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  find_cluster_in_l_list( std::list< scheduling_time_cluster * > & l_list,
    std::list< scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const {
  it_next_cluster = l_list.begin( );
  // a cluster is ready if its parents downward path status equals 2
  while ( status != 2 && it_next_cluster != l_list.end( ) ) {
    if ( ( *it_next_cluster )->get_parent( )->get_downward_path_status( ) == 2 )
      status = 2;
    else
      ++it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  find_cluster_in_m_list( std::list< scheduling_time_cluster * > & m_list,
    std::list< scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const {
  it_next_cluster = m_list.begin( );
  // a cluster is ready if its upward path counter equals 0
  while ( status != 1 && it_next_cluster != m_list.end( ) ) {
    if ( ( *it_next_cluster )->get_upward_path_counter( ) == 0 )
      status = 1;
    else
      ++it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  find_cluster_in_m2l_list(
    std::list< mesh::scheduling_time_cluster * > & m2l_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const {
  it_next_cluster = m2l_list.begin( );
  // a cluster is ready if number of ready interactions is equal to size of
  // interaction list
  while ( status != 3 && it_next_cluster != m2l_list.end( ) ) {
    if ( ( *it_next_cluster )->get_ready_interaction_list_size( )
      == ( *it_next_cluster )->get_interaction_list( )->size( ) )
      status = 3;
    else
      ++it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::provide_moments_for_m2l( scheduling_time_cluster *
                                             src_cluster,
  bool verbose, const std::string & verbose_file ) const {
  std::vector< scheduling_time_cluster * > * send_list
    = src_cluster->get_send_list( );
  std::set< lo > process_send_list;
  if ( send_list != nullptr ) {
    for ( auto it = send_list->begin( ); it != send_list->end( ); ++it ) {
      lo tar_process_id = ( *it )->get_process_id( );
      if ( tar_process_id == _my_rank ) {
        ( *it )->update_ready_interaction_size( );
      } else if ( process_send_list.count( tar_process_id ) == 0 ) {
        if ( verbose ) {
#pragma omp critical( verbose )
          {
            std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
            if ( outfile.is_open( ) ) {
              outfile << "send for m2l: data from source "
                      << src_cluster->get_global_index( ) << " at level "
                      << src_cluster->get_level( ) << " to process "
                      << tar_process_id << std::endl;
              outfile.close( );
            }
          }
        }
        lo tag = 2 * src_cluster->get_global_index( );
        sc * moment_buffer = src_cluster->get_associated_moments( );
        int buffer_size
          = src_cluster->get_associated_spacetime_clusters( )->size( )
          * _contribution_size;

        if ( _measure_tasks ) {
          _mpi_send_m2l.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        MPI_Request req;
        MPI_Isend( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ),
          tar_process_id, tag, *_comm, &req );
        MPI_Request_free( &req );
        process_send_list.insert( tar_process_id );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::provide_moments_to_parents( scheduling_time_cluster *
                                                child_cluster,
  bool verbose, const std::string & verbose_file ) const {
  scheduling_time_cluster * parent_cluster = child_cluster->get_parent( );
  // moments have to be provided only if the parent is active in the upward
  // path
  if ( parent_cluster->is_active_in_upward_path( ) ) {
    lo parent_process_id = parent_cluster->get_process_id( );
    if ( parent_process_id == _my_rank ) {
      // update local dependency counter
      parent_cluster->reduce_upward_path_counter( );
    } else if ( parent_process_id != -1 ) {
      if ( verbose ) {
#pragma omp critical( verbose )
        {
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << "send upward: from source "
                    << child_cluster->get_global_index( ) << " at level "
                    << child_cluster->get_level( ) << " to process "
                    << parent_process_id << std::endl;
            outfile.close( );
          }
        }
      }
      lo tag = 2 * parent_cluster->get_global_index( );
      sc * moment_buffer = parent_cluster->get_associated_moments( );
      int buffer_size
        = parent_cluster->get_associated_spacetime_clusters( )->size( )
        * _contribution_size;

      if ( _measure_tasks ) {
        _mpi_send_m_parent.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }

      MPI_Request req;
      MPI_Isend( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ),
        parent_process_id, tag, *_comm, &req );
      MPI_Request_free( &req );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  provide_local_contributions_to_children(
    scheduling_time_cluster * parent_cluster, bool verbose,
    const std::string & verbose_file ) const {
  std::vector< scheduling_time_cluster * > * children
    = parent_cluster->get_children( );
  if ( children != nullptr ) {
    for ( auto child : *children ) {
      lo child_process_id = child->get_process_id( );
      if ( child_process_id != _my_rank ) {
        if ( verbose ) {
#pragma omp critical( verbose )
          {
            std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
            if ( outfile.is_open( ) ) {
              outfile << "send downward: from source "
                      << parent_cluster->get_global_index( ) << " at level "
                      << parent_cluster->get_level( ) << " to process "
                      << child_process_id << std::endl;
              outfile.close( );
            }
          }
        }
        lo tag = 2 * parent_cluster->get_global_index( ) + 1;
        sc * local_contribution_buffer
          = parent_cluster->get_associated_local_contributions( );
        int buffer_size
          = parent_cluster->get_associated_spacetime_clusters( )->size( )
          * _contribution_size;

        if ( _measure_tasks ) {
          _mpi_send_l_children.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }

        MPI_Request req;
        MPI_Isend( local_contribution_buffer, buffer_size,
          get_scalar_type< sc >::MPI_SC( ), child_process_id, tag, *_comm,
          &req );
        MPI_Request_free( &req );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::start_receive_operations( std::vector< MPI_Request > &
    array_of_requests ) const {
  // start the receive operations for the moments in the upward path
  for ( lou i = 0; i < _n_moments_to_receive_upward; ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    scheduling_time_cluster * receive_cluster
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( );
    sc * moment_buffer
      = receive_cluster->get_extraneous_moment_pointer( source_id );
    int buffer_size
      = receive_cluster->get_associated_spacetime_clusters( )->size( )
      * _contribution_size;
    MPI_Irecv( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ),
      source_id, tag, *_comm, &array_of_requests[ i ] );
    // std::cout << "source: " << source_id << ", tag: " << tag << std::endl;
  }

  // start the receive operations for the moments needed for m2l
  // std::cout << "call receive operations for moments needed for m2l: "
  //           << std::endl;
  for ( lou i = _n_moments_to_receive_upward;
        i < _n_moments_to_receive_upward + _n_moments_to_receive_m2l; ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    scheduling_time_cluster * receive_cluster
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( );
    sc * moment_buffer = receive_cluster->get_associated_moments( );
    int buffer_size
      = receive_cluster->get_associated_spacetime_clusters( )->size( )
      * _contribution_size;
    MPI_Irecv( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ),
      source_id, tag, *_comm, &array_of_requests[ i ] );
    // std::cout << "source: " << source_id << ", tag: " << tag << std::endl;
  }

  // start the receive operations for the local contributions
  // std::cout << "receive operations for local contributions: " << std::endl;
  for ( lou i = _n_moments_to_receive_upward + _n_moments_to_receive_m2l;
        i < _receive_data_information.size( ); ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    scheduling_time_cluster * receive_cluster
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( ) + 1;
    sc * local_contribution_buffer
      = receive_cluster->get_associated_local_contributions( );
    int buffer_size
      = receive_cluster->get_associated_spacetime_clusters( )->size( )
      * _contribution_size;
    MPI_Irecv( local_contribution_buffer, buffer_size,
      get_scalar_type< sc >::MPI_SC( ), source_id, tag, *_comm,
      &array_of_requests[ i ] );
    // std::cout << "source: " << source_id << ", tag: " << tag << std::endl;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_chebyshev_quadrature_p0(
    const general_spacetime_cluster * source_cluster, full_matrix & T ) const {
  lo n_space_elems = source_cluster->get_n_space_elements( );
  T.resize( n_space_elems, _spat_contribution_size );
  // get some info on the current cluster
  vector_type cluster_center_space( 3 );
  vector_type cluster_half_space( 3 );
  sc dummy;
  source_cluster->get_center( cluster_center_space, dummy );
  source_cluster->get_half_size( cluster_half_space, dummy );
  sc padding = _distributed_spacetime_tree
                 ->get_spatial_paddings( )[ source_cluster->get_level( ) ];
  sc start_0 = cluster_center_space[ 0 ] - cluster_half_space[ 0 ] - padding;
  sc end_0 = cluster_center_space[ 0 ] + cluster_half_space[ 0 ] + padding;
  sc start_1 = cluster_center_space[ 1 ] - cluster_half_space[ 1 ] - padding;
  sc end_1 = cluster_center_space[ 1 ] + cluster_half_space[ 1 ] + padding;
  sc start_2 = cluster_center_space[ 2 ] - cluster_half_space[ 2 ] - padding;
  sc end_2 = cluster_center_space[ 2 ] + cluster_half_space[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );

  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = source_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  for ( lo i = 0; i < n_space_elems; ++i ) {
    // we use that the elements in the cluster are tensor products of spatial
    // elements and timesteps, and are sorted w.r.t. the timesteps. In
    // particular we get all spatial elements in the cluster by considering
    // the first n_space_elems spacetime elements.
    lo local_elem_idx = distributed_mesh->global_2_local(
      local_start_idx, source_cluster->get_element( i ) );
    lo local_elem_idx_space = local_mesh->get_space_element( local_elem_idx );

    local_mesh->get_spatial_nodes( local_elem_idx_space, y1, y2, y3 );
    sc elem_area = local_mesh->spatial_area( local_elem_idx_space );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          sc quad = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            quad += cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ] * wy[ j ];
          }
          quad *= elem_area;
          T.set( i, current_index, quad );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_normal_drv_chebyshev_quadrature_p1(
    const general_spacetime_cluster * source_cluster,
    full_matrix & T_drv ) const {
  lo n_space_elems = source_cluster->get_n_space_elements( );
  lo n_space_nodes = source_cluster->get_n_space_nodes( );
  T_drv.resize( n_space_nodes, _spat_contribution_size );
  // get some info on the current cluster
  vector_type cluster_center_space( 3 );
  vector_type cluster_half_space( 3 );
  sc dummy;
  source_cluster->get_center( cluster_center_space, dummy );
  source_cluster->get_half_size( cluster_half_space, dummy );
  sc padding = _distributed_spacetime_tree
                 ->get_spatial_paddings( )[ source_cluster->get_level( ) ];
  sc start_0 = cluster_center_space[ 0 ] - cluster_half_space[ 0 ] - padding;
  sc end_0 = cluster_center_space[ 0 ] + cluster_half_space[ 0 ] + padding;
  sc start_1 = cluster_center_space[ 1 ] - cluster_half_space[ 1 ] - padding;
  sc end_1 = cluster_center_space[ 1 ] + cluster_half_space[ 1 ] + padding;
  sc start_2 = cluster_center_space[ 2 ] - cluster_half_space[ 2 ] - padding;
  sc end_2 = cluster_center_space[ 2 ] + cluster_half_space[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );
  // same for evaluations of scaled derivatives of Chebyshev polynomials
  vector_type cheb_drv_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_drv_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_drv_dim_2( ( _spat_order + 1 ) * size_quad );

  sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  sc * y2_ref = my_quadrature._y2_ref_cheb.data( );

  linear_algebra::coordinates< 3 > grad;
  const std::vector< lo > & elems_2_local_nodes
    = source_cluster->get_elems_2_local_nodes( );
  linear_algebra::coordinates< 3 > normal;

  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = source_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  for ( lo i = 0; i < n_space_elems; ++i ) {
    lo local_elem_idx = distributed_mesh->global_2_local(
      local_start_idx, source_cluster->get_element( i ) );
    lo local_elem_idx_space = local_mesh->get_space_element( local_elem_idx );
    local_mesh->get_spatial_normal( local_elem_idx_space, normal );
    local_mesh->get_spatial_nodes( local_elem_idx_space, y1, y2, y3 );
    sc elem_area = local_mesh->spatial_area( local_elem_idx_space );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y1_polynomial, cheb_drv_dim_0 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y2_polynomial, cheb_drv_dim_1 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y3_polynomial, cheb_drv_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          sc value1 = 0.0;
          sc value2 = 0.0;
          sc value3 = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            grad[ 0 ] = cheb_drv_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ]
              / ( cluster_half_space[ 0 ] + padding );
            grad[ 1 ] = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_drv_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ]
              / ( cluster_half_space[ 1 ] + padding );
            grad[ 2 ] = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_drv_dim_2[ beta2 * size_quad + j ]
              / ( cluster_half_space[ 2 ] + padding );
            sc weighted_normal_derivative
              = wy[ j ] * elem_area * normal.dot( grad );
            value1 += weighted_normal_derivative
              * ( (sc) 1.0 - y1_ref[ j ] - y2_ref[ j ] );
            value2 += weighted_normal_derivative * y1_ref[ j ];
            value3 += weighted_normal_derivative * y2_ref[ j ];
          }

          T_drv.add_atomic(
            source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
              elems_2_local_nodes[ 6 * i ] ),
            current_index, _alpha * value1 );
          T_drv.add_atomic(
            source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
              elems_2_local_nodes[ 6 * i + 1 ] ),
            current_index, _alpha * value2 );
          T_drv.add_atomic(
            source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
              elems_2_local_nodes[ 6 * i + 2 ] ),
            current_index, _alpha * value3 );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_lagrange_quadrature(
    const mesh::general_spacetime_cluster * source_cluster,
    full_matrix & L ) const {
  lo n_temp_elems = source_cluster->get_n_time_elements( );
  lo n_spat_elems = source_cluster->get_n_space_elements( );
  L.resize( _temp_order + 1, n_temp_elems );

  const std::vector< sc, besthea::allocator_type< sc > > & line_t
    = bem::quadrature::line_x( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & line_w
    = bem::quadrature::line_w( _order_regular );

  vector_type eval_points( line_t.size( ) );
  vector_type evaluation( line_t.size( ) );

  sc cluster_t_start = source_cluster->get_time_center( )
    - source_cluster->get_time_half_size( );
  sc cluster_t_end = source_cluster->get_time_center( )
    + source_cluster->get_time_half_size( );
  sc cluster_size = cluster_t_end - cluster_t_start;

  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = source_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  linear_algebra::coordinates< 1 > elem_t_start;
  linear_algebra::coordinates< 1 > elem_t_end;
  for ( lo i = 0; i < n_temp_elems; ++i ) {
    // we use that the elements in the cluster are tensor products of spatial
    // elements and timesteps, and are sorted w.r.t. the timesteps. In
    // particular we get all temporal elements in the cluster by considering
    // every n_spat_elems spacetime element.
    lo local_elem_idx = distributed_mesh->global_2_local(
      local_start_idx, source_cluster->get_element( i * n_spat_elems ) );
    lo local_elem_idx_time = local_mesh->get_time_element( local_elem_idx );

    local_mesh->get_temporal_nodes(
      local_elem_idx_time, elem_t_start, elem_t_end );
    sc elem_size = elem_t_end[ 0 ] - elem_t_start[ 0 ];
    // compute the quadrature points in the current element in relative
    // coordinates with respect to the time cluster and transform them to
    // [-1,1]
    for ( std::vector< sc, besthea::allocator_type< sc > >::size_type j = 0;
          j < line_t.size( ); ++j ) {
      eval_points[ j ] = -1.0
        + 2.0
          * ( elem_t_start[ 0 ] + elem_size * line_t[ j ] - cluster_t_start )
          / cluster_size;
    }

    for ( lo j = 0; j <= _temp_order; ++j ) {
      _lagrange.evaluate( j, eval_points, evaluation );
      sc quad = 0.0;
      for ( lo k = 0; k < eval_points.size( ); ++k ) {
        quad += evaluation[ k ] * line_w[ k ];
      }
      quad *= elem_size;
      L.set( j, i, quad );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_coupling_coeffs( const vector_type &
                                                           src_time_nodes,
  const vector_type & tar_time_nodes, const sc half_size, const sc center_diff,
  vector_type & buffer_for_gaussians, vector_type & coupling_coeffs ) const {
  coupling_coeffs.fill( 0.0 );
  // evaluate the gaussian kernel for the numerical integration
  sc h_alpha = half_size * half_size / ( 4.0 * _alpha );
  sc scaled_center_diff = center_diff / half_size;
  lou index_gaussian = 0;

  sc * buffer_for_gaussians_data = buffer_for_gaussians.data( );
  const sc * cheb_nodes_sum_coll_data = _cheb_nodes_sum_coll.data( );
  const sc * all_poly_vals_mult_coll_data = _all_poly_vals_mult_coll.data( );

  for ( lo a = 0; a <= _temp_order; ++a ) {
    for ( lo b = 0; b <= _temp_order; ++b ) {
      sc h_delta_ab = h_alpha / ( tar_time_nodes[ a ] - src_time_nodes[ b ] );
      lou i = 0;
#pragma omp simd aligned( cheb_nodes_sum_coll_data, buffer_for_gaussians_data \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
      for ( i = 0; i < _cheb_nodes_sum_coll.size( ); ++i ) {
        buffer_for_gaussians_data[ index_gaussian + i ] = std::exp( -h_delta_ab
          * ( scaled_center_diff + cheb_nodes_sum_coll_data[ i ] )
          * ( scaled_center_diff + cheb_nodes_sum_coll_data[ i ] ) );
        //++index_gaussian;
      }
      index_gaussian += i;
    }
  }

  // compute the numerical integrals
  sc mul_factor = 4.0 / ( _cheb_nodes_sum_coll.size( ) );
  lou index_integral = 0;

  for ( lo alpha = 0; alpha <= _spat_order; ++alpha ) {
    for ( lo beta = 0; beta <= _spat_order; ++beta ) {
      index_gaussian = 0;
      for ( lo a = 0; a <= _temp_order; ++a ) {
        for ( lo b = 0; b <= _temp_order; ++b ) {
          sc val = 0.0;

          lo start_idx
            = alpha * ( _spat_order + 1 ) * _cheb_nodes_sum_coll.size( )
            + beta * _cheb_nodes_sum_coll.size( );
          const sc * curr_ptr = all_poly_vals_mult_coll_data;  // + start_idx;
          lo idx;
#pragma omp simd aligned( buffer_for_gaussians_data,curr_ptr : DATA_ALIGN ) reduction( + : val ) simdlen( DATA_WIDTH )
          for ( idx = 0; idx < _cheb_nodes_sum_coll.size( ); ++idx ) {
            val += buffer_for_gaussians_data[ index_gaussian + idx ]
              * curr_ptr[ start_idx + idx ];
          }
          index_gaussian += idx;
          coupling_coeffs[ index_integral ] += val;

          sc mul_factor_ab = mul_factor
            / std::sqrt( 4.0 * M_PI * _alpha
              * ( tar_time_nodes[ a ] - src_time_nodes[ b ] ) );
          // gamma = 2 for all alpha and beta ( wrong, correction in
          //                    case of
          // alpha == 0 or beta == 0 )
          // an attempt to compute this in a separate loop with precomputed
          // mul_factor_ab was slower
          if ( alpha == 0 ) {
            mul_factor_ab *= 0.5;
          }
          if ( beta == 0 ) {
            mul_factor_ab *= 0.5;
          }
          coupling_coeffs[ index_integral ] *= mul_factor_ab;

          ++index_integral;
        }
      }
    }
  }

  // TODO: activate (and check!) this to avoid if clauses in the above loop
  //   for ( lo k = 0; k <= _spat_order; ++ k ) {
  //     lou index_temp = 0;
  //     for ( lo a = 0; a <= _temp_order; ++ a ) {
  //       for ( lo b = 0; b <= _temp_order; ++ b ) {
  //         //corrections for alpha = 0
  //         coupling_coeffs[ ( _temp_order + 1 ) * ( _temp_order + 1 ) * k
  //                           + index_temp ] *= 0.5;
  //         //corrections for beta = 0
  //         coupling_coeffs[ ( _temp_order + 1 ) * ( _temp_order + 1 ) *
  //                           ( _spat_order + 1 ) * k + index_temp ] *= 0.5;
  //         ++ index_temp;
  //       }
  //     }
  //   }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::reset_scheduling_clusters_dependency_data( )
  const {
  // reset the upward path counters of the clusters in the _m_list.
  for ( scheduling_time_cluster * cluster : _m_list ) {
    cluster->set_upward_path_counter( cluster->get_n_children( ) );
  }

  // reset the downward path status of cluster recursively
  reset_downward_path_status_recursively(
    _distributed_spacetime_tree->get_distribution_tree( )->get_root( ) );

  // reset the m2l counter and clear the ready interaction
  // list of the clusters in the _m2l_list.
  for ( scheduling_time_cluster * cluster : _m2l_list ) {
    cluster->set_m2l_counter( 0 );
    cluster->clear_ready_interaction_list( );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  reset_downward_path_status_recursively(
    scheduling_time_cluster * root ) const {
  // consider all local clusters which are active in the downward path
  if ( root->is_active_in_downward_path( ) ) {
    if ( root->get_parent( )->is_active_in_downward_path( ) ) {
      // the cluster has to execute l2l operations -> status 0
      root->set_downward_path_status( 0 );
    } else {
      // no l2l operations necessary -> status 1
      root->set_downward_path_status( 1 );
    }
  }
  // recursive call for all children
  if ( root->get_n_children( ) > 0 ) {
    for ( auto child : *root->get_children( ) ) {
      reset_downward_path_status_recursively( child );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::init_quadrature_polynomials( quadrature_wrapper &
    my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._y1_ref_cheb = bem::quadrature::triangle_x1( _order_regular );
  my_quadrature._y2_ref_cheb = bem::quadrature::triangle_x2( _order_regular );
  my_quadrature._wy_cheb = bem::quadrature::triangle_w( _order_regular );

  lo size = my_quadrature._wy_cheb.size( );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );

  my_quadrature._y1_polynomial.resize( size );
  my_quadrature._y2_polynomial.resize( size );
  my_quadrature._y3_polynomial.resize( size );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::triangle_to_geometry( const linear_algebra::coordinates< 3 > &
                                          x1,
  const linear_algebra::coordinates< 3 > & x2,
  const linear_algebra::coordinates< 3 > & x3,
  quadrature_wrapper & my_quadrature ) const {
  const sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  const sc * y2_ref = my_quadrature._y2_ref_cheb.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo size = my_quadrature._wy_cheb.size( );

  // x1, x2, x3 are vectors in R^3,
  // y%_mapped are the %th components of the vectors to which y#_ref is
  // mapped
#pragma omp simd aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * y1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * y2_ref[ i ];
    y2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * y1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * y2_ref[ i ];
    y3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * y1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * y2_ref[ i ];
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::cluster_to_polynomials( quadrature_wrapper &
                                                          my_quadrature,
  sc start_0, sc end_0, sc start_1, sc end_1, sc start_2, sc end_2 ) const {
  for ( lo i = 0; i < my_quadrature._y1_polynomial.size( ); ++i ) {
    my_quadrature._y1_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y1[ i ] - start_0 ) / ( end_0 - start_0 );
    my_quadrature._y2_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y2[ i ] - start_1 ) / ( end_1 - start_1 );
    my_quadrature._y3_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y3[ i ] - start_2 ) / ( end_2 - start_2 );
  }
}

template< class kernel_type, class target_space, class source_space >
sc besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, target_space,
  source_space >::compute_nonzero_nearfield_ratio( ) {
  lou n_nearfield_entries = 0;

  // get the local and nearfield mesh (needed to determine correct
  // number of non-zero nearfield matrix entries)
  const mesh::spacetime_tensor_mesh * nearfield_mesh;
  const mesh::spacetime_tensor_mesh * local_mesh;
  local_mesh = _distributed_spacetime_tree->get_mesh( ).get_local_mesh( );
  nearfield_mesh
    = _distributed_spacetime_tree->get_mesh( ).get_nearfield_mesh( );
  const mesh::spacetime_tensor_mesh * tar_mesh;
  const mesh::spacetime_tensor_mesh * src_mesh;

  for ( auto it : _n_list ) {
    const std::vector< general_spacetime_cluster * > * st_targets
      = it->get_associated_spacetime_clusters( );
    if ( it->get_process_id( ) == _my_rank ) {
      tar_mesh = local_mesh;
    } else {
      tar_mesh = nearfield_mesh;
    }
    lou n_associated_leaves = it->get_n_associated_leaves( );
    for ( lou i = 0; i < n_associated_leaves; ++i ) {
      // consider all associated st target clusters
      general_spacetime_cluster * st_target = ( *st_targets )[ i ];
      lo n_tar_elements_space = st_target->get_n_space_elements( );
      lo n_tar_elements_time = st_target->get_n_time_elements( );
      lo n_tar_elements = st_target->get_n_elements( );
      std::vector< general_spacetime_cluster * > * st_nearfield_list
        = st_target->get_nearfield_list( );
      for ( lou src_index = 0; src_index < st_nearfield_list->size( );
            ++src_index ) {
        // consider all spacetime clusters in the nearfield of the
        // current target cluster
        general_spacetime_cluster * st_source
          = ( *st_nearfield_list )[ src_index ];
        if ( st_source->get_elements_are_local( ) ) {
          src_mesh = local_mesh;
        } else {
          src_mesh = nearfield_mesh;
        }
        lo n_src_elements_space = st_source->get_n_space_elements( );
        lo n_src_elements_time = st_source->get_n_time_elements( );
        lo n_src_elements = st_source->get_n_elements( );
        // depending on the configuration in time update the number of
        // nearfield entries appropriately.
        if ( st_source->get_level( ) >= st_target->get_level( ) ) {
          if ( std::abs(
                 st_target->get_time_center( ) - st_source->get_time_center( ) )
            < st_target->get_time_half_size( ) ) {
            // source cluster's temporal component is contained in
            // target cluster's
            lo src_max_time_idx = src_mesh->get_time_element(
              st_source->get_element( n_src_elements - 1 ) );
            lo tar_max_time_idx = tar_mesh->get_time_element(
              st_target->get_element( n_tar_elements - 1 ) );
            n_nearfield_entries += n_src_elements_space * n_tar_elements_space
              * ( ( n_src_elements_time * ( n_src_elements_time + 1 ) ) / 2
                + n_src_elements_time
                  * ( tar_max_time_idx - src_max_time_idx ) );
          }
        } else if ( std::abs( st_source->get_time_center( )
                      - st_target->get_time_center( ) )
          < st_source->get_time_half_size( ) ) {
          // target cluster's temporal component is contained in source
          // cluster's
          lo src_min_time_idx
            = src_mesh->get_time_element( st_source->get_element( 0 ) );
          lo tar_min_time_idx
            = tar_mesh->get_time_element( st_target->get_element( 0 ) );
          n_nearfield_entries += n_src_elements_space * n_tar_elements_space
            * ( ( n_tar_elements_time * ( n_tar_elements_time + 1 ) ) / 2
              + n_tar_elements_time * ( tar_min_time_idx - src_min_time_idx ) );
        } else {
          n_nearfield_entries += n_src_elements_space * n_tar_elements_space
            * n_src_elements_time * n_tar_elements_time;
        }
      }
    }
  }
  lou n_global_time_elements
    = _distributed_spacetime_tree->get_mesh( ).get_n_temporal_elements( );
  lou n_global_elements
    = _distributed_spacetime_tree->get_mesh( ).get_n_elements( );
  lou n_global_space_elements = n_global_elements / n_global_time_elements;
  return n_nearfield_entries
    / ( (sc) n_global_space_elements * n_global_space_elements
      * ( n_global_time_elements * ( n_global_time_elements + 1 ) / 2 ) );
}

template< class kernel_type, class target_space, class source_space >
sc besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, target_space,
  source_space >::compute_nearfield_ratio( ) {
  lou n_nearfield_entries = 0;
  for ( auto it : _n_list ) {
    const std::vector< general_spacetime_cluster * > * st_targets
      = it->get_associated_spacetime_clusters( );
    lou n_associated_leaves = it->get_n_associated_leaves( );
    for ( lou i = 0; i < n_associated_leaves; ++i ) {
      general_spacetime_cluster * st_target = ( *st_targets )[ i ];
      lo n_target_elements = st_target->get_n_elements( );
      std::vector< general_spacetime_cluster * > * st_nearfield_list
        = st_target->get_nearfield_list( );
      for ( lou src_index = 0; src_index < st_nearfield_list->size( );
            ++src_index ) {
        general_spacetime_cluster * st_source
          = ( *st_nearfield_list )[ src_index ];
        n_nearfield_entries += n_target_elements * st_source->get_n_elements( );
      }
    }
  }
  lou n_global_elements
    = _distributed_spacetime_tree->get_mesh( ).get_n_elements( );
  return n_nearfield_entries / ( (sc) n_global_elements * n_global_elements );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::count_fmm_operations_levelwise( std::vector< lou > &
                                                    n_s2m_operations,
  std::vector< lou > & n_m2m_operations, std::vector< lou > & n_m2l_operations,
  std::vector< lou > & n_l2l_operations,
  std::vector< lou > & n_l2t_operations ) {
  lo n_max_levels = _distributed_spacetime_tree->get_max_levels( );
  // count the number of s2m operations
  n_s2m_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_s2m_operations[ i ] = 0;
  }
  for ( auto it : _m_list ) {
    if ( it->get_n_associated_leaves( ) > 0 ) {
      n_s2m_operations[ it->get_level( ) ] += it->get_n_associated_leaves( );
    }
  }

  // count the number of m2m operations
  n_m2m_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_m2m_operations[ i ] = 0;
  }
  for ( auto it : _m_list ) {
    if ( it->get_parent( )->is_active_in_upward_path( ) ) {
      n_m2m_operations[ it->get_level( ) ]
        += it->get_associated_spacetime_clusters( )->size( );
    }
  }

  // count the number of m2l. in addition, count the number of l2t
  // operations for those clusters whose parents are not active in the
  // downward path
  n_m2l_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_m2l_operations[ i ] = 0;
  }
  n_l2t_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_l2t_operations[ i ] = 0;
  }
  for ( auto it : _m2l_list ) {
    std::vector< general_spacetime_cluster * > * associated_st_targets
      = it->get_associated_spacetime_clusters( );
    for ( auto spacetime_tar : *associated_st_targets ) {
      n_m2l_operations[ it->get_level( ) ]
        += spacetime_tar->get_interaction_list( )->size( );
    }
    if ( !it->get_parent( )->is_active_in_downward_path( ) ) {
      if ( it->get_n_associated_leaves( ) > 0 ) {
        n_l2t_operations[ it->get_level( ) ] += it->get_n_associated_leaves( );
      }
    }
  }

  // count the number of l2l operations
  n_l2l_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_l2l_operations[ i ] = 0;
  }
  for ( auto it : _l_list ) {
    if ( it->get_parent( )->is_active_in_downward_path( ) ) {
      n_l2l_operations[ it->get_level( ) ]
        += it->get_associated_spacetime_clusters( )->size( );
    }
  }

  // count the number of l2t operations
  for ( auto it : _l_list ) {
    if ( it->get_n_associated_leaves( ) > 0 ) {
      n_l2t_operations[ it->get_level( ) ] += it->get_n_associated_leaves( );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::m_list_task( const block_vector & x,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  call_s2m_operations( x, current_cluster, verbose, verbose_file );
  provide_moments_for_m2l( current_cluster, verbose, verbose_file );
  call_m2m_operations( current_cluster, verbose, verbose_file );

  provide_moments_to_parents( current_cluster, verbose, verbose_file );
  reduce_nn_operations( );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::l_list_task( block_vector & y_pFMM,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  call_l2l_operations( current_cluster, verbose, verbose_file );
  // check if all the m2l operations have been executed yet
  if ( current_cluster->get_interaction_list( ) == nullptr
    || current_cluster->get_m2l_counter( )
      == current_cluster->get_interaction_list( )->size( ) ) {
    // set status of parent's local contributions to completed
    current_cluster->set_downward_path_status( 2 );
    call_l2t_operations( current_cluster, y_pFMM, verbose, verbose_file );
    provide_local_contributions_to_children(
      current_cluster, verbose, verbose_file );
  } else {
    current_cluster->set_downward_path_status( 1 );
  }
  reduce_nn_operations( );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::l_list_task( distributed_block_vector & y_pFMM,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  call_l2l_operations( current_cluster, verbose, verbose_file );
  // check if all the m2l operations have been executed yet
  if ( current_cluster->get_interaction_list( ) == nullptr
    || current_cluster->get_m2l_counter( )
      == current_cluster->get_interaction_list( )->size( ) ) {
    // set status of parent's local contributions to completed
    current_cluster->set_downward_path_status( 2 );
    call_l2t_operations( current_cluster, y_pFMM, verbose, verbose_file );
    provide_local_contributions_to_children(
      current_cluster, verbose, verbose_file );
  } else {
    current_cluster->set_downward_path_status( 1 );
  }
  reduce_nn_operations( );
  if ( _measure_tasks ) {
    _l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::m2l_list_task( block_vector & y_pFMM,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  std::vector< scheduling_time_cluster * > * ready_interaction_list
    = current_cluster->get_interaction_list( );
  //  for ( slou i = current_cluster->get_m2l_counter( );
  for ( slou i = 0; i < current_cluster->get_ready_interaction_list_size( );
        ++i ) {
    call_m2l_operations( ( *ready_interaction_list )[ i ], current_cluster,
      verbose, verbose_file );
    current_cluster->set_m2l_counter( ( slou )( i + 1 ) );
  }
  // check if all the m2l operations have been executed yet
  if ( current_cluster->get_m2l_counter( )
    == current_cluster->get_interaction_list( )->size( ) ) {
    if ( current_cluster->get_downward_path_status( ) == 1 ) {
      // set status of parent's local contributions to completed
      current_cluster->set_downward_path_status( 2 );
      call_l2t_operations( current_cluster, y_pFMM, verbose, verbose_file );
      provide_local_contributions_to_children(
        current_cluster, verbose, verbose_file );
    }
  }
  reduce_nn_operations( );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply( const distributed_block_vector & /*x*/,
  distributed_block_vector & /*y*/, bool /*trans*/, sc /*alpha*/,
  sc /*beta*/ ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

//! template specialization for double layer p0p1 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::apply( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

//! template specialization for double layer p1p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_sl_dl( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  // Specialization for the single and double layer operators
  _global_timer.reset( );
  //############################################################################
  //#### multiply the global result vector by beta ####
  y.scale( beta );

  //############################################################################
  //#### setup phase ####
  // reset the contributions of all clusters to zero
  _scheduling_tree_structure->clear_moment_contributions(
    *_scheduling_tree_structure->get_root( ) );
  _scheduling_tree_structure->clear_local_contributions(
    *_scheduling_tree_structure->get_root( ) );

  // reset the dependency data of all the clusters in the 4 lists.
  reset_scheduling_clusters_dependency_data( );

  // initialize data which is used to check for received data.
  int outcount = 0;
  std::vector< int > array_of_indices( _receive_data_information.size( ) );
  for ( lou i = 0; i < _receive_data_information.size( ); ++i ) {
    array_of_indices[ i ] = 0;
  }
  // copy the 4 FMM lists to avoid reallocating them in each application
  std::list< mesh::scheduling_time_cluster * > m_list = _m_list;
  std::list< mesh::scheduling_time_cluster * > m2l_list = _m2l_list;
  std::list< mesh::scheduling_time_cluster * > l_list = _l_list;
  std::list< mesh::scheduling_time_cluster * > n_list = _n_list;

  // std::string verbose_file = verbose_dir + "/process_";
  std::string verbose_file = "verbose/process_";
  verbose_file += std::to_string( _my_rank );
  if ( _verbose ) {
    std::filesystem::create_directory( "./verbose/" );
    // remove existing verbose file and write to new one
    remove( verbose_file.c_str( ) );
  }

  //############################################################################
  //#### distributed pFMM ####
  // allocate a global result vector. Only the entries corresponding to clusters
  // assigned to the current process are computed.
  std::vector< lo > my_blocks = y.get_my_blocks( );
  distributed_block_vector y_pFMM( my_blocks, y.get_block_size( ),
    y.get_size_of_block( ), true, MPI_COMM_WORLD );

  // auxiliary arrays for OpenMP dependencis (in OpenMP 4.5 must not be members)
  auto first = m2l_list.begin( );
  auto last = m2l_list.end( );
  char * aux_dep_m2l = new char[ std::distance( first, last ) ];
  std::fill( aux_dep_m2l, aux_dep_m2l + sizeof( aux_dep_m2l ), 0 );

  char * aux_dep_m2l_send = new char[ std::distance( first, last ) ];
  std::fill(
    aux_dep_m2l_send, aux_dep_m2l_send + sizeof( aux_dep_m2l_send ), 0 );

  first = l_list.begin( );
  last = l_list.end( );
  char * aux_dep_l = new char[ std::distance( first, last ) ];
  std::fill( aux_dep_l, aux_dep_l + sizeof( aux_dep_l ), 0 );

  first = m_list.begin( );
  last = m_list.end( );
  char * aux_dep_m = new char[ std::distance( first, last ) ];
  std::fill( aux_dep_m, aux_dep_m + sizeof( aux_dep_m ), 0 );

  // allocate buffers for m2l computation
  _aux_buffer_0.resize( omp_get_max_threads( ) );
  _aux_buffer_1.resize( omp_get_max_threads( ) );

  // reset the number of non-nearfield ops
  _non_nf_op_count = 0;

  // set loop timer start
  time_type::rep loop_start;
  if ( _measure_tasks ) {
    loop_start = _global_timer.get_time_from_start< time_type >( );
  }

  // start the main "job scheduling" algorithm
  // the "master" thread checks for new available data, spawns tasks, and
  // removes clusters from lists
#pragma omp parallel
  {
    _aux_buffer_0[ omp_get_thread_num( ) ].resize(
      ( _temp_order + 1 ) * ( _temp_order + 1 ), _spat_contribution_size );
    _aux_buffer_1[ omp_get_thread_num( ) ].resize(
      ( _temp_order + 1 ) * ( _temp_order + 1 ), _spat_contribution_size );

    _m_task_times.at( omp_get_thread_num( ) ).resize( 0 );
    _m2l_task_times.at( omp_get_thread_num( ) ).resize( 0 );
    _l_task_times.at( omp_get_thread_num( ) ).resize( 0 );
    _n_task_times.at( omp_get_thread_num( ) ).resize( 0 );
    _m_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
    _m2l_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
    _l_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
    _n_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
    _mpi_send_m2l.at( omp_get_thread_num( ) ).resize( 0 );
    _mpi_send_m_parent.at( omp_get_thread_num( ) ).resize( 0 );
    _mpi_send_l_children.at( omp_get_thread_num( ) ).resize( 0 );
    _mpi_recv_m2l.at( omp_get_thread_num( ) ).resize( 0 );
    _mpi_recv_m_parent.at( omp_get_thread_num( ) ).resize( 0 );
    _mpi_recv_l_children.at( omp_get_thread_num( ) ).resize( 0 );

#pragma omp single
    {
      // start the receive operationss
      std::vector< MPI_Request > array_of_requests(
        _receive_data_information.size( ) );
      start_receive_operations( array_of_requests );
      while ( true ) {
        if ( m_list.empty( ) && m2l_list.empty( ) && l_list.empty( )
          && n_list.empty( ) ) {
          break;
        }

        // check if data has been received since the last iteration
        if ( outcount != MPI_UNDEFINED ) {
          check_for_received_data(
            array_of_requests, array_of_indices, outcount );
        }

        // we have to do this here to spawn tasks with correct dependencies
        if ( outcount != MPI_UNDEFINED && outcount > 0 ) {
          for ( lo i = 0; i < outcount; ++i ) {
            lou current_index = array_of_indices[ i ];
            scheduling_time_cluster * current_cluster
              = _receive_data_information[ current_index ].first;
            if ( _verbose ) {
#pragma omp critical( verbose )
              {
                std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
                if ( outfile.is_open( ) ) {
                  outfile << "received data of cluster "
                          << current_cluster->get_global_index( )
                          << " at level " << current_cluster->get_level( )
                          << " from process "
                          << _receive_data_information[ current_index ].second
                          << std::endl;
                  outfile.close( );
                }
              }
            }
            // distinguish which data has been received
            if ( current_index < _n_moments_to_receive_upward ) {
              if ( _measure_tasks ) {
                _mpi_recv_m_parent.at( omp_get_thread_num( ) )
                  .push_back(
                    _global_timer.get_time_from_start< time_type >( ) );
              }

              // received data are moments in the upward path. add up
              // moments and update dependencies.
              lo idx = current_cluster->get_pos_in_m_list( );

              // task depends on previously generated M-list tasks to prevent
              // collision in m2m operations
#pragma omp task depend( inout : aux_dep_m [idx:1] ) priority( 1000 )
              upward_path_task( current_index, current_cluster );
            } else if ( current_index
              < _n_moments_to_receive_upward + _n_moments_to_receive_m2l ) {
              if ( _measure_tasks ) {
                _mpi_recv_m2l.at( omp_get_thread_num( ) )
                  .push_back(
                    _global_timer.get_time_from_start< time_type >( ) );
              }
              // received data are moments for m2l. update dependencies.
              std::vector< scheduling_time_cluster * > * send_list
                = current_cluster->get_send_list( );
              if ( send_list != nullptr ) {
                for ( auto it = send_list->begin( ); it != send_list->end( );
                      ++it ) {
                  lo tar_process_id = ( *it )->get_process_id( );
                  if ( tar_process_id == _my_rank ) {
                    // lo idx_receiver = ( *it )->get_pos_in_m_list( );
                    lo idx_receiver = ( *it )->get_pos_in_m2l_list( );
                    // task depends on previously generated m2l-list task to
                    // avoid collision when adding to ready interaction list
#pragma omp task depend( inout                                 \
                         : aux_dep_m2l_send [idx_receiver:1] ) \
  priority( 1000 )
                    ( *it )->update_ready_interaction_size( );
                  }
                }
              }
            } else {
              if ( _measure_tasks ) {
                _mpi_recv_l_children.at( omp_get_thread_num( ) )
                  .push_back(
                    _global_timer.get_time_from_start< time_type >( ) );
              }
              // received data are local contributions. update dependencies.
#pragma omp task priority( 1000 )
              current_cluster->set_downward_path_status( 2 );
            }
          }
        }

        // check if there is a cluster in one of the 4 lists whose operations
        // are ready to be executed.
        char status = 0;
        std::list< scheduling_time_cluster * >::iterator it_current_cluster;
        find_cluster_in_m_list( m_list, it_current_cluster, status );
        if ( status == 0 ) {
          // search in l list if no cluster was found yet
          find_cluster_in_l_list( l_list, it_current_cluster, status );
          if ( status == 0 ) {
            // search in m2l list if no cluster was found yet
            find_cluster_in_m2l_list( m2l_list, it_current_cluster, status );
            if ( status == 0 ) {
              // take the first cluster from the n-list if it is not empty
              if ( n_list.size( ) != 0 && get_nn_operations( ) < 1 ) {
                it_current_cluster = n_list.begin( );
                status = 4;
              }
            } else {
              add_nn_operations( );
            }
          } else {
            add_nn_operations( );
          }
        } else {
          add_nn_operations( );
        }

        // interrupt the scheduling task if there is enough work to do so it can
        // join the remaining tasks
        if ( get_nn_operations( ) > 0 && status == 0 ) {
#pragma omp taskyield
        }

        // if verbose mode is chosen, write info about next operation to file
        if ( _verbose && status != 0 ) {
#pragma omp critical( verbose )
          {
            std::stringstream outss;
            outss << "executing ";
            switch ( status ) {
              case 1:
                outss << "m-list operations ";
                break;
              case 2:
                outss << "l-list operations ";
                break;
              case 3:
                outss << "m2l-list operations ";
                break;
              case 4:
                outss << "n-list operations ";
                break;
            }
            outss << "for cluster "
                  << ( *it_current_cluster )->get_global_index( )
                  << " at level " << ( *it_current_cluster )->get_level( );
            std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
            if ( outfile.is_open( ) ) {
              outfile << outss.str( ) << std::endl;
              outfile.close( );
            }
          }
        }

        scheduling_time_cluster * current_cluster = *it_current_cluster;

        // start the appropriate fmm operations according to status
        switch ( status ) {
          case 1: {
            // M-list task
            m_list.erase( it_current_cluster );

            lo idx_m_parent
              = current_cluster->get_parent( )->get_pos_in_m_list( );
            lo idx_m = current_cluster->get_pos_in_m_list( );

            std::vector< scheduling_time_cluster * > * send_list
              = current_cluster->get_send_list( );
            lo send_list_size = 0;
            if ( send_list != nullptr ) {
              send_list_size = send_list->size( );
            }
            if ( idx_m_parent != -1 ) {
              // m-list task depends on previously generated tasks
              // with the same parent (so the parent's m2m operations do not
              // collide)
              switch ( send_list_size ) {
                // m-list task depends on previously generated tasks with the
                // same cluster in the send-list (so the receiving does not
                // collide)
                case 0: {
#pragma omp task depend( inout                                               \
                         : aux_dep_m [idx_m_parent:1], aux_dep_m [idx_m:1] ) \
  priority( 500 )
                  m_list_task( x, current_cluster, _verbose, verbose_file );
                  break;
                }
                case 1: {
                  lo idx_receiver_1
                    = send_list->at( 0 )->get_pos_in_m2l_list( );
#pragma omp task depend( inout                                               \
                         : aux_dep_m [idx_m_parent:1], aux_dep_m [idx_m:1] ) \
  depend( inout                                                              \
          : aux_dep_m2l_send [idx_receiver_1:1] ) priority( 500 )
                  m_list_task( x, current_cluster, _verbose, verbose_file );
                  break;
                }
                case 2: {
                  lo idx_receiver_1
                    = send_list->at( 0 )->get_pos_in_m2l_list( );
                  lo idx_receiver_2
                    = send_list->at( 1 )->get_pos_in_m2l_list( );

#pragma omp task depend( inout                                               \
                         : aux_dep_m [idx_m_parent:1], aux_dep_m [idx_m:1] ) \
  depend( inout                                                              \
          : aux_dep_m2l_send [idx_receiver_1:1] )                            \
    depend( inout                                                            \
            : aux_dep_m2l_send[ idx_receiver_2 ] ) priority( 500 )
                  m_list_task( x, current_cluster, _verbose, verbose_file );
                  break;
                }
              }
            } else {
              // parent is not in the m list, no dependency needed
              switch ( send_list->size( ) ) {
                // m-list task depends on previously generated tasks with the
                // same cluster in the send-list (so the receiver's m2l
                // operations do not collide)
                case 0: {
#pragma omp task depend( inout : aux_dep_m [idx_m:1] ) priority( 500 )
                  m_list_task( x, current_cluster, _verbose, verbose_file );
                  break;
                }
                case 1: {
                  lo idx_receiver_1
                    = send_list->at( 0 )->get_pos_in_m2l_list( );

#pragma omp task depend( \
  inout                  \
  : aux_dep_m2l_send [idx_receiver_1:1], aux_dep_m [idx_m:1] ) priority( 500 )
                  m_list_task( x, current_cluster, _verbose, verbose_file );
                  break;
                }
                case 2: {
                  lo idx_receiver_1
                    = send_list->at( 0 )->get_pos_in_m2l_list( );
                  lo idx_receiver_2
                    = send_list->at( 1 )->get_pos_in_m2l_list( );

#pragma omp task depend(                                       \
  inout                                                        \
  : aux_dep_m2l_send [idx_receiver_1:1], aux_dep_m [idx_m:1] ) \
  depend( inout                                                \
          : aux_dep_m2l_send [idx_receiver_2:1] ) priority( 500 )
                  m_list_task( x, current_cluster, _verbose, verbose_file );
                  break;
                }
              }
            }
            break;
          }

          case 2: {
            // L-list task
            lo idx_l = current_cluster->get_pos_in_l_list( );
            l_list.erase( it_current_cluster );
            // l-list task depends on previously generated m2l tasks processing
            // the same cluster in the l-list (to prevent collision with m2l
            // tasks)
#pragma omp task depend( inout : aux_dep_l [idx_l:1] ) priority( 400 )
            l_list_task( y_pFMM, current_cluster, _verbose, verbose_file );
            break;
          }
          case 3: {
            // M2l-list task

            lo idx_l = current_cluster->get_pos_in_l_list( );
            lou ready_int_list_size
              = current_cluster->get_ready_interaction_list_size( );
            if ( ready_int_list_size
              == current_cluster->get_interaction_list( )->size( ) ) {
              m2l_list.erase( it_current_cluster );
            }
            if ( idx_l == -1 ) {
              // cluster is not in the L-list, have to depend only on the
              // previously generated tasks processing the same cluster in the
              // m2l list
#pragma omp task priority( 300 )
              m2l_list_task( y_pFMM, current_cluster, _verbose, verbose_file );
            } else {
// cluster depends additionally on the previously generated task
// with the same position in the L-list
#pragma omp task depend( inout : aux_dep_l [idx_l:1] ) priority( 300 )
              m2l_list_task( y_pFMM, current_cluster, _verbose, verbose_file );
            }
            break;
          }
          case 4: {
            // nearfiel task
            n_list.erase( it_current_cluster );
// no dependencies, possible collisions are treated by atomic
// operations
#pragma omp task priority( 200 )
            apply_nearfield_operations(
              current_cluster, x, trans, y_pFMM, _verbose, verbose_file );
            break;
          }
        }
        // @todo: is the following output of a new line useful at all?
        //         if ( verbose ) {
        // #pragma omp critical( verbose )
        //           {
        //             std::ofstream outfile( verbose_file.c_str( ),
        //             std::ios::app ); if ( outfile.is_open( ) ) {
        //               outfile << std::endl;
        //               outfile.close( );
        //             }
        //           }
        //         }
      }
    }
  }
  // set loop timer end
  time_type::rep loop_end;
  if ( _measure_tasks ) {
    loop_end = _global_timer.get_time_from_start< time_type >( );
  }

  y.add( y_pFMM, alpha );

  MPI_Barrier( y.get_comm( ) );
  y.synchronize_shared_parts( );

  delete[] aux_dep_m;
  delete[] aux_dep_l;
  delete[] aux_dep_m2l;
  delete[] aux_dep_m2l_send;

  // print out task timing
  if ( _measure_tasks ) {
    save_times( loop_end - loop_start,
      _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::upward_path_task( lou current_index,
  mesh::scheduling_time_cluster * current_cluster ) const {
  lo source_id = _receive_data_information[ current_index ].second;
  sc * current_moments = current_cluster->get_associated_moments( );
  sc * received_moments
    = current_cluster->get_extraneous_moment_pointer( source_id );
  lou n_associated_spacetime_clusters
    = current_cluster->get_associated_spacetime_clusters( )->size( );
  for ( lou j = 0; j < n_associated_spacetime_clusters * _contribution_size;
        ++j ) {
    current_moments[ j ] += received_moments[ j ];
  }
  current_cluster->reduce_upward_path_counter( );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::m_list_task( const distributed_block_vector & x,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _m_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  call_s2m_operations( x, current_cluster, verbose, verbose_file );
  provide_moments_for_m2l( current_cluster, verbose, verbose_file );
  call_m2m_operations( current_cluster, verbose, verbose_file );

  provide_moments_to_parents( current_cluster, verbose, verbose_file );
  reduce_nn_operations( );
  if ( _measure_tasks ) {
    _m_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_s2m_operations( const distributed_block_vector & sources,
  besthea::mesh::scheduling_time_cluster * time_cluster, bool verbose,
  const std::string & verbose_file ) const {
  // execute only for associated spacetime leaves
  if ( time_cluster->get_n_associated_leaves( ) > 0 ) {
    if ( verbose ) {
#pragma omp critical( verbose )
      {
        std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "call S2M for cluster "
                  << time_cluster->get_global_index( ) << " at level "
                  << time_cluster->get_level( ) << std::endl;
          outfile.close( );
        }
      }
    }
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = time_cluster->get_associated_spacetime_clusters( );
    // there is an implicit taskgroup after this taskloop
#pragma omp taskloop shared( sources )
    for ( lou i = 0; i < time_cluster->get_n_associated_leaves( ); ++i ) {
      if ( _measure_tasks ) {
        _m_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      general_spacetime_cluster * current_cluster
        = ( *associated_spacetime_clusters )[ i ];

      apply_s2m_operation( sources, current_cluster );
      if ( _measure_tasks ) {
        _m_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2m_operation( const distributed_block_vector &
  /*source_vector*/,
  general_spacetime_cluster * /*source_cluster*/ ) const {
  std::cout << "S2M operation not implemented " << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_s2m_operation( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p0( source_vector, source_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2m_operation( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operations_p1_normal_drv( source_vector, source_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_s2m_operation( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p0( source_vector, source_cluster );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2m_operation_p0( const distributed_block_vector &
                                            source_vector,
  general_spacetime_cluster * source_cluster ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  lo n_space_elements = source_cluster->get_n_space_elements( );
  full_matrix sources( n_time_elements, n_space_elements, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references of current moment and all required matrices
  sc * moment = source_cluster->get_pointer_to_moment( );

  full_matrix T;
  compute_chebyshev_quadrature_p0( source_cluster, T );
  full_matrix L;
  compute_lagrange_quadrature( source_cluster, L );

  // get the relevant entries of the block vector x and store them in sources
  const std::vector< lo > & spacetime_elements
    = source_cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = source_cluster->get_mesh( );
  // a cluster for which an S2M operation is executed is always local!
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );
  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component to
    // determine the local time index only once
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      lo global_space_index = local_mesh->get_space_element(
        distributed_mesh->global_2_local( local_start_idx,
          spacetime_elements[ i_time * n_space_elements + i_space ] ) );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      sources( i_time, i_space )
        = source_vector.get( distributed_mesh->local_2_global_time(
                               local_start_idx, local_time_index ),
          global_space_index );
    }
  }
  // compute D = Q * T and then the moment mu = L * D
  aux_matrix.multiply( sources, T );
  // mu = L * D with explicit cblas routine call
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_cols_aux_matrix = aux_matrix.get_n_columns( );
  lo n_rows_aux_matrix = aux_matrix.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_aux_matrix;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange,
    n_cols_aux_matrix, n_rows_aux_matrix, alpha, L.data( ), lda,
    aux_matrix.data( ), ldb, beta, moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_s2m_operations_p1_normal_drv(
    const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  lo n_space_elements = source_cluster->get_n_space_elements( );
  lo n_space_nodes = source_cluster->get_n_space_nodes( );
  full_matrix sources( n_time_elements, n_space_nodes, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references of current moment and all required matrices
  sc * moment = source_cluster->get_pointer_to_moment( );
  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1( source_cluster, T_drv );
  full_matrix L;
  compute_lagrange_quadrature( source_cluster, L );

  // get the relevant entries of the block vector x and store them in sources
  const std::vector< lo > & spacetime_elements
    = source_cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = source_cluster->get_mesh( );
  // a cluster for which an S2M operation is executed is always local!
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  const std::vector< lo > & local_2_global_nodes
    = source_cluster->get_local_2_global_nodes( );
  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component to
    // determine the local time index only once
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      // local_2_global_nodes gives the indices of the spacetime nodes. take
      // the rest from division by the number of global spatial nodes to get
      // the spatial node index
      lo global_space_index
        = local_2_global_nodes[ i_space ] % local_mesh->get_n_spatial_nodes( );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      sources( i_time, i_space )
        = source_vector.get( distributed_mesh->local_2_global_time(
                               local_start_idx, local_time_index ),
          global_space_index );
    }
  }
  // compute D = Q * T_drv and then the moment mu = L * D
  aux_matrix.multiply( sources, T_drv );
  // mu = L * D with explicit cblas routine call
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_cols_aux_matrix = aux_matrix.get_n_columns( );
  lo n_rows_aux_matrix = aux_matrix.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_aux_matrix;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange,
    n_cols_aux_matrix, n_rows_aux_matrix, alpha, L.data( ), lda,
    aux_matrix.data( ), ldb, beta, moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_l2t_operation( const mesh::general_spacetime_cluster *
  /*cluster*/,
  distributed_block_vector & /*output_vector*/ ) const {
  std::cout << "L2T operation not implemented!" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_l2t_operation( const mesh::general_spacetime_cluster * cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p0( cluster, output_vector );
}

//! template specialization for double layer p0p1 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation( const mesh::general_spacetime_cluster * cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p0( cluster, output_vector );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_l2t_operation( const mesh::general_spacetime_cluster * cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p1_normal_drv( cluster, output_vector );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p0( const mesh::general_spacetime_cluster * cluster,
    distributed_block_vector & output_vector ) const {
  lo n_time_elements = cluster->get_n_time_elements( );
  lo n_space_elements = cluster->get_n_space_elements( );
  full_matrix targets( n_time_elements, n_space_elements, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution = cluster->get_pointer_to_local_contribution( );

  full_matrix T;
  compute_chebyshev_quadrature_p0( cluster, T );
  full_matrix L;
  compute_lagrange_quadrature( cluster, L );

  // compute D = trans(L) * lambda and then the result Y = D * trans(T)
  //  D = trans(L) * lambda with explicit cblas routine call:
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo n_rows_lagrange = L.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_lagrange,
    n_cols_local, n_rows_lagrange, alpha, L.data( ), lda, local_contribution,
    ldb, beta, aux_matrix.data( ), n_cols_lagrange );
  // compute Y = D * trans(T)
  targets.multiply( aux_matrix, T, false, true );

  // add the results to the correct positions of the output vector
  const std::vector< lo > & spacetime_elements = cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = cluster->get_mesh( );
  // a cluster for which an S2M operation is executed is always local!
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      lo global_space_index = local_mesh->get_space_element(
        distributed_mesh->global_2_local( local_start_idx,
          spacetime_elements[ i_time * n_space_elements + i_space ] ) );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.

      output_vector.add_atomic( distributed_mesh->local_2_global_time(
                                  local_start_idx, local_time_index ),
        global_space_index, targets( i_time, i_space ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * cluster,
    distributed_block_vector & output_vector ) const {
  lo n_time_elements = cluster->get_n_time_elements( );
  lo n_space_elements = cluster->get_n_space_elements( );
  lo n_space_nodes = cluster->get_n_space_nodes( );
  full_matrix targets( n_time_elements, n_space_nodes, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution = cluster->get_pointer_to_local_contribution( );
  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1( cluster, T_drv );
  full_matrix L;
  compute_lagrange_quadrature( cluster, L );

  // compute D = trans(L) * lambda and then the result Y = D * trans(T)
  //  D = trans(L) * lambda with explicit cblas routine call:
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo n_rows_lagrange = L.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_lagrange,
    n_cols_local, n_rows_lagrange, alpha, L.data( ), lda, local_contribution,
    ldb, beta, aux_matrix.data( ), n_cols_lagrange );
  // compute Y = D * trans(T)
  targets.multiply( aux_matrix, T_drv, false, true );

  // add the results to the correct positions of the output vector
  const std::vector< lo > & spacetime_elements = cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = cluster->get_mesh( );
  // a cluster for which an S2M operation is executed is always local!
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  const std::vector< lo > & local_2_global_nodes
    = cluster->get_local_2_global_nodes( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      // local_2_global_nodes gives the indices of the spacetime nodes. take
      // the rest from division by the number of global spatial nodes to get
      // the spatial node index
      lo global_space_index
        = local_2_global_nodes[ i_space ] % local_mesh->get_n_spatial_nodes( );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      output_vector.add_atomic( distributed_mesh->local_2_global_time(
                                  local_start_idx, local_time_index ),
        global_space_index, targets( i_time, i_space ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::m2l_list_task( distributed_block_vector &
                                                 y_pFMM,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _m2l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  std::vector< scheduling_time_cluster * > * ready_interaction_list
    = current_cluster->get_interaction_list( );
  //  for ( slou i = current_cluster->get_m2l_counter( );
  for ( slou i = 0; i < current_cluster->get_ready_interaction_list_size( );
        ++i ) {
    call_m2l_operations( ( *ready_interaction_list )[ i ], current_cluster,
      verbose, verbose_file );
    current_cluster->set_m2l_counter( ( slou )( i + 1 ) );
  }
  // check if all the m2l operations have been executed yet
  if ( current_cluster->get_m2l_counter( )
    == current_cluster->get_interaction_list( )->size( ) ) {
    if ( current_cluster->get_downward_path_status( ) == 1 ) {
      // set status of parent's local contributions to completed
      current_cluster->set_downward_path_status( 2 );
      call_l2t_operations( current_cluster, y_pFMM, verbose, verbose_file );
      provide_local_contributions_to_children(
        current_cluster, verbose, verbose_file );
    }
  }
  reduce_nn_operations( );
  if ( _measure_tasks ) {
    _m2l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_nearfield_operations( const mesh::scheduling_time_cluster * cluster,
    const distributed_block_vector & sources, bool trans,
    distributed_block_vector & output_vector, bool verbose,
    const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _n_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  if ( verbose ) {
#pragma omp critical( verbose )
    {
      std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "apply NF operations for cluster "
                << cluster->get_global_index( ) << " at level "
                << cluster->get_level( ) << std::endl;
        outfile.close( );
      }
    }
  }
  vector_type local_sources;
  const std::vector< general_spacetime_cluster * > *
    associated_spacetime_targets
    = cluster->get_associated_spacetime_clusters( );
  lou n_associated_leaves = cluster->get_n_associated_leaves( );
  // there is an implicit taskgroup after this taskloop
#pragma omp taskloop shared( output_vector, _clusterwise_nearfield_matrices )
  for ( lou i = 0; i < n_associated_leaves; ++i ) {
    if ( _measure_tasks ) {
      _n_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
    general_spacetime_cluster * current_spacetime_target
      = ( *associated_spacetime_targets )[ i ];
    // construct a local result_vector
    vector_type local_result(
      current_spacetime_target->get_n_dofs< target_space >( ), true );
    // get the nearfield list of the current spacetime target cluster and
    // apply the nearfield operations for all the clusters in this list.
    std::vector< general_spacetime_cluster * > * spacetime_nearfield_list
      = current_spacetime_target->get_nearfield_list( );
    for ( lou src_index = 0; src_index < spacetime_nearfield_list->size( );
          ++src_index ) {
      general_spacetime_cluster * current_spacetime_source
        = ( *spacetime_nearfield_list )[ src_index ];
      local_sources.resize(
        current_spacetime_source->get_n_dofs< source_space >( ) );
      // get the sources corresponding to the current spacetime source cluster
      sources.get_local_part< source_space >(
        current_spacetime_source, local_sources );

      full_matrix * current_block = _clusterwise_nearfield_matrices.at(
        current_spacetime_target )[ src_index ];
      // apply the nearfield matrix and add the result to local_result
      current_block->apply( local_sources, local_result, trans, 1.0, 1.0 );
    }
    // add the local result to the output vector
    output_vector.add_local_part< target_space >(
      current_spacetime_target, local_result );
    if ( _measure_tasks ) {
      _n_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
  }
  if ( _measure_tasks ) {
    _n_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::save_times( time_type::rep total_loop_duration,
  time_type::rep total_apply_duration ) const {
  std::filesystem::create_directory( "./task_timer/" );

  std::string timer_file = "task_timer/process_";
  timer_file += std::to_string( _my_rank );
  timer_file += ".m";
  remove( timer_file.c_str( ) );

  std::ofstream outfile( timer_file.c_str( ), std::ios::app );

  outfile << "% Total apply duration [us]: " << std::endl;
  outfile << "T = " << total_apply_duration << ";" << std::endl;

  if ( outfile.is_open( ) ) {
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      // compute thread total execution time in individual tasks
      time_type::rep us_m_sub = 0;
      time_type::rep us_m2l_sub = 0;
      time_type::rep us_l_sub = 0;
      time_type::rep us_n_sub = 0;
      time_type::rep total_time = 0;
      for ( std::size_t j = 0; j < _m_subtask_times.at( i ).size( ) / 2; ++j ) {
        us_m_sub += _m_subtask_times.at( i ).at( 2 * j + 1 )
          - _m_subtask_times.at( i ).at( 2 * j );
      }
      for ( std::size_t j = 0; j < _m2l_subtask_times.at( i ).size( ) / 2;
            ++j ) {
        us_m2l_sub += _m2l_subtask_times.at( i ).at( 2 * j + 1 )
          - _m2l_subtask_times.at( i ).at( 2 * j );
      }
      for ( std::size_t j = 0; j < _l_subtask_times.at( i ).size( ) / 2; ++j ) {
        us_l_sub += _l_subtask_times.at( i ).at( 2 * j + 1 )
          - _l_subtask_times.at( i ).at( 2 * j );
      }
      for ( std::size_t j = 0; j < _n_subtask_times.at( i ).size( ) / 2; ++j ) {
        us_n_sub += _n_subtask_times.at( i ).at( 2 * j + 1 )
          - _n_subtask_times.at( i ).at( 2 * j );
      }
      total_time = us_m_sub + us_m2l_sub + us_l_sub + us_n_sub;
      double perc_subtasks_apply
        = (double) total_time / (double) total_apply_duration;
      double perc_subtasks_loop
        = (double) total_time / (double) total_loop_duration;

      outfile << "% Thread " << i << ": " << std::endl;

      outfile << "% M subtasks duration: " << us_m_sub << " us" << std::endl;
      outfile << "% M2L subtasks duration: " << us_m2l_sub << " us"
              << std::endl;
      outfile << "% L subtasks duration: " << us_l_sub << " us" << std::endl;
      outfile << "% N subtasks duration: " << us_n_sub << " us" << std::endl;
      outfile << "% Sum: " << us_m_sub + us_m2l_sub + us_l_sub + us_n_sub
              << " us (" << perc_subtasks_loop * 100.0 << " % [loop], "
              << perc_subtasks_apply * 100.0 << " % [total])\n\n";

      // output main tasks
      outfile << "% M tasks: " << std::endl;
      outfile << "M" << i << " = [";
      auto it = _m_task_times.at( i ).begin( );
      for ( ; it != _m_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl;

      outfile << "% M2L tasks: " << std::endl;
      outfile << "M2L" << i << " = [";
      it = _m2l_task_times.at( i ).begin( );
      for ( ; it != _m2l_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl;

      outfile << std::endl << std::endl;
      outfile << "% L tasks: " << std::endl;
      outfile << "L" << i << " = [";
      it = _l_task_times.at( i ).begin( );
      for ( ; it != _l_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% N tasks: " << std::endl;
      outfile << "N" << i << " = [";
      it = _n_task_times.at( i ).begin( );
      for ( ; it != _n_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      // output subtasks
      outfile << "% M subtasks: " << std::endl;
      outfile << "Ms" << i << " = [";
      it = _m_subtask_times.at( i ).begin( );
      for ( ; it != _m_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% M2L subtasks: " << std::endl;
      it = _m2l_subtask_times.at( i ).begin( );
      outfile << "M2Ls" << i << " = [";
      for ( ; it != _m2l_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% L subtasks: " << std::endl;
      it = _l_subtask_times.at( i ).begin( );
      outfile << "Ls" << i << " = [";
      for ( ; it != _l_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% N subtasks: " << std::endl;
      it = _n_subtask_times.at( i ).begin( );
      outfile << "Ns" << i << " = [";
      for ( ; it != _n_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      // output MPI communication
      outfile << "% M2L send: " << std::endl;
      it = _mpi_send_m2l.at( i ).begin( );
      outfile << "M2L_send" << i << " = [";
      for ( ; it != _mpi_send_m2l.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% parent send: " << std::endl;
      it = _mpi_send_m_parent.at( i ).begin( );
      outfile << "parent_send" << i << " = [";
      for ( ; it != _mpi_send_m_parent.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% local send: " << std::endl;
      it = _mpi_send_l_children.at( i ).begin( );
      outfile << "local_send" << i << " = [";
      for ( ; it != _mpi_send_l_children.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% M2L receive: " << std::endl;
      it = _mpi_recv_m2l.at( i ).begin( );
      outfile << "M2L_recv" << i << " = [";
      for ( ; it != _mpi_recv_m2l.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% parent receive: " << std::endl;
      it = _mpi_recv_m_parent.at( i ).begin( );
      outfile << "parent_recv" << i << " = [";
      for ( ; it != _mpi_recv_m_parent.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% local receive: " << std::endl;
      it = _mpi_recv_l_children.at( i ).begin( );
      outfile << "local_recv" << i << " = [";
      for ( ; it != _mpi_recv_l_children.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    }
    outfile << "M = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2L = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2L" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "L = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "L" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "N = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "N" << i << ", ";
    }
    outfile << "};" << std::endl;

    outfile << "Ms = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "Ms" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2Ls = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2Ls" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "Ls = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "Ls" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "Ns = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "Ns" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2L_send = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2L_send" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "parent_send = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "parent_send" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "local_send = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "local_send" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2L_recv = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2L_recv" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "parent_recv = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "parent_recv" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "local_recv = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "local_recv" << i << ", ";
    }
    outfile << "};" << std::endl;

    outfile.close( );
  }
}

template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >;

template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;

template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >;
