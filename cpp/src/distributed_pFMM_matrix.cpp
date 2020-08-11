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

#include "besthea/blas_lapack_wrapper.h"
#include "besthea/distributed_pFMM_matrix.h"
#include "besthea/quadrature.h"

using besthea::mesh::distributed_spacetime_cluster_tree;
using besthea::linear_algebra::full_matrix;
using besthea::mesh::general_spacetime_cluster;
using besthea::mesh::scheduling_time_cluster;

#include <sstream> 

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::apply( const block_vector & x, 
  block_vector & y, bool trans, sc alpha, sc beta ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply( const block_vector & x, block_vector & y, bool trans,
    sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}


template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::apply_sl_dl( const block_vector & x,
  block_vector & y, bool trans, sc alpha, sc beta ) const {
  // Specialization for the single and double layer operators

  // @todo should the multiplication be done like this or only in a local part
  // of the result vector?
  // #pragma omp parallel for schedule( static )
  for ( lo i = 0; i < y.get_block_size( ); ++i ) {
    for ( lo j = 0; j < y.get_size_of_block( ); ++j ) {
      y.set( i, j, y.get( i, j ) * beta );
    }
  }

  //reset the contributions of all clusters to zero
  _scheduling_tree_structure->clear_moment_contributions( 
    *_scheduling_tree_structure->get_root( ) );
  _scheduling_tree_structure->clear_local_contributions(
    *_scheduling_tree_structure->get_root( ) );

  // Next, use the pFMM for the computation of the farfield contribution
  // according to the respective spaces. The result is stored in an auxiliary
  // vector y_pFMM and then added to y

  // allocate buffers for the operations
  std::vector< full_matrix > buffer_matrices;
  buffer_matrices.resize( 8 );
  for ( auto it = buffer_matrices.begin( ); it != buffer_matrices.end( );
        ++it ) {
    ( *it ).resize( _temp_order + 1,
      ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  }
  vector_type buffer_for_gaussians( ( _m2l_integration_order + 1 )
      * ( _m2l_integration_order + 1 ) * ( _temp_order + 1 )
      * ( _temp_order + 1 ),
    false );
  vector_type buffer_for_coeffs( ( _spat_order + 1 ) * ( _spat_order + 1 )
      * ( _temp_order + 1 ) * ( _temp_order + 1 ),
    false );
  full_matrix aux_buffer_0( ( _temp_order + 1 ) * ( _temp_order + 1 ),
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  full_matrix aux_buffer_1( ( _temp_order + 1 ) * ( _temp_order + 1 ),
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  block_vector y_pFMM( y.get_block_size( ), y.get_size_of_block( ), true );

  // @todo: add appropriate verbose mode if necessary
  bool verbose = false;
  // std::string verbose_file = verbose_dir + "/process_";
  std::string verbose_file = "verbose/process_";
  verbose_file += std::to_string( _my_rank );
  if ( verbose ) {
    // remove existing verbose file and write to new one
    remove( verbose_file.c_str( ) ); 
  }
  
  // start the receive operations
  MPI_Request array_of_requests[ _receive_data_information.size( ) ];
  start_receive_operations( array_of_requests );
  
  // initialize data which is used to check for received data.
  int outcount = 0;
  int array_of_indices[ _receive_data_information.size( ) ];
  for ( lou i = 0; i < _receive_data_information.size( ); ++i ) {
    array_of_indices[ i ] = 0;
  }

  // copy the 4 FMM lists to avoid reallocating them in each application
  std::list< mesh::scheduling_time_cluster* > m_list = _m_list;
  std::list< mesh::scheduling_time_cluster* > m2l_list = _m2l_list;
  std::list< mesh::scheduling_time_cluster* > l_list = _l_list;
  std::list< mesh::scheduling_time_cluster* > n_list = _n_list;

  while ( !m_list.empty( ) || !m2l_list.empty( ) || !l_list.empty( ) 
          || !n_list.empty( ) ) {
    if ( outcount != MPI_UNDEFINED ) {
      check_for_received_data( array_of_requests, array_of_indices, outcount, 
        verbose, verbose_file );
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
    // if verbose mode is chosen, write info about next operation to file
    if ( verbose && status != 0 ) {
      std::stringstream outss;
      outss << "executing ";
      switch (status)
      {
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
      outss << "for cluster " << ( *it_current_cluster )->get_global_index( );
      std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << outss.str( ) << std::endl;
        outfile.close( );
      }
    }
    // start the appropriate fmm operations according to status
    // TODO: the switch routine could be avoided by using several else 
    // statements above. Switch is used for better clarity.
    switch ( status ) {
      case 1: {
        // apply S2M operations for leaf clusters
        call_s2m_operations( 
          x, *it_current_cluster, verbose, verbose_file );
        // update dependencies of targets in send list if they are local or send
        // data for interactions 
        // provide_moments_for_m2l( communicator, *it_current_cluster );
        // apply M2M operations 
        // call_m2m_operations( *it_current_cluster, verbose, verbose_file );       
        // update dependencies of parent if it is handled by the same process or
        // send data to other process if not
        // provide_moments_to_parents( communicator, *it_current_cluster );
        // remove cluster from the list
        m_list.erase( it_current_cluster );
        break;
      }
  //     case 2: {
  //       // apply L2L operations
  //       call_l2l_operations( *it_current_cluster, verbose, verbose_file );
  //       if ( ( *it_current_cluster )->get_interaction_list( ) == nullptr ||
  //            ( *it_current_cluster )->get_m2l_counter( ) 
  //             == ( *it_current_cluster )->get_interaction_list( )->size( ) ) {
  //         // set status of parent's local contributions to completed
  //         ( *it_current_cluster )->set_downward_path_status( 2 );
  //         // apply L2T operations
  //         call_l2t_operations( 
  //           *it_current_cluster, output_vector, verbose, verbose_file );
  //         // send data to all other processes which handle a child
  //         provide_local_contributions_to_children( 
  //           communicator, *it_current_cluster );
  //       } else {
  //         ( *it_current_cluster )->set_downward_path_status( 1 );
  //       }
  //       l_list.erase( it_current_cluster );
  //       break; 
  //     }
  //     case 3: {
  //       // std::cout << "applying m2l-list operation" << std::endl;
  //       std::vector< scheduling_time_cluster* > * ready_interaction_list
  //         = ( *it_current_cluster )->get_ready_interaction_list( );
  //       for ( lou i = ( *it_current_cluster )->get_m2l_counter( ); 
  //             i < ready_interaction_list->size( ); ++i ) {
  //         call_m2l_operations( ( 
  //           *ready_interaction_list )[ i ], *it_current_cluster, verbose, 
  //           verbose_file );
  //         ( *it_current_cluster )->increase_m2l_counter( );
  //       }
  //       if ( ( *it_current_cluster )->get_m2l_counter( ) ==
  //           ( *it_current_cluster )->get_interaction_list( )->size( ) ) {
  //         if ( ( *it_current_cluster )->get_downward_path_status( ) == 1 ) {
  //           // set status of parent's local contributions to completed
  //           ( *it_current_cluster )->set_downward_path_status( 2 );
  //           // apply L2T operations
  //           call_l2t_operations( *it_current_cluster, output_vector, verbose,
  //             verbose_file );
  //           // send data to all other processes which handle a child
  //           provide_local_contributions_to_children( 
  //             communicator, *it_current_cluster );
  //         }
  //         m2l_list.erase( it_current_cluster );
  //       }
  //       break;
  //     }
  //     case 4: {
  //       // std::cout << "applying n-list operation" << std::endl;
  //       std::vector< scheduling_time_cluster* > * nearfield 
  //         = ( *it_current_cluster )->get_nearfield_list( );
  //       for ( auto it = nearfield->begin( ); it != nearfield->end( ); ++it ) {
  //         call_nearfield_operations(
  //           input_vector, *it, *it_current_cluster, output_vector, verbose,
  //           verbose_file );
  //       }
  //       n_list.erase( it_current_cluster );
  //       break;
  //     }
  //   }
  //   if ( verbose ) {
  //     std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
  //     if ( outfile.is_open( ) ) {
  //       outfile << std::endl;
  //       outfile.close( );
  //     }
    }
  }
  // Add the scaled result to y.
  y.add( y_pFMM, alpha );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::set_trees( 
  distributed_spacetime_cluster_tree * distributed_spacetime_tree ) {
  _distributed_spacetime_tree = distributed_spacetime_tree;
  _scheduling_tree_structure 
    = _distributed_spacetime_tree->get_distribution_tree( );
  const std::vector< general_spacetime_cluster * > & local_leaves
    = distributed_spacetime_tree->get_local_leaves( );
  _clusterwise_nearfield_matrices.resize( local_leaves.size( ) );
  for ( lou i = 0; i < _clusterwise_nearfield_matrices.size( ); ++i ) {
    _clusterwise_nearfield_matrices[ i ].resize(
      local_leaves[ i ]->get_nearfield_list( )->size( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::prepare_fmm(  ) {
  _scheduling_tree_structure->init_fmm_lists_and_dependency_data( 
    *_scheduling_tree_structure->get_root( ), 
    _m_list, _m2l_list, _l_list, _n_list );
  // sort the m_list from bottom up, right to left
  _m_list.sort( 
    _scheduling_tree_structure->compare_clusters_bottom_up_right_2_left );
  _m2l_list.sort( 
    _scheduling_tree_structure->compare_clusters_top_down_right_2_left );
  _l_list.sort( 
    _scheduling_tree_structure->compare_clusters_top_down_right_2_left );

  // fill the receive list by determining all incoming data.
  // check for receive operations in the upward path
  for ( auto it = _m_list.begin( ); it != _m_list.end( ); ++it ) {
    std::vector< scheduling_time_cluster* > * children
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
    std::vector< scheduling_time_cluster* > * interaction_list 
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
    [ & ]( const std::pair< scheduling_time_cluster*, lo > pair_one,
        const std::pair< scheduling_time_cluster*, lo > pair_two ) {
          return _scheduling_tree_structure->
            compare_clusters_top_down_right_2_left( 
            pair_one.first, pair_two.first );
        } );
  auto new_end 
    = std::unique( _receive_data_information.begin( ) 
      + _n_moments_to_receive_upward, _receive_data_information.end( ),
      []( const std::pair< scheduling_time_cluster*, lo > pair_one,
          const std::pair< scheduling_time_cluster*, lo > pair_two ) {
            return pair_one.first == pair_two.first;
          } );
  _receive_data_information.resize( 
    std::distance( _receive_data_information.begin(), new_end ) );
  _n_moments_to_receive_m2l 
    = _receive_data_information.size( ) - _n_moments_to_receive_upward;
  // check for receive operations in the downward path
  for ( auto it = _l_list.begin( ); it != _l_list.end( ); ++it ) {
    scheduling_time_cluster * parent = ( *it )->get_parent( );
    // if the parent cluster is handled by a different process p the current 
    // process has to receive its local contributions from p.
    if ( parent->get_process_id( ) != _my_rank && 
         parent->get_process_id( ) != -1 ) {
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
  full_matrix * local_matrix
    = new full_matrix( n_dofs_target, n_dofs_source );

  _clusterwise_nearfield_matrices[ leaf_index ][ source_index ] = local_matrix;

  return local_matrix;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::call_s2m_operations( 
    const block_vector & sources, 
    besthea::mesh::scheduling_time_cluster* time_cluster, bool verbose, 
    std::string verbose_file ) const {
  // execute only for associated spacetime leaves
  if ( time_cluster->get_n_associated_leaves( ) > 0 ) {
    if ( verbose ) {
      std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "call S2M for cluster " << time_cluster->get_global_index( )
                << std::endl; 
        outfile.close( );
      }
    }
    std::vector< general_spacetime_cluster* >* associated_spacetime_clusters 
      = time_cluster->get_associated_spacetime_clusters( );
    for ( lou i = 0; i < time_cluster->get_n_associated_leaves( ); ++i ) {
      apply_s2m_operation( sources, (* associated_spacetime_clusters )[ i ] );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::apply_s2m_operation( 
    const block_vector & source_vector, 
    general_spacetime_cluster* source_cluster ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  lo n_space_elements = source_cluster->get_n_space_elements( );
  full_matrix sources( n_time_elements, n_space_elements, false );
  full_matrix aux_matrix( n_time_elements,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );

  // get references of current moment and all required matrices
  sc* moment = source_cluster->get_pointer_to_moment( );
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
      lo global_space_index 
        = local_mesh->get_space_element( distributed_mesh->global_2_local(
          local_start_idx, spacetime_elements[ i_time * n_space_elements
                                                + i_space ] ) );
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
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange, 
    n_cols_aux_matrix, n_rows_aux_matrix, 1.0, L.data( ), lda, 
    aux_matrix.data( ), ldb, 0.0, moment, n_rows_lagrange );
}

// void call_m2m_operations( scheduling_time_cluster* time_cluster, bool verbose, 
//   std::string verbose_file ) {
//   // add child moment to parent moment
//   sc parent_moment = time_cluster->get_parent( )->get_moment( );
//   time_cluster->get_parent( )->set_moment( 
//     parent_moment + time_cluster->get_moment( ) );
//   // artificial wait to simulate real operation
//   lo level = time_cluster->get_level( );
//   lou wait;
//   if ( level % 2 ) {
//     wait = ( lou ) N_CLUSTERS_SPACE_INIT 
//             * std::pow( N_CLUSTERS_SPACE_REF, level / 2 ) * DEG_TIME_P1
//             * DEG_TIME_P1 * SPACE_COEFFS * EST_TIME_MUL;
//   } else {
//     wait = ( lou ) N_CLUSTERS_SPACE_INIT 
//             * std::pow( N_CLUSTERS_SPACE_REF, level / 2 ) * EST_TIME_MUL 
//             * ( DEG_TIME_P1 * DEG_TIME_P1 * SPACE_COEFFS 
//               + 3 * N_CLUSTERS_SPACE_REF * DEG_TIME_P1 * SPACE_COEFFS
//                 * DEG_SPACE / 4 );
//   }
//   if ( verbose ) {
//     std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
//     if ( outfile.is_open( ) ) {
//       outfile << "call M2M: waiting " << wait * 1e-9 << " seconds"
//               << std::endl; 
//       outfile.close( );
//     }
//   }
//   std::this_thread::sleep_for(std::chrono::nanoseconds( wait ));
// }

// void call_m2l_operations( scheduling_time_cluster* src_cluster,
//   scheduling_time_cluster* tar_cluster, bool verbose, 
//   std::string verbose_file ) {
//   // add moment of src_cluster to local contribution of tar_cluster
//   sc tar_local = tar_cluster->get_local_contribution( );
//   tar_cluster->set_local_contribution( tar_local + src_cluster->get_moment( ) );
//   // artificial wait to simulate real operation
//   lo level = tar_cluster->get_level( );
//   lou n_clusters_at_level = ( lou ) N_CLUSTERS_SPACE_INIT 
//             * std::pow( N_CLUSTERS_SPACE_REF, level / 2 );
//   lou n_est_max_interactions = ( lou ) N_CLUSTERS_SPACE_REF / 8.0
//             * std::pow( ( N_IA_BOX + 1 ), 3 );
//   lou n_est_interact = std::min( n_clusters_at_level, n_est_max_interactions );
//   lou wait = n_clusters_at_level * n_est_interact * DEG_TIME_P1 * DEG_TIME_P1
//             * SPACE_COEFFS * ( DEG_SPACE + 2 ) * 1.5 * EST_TIME_MUL;
//   if ( verbose ) {
//     std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
//     if ( outfile.is_open( ) ) {
//       outfile << "call M2L: waiting " << wait * 1e-9 << " seconds"
//               << std::endl; 
//       outfile.close( );
//     }
//   }
//   std::this_thread::sleep_for(std::chrono::nanoseconds( wait ));
// }

// void call_l2l_operations( scheduling_time_cluster* time_cluster, bool verbose, 
//   std::string verbose_file ) {
//   // add local contribution of the parent of the time cluster to its own
//   sc child_local = time_cluster->get_local_contribution( );
//   time_cluster->set_local_contribution( 
//     child_local + time_cluster->get_parent( )->get_local_contribution( ) );
//   // artificial wait to simulate real operation
//   lo level = time_cluster->get_level( );
//   lou wait;
//   if ( level % 2 ) {
//     wait = ( lou ) N_CLUSTERS_SPACE_INIT 
//             * std::pow( N_CLUSTERS_SPACE_REF, level / 2 ) * DEG_TIME_P1
//             * DEG_TIME_P1 * SPACE_COEFFS * EST_TIME_MUL;
//   } else {
//     wait = ( lou ) N_CLUSTERS_SPACE_INIT 
//             * std::pow( N_CLUSTERS_SPACE_REF, level / 2 ) * EST_TIME_MUL 
//             * ( DEG_TIME_P1 * DEG_TIME_P1 * SPACE_COEFFS 
//               + 3 * N_CLUSTERS_SPACE_REF * DEG_TIME_P1 * SPACE_COEFFS
//                 * DEG_SPACE / 4 );
//   }
//   if ( verbose ) {
//     std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
//     if ( outfile.is_open( ) ) {
//       outfile << "call L2L: waiting " << wait * 1e-9 << " seconds"
//               << std::endl; 
//       outfile.close( );
//     }
//   }
//   std::this_thread::sleep_for(std::chrono::nanoseconds( wait ));
// }

// void call_l2t_operations( scheduling_time_cluster* time_cluster, 
//   std::vector< sc > & output_vector, bool verbose, std::string verbose_file ) {
//   // execute only for leaf clusters
//   if ( time_cluster->get_n_children( ) == 0 ) {
//     // subtract local contribution from correct position of output vector
//     lo cluster_index = time_cluster->get_leaf_index( );
//     output_vector[ cluster_index ] -= time_cluster->get_local_contribution( );
//     // artificial wait to simulate real operation
//     lo level = time_cluster->get_level( );
//     lou wait = ( lou ) N_CLUSTERS_SPACE_INIT 
//                 * std::pow( N_CLUSTERS_SPACE_REF, level / 2 ) * N_ELEMENTS_LEAF
//                 * DEG_TIME_P1 * SPACE_COEFFS * EFFORT_INT_FMM * EST_TIME_MUL;
//     std::this_thread::sleep_for(std::chrono::nanoseconds( wait ));
//     if ( verbose ) {
//       std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
//       if ( outfile.is_open( ) ) {
//         outfile << "call L2T: waiting " << wait * 1e-9 << " seconds"
//                 << std::endl; 
//         outfile.close( );
//       }
//     }
//   }
// }

// void call_nearfield_operations( const std::vector< sc > & sources,
//   scheduling_time_cluster* src_cluster, scheduling_time_cluster* tar_cluster, 
//   std::vector< sc > & output_vector, bool verbose, std::string verbose_file ) {
//   lo tar_ind = tar_cluster->get_leaf_index( );
//   lo src_ind = src_cluster->get_leaf_index( );
//   output_vector[ tar_ind ] -= sources[ src_ind ];
//   // artificial wait to simulate real operation
//   lo level = tar_cluster->get_level( );
//   lou n_clusters_at_level = ( lou ) N_CLUSTERS_SPACE_INIT 
//             * std::pow( N_CLUSTERS_SPACE_REF, level / 2 );
//   lou n_est_max_interactions = ( lou ) N_CLUSTERS_SPACE_REF / 8.0
//             * std::pow( ( N_IA_BOX + 1 ), 3 );
//   lou n_est_interact = std::min( n_clusters_at_level, n_est_max_interactions );
//   lou wait = n_clusters_at_level * n_est_interact * N_ELEMENTS_LEAF 
//             * N_ELEMENTS_LEAF * EFFORT_INT_NF * EST_TIME_MUL;
//   if ( verbose ) {
//     std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
//     if ( outfile.is_open( ) ) {
//       outfile << "call NF: waiting " << wait * 1e-9 << " seconds"
//               << std::endl; 
//       outfile.close( );
//     }
//   }
//   std::this_thread::sleep_for(std::chrono::nanoseconds( wait ));
// }

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::check_for_received_data( 
    MPI_Request * array_of_requests, int array_of_indices[ ], int & outcount, 
    bool verbose, std::string verbose_file ) const {
  MPI_Testsome( _receive_data_information.size( ), array_of_requests, &outcount, 
    array_of_indices, MPI_STATUSES_IGNORE );
  if ( outcount != MPI_UNDEFINED && outcount > 0 ) {
    for ( lo i = 0; i < outcount; ++i ) {
      lou current_index = array_of_indices[ i ];
      scheduling_time_cluster* current_cluster 
        = _receive_data_information[ current_index ].first;
      if ( verbose ) {
        std::ofstream outfile ( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "received data of cluster " 
                  << current_cluster->get_global_index( ) << " from process " 
                  << _receive_data_information[ current_index ].second
                  << std::endl; 
          outfile.close( );
        }
      }
      // distinguish which data has been received
      if ( current_index < _n_moments_to_receive_upward ) {
        // received data are moments in the upward path. add up 
        // moments and update dependencies.
        lo source_id = _receive_data_information[ current_index ].second;
        sc* current_moments = current_cluster->get_associated_moments( );
        sc* received_moments 
          = current_cluster->get_extraneous_moment_pointer( source_id );
        lou buffer_size 
          = current_cluster->get_associated_spacetime_clusters( )->size( );
        for ( lou i = 0; i < buffer_size; ++ i ) {
          current_moments[ i ] += received_moments[ i ];
        }
        current_cluster->reduce_upward_path_counter( );
      }
      else if ( current_index < 
                _n_moments_to_receive_upward + _n_moments_to_receive_m2l ) {
        // received data are moments for m2l. update dependencies.
        std::vector< scheduling_time_cluster* > * send_list 
          = current_cluster->get_send_list( );
        if ( send_list != nullptr ) {
          for ( auto it = send_list->begin( ); it != send_list->end( ); ++it ) {
            lo tar_process_id = ( *it )->get_process_id( );
            if ( tar_process_id == _my_rank ) {
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

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::find_cluster_in_l_list( 
    std::list< scheduling_time_cluster* > & l_list, 
    std::list< scheduling_time_cluster* >::iterator & it_next_cluster,
  char & status ) const {
  it_next_cluster = l_list.begin( );
  // a cluster is ready if its parents downward path status equals 2
  while ( status != 2 && it_next_cluster != l_list.end( ) ) {
    if ( ( *it_next_cluster )->get_parent( )->get_downward_path_status( ) == 2 )
      status = 2;
    else
      ++ it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::find_cluster_in_m_list( 
    std::list< scheduling_time_cluster* > & m_list, 
    std::list< scheduling_time_cluster* >::iterator & it_next_cluster,
    char & status ) const {
  it_next_cluster = m_list.begin( );
  // a cluster is ready if its upward path counter equals 0
  while ( status != 1 && it_next_cluster != m_list.end( ) ) {
    if ( ( *it_next_cluster )->get_upward_path_counter( ) == 0 )
      status = 1;
    else
      ++ it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::find_cluster_in_m2l_list( 
    std::list< mesh::scheduling_time_cluster* > & m2l_list,
    std::list< mesh::scheduling_time_cluster* >::iterator & it_next_cluster, 
    char & status ) const {
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

// void provide_moments_for_m2l( MPI_Comm communicator, 
//   scheduling_time_cluster* src_cluster ) {
//   int my_process_id;
//   MPI_Comm_rank( communicator, &my_process_id );
//   std::vector< scheduling_time_cluster* > * send_list 
//     = src_cluster->get_send_list( );
//   if ( send_list != nullptr ) {
//     for ( auto it = send_list->begin( ); it != send_list->end( ); ++it ) {
//       lo tar_process_id = ( *it )->get_process_id( );
//       if ( tar_process_id == my_process_id ) {
//         ( *it )->add_to_ready_interaction_list( src_cluster );
//       } else {
//         lo tag = 2 * src_cluster->get_global_index( );
//         sc* moment_buffer = src_cluster->get_moment_pointer( );
//         MPI_Request req;
//         MPI_Isend( moment_buffer, 1, MPI_DOUBLE, tar_process_id, tag, 
//           communicator, &req );
//         MPI_Request_free( &req ); 
//       } 
//     } 
//   }
// }

// void provide_moments_to_parents( const MPI_Comm communicator, 
//   scheduling_time_cluster* child_cluster ) {
//   int my_process_id;
//   MPI_Comm_rank( communicator, &my_process_id );
//   scheduling_time_cluster* parent_cluster = child_cluster->get_parent( );
//   lo parent_process_id = parent_cluster->get_process_id( );
//   if ( parent_process_id == my_process_id ) {
//     parent_cluster->reduce_upward_path_counter( );
//   } 
//   else if ( parent_process_id != -1 )  {
//     lo tag = 2 * parent_cluster->get_global_index( );
//     sc* moment_buffer = parent_cluster->get_moment_pointer( );
//     MPI_Request req;
//     MPI_Isend( moment_buffer, 1, MPI_DOUBLE, parent_process_id, tag, 
//       communicator, &req );
//     MPI_Request_free( &req );
//   }
// } 

// void provide_local_contributions_to_children( const MPI_Comm communicator, 
//   scheduling_time_cluster* parent_cluster ) {
//   int my_process_id;
//   MPI_Comm_rank( communicator, &my_process_id );
//   std::vector< scheduling_time_cluster* > * children 
//     = parent_cluster->get_children( );
//   if ( children != nullptr ) {
//     for ( auto it = children->begin( ); it != children->end( ); ++it ) {
//       lo child_process_id = ( *it )->get_process_id( );
//       if ( child_process_id != my_process_id ) {
//         lo tag = 2 * parent_cluster->get_global_index( ) + 1;
//         sc* local_contribution_buffer 
//           = parent_cluster->get_local_contribution_pointer( );
//         MPI_Request req;
//         MPI_Isend( local_contribution_buffer, 1, MPI_DOUBLE, child_process_id, 
//           tag, communicator, &req );
//         MPI_Request_free( &req );
//       }
//     }
//   }
// }

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, 
  target_space, source_space >::start_receive_operations( 
    MPI_Request array_of_requests[ ] ) const {
  // start the receive operations for the moments in the upward path
  for ( lou i = 0; i < _n_moments_to_receive_upward; ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    scheduling_time_cluster* receive_cluster
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( );
    sc* moment_buffer 
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
    scheduling_time_cluster* receive_cluster 
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( );
    sc* moment_buffer = receive_cluster->get_associated_moments( );
    int buffer_size 
      = receive_cluster->get_associated_spacetime_clusters( )->size( )
      * _contribution_size;
    MPI_Irecv( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ), 
                source_id, tag, *_comm, &array_of_requests[ i ] );
    // std::cout << "source: " << source_id << ", tag: " << tag << std::endl;
  }

  // start the receive operations for the local contributions
  // std::cout << "receive operations for local contributions: " << std::endl;
  for ( lou i =  _n_moments_to_receive_upward + _n_moments_to_receive_m2l; 
        i < _receive_data_information.size( ); ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    scheduling_time_cluster* receive_cluster 
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( ) + 1;
    sc* local_contribution_buffer 
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
  target_space, source_space >::compute_chebyshev_quadrature_p0( 
    general_spacetime_cluster* source_cluster,
    full_matrix & T ) const {

  lo n_space_elems = source_cluster->get_n_space_elements( );
  T.resize( n_space_elems,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  // get some info on the current cluster
  vector_type cluster_center_space( 3 );
  vector_type cluster_half_space( 3 );
  sc dummy;
  source_cluster->get_center( cluster_center_space, dummy );
  source_cluster->get_half_size( cluster_half_space, dummy );
  sc padding = 0;
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
    // particular we get all spatial elements in the cluster by considering the
    // first n_space_elems spacetime elements.
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
  target_space, source_space >::compute_lagrange_quadrature(
    mesh::general_spacetime_cluster* source_cluster, 
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

  sc cluster_t_start
    = source_cluster->get_time_center( ) 
    - source_cluster->get_time_half_size( );
  sc cluster_t_end
    = source_cluster->get_time_center( ) 
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
    // coordinates with respect to the time cluster and transform them to [-1,1]
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
  target_space, source_space >::init_quadrature_polynomials( 
    quadrature_wrapper & my_quadrature ) const {
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
  target_space, source_space >::triangle_to_geometry( 
    const linear_algebra::coordinates< 3 > & x1,
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
  // y%_mapped are the %th components of the vectors to which y#_ref is mapped
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
  target_space, source_space >::cluster_to_polynomials( 
    quadrature_wrapper & my_quadrature, sc start_0, sc end_0, 
    sc start_1, sc end_1, sc start_2, sc end_2 ) const {
  for ( lo i = 0; i < my_quadrature._y1_polynomial.size( ); ++i ) {
    my_quadrature._y1_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y1[ i ] - start_0 ) / ( end_0 - start_0 );
    my_quadrature._y2_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y2[ i ] - start_1 ) / ( end_1 - start_1 );
    my_quadrature._y3_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y3[ i ] - start_2 ) / ( end_2 - start_2 );
  }
}

template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space< 
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space< 
    besthea::bem::basis_tri_p0 > >;