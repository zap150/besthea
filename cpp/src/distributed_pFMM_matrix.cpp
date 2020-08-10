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

#include "besthea/distributed_pFMM_matrix.h"

using besthea::mesh::distributed_spacetime_cluster_tree;
using besthea::mesh::general_spacetime_cluster;
using besthea::mesh::scheduling_time_cluster;

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
  full_matrix_type * local_matrix
    = new full_matrix_type( n_dofs_target, n_dofs_source );

  _clusterwise_nearfield_matrices[ leaf_index ][ source_index ] = local_matrix;

  return local_matrix;
}





template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space< 
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space< 
    besthea::bem::basis_tri_p0 > >;