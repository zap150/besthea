/*
Copyright (c) 2021, VSB - Technical University of Ostrava and Graz University of
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

#include "besthea/distributed_initial_pFMM_matrix.h"

#include "besthea/fmm_routines.h"
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
using besthea::mesh::volume_space_cluster;

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::determine_interacting_time_clusters( scheduling_time_cluster &
    current_cluster ) {
  if ( current_cluster.get_status_in_initial_op_downward_path( ) > 0 ) {
    if ( current_cluster.get_configuration( ) == 1 ) {
      _time_clusters_for_m2l.push_back( &current_cluster );
    } else {
      if ( current_cluster.get_process_id( ) == _my_rank
        && current_cluster.get_n_associated_leaves( ) > 0 ) {
        _time_clusters_for_nf.push_back( &current_cluster );
      }
      // call the routine recursively for all children, if current_cluster is
      // not a leaf
      if ( current_cluster.get_n_children( ) > 0 ) {
        for ( auto child : *current_cluster.get_children( ) ) {
          determine_interacting_time_clusters( *child );
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::initialize_nearfield_and_interaction_lists( ) {
  // Initialize first the nearfield lists vector: determine all nearfields of
  // space-time leaf clusters associated with temporal clusters in the list
  // _time_clusters_for_nf.
  for ( auto nf_time_cluster : _time_clusters_for_nf ) {
    lo n_associated_leaves = nf_time_cluster->get_n_associated_leaves( );
    const std::vector< general_spacetime_cluster * > * associated_clusters
      = nf_time_cluster->get_associated_spacetime_clusters( );
    for ( lo i = 0; i < n_associated_leaves; ++i ) {
      general_spacetime_cluster * spacetime_leaf
        = ( *associated_clusters )[ i ];
      std::vector< slou > cluster_coords
        = spacetime_leaf->get_box_coordinate( );
      std::vector< slou > spatial_grid_coords
        = { cluster_coords[ 1 ], cluster_coords[ 2 ], cluster_coords[ 3 ] };
      std::vector< mesh::volume_space_cluster * > neighbors;
      lo spat_level, dummy;
      spacetime_leaf->get_n_divs( spat_level, dummy );
      slou spatial_nearfield_limit
        = _distributed_spacetime_target_tree->get_spatial_nearfield_limit( );
      _space_source_tree->find_neighbors(
        spat_level, spatial_grid_coords, spatial_nearfield_limit, neighbors );
      // check if the determined nearfield clusters are leaves. If they are not,
      // add their leaf descendants to the nearfield list of the current target
      // cluster
      std::vector< mesh::volume_space_cluster * > neighboring_leaves;
      for ( auto neighbor : neighbors ) {
        if ( neighbor->get_n_children( ) > 0 ) {
          _space_source_tree->collect_leaf_descendants(
            *neighbor, neighboring_leaves );
        } else {
          neighboring_leaves.push_back( neighbor );
        }
      }
      if ( neighboring_leaves.size( ) > 0 ) {
        neighboring_leaves.shrink_to_fit( );
        _nearfield_list_vector.push_back(
          { spacetime_leaf, neighboring_leaves } );
      }
    }
  }
  // Next, initialize the interaction list vectors by traversing similarly the
  // space-time clusters associated with temporal clusters in the list
  // _time_clusters_for_m2l (and update the nearfield list vector when
  // encountering exceptional situations)
  for ( auto m2l_time_cluster : _time_clusters_for_m2l ) {
    const std::vector< general_spacetime_cluster * > * associated_clusters
      = m2l_time_cluster->get_associated_spacetime_clusters( );
    lo n_relevant_st_clusters
      = m2l_time_cluster->get_n_st_clusters_w_local_contributions( );
    // by construction, the clusters with local contributions are first in the
    // list of all associated space-time clusters.
    for ( lo i = 0; i < n_relevant_st_clusters; ++i ) {
      general_spacetime_cluster * st_cluster = ( *associated_clusters )[ i ];
      std::vector< slou > cluster_coords = st_cluster->get_box_coordinate( );
      std::vector< slou > spatial_grid_coords
        = { cluster_coords[ 1 ], cluster_coords[ 2 ], cluster_coords[ 3 ] };
      std::vector< volume_space_cluster * > neighbors;
      lo spat_level, dummy;
      st_cluster->get_n_divs( spat_level, dummy );
      slou spatial_nearfield_limit
        = _distributed_spacetime_target_tree->get_spatial_nearfield_limit( );
      _space_source_tree->find_neighbors(
        spat_level, spatial_grid_coords, spatial_nearfield_limit, neighbors );
      std::vector< volume_space_cluster * > clusters_requiring_nearfield_ops;
      // If neighboring volume cluster at lower levels exist, they are
      // positioned at the end of the vector neighbors. For such clusters
      // nearfield operations have to be executed.
      auto it = neighbors.rbegin( );
      while ( it != neighbors.rend( ) && ( *it )->get_level( ) < spat_level ) {
        clusters_requiring_nearfield_ops.push_back( *it );
        ++it;
        // remove the detected early volume leaf cluster from the neighbors
        neighbors.pop_back( );
      }
      if ( neighbors.size( ) > 0 ) {
        neighbors.shrink_to_fit( );
        _interaction_list_vector.push_back( { st_cluster, neighbors } );
      }
      if ( clusters_requiring_nearfield_ops.size( ) > 0 ) {
        clusters_requiring_nearfield_ops.shrink_to_fit( );
        // get all leaf descendants of the current st_cluster
        std::vector< general_spacetime_cluster * > local_leaf_descendants;
        _distributed_spacetime_target_tree->collect_local_leaves(
          *st_cluster, local_leaf_descendants );
        // Add all leaf descendants with the corresponding list of nearfield
        // volume clusters to the nearfield list vector.
        // Note: The newly added space-time leaves are guaranteed to be
        // different from the ones already in the list by construction.
        for ( auto st_leaf : local_leaf_descendants ) {
          _nearfield_list_vector.push_back(
            { st_leaf, clusters_requiring_nearfield_ops } );
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::resize_nearfield_matrix_container( ) {
  // resize the container for the clusterwise nearfield matrices
  _clusterwise_nf_matrices.resize( _nearfield_list_vector.size( ) );
  for ( lou i = 0; i < _nearfield_list_vector.size( ); ++i ) {
    _clusterwise_nf_matrices[ i ].resize(
      _nearfield_list_vector[ i ].second.size( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_all_nearfield_operations( const vector &
                                                                  sources,
  distributed_block_vector & output_vector ) const {
  vector_type local_sources;
  for ( lou i_tar = 0; i_tar < _nearfield_list_vector.size( ); ++i_tar ) {
    const mesh::general_spacetime_cluster * st_tar_cluster
      = _nearfield_list_vector[ i_tar ].first;
    // initialize a local vector where the results of the nearfield operations
    // for the current cluster are stored.
    vector_type local_result(
      st_tar_cluster->get_n_dofs< target_space >( ), true );
    vector_type local_sources;
    for ( lou i_src = 0; i_src < _nearfield_list_vector[ i_tar ].second.size( );
          ++i_src ) {
      const mesh::volume_space_cluster * space_src_cluster
        = _nearfield_list_vector[ i_tar ].second[ i_src ];
      local_sources.resize( space_src_cluster->get_n_dofs< source_space >( ) );
      // get the sources corresponding to the current spacetime source
      // cluster
      sources.get_local_part< source_space >(
        space_src_cluster, local_sources );

      full_matrix * current_block = _clusterwise_nf_matrices[ i_tar ][ i_src ];
      // apply the nearfield matrix and add the result to local_result
      current_block->apply( local_sources, local_result, false, 1.0, 1.0 );
    }
    // add the local result to the output vector
    output_vector.add_local_part< target_space >(
      st_tar_cluster, local_result );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_moments_upward_path( const vector &
                                                               sources,
  mesh::volume_space_cluster * current_cluster ) const {
  if ( current_cluster->get_n_children( ) > 0 ) {
    const std::vector< volume_space_cluster * > * children
      = current_cluster->get_children( );
    for ( auto child : *children ) {
      compute_moments_upward_path( sources, child );
    }
    apply_grouped_m2m_operation( current_cluster );
  } else {
    apply_s2m_operation( sources, current_cluster );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_all_m2l_operations( ) const {
  for ( lou i = 0; i < _interaction_list_vector.size( ); ++i ) {
    mesh::general_spacetime_cluster * st_target
      = _interaction_list_vector[ i ].first;
    for ( auto space_source_cluster : _interaction_list_vector[ i ].second ) {
      apply_m2l_operation( space_source_cluster, st_target );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  evaluate_local_contributions_downward_path(
    mesh::scheduling_time_cluster * current_cluster,
    distributed_block_vector & output_vector ) const {
  char downpard_path_status
    = current_cluster->get_status_in_initial_op_downward_path( );
  if ( downpard_path_status == 1 ) {
    const mesh::scheduling_time_cluster * current_parent
      = current_cluster->get_parent( );
    if ( current_parent != nullptr
      && ( current_parent->get_status_in_initial_op_downward_path( ) == 1 ) ) {
      call_l2l_operations( current_cluster );
    }
    call_l2t_operations( current_cluster, output_vector );
  }
  if ( downpard_path_status > 0 && current_cluster->get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster->get_children( ) ) {
      evaluate_local_contributions_downward_path( child, output_vector );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_s2m_operation( const vector & sources,
  mesh::volume_space_cluster * leaf ) const {
  full_matrix T_vol;
  compute_chebyshev_quadrature_p1_volume( T_vol, leaf );
  vector & moments = leaf->get_moments( );
  vector sources_in_leaf;
  sources.get_local_part< source_space >( leaf, sources_in_leaf );
  T_vol.apply( sources_in_leaf, moments );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_grouped_m2m_operation( mesh::volume_space_cluster *
    parent_cluster ) const {
  const std::vector< mesh::volume_space_cluster * > * children
    = parent_cluster->get_children( );
  // declare auxiliary vectors lambda_1/2 to store intermediate results in m2m
  // operations
  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  vector_type lambda_1( n_coeffs_s, false );
  vector_type lambda_2( n_coeffs_s, false );

  lo parent_level = parent_cluster->get_level( );
  vector_type & parent_moment = parent_cluster->get_moments( );
  for ( auto child : *children ) {
    // execute the m2m operation for each child
    short child_octant = child->get_octant( );
    const vector_type & child_moment = child->get_moments( );

    const vector_type * m2m_coeffs_s_dim_0;
    const vector_type * m2m_coeffs_s_dim_1;
    const vector_type * m2m_coeffs_s_dim_2;
    switch ( child_octant ) {
      case 0:
        m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_right[ parent_level ] );
        m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_right[ parent_level ] );
        m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_right[ parent_level ] );
        break;
      case 1:
        m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_left[ parent_level ] );
        m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_right[ parent_level ] );
        m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_right[ parent_level ] );
        break;
      case 2:
        m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_left[ parent_level ] );
        m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_left[ parent_level ] );
        m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_right[ parent_level ] );
        break;
      case 3:
        m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_right[ parent_level ] );
        m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_left[ parent_level ] );
        m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_right[ parent_level ] );
        break;
      case 4:
        m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_right[ parent_level ] );
        m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_right[ parent_level ] );
        m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_left[ parent_level ] );
        break;
      case 5:
        m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_left[ parent_level ] );
        m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_right[ parent_level ] );
        m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_left[ parent_level ] );
        break;
      case 6:
        m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_left[ parent_level ] );
        m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_left[ parent_level ] );
        m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_left[ parent_level ] );
        break;
      case 7:
        m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_right[ parent_level ] );
        m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_left[ parent_level ] );
        m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_left[ parent_level ] );
        break;
      default:  // default case should never be used, program will crash!
        m2m_coeffs_s_dim_0 = nullptr;
        m2m_coeffs_s_dim_1 = nullptr;
        m2m_coeffs_s_dim_2 = nullptr;
    }

    lambda_1.fill( 0.0 );
    lambda_2.fill( 0.0 );

    for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
      lo child_index = 0;
      for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
        for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0;
              ++alpha1 ) {
          lo alpha2;
          for ( alpha2 = 0; alpha2 <= beta2; ++alpha2 ) {
            lambda_1[ ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
              + ( _spat_order + 1 ) * alpha0 + alpha1 ]
              += ( *m2m_coeffs_s_dim_2 )[ beta2 * ( _spat_order + 1 ) + alpha2 ]
              * child_moment[ child_index ];
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

    // compute intermediate result lambda_1 not exploiting zero entries for the
    // sake of better readability
    for ( lo beta1 = 0; beta1 <= _spat_order; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta1; ++beta2 ) {
        for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
          for ( lo alpha1 = 0; alpha1 <= beta1; ++alpha1 ) {
            lambda_2[ ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
              + ( _spat_order + 1 ) * beta2 + alpha0 ]
              += ( *m2m_coeffs_s_dim_1 )[ beta1 * ( _spat_order + 1 ) + alpha1 ]
              * lambda_1[ ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 ];
          }
        }
      }
    }

    lo parent_moment_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2;
                ++alpha0 ) {
            parent_moment[ parent_moment_index ]
              += ( *m2m_coeffs_s_dim_0 )[ beta0 * ( _spat_order + 1 ) + alpha0 ]
              * lambda_2[ ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                + ( _spat_order + 1 ) * beta2 + alpha0 ];
          }
          ++parent_moment_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_m2l_operation( const mesh::volume_space_cluster *
                                         s_src_cluster,
  mesh::general_spacetime_cluster * st_tar_cluster ) const {
  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2l coefficients
  vector_type buffer_for_gaussians(
    ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  // buffer to store m2l coefficients.
  vector_type buffer_for_coeffs(
    ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  // buffer matrices to store intermediate m2l results.
  lo thread_num = omp_get_thread_num( );

  _aux_buffer_0[ thread_num ].fill( 0.0 );
  _aux_buffer_1[ thread_num ].fill( 0.0 );

  // get geometrical data of the clusters
  vector_type half_size_space( 3, false );
  s_src_cluster->get_half_size( half_size_space );
  sc tar_half_size_time = st_tar_cluster->get_time_half_size( );
  sc tar_center_time;
  vector_type src_center_space( 3, false );
  vector_type tar_center_space( 3, false );
  s_src_cluster->get_center( src_center_space );
  st_tar_cluster->get_center( tar_center_space, tar_center_time );

  // initialize temporal interpolation nodes in target cluster
  vector_type tar_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    tar_time_nodes[ i ]
      = tar_center_time + tar_half_size_time * time_nodes[ i ];
  }

  // get spatial properties ( difference of cluster, half length )
  vector_type center_diff_space( tar_center_space );
  for ( lo i = 0; i < 3; ++i ) {
    center_diff_space[ i ] -= src_center_space[ i ];
  }

  sc padding_space = _maximal_spatial_paddings[ s_src_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // compute coupling coefficients for dimension 2
  compute_coupling_coeffs_initial_op( tar_time_nodes, half_size_space[ 2 ],
    center_diff_space[ 2 ], buffer_for_gaussians, buffer_for_coeffs );

  const vector_type & src_moment = s_src_cluster->get_moments( );
  sc * tar_local = st_tar_cluster->get_pointer_to_local_contribution( );
  // efficient m2l operation similar to Tausch, 2009, p. 3558
  // help variables for accessing right values in coefficient buffer
  lo hlp_acs_alpha = ( _spat_order + 1 ) * ( _temp_order + 1 );
  lo hlp_acs_beta = ( _temp_order + 1 );

  sc * aux_buffer_0_data = _aux_buffer_0[ thread_num ].data( );
  sc * buffer_for_coeffs_data = buffer_for_coeffs.data( );

  // compute first intermediate product and store it in aux_buffer_0
  lo buffer_0_index = 0;
  for ( lo alpha2 = 0; alpha2 <= _spat_order; ++alpha2 ) {
    lo moment_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order - alpha2; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - alpha2 - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          // no need for reduction, in a single inner cycle data are written
          // on unique positions
          for ( lo a = 0; a <= _temp_order; ++a ) {
            aux_buffer_0_data[ buffer_0_index * hlp_acs_beta + a ]
              += buffer_for_coeffs_data[ alpha2 * hlp_acs_alpha
                   + beta2 * hlp_acs_beta + a ]
              * src_moment[ moment_index ];
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
  compute_coupling_coeffs_initial_op( tar_time_nodes, half_size_space[ 1 ],
    center_diff_space[ 1 ], buffer_for_gaussians, buffer_for_coeffs );

  sc * aux_buffer_1_data = _aux_buffer_1[ thread_num ].data( );

  lo buffer_1_index = 0;
  for ( lo alpha1 = 0; alpha1 <= _spat_order; ++alpha1 ) {
    buffer_0_index = 0;
    for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha1; ++alpha2 ) {
      for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
        for ( lo beta1 = 0; beta1 <= _spat_order - beta0 - alpha2; ++beta1 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            aux_buffer_1_data[ buffer_1_index * hlp_acs_beta + a ]
              += buffer_for_coeffs_data[ alpha1 * hlp_acs_alpha
                   + beta1 * hlp_acs_beta + a ]
              * aux_buffer_0_data[ buffer_0_index * hlp_acs_beta + a ];
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
  compute_coupling_coeffs_initial_op( tar_time_nodes, half_size_space[ 0 ],
    center_diff_space[ 0 ], buffer_for_gaussians, buffer_for_coeffs );
  int local_index = 0;
  for ( lo alpha0 = 0; alpha0 <= _spat_order; ++alpha0 ) {
    buffer_1_index = 0;
    for ( lo alpha1 = 0; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
      for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha0 - alpha1; ++alpha2 ) {
        for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            tar_local[ a + local_index * ( _temp_order + 1 ) ]
              += buffer_for_coeffs_data[ alpha0 * hlp_acs_alpha
                   + beta0 * hlp_acs_beta + a ]
              * aux_buffer_1_data[ buffer_1_index * hlp_acs_beta + a ];
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_l2l_operations( mesh::scheduling_time_cluster *
    child_cluster ) const {
  scheduling_time_cluster * parent_cluster = child_cluster->get_parent( );
  slou configuration = child_cluster->get_configuration( );
  std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
    = parent_cluster->get_associated_spacetime_clusters( );
  lou n_associated_leaves = parent_cluster->get_n_associated_leaves( );
  // call the l2l operations for all non-leaf spacetime clusters which are
  // associated with the parent scheduling time cluster (note: there cannot
  // be any auxiliary clusters associated with the parent)

  // there is an implicit taskgroup associated with this taskloop
#pragma omp taskloop
  for ( lou i = n_associated_leaves; i < associated_spacetime_clusters->size( );
        ++i ) {
    apply_grouped_l2l_operation(
      ( *associated_spacetime_clusters )[ i ], configuration );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
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
    nodes_child[ j ] = ( child_time_center + child_time_half_size * nodes[ j ]
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
    apply_temporal_l2l_operation(
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_spatial_l2l_operation( const sc *
                                                               parent_local,
  const lo n_space_div_parent, const slou octant, sc * child_local ) const {
  const vector_type * m2m_coeffs_s_dim_0;
  const vector_type * m2m_coeffs_s_dim_1;
  const vector_type * m2m_coeffs_s_dim_2;
  switch ( octant ) {
    case 0:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      break;
    case 1:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      break;
    case 2:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      break;
    case 3:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      break;
    case 4:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      break;
    case 5:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      break;
    case 6:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      break;
    case 7:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_left[ n_space_div_parent ] );
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_l2t_operations( mesh::scheduling_time_cluster *
                                         t_cluster,
  distributed_block_vector & output_vector ) const {
  // execute only for associated spacetime leaves
  if ( t_cluster->get_n_associated_leaves( ) > 0 ) {
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = t_cluster->get_associated_spacetime_clusters( );
    lou i = 0;
    lou n = t_cluster->get_n_associated_leaves( );
    // there is an implicit taskgroup associated with this taskloop
#pragma omp taskloop shared( output_vector, associated_spacetime_clusters )
    for ( i = 0; i < n; ++i ) {
      apply_l2t_operation(
        ( *associated_spacetime_clusters )[ i ], output_vector );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_l2t_operation( const mesh::general_spacetime_cluster *
  /*st_cluster*/,
  distributed_block_vector & /*output_vector*/ ) const {
  std::cout << "L2T operation not implemented!" << std::endl;
}

//! template specialization for initial potential M0 p0-p1 PFMM matrix
template<>
void besthea::linear_algebra::distributed_initial_pFMM_matrix<
  besthea::bem::spacetime_heat_initial_m0_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >::
  apply_l2t_operation( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p0( st_cluster, output_vector );
}

//! template specialization for initial potential M1 p1-p1 PFMM matrix
template<>
void besthea::linear_algebra::distributed_initial_pFMM_matrix<
  besthea::bem::spacetime_heat_initial_m1_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >::
  apply_l2t_operation( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p1_normal_drv( st_cluster, output_vector );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p0( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  lo n_time_elements = st_cluster->get_n_time_elements( );
  lo n_space_elements = st_cluster->get_n_space_elements( );
  full_matrix targets( n_time_elements, n_space_elements, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution
    = st_cluster->get_pointer_to_local_contribution( );

  full_matrix T;
  compute_chebyshev_quadrature_p0( T, st_cluster );
  full_matrix L;
  compute_lagrange_quadrature( L, st_cluster );

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
  output_vector.add_local_part< target_space >( st_cluster, targets );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  lo n_time_elements = st_cluster->get_n_time_elements( );
  lo n_space_nodes = st_cluster->get_n_space_nodes( );
  full_matrix targets( n_time_elements, n_space_nodes, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution
    = st_cluster->get_pointer_to_local_contribution( );
  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1( T_drv, st_cluster );
  full_matrix L;
  compute_lagrange_quadrature( L, st_cluster );

  // compute D = trans(L) * lambda and then the result Y = D * trans(T_drv)
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
  // compute Y = D * trans(T_drv)
  targets.multiply( aux_matrix, T_drv, false, true );

  // add the results to the correct positions of the output vector
  output_vector.add_local_part< target_space >( st_cluster, targets );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::compute_coupling_coeffs_initial_op( const vector_type &
                                                        tar_time_nodes,
  const sc spat_half_size, const sc spat_center_diff,
  vector_type & buffer_for_gaussians, vector_type & coupling_coeffs ) const {
  // evaluate the gaussian kernel for the numerical integration
  sc h_alpha = spat_half_size * spat_half_size / ( 4.0 * _alpha );
  sc scaled_center_diff = spat_center_diff / spat_half_size;
  lou index_gaussian = 0;

  sc * buffer_for_gaussians_data = buffer_for_gaussians.data( );
  const sc * cheb_nodes_sum_coll_data = _cheb_nodes_sum_coll.data( );
  const sc * all_poly_vals_mult_coll_data = _all_poly_vals_mult_coll.data( );

  for ( lo a = 0; a <= _temp_order; ++a ) {
    sc h_delta_a = h_alpha / ( tar_time_nodes[ a ] );
    lou i = 0;
#pragma omp simd aligned( cheb_nodes_sum_coll_data, buffer_for_gaussians_data \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
    for ( i = 0; i < _cheb_nodes_sum_coll.size( ); ++i ) {
      buffer_for_gaussians_data[ index_gaussian + i ] = std::exp( -h_delta_a
        * ( scaled_center_diff + cheb_nodes_sum_coll_data[ i ] )
        * ( scaled_center_diff + cheb_nodes_sum_coll_data[ i ] ) );
    }
    index_gaussian += i;
  }

  // compute the numerical integrals
  sc mul_factor = 4.0 / ( _cheb_nodes_sum_coll.size( ) );
  lou index_integral = 0;

  for ( lo alpha = 0; alpha <= _spat_order; ++alpha ) {
    for ( lo beta = 0; beta <= _spat_order; ++beta ) {
      index_gaussian = 0;
      for ( lo a = 0; a <= _temp_order; ++a ) {
        sc val = 0.0;

        lo start_idx
          = alpha * ( _spat_order + 1 ) * _cheb_nodes_sum_coll.size( )
          + beta * _cheb_nodes_sum_coll.size( );
        const sc * curr_ptr = all_poly_vals_mult_coll_data;  // + start_idx;
        std::vector< sc, besthea::allocator_type< sc > >::size_type idx;
#pragma omp simd aligned( buffer_for_gaussians_data,curr_ptr : \
                       DATA_ALIGN ) reduction( + : val ) simdlen( BESTHEA_SIMD_WIDTH )
        for ( idx = 0; idx < _cheb_nodes_sum_coll.size( ); ++idx ) {
          val += buffer_for_gaussians_data[ index_gaussian + idx ]
            * curr_ptr[ start_idx + idx ];
        }
        index_gaussian += idx;
        coupling_coeffs[ index_integral ] = val;

        sc mul_factor_a
          = mul_factor / std::sqrt( 4.0 * M_PI * _alpha * tar_time_nodes[ a ] );
        // In the multiplicative factor a factor of 2 (gamma) is used for all
        // alpha and beta. For alpha == 0 or beta == 0 a correction is
        // required)
        // an attempt to compute this in a separate loop with precomputed
        // mul_factor_ab was slower
        if ( alpha == 0 ) {
          mul_factor_a *= 0.5;
        }
        if ( beta == 0 ) {
          mul_factor_a *= 0.5;
        }
        coupling_coeffs[ index_integral ] *= mul_factor_a;

        ++index_integral;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::compute_chebyshev_quadrature_p1_volume( full_matrix & T_vol,
  const mesh::volume_space_cluster * source_cluster ) const {
  lo n_space_elems = source_cluster->get_n_elements( );
  lo n_space_nodes = source_cluster->get_n_nodes( );
  T_vol.resize( _spat_contribution_size, n_space_nodes );
  T_vol.fill( 0.0 );

  // get some info on the current cluster
  vector_type cluster_center( 3 );
  vector_type cluster_half_size( 3 );
  source_cluster->get_center( cluster_center );
  source_cluster->get_half_size( cluster_half_size );
  sc padding = _maximal_spatial_paddings[ source_cluster->get_level( ) ];
  sc start_0 = cluster_center[ 0 ] - cluster_half_size[ 0 ] - padding;
  sc end_0 = cluster_center[ 0 ] + cluster_half_size[ 0 ] + padding;
  sc start_1 = cluster_center[ 1 ] - cluster_half_size[ 1 ] - padding;
  sc end_1 = cluster_center[ 1 ] + cluster_half_size[ 1 ] + padding;
  sc start_2 = cluster_center[ 2 ] - cluster_half_size[ 2 ] - padding;
  sc end_2 = cluster_center[ 2 ] + cluster_half_size[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials_tetrahedron( my_quadrature );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3, y4;

  // for storing the result of the Chebyshev evaluation in quadrature points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );

  sc * y1_ref = my_quadrature._y1_ref.data( );
  sc * y2_ref = my_quadrature._y2_ref.data( );
  sc * y3_ref = my_quadrature._y3_ref.data( );

  const std::vector< lo > & elems_2_local_nodes
    = source_cluster->get_elems_2_local_nodes( );

  const mesh::tetrahedral_volume_mesh & volume_mesh
    = source_cluster->get_mesh( );
  for ( lo i = 0; i < n_space_elems; ++i ) {
    lo elem_idx = source_cluster->get_element( i );
    volume_mesh.get_nodes( elem_idx, y1, y2, y3, y4 );
    sc elem_volume = volume_mesh.area( elem_idx );

    tetrahedron_to_geometry( y1, y2, y3, y4, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          sc value1 = 0.0;
          sc value2 = 0.0;
          sc value3 = 0.0;
          sc value4 = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            sc weighted_poly_value = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ] * wy[ j ] * elem_volume;
            value1 += weighted_poly_value
              * ( (sc) 1.0 - y1_ref[ j ] - y2_ref[ j ] - y3_ref[ j ] );
            value2 += weighted_poly_value * y1_ref[ j ];
            value3 += weighted_poly_value * y2_ref[ j ];
            value4 += weighted_poly_value * y3_ref[ j ];
          }
          T_vol.add( current_index, elems_2_local_nodes[ 4 * i ], value1 );
          T_vol.add( current_index, elems_2_local_nodes[ 4 * i + 1 ], value2 );
          T_vol.add( current_index, elems_2_local_nodes[ 4 * i + 2 ], value3 );
          T_vol.add( current_index, elems_2_local_nodes[ 4 * i + 3 ], value4 );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_lagrange_quadrature( full_matrix & L,
  const mesh::general_spacetime_cluster * source_cluster ) const {
  lo n_temp_elems = source_cluster->get_n_time_elements( );
  lo n_spat_elems = source_cluster->get_n_space_elements( );
  L.resize( _temp_order + 1, n_temp_elems );

  const std::vector< sc, besthea::allocator_type< sc > > & line_t
    = bem::quadrature::line_x( _order_regular_line );
  const std::vector< sc, besthea::allocator_type< sc > > & line_w
    = bem::quadrature::line_w( _order_regular_line );

  vector_type eval_points( line_t.size( ) );
  vector_type evaluation( line_t.size( ) );

  sc cluster_t_start = source_cluster->get_time_center( )
    - source_cluster->get_time_half_size( );
  sc cluster_t_end = source_cluster->get_time_center( )
    + source_cluster->get_time_half_size( );
  sc cluster_size = cluster_t_end - cluster_t_start;

  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = source_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  linear_algebra::coordinates< 1 > elem_t_start;
  linear_algebra::coordinates< 1 > elem_t_end;
  for ( lo i = 0; i < n_temp_elems; ++i ) {
    // we use that the elements in the cluster are tensor products of spatial
    // elements and timesteps, and are sorted w.r.t. the timesteps. In
    // particular we get all temporal elements in the cluster by considering
    // every n_spat_elems spacetime element.
    lo local_elem_idx = distributed_mesh.global_2_local(
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_chebyshev_quadrature_p0( full_matrix &
                                                                   T,
  const general_spacetime_cluster * source_cluster ) const {
  lo n_space_elems = source_cluster->get_n_space_elements( );
  T.resize( n_space_elems, _spat_contribution_size );
  // get some info on the current cluster
  vector_type cluster_center_space( 3 );
  vector_type cluster_half_space( 3 );
  sc sc_dummy;
  source_cluster->get_center( cluster_center_space, sc_dummy );
  source_cluster->get_half_size( cluster_half_space, sc_dummy );
  lo spat_level, lo_dummy;
  source_cluster->get_n_divs( spat_level, lo_dummy );
  sc padding = _maximal_spatial_paddings[ spat_level ];
  sc start_0 = cluster_center_space[ 0 ] - cluster_half_space[ 0 ] - padding;
  sc end_0 = cluster_center_space[ 0 ] + cluster_half_space[ 0 ] + padding;
  sc start_1 = cluster_center_space[ 1 ] - cluster_half_space[ 1 ] - padding;
  sc end_1 = cluster_center_space[ 1 ] + cluster_half_space[ 1 ] + padding;
  sc start_2 = cluster_center_space[ 2 ] - cluster_half_space[ 2 ] - padding;
  sc end_2 = cluster_center_space[ 2 ] + cluster_half_space[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials_triangle( my_quadrature );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );

  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = source_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  for ( lo i = 0; i < n_space_elems; ++i ) {
    // we use that the elements in the cluster are tensor products of spatial
    // elements and timesteps, and are sorted w.r.t. the timesteps. In
    // particular we get all spatial elements in the cluster by considering
    // the first n_space_elems spacetime elements.
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, source_cluster->get_element( i ) );
    lo local_elem_idx_space
      = local_mesh->get_space_element_index( local_elem_idx );

    local_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc elem_area = local_mesh->get_spatial_area_using_spatial_index(
      local_elem_idx_space );

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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::compute_normal_drv_chebyshev_quadrature_p1( full_matrix &
                                                                T_drv,
  const general_spacetime_cluster * source_cluster ) const {
  lo n_space_elems = source_cluster->get_n_space_elements( );
  lo n_space_nodes = source_cluster->get_n_space_nodes( );
  T_drv.resize( n_space_nodes, _spat_contribution_size );
  T_drv.fill( 0.0 );
  // get some info on the current cluster
  vector_type cluster_center_space( 3 );
  vector_type cluster_half_space( 3 );
  sc sc_dummy;
  source_cluster->get_center( cluster_center_space, sc_dummy );
  source_cluster->get_half_size( cluster_half_space, sc_dummy );
  lo spat_level, lo_dummy;
  source_cluster->get_n_divs( spat_level, lo_dummy );
  sc padding = _maximal_spatial_paddings[ spat_level ];
  sc start_0 = cluster_center_space[ 0 ] - cluster_half_space[ 0 ] - padding;
  sc end_0 = cluster_center_space[ 0 ] + cluster_half_space[ 0 ] + padding;
  sc start_1 = cluster_center_space[ 1 ] - cluster_half_space[ 1 ] - padding;
  sc end_1 = cluster_center_space[ 1 ] + cluster_half_space[ 1 ] + padding;
  sc start_2 = cluster_center_space[ 2 ] - cluster_half_space[ 2 ] - padding;
  sc end_2 = cluster_center_space[ 2 ] + cluster_half_space[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials_triangle( my_quadrature );
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

  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = source_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  for ( lo i = 0; i < n_space_elems; ++i ) {
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, source_cluster->get_element( i ) );
    lo local_elem_idx_space
      = local_mesh->get_space_element_index( local_elem_idx );
    local_mesh->get_spatial_normal_using_spatial_element_index(
      local_elem_idx_space, normal );
    local_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc elem_area = local_mesh->get_spatial_area_using_spatial_index(
      local_elem_idx_space );

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

          T_drv.add(
            source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
              elems_2_local_nodes[ 6 * i ] ),
            current_index, _alpha * value1 );
          T_drv.add(
            source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
              elems_2_local_nodes[ 6 * i + 1 ] ),
            current_index, _alpha * value2 );
          T_drv.add(
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  initialize_fmm_data(
    mesh::distributed_spacetime_cluster_tree * spacetime_target_tree,
    mesh::volume_space_cluster_tree * space_source_tree ) {
  _distributed_spacetime_target_tree = spacetime_target_tree;
  _space_source_tree = space_source_tree;

  // determine the maximal spatial padding at each spatial level in both trees
  // initialize the vector with the paddings in the volume space tree.
  _maximal_spatial_paddings = _space_source_tree->get_paddings( );
  std::vector< sc > spatial_paddings_target_tree
    = _distributed_spacetime_target_tree
        ->get_spatial_paddings_per_spatial_level( );
  for ( lou i = 0; i < _maximal_spatial_paddings.size( ); ++i ) {
    if ( spatial_paddings_target_tree[ i ] > _maximal_spatial_paddings[ i ] ) {
      _maximal_spatial_paddings[ i ] = spatial_paddings_target_tree[ i ];
    }
  }

  // determine the clusters in the target tree for which m2l or nearfield
  // operations have to be executed
  determine_interacting_time_clusters(
    *_distributed_spacetime_target_tree->get_distribution_tree( )
       ->get_root( ) );
  // determine the related nearfield and interaction lists of space-time
  // clusters
  initialize_nearfield_and_interaction_lists( );

  resize_nearfield_matrix_container( );

  // initialize the buffers used for the matrix-vector multiplication
  _aux_buffer_0.resize( omp_get_max_threads( ) );
  _aux_buffer_1.resize( omp_get_max_threads( ) );

#pragma omp parallel
  {
    _aux_buffer_0[ omp_get_thread_num( ) ].resize(
      ( _temp_order + 1 ), _spat_contribution_size );
    _aux_buffer_1[ omp_get_thread_num( ) ].resize(
      ( _temp_order + 1 ), _spat_contribution_size );
  }
}

template< class kernel_type, class target_space, class source_space >
besthea::linear_algebra::full_matrix *
besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::create_nearfield_matrix( lou leaf_index,
  lou source_index ) {
  mesh::general_spacetime_cluster * target_cluster
    = _nearfield_list_vector[ leaf_index ].first;
  mesh::volume_space_cluster * source_cluster
    = _nearfield_list_vector[ leaf_index ].second[ source_index ];
  lo n_dofs_source = source_cluster->get_n_dofs< source_space >( );
  lo n_dofs_target = target_cluster->get_n_dofs< target_space >( );
  full_matrix * local_matrix = new full_matrix( n_dofs_target, n_dofs_source );

  _clusterwise_nf_matrices[ leaf_index ][ source_index ] = local_matrix;

  return local_matrix;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::initialize_spatial_m2m_coeffs( ) {
  lo n_space_levels
    = _distributed_spacetime_target_tree->get_global_n_space_levels( );
  vector_type root_half_size( 3, false );
  _space_source_tree->get_root( )->get_half_size( root_half_size );
  compute_spatial_m2m_coeffs( n_space_levels, _spat_order, root_half_size[ 0 ],
    _maximal_spatial_paddings, _m2m_coeffs_s_left, _m2m_coeffs_s_right );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
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
    != (lou) ( _spat_order + 1 ) * ( _spat_order + 1 ) * cheb_nodes.size( )
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::init_quadrature_polynomials_triangle( quadrature_wrapper &
    my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._y1_ref_cheb
    = bem::quadrature::triangle_x1( _order_regular_tri );
  my_quadrature._y2_ref_cheb
    = bem::quadrature::triangle_x2( _order_regular_tri );
  my_quadrature._wy_cheb = bem::quadrature::triangle_w( _order_regular_tri );

  lo size = my_quadrature._wy_cheb.size( );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );

  my_quadrature._y1_polynomial.resize( size );
  my_quadrature._y2_polynomial.resize( size );
  my_quadrature._y3_polynomial.resize( size );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::init_quadrature_polynomials_tetrahedron( quadrature_wrapper &
    my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._y1_ref
    = bem::quadrature::tetrahedron_x1( _order_regular_tetra );
  my_quadrature._y2_ref
    = bem::quadrature::tetrahedron_x2( _order_regular_tetra );
  my_quadrature._y3_ref
    = bem::quadrature::tetrahedron_x3( _order_regular_tetra );
  my_quadrature._wy_cheb
    = bem::quadrature::tetrahedron_w( _order_regular_tetra );

  lo size = my_quadrature._wy_cheb.size( );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );

  my_quadrature._y1_polynomial.resize( size );
  my_quadrature._y2_polynomial.resize( size );
  my_quadrature._y3_polynomial.resize( size );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  tetrahedron_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & x4,
    quadrature_wrapper & my_quadrature ) const {
  const sc * y1_ref = my_quadrature._y1_ref.data( );
  const sc * y2_ref = my_quadrature._y2_ref.data( );
  const sc * y3_ref = my_quadrature._y3_ref.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo size = my_quadrature._wy_cheb.size( );

  // x1, x2, x3, x4 are vectors in R^3,
  // y%_mapped are the %th components of the vectors to which y#_ref is
  // mapped
#pragma omp simd aligned(                                 \
  y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, y3_ref \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * y1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * y2_ref[ i ]
      + ( x4[ 0 ] - x1[ 0 ] ) * y3_ref[ i ];
    y2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * y1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * y2_ref[ i ]
      + ( x4[ 1 ] - x1[ 1 ] ) * y3_ref[ i ];
    y3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * y1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * y2_ref[ i ]
      + ( x4[ 2 ] - x1[ 2 ] ) * y3_ref[ i ];
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
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
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::apply( const vector & /*x*/, vector & /*y*/,
  bool /*trans*/, sc /*alpha*/, sc /*beta*/ ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED for this combination of vectors. Please "
               "use a vector x and a distributed block y!"
            << std::endl;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::apply( const vector & x,
  distributed_block_vector & y, [[maybe_unused]] bool trans, sc alpha,
  sc beta ) const {
  // first scaling
  y.scale( beta );

  // (re)set all moment and local contributions to 0
  mesh::tree_structure * scheduling_tree
    = _distributed_spacetime_target_tree->get_distribution_tree( );
  scheduling_tree->clear_local_contributions( *scheduling_tree->get_root( ) );
  _space_source_tree->clear_moment_contributions(
    *_space_source_tree->get_root( ) );

  // allocate a global result vector to store the result of the pFMM procedure.
  std::vector< lo > my_blocks = y.get_my_blocks( );
  distributed_block_vector y_pFMM(
    my_blocks, y.get_n_blocks( ), y.get_size_of_block( ), true, y.get_comm( ) );

  // farfield part (approximated with FMM)
  compute_moments_upward_path( x, _space_source_tree->get_root( ) );
  apply_all_m2l_operations( );
  evaluate_local_contributions_downward_path(
    _distributed_spacetime_target_tree->get_distribution_tree( )->get_root( ),
    y_pFMM );
  // nearfield part
  apply_all_nearfield_operations( x, y_pFMM );

  y.add( y_pFMM, alpha );
}

template< class kernel_type, class target_space, class source_space >
sc besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_nearfield_ratio( ) const {
  lou n_nearfield_entries = 0;
  for ( auto nearfield_pair : _nearfield_list_vector ) {
    general_spacetime_cluster * st_target = nearfield_pair.first;
    lo n_target_dofs = st_target->get_n_dofs< target_space >( );
    for ( auto src_cluster : nearfield_pair.second ) {
      n_nearfield_entries
        += n_target_dofs * src_cluster->template get_n_dofs< source_space >( );
    }
  }
  lou n_global_target_dofs = _distributed_spacetime_target_tree->get_mesh( )
                               .get_n_dofs< target_space >( );
  lou n_global_source_dofs
    = _space_source_tree->get_mesh( ).get_n_dofs< source_space >( );
  return n_nearfield_entries
    / ( (sc) n_global_target_dofs * n_global_source_dofs );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::count_fmm_operations_levelwise( std::vector< long long > &
                                                    n_s2m_operations,
  std::vector< long long > & n_m2m_operations,
  std::vector< long long > & n_m2l_operations,
  std::vector< long long > & n_l2l_operations,
  std::vector< long long > & n_l2t_operations ) const {
  lo n_source_levels = _space_source_tree->get_max_n_levels( );
  lo n_target_levels = _distributed_spacetime_target_tree->get_n_levels( );
  // initialize all lists of operation counters appropriately
  n_s2m_operations.resize( n_source_levels );
  n_m2m_operations.resize( n_source_levels );
  for ( lo i = 0; i < n_source_levels; ++i ) {
    n_s2m_operations[ i ] = 0;
    n_m2m_operations[ i ] = 0;
  }
  n_m2l_operations.resize( n_target_levels );
  n_l2l_operations.resize( n_target_levels );
  n_l2t_operations.resize( n_target_levels );
  for ( lo i = 0; i < n_target_levels; ++i ) {
    n_m2l_operations[ i ] = 0;
    n_l2l_operations[ i ] = 0;
    n_l2t_operations[ i ] = 0;
  }

  // count the number of s2m operations
  const std::vector< mesh::volume_space_cluster * > & leaves_source_tree
    = _space_source_tree->get_leaves( );
  for ( mesh::volume_space_cluster * leaf : leaves_source_tree ) {
    n_s2m_operations[ leaf->get_level( ) ] += 1;
  }

  // count the number of m2m operations
  count_m2m_operations_recursively(
    *_space_source_tree->get_root( ), n_m2m_operations );

  // count the number of m2l operations.
  for ( auto m2l_pair : _interaction_list_vector ) {
    lo current_level = m2l_pair.first->get_level( );
    n_m2l_operations[ current_level ] += m2l_pair.second.size( );
  }

  // count the number of l2l and l2t operations
  mesh::scheduling_time_cluster * root_distribution_tree
    = _distributed_spacetime_target_tree->get_distribution_tree( )->get_root( );
  count_l2l_and_l2t_operations_recursively(
    *root_distribution_tree, n_l2l_operations, n_l2t_operations );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  count_m2m_operations_recursively(
    const mesh::volume_space_cluster & current_cluster,
    std::vector< long long > & n_m2m_operations ) const {
  if ( current_cluster.get_parent( ) != nullptr ) {
    n_m2m_operations[ current_cluster.get_level( ) ] += 1;
  }
  if ( current_cluster.get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster.get_children( ) ) {
      count_m2m_operations_recursively( *child, n_m2m_operations );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  count_l2l_and_l2t_operations_recursively(
    const mesh::scheduling_time_cluster & current_cluster,
    std::vector< long long > & n_l2l_operations,
    std::vector< long long > & n_l2t_operations ) const {
  char downpard_path_status
    = current_cluster.get_status_in_initial_op_downward_path( );
  lo current_level = current_cluster.get_level( );
  if ( downpard_path_status == 1 ) {
    // l2l operations are executed for all associated space-time clusters for
    // which local contributions are allocated (i.e. all non-auxiliary
    // associated clusters).
    lo n_assoc_st_clusters_w_local_contributions
      = current_cluster.get_n_st_clusters_w_local_contributions( );
    lo n_associated_leaves = current_cluster.get_n_associated_leaves( );
    const mesh::scheduling_time_cluster * current_parent
      = current_cluster.get_parent( );
    if ( current_parent != nullptr
      && ( current_parent->get_status_in_initial_op_downward_path( ) == 1 ) ) {
      n_l2l_operations[ current_level ]
        += n_assoc_st_clusters_w_local_contributions;
    }
    n_l2t_operations[ current_level ] += n_associated_leaves;
  }
  if ( downpard_path_status > 0 && current_cluster.get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster.get_children( ) ) {
      count_l2l_and_l2t_operations_recursively(
        *child, n_l2l_operations, n_l2t_operations );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::print_information( const int root_process )
  const {
  // print rough nearfield percentage
  // compute the nearfield ratio on each process.
  sc local_nearfield_ratio = compute_nearfield_ratio( );
  sc global_nearfield_ratio( 0 );
  // gather the nearfield ratios at the root process
  int n_processes;
  MPI_Comm_size( *_comm, &n_processes );
  sc * all_local_nearfield_ratios = nullptr;
  if ( _my_rank == root_process ) {
    all_local_nearfield_ratios = new sc[ n_processes ];
  }

  MPI_Gather( &local_nearfield_ratio, 1, get_scalar_type< sc >::MPI_SC( ),
    all_local_nearfield_ratios, 1, get_scalar_type< sc >::MPI_SC( ),
    root_process, *_comm );
  if ( _my_rank == root_process ) {
    for ( lo i = 0; i < n_processes; ++i ) {
      global_nearfield_ratio += all_local_nearfield_ratios[ i ];
    }
    std::cout << std::endl;
    std::cout << "nearfield ratio = " << global_nearfield_ratio << std::endl;
  }
  // count the fmm operations levelwise
  std::vector< long long > n_s2m_operations, n_m2m_operations, n_m2l_operations,
    n_l2l_operations, n_l2t_operations;
  count_fmm_operations_levelwise( n_s2m_operations, n_m2m_operations,
    n_m2l_operations, n_l2l_operations, n_l2t_operations );
  // collect the numbers of m2l, l2l and l2t operations at the root
  // process via reduce operations
  lo n_target_levels = _distributed_spacetime_target_tree->get_n_levels( );
  lo n_source_levels = _space_source_tree->get_max_n_levels( );
  if ( _my_rank == root_process ) {
    MPI_Reduce( MPI_IN_PLACE, n_m2l_operations.data( ), n_target_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_l2l_operations.data( ), n_target_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_l2t_operations.data( ), n_target_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
  } else {
    MPI_Reduce( n_m2l_operations.data( ), nullptr, n_target_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_l2l_operations.data( ), nullptr, n_target_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_l2t_operations.data( ), nullptr, n_target_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
  }
  if ( _my_rank == root_process ) {
    lo start_space_refinement
      = _distributed_spacetime_target_tree->get_start_space_refinement( );
    std::cout << "number of s2m operations: " << std::endl;
    for ( lo i = 0; i < n_source_levels; ++i ) {
      std::cout << "level " << i << ": " << n_s2m_operations[ i ] << std::endl;
    }
    std::cout << "number of m2m operations: " << std::endl;
    for ( lo i = 0; i < n_source_levels; ++i ) {
      std::cout << "level " << i << ": " << n_m2m_operations[ i ] << std::endl;
    }
    std::cout << "number of m2l operations: " << std::endl;
    for ( lo i = 0; i < n_target_levels; ++i ) {
      std::cout << "level " << i << ": " << n_m2l_operations[ i ] << std::endl;
    }
    std::cout << "number of l2l operations: " << std::endl;
    for ( lo i = 0; i < n_target_levels; ++i ) {
      std::cout << "level " << i << ": " << n_l2l_operations[ i ];
      if ( i >= start_space_refinement
        && ( ( i - start_space_refinement ) ) % 2 == 0 ) {
        std::cout << " spacetime l2l" << std::endl;
      } else {
        std::cout << " temporal l2l" << std::endl;
      }
    }
    std::cout << "number of l2t operations: " << std::endl;
    for ( lo i = 0; i < n_target_levels; ++i ) {
      std::cout << "level " << i << ": " << n_l2t_operations[ i ] << std::endl;
    }

    std::cout << "rough memory estimates per process (nearfield only): "
              << std::endl;

    lo n_target_dofs = _distributed_spacetime_target_tree->get_mesh( )
                         .get_n_dofs< target_space >( );
    lo n_source_dofs
      = _space_source_tree->get_mesh( ).get_n_dofs< source_space >( );
    sc total_storage_nearfield = 0.0;
    // sc total_storage_contributions = 0.0;
    for ( int i = 0; i < n_processes; ++i ) {
      sc local_storage_nearfield
        = n_target_dofs * n_source_dofs * all_local_nearfield_ratios[ i ];
      local_storage_nearfield *= 8. / 1024. / 1024. / 1024.;
      // get memory for double entries in GiB.
      total_storage_nearfield += local_storage_nearfield;
      // total_storage_contributions += local_storage_contributions;
      std::cout << "process " << i
                << ": nearfield_matrices: " << local_storage_nearfield << " GiB"
                << std::endl;
    }
    std::cout << "total storage: nearfield matrices: "
              << total_storage_nearfield << " GiB" << std::endl;
    std::cout << "storage per allocated vector (source): "
              << n_source_dofs * 8. / 1024. / 1024. / 1024. << " GiB."
              << std::endl;
    std::cout << "storage per allocated vector (target): "
              << n_target_dofs * 8. / 1024. / 1024. / 1024. << " GiB."
              << std::endl;
    delete[] all_local_nearfield_ratios;
    std::cout << "#############################################################"
              << "###########################" << std::endl;
    std::cout << std::endl;
  }
}

template class besthea::linear_algebra::distributed_initial_pFMM_matrix<
  besthea::bem::spacetime_heat_initial_m0_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >;

template class besthea::linear_algebra::distributed_initial_pFMM_matrix<
  besthea::bem::spacetime_heat_initial_m1_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >;
