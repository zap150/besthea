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
  if ( _distributed_spacetime_target_tree->get_distribution_tree( )
         ->subtree_contains_local_spacetime_leaves( current_cluster ) ) {
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
      std::vector< volume_space_cluster * > neighbors;
      lo spat_level, dummy;
      spacetime_leaf->get_n_divs( spat_level, dummy );
      _space_source_tree->find_neighbors( spat_level, spatial_grid_coords,
        _distributed_spacetime_target_tree->get_spatial_nearfield_limit( ),
        neighbors );
      if ( neighbors.size( ) > 0 ) {
        neighbors.shrink_to_fit( );
        _nearfield_list_vector.push_back( { spacetime_leaf, neighbors } );
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
    for ( size_t i = 0; i < associated_clusters->size( ); ++i ) {
      general_spacetime_cluster * st_cluster = ( *associated_clusters )[ i ];
      std::vector< slou > cluster_coords = st_cluster->get_box_coordinate( );
      std::vector< slou > spatial_grid_coords
        = { cluster_coords[ 1 ], cluster_coords[ 2 ], cluster_coords[ 3 ] };
      std::vector< volume_space_cluster * > neighbors;
      lo spat_level, dummy;
      st_cluster->get_n_divs( spat_level, dummy );
      _space_source_tree->find_neighbors( spat_level, spatial_grid_coords,
        _distributed_spacetime_target_tree->get_spatial_nearfield_limit( ),
        neighbors );
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
  target_space, source_space >::compute_moments_upward_path( const vector &
                                                               sources,
  mesh::volume_space_cluster * current_cluster ) const {
  if ( current_cluster->get_n_children( ) > 0 ) {
    std::vector< volume_space_cluster * > * children
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
  for ( size_t i = 0; i < _interaction_list_vector.size( ); ++i ) {
    general_spacetime_cluster * st_target = _interaction_list_vector[ i ].first;
    for ( auto space_source_cluster : _interaction_list_vector[ i ].second ) {
      apply_m2l_operation( space_source_cluster, st_target );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_s2m_operation( const vector & sources,
  mesh::volume_space_cluster * leaf ) const {
  full_matrix T_vol;
  compute_chebyshev_quadrature_p1( leaf, T_vol );
  vector moments = leaf->get_moments( );
  vector sources_in_leaf;
  sources_in_leaf.get_local_part< source_space >( leaf, sources_in_leaf );
  T_vol.apply( sources_in_leaf, moments );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_grouped_m2m_operation( mesh::volume_space_cluster *
    parent_cluster ) const {
  std::vector< mesh::volume_space_cluster * > * children
    = parent_cluster->get_children( );
  // declare auxiliary vectors lambda_1/2 to store intermediate results in m2m
  // operations
  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  vector_type lambda_1( n_coeffs_s, true );
  vector_type lambda_2( n_coeffs_s, true );

  lo parent_level = parent_cluster->get_level( );
  vector_type & parent_moment = parent_cluster->get_moments( );
  for ( auto child : *children ) {
    // execute the m2m operation for each child
    short child_octant = child->get_octant( );
    const vector_type child_moment = child->get_moments( );

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

    // compute intermediate result lambda_1 ignoring zero entries for the sake
    // of better readability
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

  const vector_type src_moment = s_src_cluster->get_moments( );
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
          // no need for reduction, in a single inner cycle data are written on
          // unique positions
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
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
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
                       DATA_ALIGN ) reduction( + : val ) simdlen( DATA_WIDTH )
        for ( idx = 0; idx < _cheb_nodes_sum_coll.size( ); ++idx ) {
          val += buffer_for_gaussians_data[ index_gaussian + idx ]
            * curr_ptr[ start_idx + idx ];
        }
        index_gaussian += idx;
        coupling_coeffs[ index_integral ] = val;

        sc mul_factor_ab
          = mul_factor / std::sqrt( 4.0 * M_PI * _alpha * tar_time_nodes[ a ] );
        // In the multiplicative factor a factor of 2 (gamma) is used for all
        // alpha and beta. For alpha == 0 or beta == 0 a correction is
        // required)
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

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_chebyshev_quadrature_p1(
    const mesh::volume_space_cluster * source_cluster,
    full_matrix & T_vol ) const {
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
  lo size_quad = my_quadrature._w.size( );
  sc * wy = my_quadrature._w.data( );
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
  target_space, source_space >::apply( const block_vector & /*x*/,
  block_vector & /*y*/, bool /*trans*/, sc /*alpha*/, sc /*beta*/ ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED for standard block vectors. Please use "
               "distributed block vectors!"
            << std::endl;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::
  set_trees_and_operation_lists(
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
  for ( size_t i = 0; i < _maximal_spatial_paddings.size( ); ++i ) {
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
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::prepare_fmm( ) {
}

template< class kernel_type, class target_space, class source_space >
besthea::linear_algebra::full_matrix *
besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::create_nearfield_matrix( lou leaf_index,
  lou source_index ) {
  return nullptr;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space, source_space >::initialize_spatial_m2m_coeffs( ) {
  lo max_space_level
    = _distributed_spacetime_target_tree->get_global_max_space_level( );
  vector_type root_half_size( 3, false );
  _space_source_tree->get_root( )->get_half_size( root_half_size );
  compute_spatial_m2m_coeffs( max_space_level, _spat_order, root_half_size[ 0 ],
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
void besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
  target_space,
  source_space >::init_quadrature_polynomials_tetrahedron( quadrature_wrapper &
    my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._y1_ref = bem::quadrature::tetrahedron_x1( _order_regular );
  my_quadrature._y2_ref = bem::quadrature::tetrahedron_x2( _order_regular );
  my_quadrature._y3_ref = bem::quadrature::tetrahedron_x3( _order_regular );
  my_quadrature._w = bem::quadrature::tetrahedron_w( _order_regular );

  lo size = my_quadrature._w.size( );
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
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
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
  target_space, source_space >::apply( const distributed_block_vector & /*x*/,
  distributed_block_vector & /*y*/, bool /*trans*/, sc /*alpha*/,
  sc /*beta*/ ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED" << std::endl;
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
