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

#include <besthea/spacetime_cluster_tree.h>

#include <algorithm>
#include <cmath>
#include <vector>

besthea::mesh::spacetime_cluster_tree::spacetime_cluster_tree(
  const spacetime_tensor_mesh & spacetime_mesh, lo time_levels,
  lo n_min_time_elems, lo n_min_space_elems, sc st_coeff )
  : _spacetime_mesh( spacetime_mesh ),
    _space_mesh( *( spacetime_mesh.get_spatial_mesh( ) ) ),
    _time_mesh( *( spacetime_mesh.get_temporal_mesh( ) ) ),
    _s_t_coeff( st_coeff ) {
  // first, we create the temporal and spatial trees

  _time_tree
    = new time_cluster_tree( _time_mesh, time_levels, n_min_time_elems );
  sc xmin, xmax, ymin, ymax, zmin, zmax;
  _space_mesh.compute_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );

  sc max_half_size = std::max( { ( xmax - xmin ) / 2.0, ( ymax - ymin ) / 2.0,
                       ( zmax - zmin ) / 2.0 } )
    / 2.0;

  sc delta = _time_mesh.get_end( ) - _time_mesh.get_start( );

  _start_spatial_level = 0;

  // determine the number of initial octasections that has to be performed to
  // get to the level of the spatial tree satisfying the condition h_x^l \approx
  // sqrt(delta)
  while ( max_half_size > st_coeff * sqrt( delta ) ) {
    max_half_size *= 0.5;
    _start_spatial_level += 1;
  }

  lo n_t_levels = _time_tree->get_levels( );
  lo n_s_levels = n_t_levels / 2;
  if ( n_t_levels % 2 ) {
    ++n_s_levels;
  }
  _space_tree = new space_cluster_tree(
    _space_mesh, _start_spatial_level + n_s_levels, n_min_space_elems );

  // determine for which temporal level the first spatial refinement is needed
  _start_temporal_level = 0;
  if ( _start_spatial_level == 0 ) {
    while ( max_half_size <= st_coeff * sqrt( delta ) ) {
      delta *= 0.5;
      _start_temporal_level += 1;
    }
    // shift _start_temporal_level if necessary to guarantee levels as in
    // Messner's work
    if ( ( n_t_levels - _start_temporal_level ) % 2 ) {
      _start_temporal_level -= 1;
    }
  }

  // next, we assemble their combination into a space-time tree
  // the level -1 node is always the combination of the whole space & time
  _root = new spacetime_cluster(
    *_space_tree->get_root( ), *_time_tree->get_root( ), nullptr, -1 );

  // if the space has to be split to fullfil the condition, the individual roots
  // are stored in the space_roots vector
  std::vector< space_cluster * > space_roots;
  get_space_clusters_on_level(
    _space_tree->get_root( ), _start_spatial_level, space_roots );

  // determine whether the individual roots are split in the first step of the
  // building of the cluster tree (guarantees levels as in Messner's work)
  bool split_space = false;
  if ( ( _start_temporal_level == 0 ) && ( n_t_levels % 2 ) ) {
    split_space = true;
  }

  for ( auto it = space_roots.begin( ); it != space_roots.end( ); ++it ) {
    spacetime_cluster * cluster
      = new spacetime_cluster( **it, *_time_tree->get_root( ), _root, 0 );
    build_tree( cluster, 1, split_space );
    // the roots of the subtrees are linked to the level -1 global root
    _root->add_child( cluster );
  }
}

void besthea::mesh::spacetime_cluster_tree::build_tree(
  spacetime_cluster * root, lo level, bool split_space ) {
  std::vector< space_cluster * > * space_children;
  bool split_space_descendant;
  if ( !split_space ) {
    space_children = new std::vector< space_cluster * >;
    space_children->push_back( &root->get_space_cluster( ) );
    if ( level + 1 < _start_temporal_level ) {
      // no spatial refinement as long as next level < _start_temporal_level
      split_space_descendant = false;
    } else {
      // alternate between refinement and non refinement
      split_space_descendant = true;
    }
  } else {
    space_children = root->get_space_cluster( ).get_children( );
    split_space_descendant = false;
  }

  std::vector< time_cluster * > * time_children
    = root->get_time_cluster( ).get_children( );

  if ( space_children != nullptr && time_children != nullptr ) {
    root->set_n_children( time_children->size( ) * space_children->size( ) );
    for ( auto it = time_children->begin( ); it != time_children->end( );
          ++it ) {
      for ( auto it2 = space_children->begin( ); it2 != space_children->end( );
            ++it2 ) {
        spacetime_cluster * cluster
          = new spacetime_cluster( **it2, **it, root, level );
        root->add_child( cluster );
        build_tree( cluster, level + 1, split_space_descendant );
      }
    }
  }
  if ( ( split_space_descendant ) || ( level < _start_temporal_level ) ) {
    delete space_children;
  }
}

void besthea::mesh::spacetime_cluster_tree::get_space_clusters_on_level(
  space_cluster * root, lo level, std::vector< space_cluster * > & clusters ) {
  if ( root->get_level( ) < level ) {
    std::vector< space_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      get_space_clusters_on_level( *it, level, clusters );
    }
  } else {
    clusters.push_back( root );
  }
}

void besthea::mesh::spacetime_cluster_tree::apply_temporal_m2m(
  full_matrix_type const & child_moment, const lo level,
  const bool is_left_child, full_matrix_type & parent_moment ) {
  if ( is_left_child )
    parent_moment.multiply(
      _m2m_matrices_t_left[ level ], child_moment, false, false, 1.0, 1.0 );
  else
    parent_moment.multiply(
      _m2m_matrices_t_right[ level ], child_moment, false, false, 1.0, 1.0 );
}

void besthea::mesh::spacetime_cluster_tree::apply_spatial_m2m(
  full_matrix_type const & child_moment, const lo level, const slou octant,
  full_matrix_type & parent_moment ) {
  const vector_type * m2m_coeffs_s_dim_0;
  const vector_type * m2m_coeffs_s_dim_1;
  const vector_type * m2m_coeffs_s_dim_2;

  switch ( octant ) {
    case 0:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ level ] );
      break;
    case 1:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ level ] );
      break;
    case 2:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ level ] );
      break;
    case 3:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ level ] );
      break;
    case 4:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ level ] );
      break;
    case 5:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ level ] );
      break;
    case 6:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ level ] );
      break;
    case 7:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ level ] );
      break;
    default:  // default case should never be used, programm will crash!
      m2m_coeffs_s_dim_0 = nullptr;
      m2m_coeffs_s_dim_1 = nullptr;
      m2m_coeffs_s_dim_2 = nullptr;
  }

  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  // initialize auxiliary matrices lambda_1/2 for intermediate results with 0
  full_matrix_type lambda_1( _temp_order + 1, n_coeffs_s, true );
  full_matrix_type lambda_2( _temp_order + 1, n_coeffs_s, true );

  for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
    lo crnt_indx = 0;
    for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
      for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0; ++alpha1 ) {
        lo alpha2;
        for ( alpha2 = 0; alpha2 <= beta2; ++alpha2 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            lambda_1( b,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 )
              += ( *m2m_coeffs_s_dim_2 )[ beta2 * ( _spat_order + 1 ) + alpha2 ]
              * child_moment( b, crnt_indx );
          }
          ++crnt_indx;
        }
        // correction needed for skipt entries of child_moment
        crnt_indx += _spat_order + 1 - alpha0 - alpha1 - alpha2;
      }
      // correction for current index; necessary since alpha1 does not run until
      // _spat_order - alpha0 as it does in stored child_moment
      crnt_indx += ( ( beta2 + 1 ) * beta2 ) / 2;
    }
  }

  // compute intermediate result lambda_1 ignoring zero entries for the sake of
  // better readability
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

  lo crnt_indx = 0;
  for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
    for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
        for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            parent_moment( b, crnt_indx )
              += ( *m2m_coeffs_s_dim_0 )[ beta0 * ( _spat_order + 1 ) + alpha0 ]
              * lambda_2( b,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                  + ( _spat_order + 1 ) * beta2 + alpha0 );
          }
          ++crnt_indx;
        }
      }
    }
  }
}
