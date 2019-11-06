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

  // initialize temporal m2m matrices
  set_temporal_m2m_matrices( );
  set_spatial_m2m_coeffs( );
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

void besthea::mesh::spacetime_cluster_tree::set_temporal_m2m_matrices( ) {
  lo n_levels = _time_tree->get_levels( );
  // Declare the two structures containing matrices of appropriate size.
  // NOTE: For level 0 and 1 matrices are stored, but not needed. This allows
  //       for a direct access of the matrices via their level. The matrix for
  //       level n_levels is not needed and hence not allocated.

  _m2m_matrices_t_left.resize( n_levels );
  _m2m_matrices_t_right.resize( n_levels );

  for ( auto it = _m2m_matrices_t_left.begin( );
        it != _m2m_matrices_t_left.end( ); ++it ) {
    ( *it ).resize( _temp_order + 1, _temp_order + 1 );
  }
  for ( auto it = _m2m_matrices_t_right.begin( );
        it != _m2m_matrices_t_right.end( ); ++it ) {
    ( *it ).resize( _temp_order + 1, _temp_order + 1 );
  }

  const std::vector< sc > & paddings = _time_tree->get_paddings( );
  sc h_root_no_pad = _time_tree->get_root( )->get_half_size( );
  sc h_par_no_pad = h_root_no_pad / 4.0;

  // Initialize class for evaluation of lagrange polynomials and get
  // interpolation nodes in the interval [-1, 1].
  besthea::bem::lagrange_interpolant lagrange( _temp_order );
  const vector_type & nodes = lagrange.get_nodes( );
  vector_type nodes_l_child( _temp_order + 1, false );
  vector_type nodes_r_child( _temp_order + 1, false );
  vector_type values_lagrange( _temp_order + 1, false );
  for ( lo curr_level = 2; curr_level < n_levels; ++curr_level ) {
    sc h_child_no_pad = h_par_no_pad / 2.0;
    sc padding_par = paddings[ curr_level ];
    sc padding_child = paddings[ curr_level + 1 ];

    // transform the nodes from [-1, 1] to the child interval and then back to
    // [-1, 1] with the transformation of the parent interval:
    for ( lo j = 0; j <= _temp_order; ++j ) {
      nodes_l_child[ j ] = 1.0 / ( h_par_no_pad + padding_par )
        * ( -h_child_no_pad + ( h_child_no_pad + padding_child ) * nodes[ j ] );
      nodes_r_child[ j ] = 1.0 / ( h_par_no_pad + padding_par )
        * ( h_child_no_pad + ( h_child_no_pad + padding_child ) * nodes[ j ] );
    }

    // compute left m2m matrix at current level
    for ( lo j = 0; j <= _temp_order; ++j ) {
      lagrange.evaluate( j, nodes_l_child, values_lagrange );
      for ( lo k = 0; k <= _temp_order; ++k )
        _m2m_matrices_t_left[ curr_level ].set( j, k, values_lagrange[ k ] );
    }

    // compute right m2m matrix at current level
    for ( lo j = 0; j <= _temp_order; ++j ) {
      lagrange.evaluate( j, nodes_r_child, values_lagrange );
      for ( lo k = 0; k <= _temp_order; ++k )
        _m2m_matrices_t_right[ curr_level ].set( j, k, values_lagrange[ k ] );
    }

    // TODO: The construction of the matrices is probably far from optimal: The
    // values are computed in row major order, but matrix memory is column major
    // Idea: Compute L2L matrices instead of M2M matrices?

    // update for next iteration
    h_par_no_pad = h_child_no_pad;
  }
}

void besthea::mesh::spacetime_cluster_tree::set_spatial_m2m_coeffs( ) {
  lo n_levels = _space_tree->get_levels( );
  // Declare the structures containing coefficients of appropriate size.
  // NOTE: The M2M coefficients are computed for all levels except the last one,
  //       even in case they are not needed for the first few levels.
  _m2m_coeffs_s_dim_0_left.resize( n_levels );
  _m2m_coeffs_s_dim_0_right.resize( n_levels );
  _m2m_coeffs_s_dim_1_left.resize( n_levels );
  _m2m_coeffs_s_dim_1_right.resize( n_levels );
  _m2m_coeffs_s_dim_2_left.resize( n_levels );
  _m2m_coeffs_s_dim_2_right.resize( n_levels );
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

  const std::vector< sc > & paddings = _space_tree->get_paddings( );

  // declare half box side lengths of parent and child cluster + initialize
  vector_type h_par_no_pad( 3, false ), h_child_no_pad( 3, false );
  _space_tree->get_root( )->get_half_size( h_par_no_pad );
  // Initialize class for evaluation of Chebyshev polynomials and compute
  // Chebyshev nodes in the interval [-1, 1].
  besthea::bem::chebyshev_evaluator chebyshev( _spat_order );
  vector_type nodes( _spat_order + 1, false );
  for ( lo i = 0; i <= _spat_order; ++i )
    nodes[ i ] = cos( ( M_PI * ( 2 * i + 1 ) ) / ( 2 * ( _spat_order + 1 ) ) );
  // evaluate Chebyshev polynomials at the nodes (needed for coefficients)
  vector_type all_values_cheb_std_intrvl(
    ( _spat_order + 1 ) * ( _spat_order + 1 ), false );
  chebyshev.evaluate( nodes, all_values_cheb_std_intrvl );
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
  for ( lo curr_level = 0; curr_level < n_levels; ++curr_level ) {
    h_child_no_pad[ 0 ] = h_par_no_pad[ 0 ] / 2.0;
    h_child_no_pad[ 1 ] = h_par_no_pad[ 1 ] / 2.0;
    h_child_no_pad[ 2 ] = h_par_no_pad[ 2 ] / 2.0;
    sc padding_par = paddings[ curr_level ];
    sc padding_child = paddings[ curr_level + 1 ];
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
    // for i1 < i0 the coefficients are knwon to be zero
    chebyshev.evaluate( nodes_l_child_dim_0, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = i0; i1 <= _spat_order; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i0 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i1 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1.0 );
        if ( i0 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_0_left[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    chebyshev.evaluate( nodes_r_child_dim_0, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = i0; i1 <= _spat_order; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i0 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i1 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i0 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_0_right[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    // compute m2m coefficients at current level along all dimensions
    chebyshev.evaluate( nodes_l_child_dim_1, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = i0; i1 <= _spat_order; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i0 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i1 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i0 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_1_left[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    chebyshev.evaluate( nodes_r_child_dim_1, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = i0; i1 <= _spat_order; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i0 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i1 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i0 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_1_right[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    // compute m2m coefficients at current level along all dimensions
    chebyshev.evaluate( nodes_l_child_dim_2, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = i0; i1 <= _spat_order; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i0 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i1 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i0 == 0 ) {
          coeff /= 2.0;
        }
        _m2m_coeffs_s_dim_2_left[ curr_level ][ ( _spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }

    chebyshev.evaluate( nodes_r_child_dim_2, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= _spat_order; ++i0 ) {
      for ( lo i1 = i0; i1 <= _spat_order; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= _spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i0 * ( _spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i1 * ( _spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( _spat_order + 1 );
        if ( i0 == 0 ) {
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
