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

#include "besthea/pFMM_matrix.h"

besthea::linear_algebra::sparse_matrix *
besthea::linear_algebra::pFMM_matrix::create_nearfield_matrix(
  lo test_idx, lo trial_idx, lo n_duplications ) {
  sparse_matrix_type * local_matrix = new sparse_matrix_type( );

  _nearfield_matrices.push_back( local_matrix );

  _nearfield_block_map.push_back( std::make_pair( test_idx, trial_idx ) );

  // duplicate diagonals
  for ( lo i = 1; i < n_duplications; ++i ) {
    _nearfield_matrices.push_back( local_matrix );
    _nearfield_block_map.push_back(
      std::make_pair( test_idx + i, trial_idx + i ) );
    _uniform = true;
  }
  
  return local_matrix;
}

besthea::linear_algebra::sparse_matrix *
besthea::linear_algebra::pFMM_matrix::create_farfield_matrix(
  lo test_idx, lo trial_idx ) {
  sparse_matrix_type * local_matrix = new sparse_matrix_type( );

  _farfield_matrices.push_back( local_matrix );

  _farfield_block_map.push_back( std::make_pair( test_idx, trial_idx ) );

  return local_matrix;
}

void besthea::linear_algebra::pFMM_matrix::apply( const block_vector_type & x,
  block_vector_type & y, bool trans, sc alpha, sc beta ) const {
#pragma omp parallel for schedule( static )
  for ( lo i = 0; i < y.get_block_size( ); ++i ) {
    for ( lo j = 0; j < y.get_size_of_block( ); ++j ) {
      y.set( i, j, y.get( i, j ) * beta );
    }
  }

  sparse_matrix_type * current_block;

#pragma omp parallel
  {
    vector_type local_y( y.get_size_of_block( ) );
    // first, multiply by the nearfield blocks
#pragma omp for
    for ( std::vector< sparse_matrix_type * >::size_type i = 0;
          i < _nearfield_matrices.size( ); ++i ) {
      current_block = _nearfield_matrices.at( i );
      const std::pair< lo, lo > & indices = _nearfield_block_map.at( i );

      const vector_type & local_x = x.get_block( indices.second );

      current_block->apply( local_x, local_y, trans, alpha, 0.0 );
      for ( lo j = 0; j < local_y.size( ); ++j ) {
        y.add_atomic( indices.first, j, local_y[ j ] );
      }
    }

    // next, multiply by the farfield blocks
#pragma omp for
    for ( std::vector< sparse_matrix_type * >::size_type i = 0;
          i < _farfield_matrices.size( ); ++i ) {
      current_block = _farfield_matrices.at( i );
      const std::pair< lo, lo > & indices = _farfield_block_map.at( i );

      const vector_type & local_x = x.get_block( indices.second );

      current_block->apply( local_x, local_y, trans, alpha, 0.0 );
      for ( lo j = 0; j < local_y.size( ); ++j ) {
        y.add_atomic( indices.first, j, local_y[ j ] );
      }
    }
  }

  // lo matrix_idx = 0;
  //#pragma omp parallel

  //  for ( auto it = _nearfield_matrices.begin( );
  //        it != _nearfield_matrices.end( ); ++it ) {
  //    current_block = *it;
  //    const std::pair< lo, lo > & indices = _nearfield_block_map.at(
  //    matrix_idx );
  //
  //    const vector_type & local_x = x.get_block( indices.second );
  //    vector_type & local_y = y.get_block( indices.first );
  //
  //    current_block->apply( local_x, local_y, trans, alpha, 1.0 );
  //
  //    matrix_idx++;
  //  }
  // next, multiply the farfield blocks
  //  matrix_idx = 0;
  //  for ( auto it = _farfield_matrices.begin( ); it != _farfield_matrices.end(
  //  );
  //        ++it ) {
  //    current_block = *it;
  //    const std::pair< lo, lo > & indices = _farfield_block_map.at( matrix_idx
  //    );
  //
  //    const vector_type & local_x = x.get_block( indices.second );
  //    vector_type & local_y = y.get_block( indices.first );
  //
  //    current_block->apply( local_x, local_y, trans, alpha, 1.0 );
  //
  //    matrix_idx++;
  //  }
}

void besthea::linear_algebra::pFMM_matrix::compute_temporal_m2m_matrices( ) {
  mesh::time_cluster_tree * time_tree
    = _spacetime_tree->get_time_cluster_tree( );
  lo n_levels = time_tree->get_levels( );
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

  const std::vector< sc > & paddings = time_tree->get_paddings( );
  sc h_root_no_pad = time_tree->get_root( )->get_half_size( );
  sc h_par_no_pad = h_root_no_pad / 4.0;

  // Initialize class for evaluation of lagrange polynomials and get
  // interpolation nodes in the interval [-1, 1].
  besthea::bem::lagrange_interpolant lagrange( _temp_order );
  const vector_type & nodes = lagrange.get_nodes( );
  vector_type nodes_l_child( _temp_order + 1, false );
  vector_type nodes_r_child( _temp_order + 1, false );
  vector_type values_lagrange( _temp_order + 1, false );
  for ( lo curr_level = 2; curr_level < n_levels - 1; ++curr_level ) {
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

void besthea::linear_algebra::pFMM_matrix::compute_spatial_m2m_coeffs( ) {
  mesh::space_cluster_tree * space_tree
    = _spacetime_tree->get_space_cluster_tree( );
  lo n_levels = space_tree->get_levels( );

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

  const std::vector< sc > & paddings = space_tree->get_paddings( );

  // declare half box side lengths of parent and child cluster + initialize
  vector_type h_par_no_pad( 3, false ), h_child_no_pad( 3, false );
  space_tree->get_root( )->get_half_size( h_par_no_pad );
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
  for ( lo curr_level = 0; curr_level < n_levels - 1; ++curr_level ) {
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
    _chebyshev.evaluate( nodes_l_child_dim_0, all_values_cheb_trf_intrvl );
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

    _chebyshev.evaluate( nodes_r_child_dim_0, all_values_cheb_trf_intrvl );
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
    _chebyshev.evaluate( nodes_l_child_dim_1, all_values_cheb_trf_intrvl );
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

    _chebyshev.evaluate( nodes_r_child_dim_1, all_values_cheb_trf_intrvl );
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
    _chebyshev.evaluate( nodes_l_child_dim_2, all_values_cheb_trf_intrvl );
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

    _chebyshev.evaluate( nodes_r_child_dim_2, all_values_cheb_trf_intrvl );
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

void besthea::linear_algebra::pFMM_matrix::apply_temporal_m2m(
  full_matrix_type const & child_moment, const lo level,
  const bool is_left_child, full_matrix_type & parent_moment ) {
  if ( is_left_child )
    parent_moment.multiply(
      _m2m_matrices_t_left[ level ], child_moment, false, false, 1.0, 1.0 );
  else
    parent_moment.multiply(
      _m2m_matrices_t_right[ level ], child_moment, false, false, 1.0, 1.0 );
}

void besthea::linear_algebra::pFMM_matrix::apply_spatial_m2m(
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
  // TODO move as a member variables to avoid allocations
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

// template< class basis_type >
// void besthea::linear_algebra::pFMM_matrix::apply_s2m_operations( 
//   besthea::bem::fast_spacetime_be_space< basis_type > be_space,
//   block_vector_type const & x ) const { 
// }


// // TODO current implementation just for spatial space p0
// template<>
// void besthea::linear_algebra::pFMM_matrix::apply_s2m_operations( 
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > be_space,
//   block_vector_type const & x ) const 

void besthea::linear_algebra::pFMM_matrix::apply_s2m_operations_p0( 
  block_vector_type const & x ) const {
  lo max_elem_time_cluster = _spacetime_tree->get_time_cluster_tree( )->
    get_n_max_elems_leaf( );
  // TODO needs to be changed for spaces other than p0
  lo max_elem_space_cluster = _spacetime_tree->get_space_cluster_tree( )->
    get_n_max_elems_leaf( );
  full_matrix_type sources( max_elem_time_cluster, max_elem_space_cluster, 
                            false);
  full_matrix_type aux_matrix( max_elem_time_cluster, 
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( ); 
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of current moment and all required matrices
    full_matrix_type & moment = *( ( *it )->get_moment_contribution( ) );
    full_matrix_type & T = (*it)->get_space_cluster( ).get_chebyshev_quad( );
    full_matrix_type & L = (*it)->get_time_cluster( ).get_lagrange_quad( );
    
    // get the relevant entries of the block vector x and store them in sources
    lo n_space_elements = (*it)->get_space_cluster( ).get_n_elements( );
    lo n_time_elements = (*it)->get_time_cluster( ).get_n_elements( );
    sources.resize( n_time_elements, n_space_elements );
    for ( lo i_time = 0; i_time < n_time_elements; ++ i_time ) {
      for ( lo i_space = 0; i_space < n_space_elements; ++ i_space ) {
        sources( i_time, i_space ) = x.get( i_time, i_space );
      }
    }
    
    // compute D = Q * T and then the moment mu = L * D 
    aux_matrix.resize( n_time_elements, 
      ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
    aux_matrix.multiply( sources, T );
    moment.multiply( L, aux_matrix );
  }
}

void besthea::linear_algebra::pFMM_matrix::apply_s2m_operations_p1( 
  block_vector_type const & x ) const {
  lo max_elem_time_cluster = _spacetime_tree->get_time_cluster_tree( )->
    get_n_max_elems_leaf( );
  // TODO needs to be changed for spaces other than p0
  lo max_elem_space_cluster = _spacetime_tree->get_space_cluster_tree( )->
    get_n_max_elems_leaf( );
  // allocate large enough matrix for sources to avoid reallocation
  // (3 * nr_elem is used as bound for the number of vertices in a cluster)
  full_matrix_type sources( max_elem_time_cluster, 3 * max_elem_space_cluster, 
                            false);
  full_matrix_type aux_matrix( max_elem_time_cluster, 
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( ); 
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of current moment and all required matrices
    full_matrix_type & moment = *( ( *it )->get_moment_contribution( ) );
    full_matrix_type & T = (*it)->get_space_cluster( ).get_chebyshev_quad( );
    full_matrix_type & L = (*it)->get_time_cluster( ).get_lagrange_quad( );
    
    // get the relevant entries of the block vector x and store them in sources
    lo n_space_nodes = (*it)->get_space_cluster( ).get_n_nodes( );
    const std::vector< lo > & local_2_global_nodes = 
      (*it)->get_space_cluster( ).get_local_2_global_nodes( );
    lo n_time_elements = (*it)->get_time_cluster( ).get_n_elements( );
    sources.resize( n_time_elements, n_space_nodes );
    for ( lo i_time = 0; i_time < n_time_elements; ++ i_time ) {
      for ( lo i_space = 0; i_space < n_space_nodes; ++ i_space ) {
        sources( i_time, i_space ) = x.get( i_time, 
                                            local_2_global_nodes[ i_space ] );
      }
    }
    
    // compute D = Q * T and then the moment mu = L * D 
    aux_matrix.resize( n_time_elements, 
      ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
    aux_matrix.multiply( sources, T );
    moment.multiply( L, aux_matrix );
  }
}

void besthea::linear_algebra::pFMM_matrix::call_m2m_operations( 
  spacetime_cluster * root, 
  std::vector< full_matrix_type >& buffer_matrices ) {
  if ( root->get_n_children( ) > 0 ) {
    std::vector< spacetime_cluster * > * children = root->get_children( );
    // compute the moment contributions for all children recursively
    for ( std::vector< spacetime_cluster * >::iterator it = children->begin( );
          it != children->end( ); ++ it ) {
      apply_m2m_operations( *it, buffer_matrices );
    }
    lo temporal_level = root->get_time_cluster( ).get_level( );
    full_matrix_type * parent_moment = root->get_moment_contribution( );
    // decide whether to use only temporal m2m or spacetime m2m
    space_cluster & root_space_cluster = root->get_space_cluster( ); 
    space_cluster & child_space_cluster = ( *children )[ 0 ]->
                                            get_space_cluster( );
    bool temporal_only = ( &root_space_cluster == &child_space_cluster );
    if ( temporal_only ) {
      // execute only temporal m2m operation
      for ( std::vector< spacetime_cluster * >::iterator it = children->begin( );
          it != children->end( ); ++ it ) {
        full_matrix_type * child_moment = ( *it )->get_moment_contribution( );
        // determine configuration in time for temporal m2m
        bool is_left_time_child = ( *it )->get_time_cluster( ).is_left_child( );
        apply_temporal_m2m( *child_moment, temporal_level, is_left_time_child,
                            *parent_moment );
      }
    }
    else {
      bool octant_used[ 8 ];
      // apply temporal m2m to all child moments
      for ( std::vector< spacetime_cluster * >::iterator it = children->begin( );
          it != children->end( ); ++ it ) {
        full_matrix_type * child_moment = ( *it )->get_moment_contribution( );
        // determine configuration in space to determine auxiliary target matrix
        // to store result of temporal m2m
        slou octant = ( *it )->get_space_cluster( ).get_octant( );
        // determine configuration in time for temporal m2m
        bool is_left_time_child = ( *it )->get_time_cluster( ).is_left_child( );
        octant_used[ octant ] = true; // remember that octant was used
        apply_temporal_m2m( *child_moment, temporal_level, is_left_time_child,
                            buffer_matrices[ octant ] );
      }
      // compute parent moment with spatial m2m using the auxiliary matrices
      lo spatial_level = root_space_cluster.get_level( );
      for ( slou octant = 0; octant < 8; ++ octant ) {
        if ( octant_used[ octant ] ) {
          apply_spatial_m2m( buffer_matrices[ octant ], spatial_level, octant,
                              *parent_moment );
        }
      }
      // reset buffer matrices to zero if they were used
      for ( slou octant = 0; octant < 8; ++ octant ) {
        if ( octant_used[ octant ] ) {
          buffer_matrices[ octant ].fill( 0.0 );
        }
      }
    }
  }
}

void besthea::linear_algebra::pFMM_matrix::apply_m2l_operation( 
  spacetime_cluster * target_cluster, spacetime_cluster * source_cluster,
  vector_type * buffer_for_integrals ) {
  
  
  slou dimension = 0;
  for (lo i = 0; i < ( _spat_order + 1 ) * ( _spat_order + 1 ) 
        * ( _temp_order + 1) * ( _temp_order + 1 ); ++i )
    buffer_for_integrals[ i ] = 0;
    
  // initialize Chebyshev nodes for numerical integration
  vector_type cheb_nodes( _spat_order + 1, false );
  for ( lo i = 0 ; i <= _spat_order; ++i ) {
    cheb_nodes[ i ] = 
      std::cos( M_PI * ( 2 * i + 1 ) / ( 2 * ( _spat_order + 1 ) ) );
  }
  // evaluate Chebyshev polynomials for all degrees <= _spat_order for integrals
  vector_type all_poly_vals( ( _spat_order + 1 ) * ( _spat_order + 1 ), false );
  _chebyshev.evaluate( cheb_nodes, all_poly_vals );
  
  // get spatial properties ( difference of cluster, half length )
  vector_type source_center( 3, false );
  source_cluster->get_space_cluster( ).get_center( source_center );
  vector_type center_diff( 3, false );
  target_cluster->get_space_cluster( ).get_center( center_diff );
  for ( lo i = 0; i < 3; ++i ) {
    center_diff [ i ] -= source_center[ i ];
  }
  vector_type half_size( 3, false );
  target_cluster->get_space_cluster( ).get_half_size( half_size );
  
  sc h_delta = half_size[ dimension ];
  int center_diff = 2*(i-n_int_boxes);
  //evaluate the gaussian kernel for the numerical integration
  double h_delta = h*h/delta;
  double eval_gaussian[(rho+1)*(rho+1)];
  for (int mu=0; mu<rho+1; mu++) 
  {
      for (int nu=0; nu<rho+1; nu++) 
      {
          eval_gaussian[mu*(rho+1)+nu] = exp(-h_delta * (center_diff+cheb_nodes[nu]-cheb_nodes[mu]) * (center_diff+cheb_nodes[nu]-cheb_nodes[mu]));
      }
  }
  //compute the numerical integrals
  double inv_rho = 1.0/(rho+1)/(rho+1);
  for (int beta=0; beta<=p; ++beta)
      for (int alpha=0; alpha<=p; ++alpha)
      {
          for (int mu=0; mu<=rho; ++mu)
              for (int nu=0; nu<=rho; ++nu)
                  expansion_integrals[crnt_indx] +=  eval_gaussian[mu*(rho+1)+nu] * cheb_poly_eval[alpha*(rho+1)+mu] * cheb_poly_eval[beta*(rho+1)+nu];
          expansion_integrals[crnt_indx] *= inv_rho * 4; //gamma = 2 everywhere (correction later)
          ++crnt_indx;
      }
  for (int k=0; k<=p; ++k) //correction for alpha2=0 or beta2=0
  {   
      expansion_integrals[(p+1)*(p+1)*i+k] *= 0.5;
      expansion_integrals[(p+1)*(p+1)*i+k*(p+1)] *= 0.5;
  }
  
  
  
  
  int nr_coeffs = (p+3)*(p+2)*(p+1)/6;
  double lambda[2][nr_coeffs]; //2d to store lambda^(1) and lambda^(2)
  for (int k=0; k<nr_coeffs; k++) 
  {
      lambda[0][k] = 0;
      lambda[1][k] = 0;
  }
  // efficient calculation of c_beta (Tausch, 2009, p. 3558) (index shift -1 compared to notation there)
  // lambda^(0)(b2,a0,a1) = sum_{a2=0}^{p-a0-a1} E_{b2,a2}*A_{a0,a1,a2}
  int crnt_indx_lmbd_0=0;
  for (int beta2=0; beta2<=p; ++beta2)
  {
      int crnt_indx_a = 0;
      for (int alpha0=0; alpha0<=p-beta2; ++alpha0)
      {
          for (int alpha1=0; alpha1<=p-beta2-alpha0; ++alpha1)
          {
              for (int alpha2=0; alpha2<=p-alpha0-alpha1; ++alpha2)
              {
                  lambda[0][crnt_indx_lmbd_0] += expansion_coeffs[2][beta2*(p+1)+alpha2] * a_alpha[crnt_indx_a];
                  ++crnt_indx_a;
              }
              ++crnt_indx_lmbd_0;
          }
          crnt_indx_a += ((beta2+1)*beta2)/2; //correction for current index; necessary since alpha1 does not run until p-alpha0 as it does in a_alpha;
      }
  }
  // lambda^(1)(b1,b2,a0) = sum_{a1=0}^{p-a0-b2} E_{b1,a1}*lambda^(0)(b2,a0,a1)
  int crnt_indx_lmbd_1 = 0;
  for (int beta1=0; beta1<=p; ++beta1)
  {
      crnt_indx_lmbd_0 = 0;
      for (int beta2=0; beta2<=p-beta1; ++beta2)
      {
          for (int alpha0=0; alpha0<=p-beta1-beta2; ++alpha0)
          {
              for (int alpha1=0; alpha1<=p-alpha0-beta2; ++ alpha1)
              {
                  lambda[1][crnt_indx_lmbd_1] += expansion_coeffs[1][beta1*(p+1)+alpha1] * lambda[0][crnt_indx_lmbd_0];
                  ++crnt_indx_lmbd_0;
              }
              ++crnt_indx_lmbd_1;
          }
          crnt_indx_lmbd_0 += ((beta1+1)*beta1)/2; //correction for current index; necessary since alpha0 does not run until p-beta2 as it does in lambda[0];
      }
  }
  // C(b0,b1,b2) = sum_{a0=0}^{p-b2-b1} E_{b0,a0}*lambda^(1)(b1,b2,a0)
  int crnt_indx_c = 0;
  for (int beta0=0; beta0<=p; ++beta0)
  {
      crnt_indx_lmbd_1 = 0;
      for (int beta1=0; beta1<=p-beta0; ++beta1)
      {
          for (int beta2=0; beta2<=p-beta0-beta1; ++beta2)
          {
              for (int alpha0=0; alpha0<=p-beta1-beta2; ++alpha0)
              {
                  c_beta[crnt_indx_c] += expansion_coeffs[0][beta0*(p+1)+alpha0] * lambda[1][crnt_indx_lmbd_1];
                  ++crnt_indx_lmbd_1;
              }
              ++crnt_indx_c;
          }
          crnt_indx_lmbd_1 += ((beta0+1)*beta0)/2; //correction for current index; necessary since beta2 does not run until p-beta1 as it does in lambda[0];
      }
  }
}

