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

template< class kernel_type, class target_space, class source_space >
besthea::linear_algebra::full_matrix *
besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::create_nearfield_matrix( lo test_idx, lo trial_idx,
  lo n_duplications ) {
  full_matrix_type * local_matrix
    = new full_matrix_type( _dim_domain, _dim_range );

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

template< class kernel_type, class target_space, class source_space >
besthea::linear_algebra::full_matrix *
besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::create_farfield_matrix( lo test_idx, lo trial_idx ) {
  full_matrix_type * local_matrix
    = new full_matrix_type( _dim_domain, _dim_range );

  _farfield_matrices.push_back( local_matrix );

  _farfield_block_map.push_back( std::make_pair( test_idx, trial_idx ) );

  return local_matrix;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply( const block_vector_type & x, block_vector_type & y,
  bool trans, sc alpha, sc beta ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED" << std::endl;
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply( const block_vector_type & x, block_vector_type & y, bool trans,
    sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply( const block_vector_type & x, block_vector_type & y, bool trans,
    sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply( const block_vector_type & x, block_vector_type & y, bool trans,
    sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply( const block_vector_type & x, block_vector_type & y, bool trans,
    sc alpha, sc beta ) const {
  apply_hs( x, y, trans, alpha, beta );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_sl_dl( const block_vector_type & x,
  block_vector_type & y, bool trans, sc alpha, sc beta ) const {
// Specialization for the single and double layer operators
#pragma omp parallel for schedule( static )
  for ( lo i = 0; i < y.get_block_size( ); ++i ) {
    for ( lo j = 0; j < y.get_size_of_block( ); ++j ) {
      y.set( i, j, y.get( i, j ) * beta );
    }
  }

  full_matrix_type * current_block;

#pragma omp parallel
  {
    vector_type local_y( y.get_size_of_block( ) );
    // first, multiply by the nearfield blocks
#pragma omp for
    for ( std::vector< full_matrix_type * >::size_type i = 0;
          i < _nearfield_matrices.size( ); ++i ) {
      current_block = _nearfield_matrices.at( i );
      const std::pair< lo, lo > & indices = _nearfield_block_map.at( i );

      const vector_type & local_x = x.get_block( indices.second );

      current_block->apply( local_x, local_y, trans, alpha, 0.0 );
      for ( lo j = 0; j < local_y.size( ); ++j ) {
        y.add_atomic( indices.first, j, local_y[ j ] );
      }
    }
  }

  // Next, use the pFMM for the computation of the farfield contribution
  // according to the respective spaces. The result is stored in an auxiliary
  // vector y_pFMM and then added to y

  // reset all moment and local contributions to zero
  _spacetime_tree->clean_local_contributions( _spacetime_tree->get_root( ) );
  _spacetime_tree->clean_moment_contributions( _spacetime_tree->get_root( ) );

  // allocate buffers for the operations
  std::vector< full_matrix_type > buffer_matrices;
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
  full_matrix_type aux_buffer_0( ( _temp_order + 1 ) * ( _temp_order + 1 ),
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  full_matrix_type aux_buffer_1( ( _temp_order + 1 ) * ( _temp_order + 1 ),
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  block_vector_type y_pFMM( y.get_block_size( ), y.get_size_of_block( ), true );

  // S2M, M2M, M2L, L2L and L2T steps:

  apply_s2m_operations( x, trans );

  call_m2m_operations( _spacetime_tree->get_root( ), buffer_matrices );
  call_m2l_operations( _spacetime_tree->get_root( ), buffer_for_gaussians,
    buffer_for_coeffs, aux_buffer_0, aux_buffer_1 );
  call_l2l_operations( _spacetime_tree->get_root( ), buffer_matrices );

  apply_l2t_operations( y_pFMM, trans );

  // Add the scaled result to y.
  y.add( y_pFMM, alpha );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_hs( const block_vector_type & x, block_vector_type & y,
  bool trans, sc alpha, sc beta ) const {
  // Specialization for hypersingular operator

#pragma omp parallel for schedule( static )
  for ( lo i = 0; i < y.get_block_size( ); ++i ) {
    for ( lo j = 0; j < y.get_size_of_block( ); ++j ) {
      y.set( i, j, y.get( i, j ) * beta );
    }
  }

  full_matrix_type * current_block;

#pragma omp parallel
  {
    vector_type local_y( y.get_size_of_block( ) );
    // first, multiply by the nearfield blocks
#pragma omp for
    for ( std::vector< full_matrix_type * >::size_type i = 0;
          i < _nearfield_matrices.size( ); ++i ) {
      current_block = _nearfield_matrices.at( i );
      const std::pair< lo, lo > & indices = _nearfield_block_map.at( i );

      const vector_type & local_x = x.get_block( indices.second );

      current_block->apply( local_x, local_y, trans, alpha, 0.0 );
      for ( lo j = 0; j < local_y.size( ); ++j ) {
        y.add_atomic( indices.first, j, local_y[ j ] );
      }
    }
  }

  // Next, use the pFMM for the computation of the farfield contribution
  // according to the respective spaces. The result is stored in an auxiliary
  // vector y_pFMM and then added to y

  //  Allocate buffers for the operations
  std::vector< full_matrix_type > buffer_matrices;
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
  full_matrix_type aux_buffer_0( ( _temp_order + 1 ) * ( _temp_order + 1 ),
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  full_matrix_type aux_buffer_1( ( _temp_order + 1 ) * ( _temp_order + 1 ),
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  block_vector_type y_pFMM( y.get_block_size( ), y.get_size_of_block( ), true );

  // 3 "pFMM steps" for first term of hypersingular operator (curls)
  for ( lo dim = 0; dim < 3; ++dim ) {
    // reset all moment and local contributions to zero
    _spacetime_tree->clean_local_contributions( _spacetime_tree->get_root( ) );
    _spacetime_tree->clean_moment_contributions( _spacetime_tree->get_root( ) );
    apply_s2m_operations_curl_p1_hs( x, dim );
    call_m2m_operations( _spacetime_tree->get_root( ), buffer_matrices );
    call_m2l_operations( _spacetime_tree->get_root( ), buffer_for_gaussians,
      buffer_for_coeffs, aux_buffer_0, aux_buffer_1 );
    call_l2l_operations( _spacetime_tree->get_root( ), buffer_matrices );
    apply_l2t_operations_curl_p1_hs( y_pFMM, dim );
  }
  // Add the scaled result to y.
  // (additional scaling by square of heat coefficient)
  y.add( y_pFMM, alpha * _alpha * _alpha );

  // reset y_pFMM to zero
  y_pFMM.fill( 0.0 );

  // 3 "pFMM steps" for the second term of hypersingular operator (normals and
  // time derivative in source term)
  for ( lo dim = 0; dim < 3; ++dim ) {
    // reset all moment and local contributions to zero
    _spacetime_tree->clean_local_contributions( _spacetime_tree->get_root( ) );
    _spacetime_tree->clean_moment_contributions( _spacetime_tree->get_root( ) );
    apply_s2m_operations_p1_normal_hs( x, dim );
    call_m2m_operations( _spacetime_tree->get_root( ), buffer_matrices );
    call_m2l_operations( _spacetime_tree->get_root( ), buffer_for_gaussians,
      buffer_for_coeffs, aux_buffer_0, aux_buffer_1 );
    call_l2l_operations( _spacetime_tree->get_root( ), buffer_matrices );
    apply_l2t_operations_p1_normal_hs( y_pFMM, dim );
  }

  // Subtract the scaled result from y.
  // (additional scaling by heat coefficient)
  y.add( y_pFMM, -alpha * _alpha );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::compute_temporal_m2m_matrices( ) {
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

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::compute_spatial_m2m_coeffs( ) {
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
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_temporal_m2m( full_matrix_type const & child_moment,
  const lo level, const bool is_left_child,
  full_matrix_type & parent_moment ) const {
  if ( is_left_child )
    parent_moment.multiply(
      _m2m_matrices_t_left[ level ], child_moment, false, false, 1.0, 1.0 );
  else
    parent_moment.multiply(
      _m2m_matrices_t_right[ level ], child_moment, false, false, 1.0, 1.0 );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_spatial_m2m( full_matrix_type const & child_moment,
  const lo level, const slou octant, full_matrix_type & parent_moment ) const {
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
  // TODO use buffers to avoid allocations
  full_matrix_type lambda_1( _temp_order + 1, n_coeffs_s, true );
  full_matrix_type lambda_2( _temp_order + 1, n_coeffs_s, true );

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
              * child_moment( b, child_index );
          }
          ++child_index;
        }
        // correction needed for skipt entries of child_moment
        child_index += _spat_order + 1 - alpha0 - alpha1 - alpha2;
      }
      // correction for current index; necessary since alpha1 does not run until
      // _spat_order - alpha0 as it does in stored child_moment
      child_index += ( ( beta2 + 1 ) * beta2 ) / 2;
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

  lo parent_index = 0;
  for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
    for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
        for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            parent_moment( b, parent_index )
              += ( *m2m_coeffs_s_dim_0 )[ beta0 * ( _spat_order + 1 ) + alpha0 ]
              * lambda_2( b,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                  + ( _spat_order + 1 ) * beta2 + alpha0 );
          }
        }
        ++parent_index;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_temporal_l2l( full_matrix_type const & parent_local,
  const lo level, const bool is_left_child,
  full_matrix_type & child_local ) const {
  if ( is_left_child )
    // use transposed left temporal m2m matrix for multiplication
    child_local.multiply(
      _m2m_matrices_t_left[ level ], parent_local, true, false, 1.0, 1.0 );
  else
    // use transposed left temporal m2m matrix for multiplication
    child_local.multiply(
      _m2m_matrices_t_right[ level ], parent_local, true, false, 1.0, 1.0 );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_spatial_l2l( full_matrix_type const & parent_local,
  const lo level, const slou octant, full_matrix_type & child_local ) const {
  // get all m2m coefficients ( i.e. transposed l2l coefficients )
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
  // TODO instead of allocating every time use buffers
  full_matrix_type lambda_1( _temp_order + 1, n_coeffs_s, true );
  full_matrix_type lambda_2( _temp_order + 1, n_coeffs_s, true );

  for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
    lou parent_index = 0;
    for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
      for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0; ++alpha1 ) {
        // correction for skipt entries of parent_local due to starting point
        // alpha2 = beta2 in the next loop
        parent_index += beta2;
        for ( lo alpha2 = beta2; alpha2 <= _spat_order - alpha0 - alpha1;
              ++alpha2 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            lambda_1( a,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 )
              += ( *m2m_coeffs_s_dim_2 )[ alpha2 * ( _spat_order + 1 ) + beta2 ]
              * parent_local( a, parent_index );
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
            child_local( a, child_index )
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
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_s2m_operations( block_vector_type const & x,
  bool trans ) const {
  std::cout << "apply_s2m_operations: NOT IMPLEMENTED" << std::endl;
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_s2m_operations( block_vector_type const & x, bool trans ) const {
  apply_s2m_operations_p0( x );
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_s2m_operations( block_vector_type const & x, bool trans ) const {
  apply_s2m_operations_p1( x );
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_s2m_operations( block_vector_type const & x, bool trans ) const {
  if ( trans ) {
    apply_s2m_operations_p0( x );
  } else {
    apply_s2m_operations_p1( x );
  }
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_s2m_operations( block_vector_type const & x, bool trans ) const {
  apply_s2m_operations_p0( x );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_s2m_operations_p0( block_vector_type const & x ) const {
  lo max_elem_time_cluster
    = _spacetime_tree->get_time_cluster_tree( )->get_n_max_elems_leaf( );
  lo max_elem_space_cluster
    = _spacetime_tree->get_space_cluster_tree( )->get_n_max_elems_leaf( );
  full_matrix_type sources(
    max_elem_time_cluster, max_elem_space_cluster, false );
  full_matrix_type aux_matrix( max_elem_time_cluster,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( );
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of current moment and all required matrices
    full_matrix_type & moment = *( ( *it )->get_moment_contribution( ) );
    full_matrix_type & T = ( *it )->get_space_cluster( ).get_chebyshev_quad( );
    full_matrix_type & L = ( *it )->get_time_cluster( ).get_lagrange_quad( );
    // get the relevant entries of the block vector x and store them in sources
    lo n_space_elements = ( *it )->get_space_cluster( ).get_n_elements( );
    lo n_time_elements = ( *it )->get_time_cluster( ).get_n_elements( );
    const std::vector< lo > & space_elements
      = ( *it )->get_space_cluster( ).get_all_elements( );
    const std::vector< lo > & time_elements
      = ( *it )->get_time_cluster( ).get_all_elements( );
    sources.resize( n_time_elements, n_space_elements );
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
        sources( i_time, i_space )
          = x.get( time_elements[ i_time ], space_elements[ i_space ] );
      }
    }

    // compute D = Q * T and then the moment mu = L * D
    aux_matrix.resize( n_time_elements,
      ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
    aux_matrix.multiply( sources, T );
    moment.multiply( L, aux_matrix );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_s2m_operations_p1( block_vector_type const & x ) const {
  lo max_elem_time_cluster
    = _spacetime_tree->get_time_cluster_tree( )->get_n_max_elems_leaf( );
  lo max_elem_space_cluster
    = _spacetime_tree->get_space_cluster_tree( )->get_n_max_elems_leaf( );
  // allocate large enough matrix for sources to avoid reallocation
  // (3 * nr_elem is used as bound for the number of vertices in a cluster)
  full_matrix_type sources(
    max_elem_time_cluster, 3 * max_elem_space_cluster, false );
  full_matrix_type aux_matrix( max_elem_time_cluster,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( );
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of current moment and all required matrices
    full_matrix_type & moment = *( ( *it )->get_moment_contribution( ) );
    full_matrix_type & T_drv
      = ( *it )->get_space_cluster( ).get_normal_drv_chebyshev_quad( );
    full_matrix_type & L = ( *it )->get_time_cluster( ).get_lagrange_quad( );

    // get the relevant entries of the block vector x and store them in sources
    lo n_space_nodes = ( *it )->get_space_cluster( ).get_n_nodes( );
    const std::vector< lo > & local_2_global_nodes
      = ( *it )->get_space_cluster( ).get_local_2_global_nodes( );
    lo n_time_elements = ( *it )->get_time_cluster( ).get_n_elements( );
    const std::vector< lo > & time_elements
      = ( *it )->get_time_cluster( ).get_all_elements( );
    sources.resize( n_time_elements, n_space_nodes );
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
        sources( i_time, i_space )
          = x.get( time_elements[ i_time ], local_2_global_nodes[ i_space ] );
      }
    }

    // compute D = Q * T_drv and then the moment mu = L * D
    aux_matrix.resize( n_time_elements,
      ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
    aux_matrix.multiply( sources, T_drv );
    moment.multiply( L, aux_matrix );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_s2m_operations_curl_p1_hs( block_vector_type const & x,
  const lo dim ) const {
  lo max_elem_time_cluster
    = _spacetime_tree->get_time_cluster_tree( )->get_n_max_elems_leaf( );
  lo max_elem_space_cluster
    = _spacetime_tree->get_space_cluster_tree( )->get_n_max_elems_leaf( );
  lo n_multi_indices
    = ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
  // allocate large enough matrix for sources to avoid reallocation
  // (3 * nr_elem is used as bound for the number of vertices in a cluster)
  full_matrix_type sources(
    max_elem_time_cluster, 3 * max_elem_space_cluster, false );
  full_matrix_type T_curl( 3 * max_elem_space_cluster, n_multi_indices, false );
  full_matrix_type aux_matrix( max_elem_time_cluster, n_multi_indices, false );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( );
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of current moment and all required matrices
    full_matrix_type & moment = *( ( *it )->get_moment_contribution( ) );
    full_matrix_type & T = ( *it )->get_space_cluster( ).get_chebyshev_quad( );
    full_matrix_type & L = ( *it )->get_time_cluster( ).get_lagrange_quad( );

    // get the relevant entries of the block vector x and store them in sources
    lo n_space_nodes = ( *it )->get_space_cluster( ).get_n_nodes( );
    lo n_space_elements = ( *it )->get_space_cluster( ).get_n_elements( );
    const std::vector< lo > & local_2_global_nodes
      = ( *it )->get_space_cluster( ).get_local_2_global_nodes( );
    lo n_time_elements = ( *it )->get_time_cluster( ).get_n_elements( );
    const std::vector< lo > & time_elements
      = ( *it )->get_time_cluster( ).get_all_elements( );
    sources.resize( n_time_elements, n_space_nodes );
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
        sources( i_time, i_space )
          = x.get( time_elements[ i_time ], local_2_global_nodes[ i_space ] );
      }
    }
    T_curl.resize( n_space_nodes, n_multi_indices );
    T_curl.fill( 0.0 );
    const std::vector< lo > & elems_2_local_nodes
      = ( *it )->get_space_cluster( ).get_elems_2_local_nodes( );
    const std::vector< sc > & surf_curls_curr_dim
      = ( *it )->get_space_cluster( ).get_surf_curls( dim );
    // use quadrature of plain Chebyshev polynomials to compute quadrature of
    // Chebyshev polynomials times surface curls of p1 basis functions.
    for ( lo i_beta = 0; i_beta < n_multi_indices; ++i_beta ) {
      for ( lo i_el = 0; i_el < n_space_elements; ++i_el ) {
        T_curl.add_atomic( elems_2_local_nodes[ 3 * i_el ], i_beta,
          surf_curls_curr_dim[ 3 * i_el ] * T.get( i_el, i_beta ) );
        T_curl.add_atomic( elems_2_local_nodes[ 3 * i_el + 1 ], i_beta,
          surf_curls_curr_dim[ 3 * i_el + 1 ] * T.get( i_el, i_beta ) );
        T_curl.add_atomic( elems_2_local_nodes[ 3 * i_el + 2 ], i_beta,
          surf_curls_curr_dim[ 3 * i_el + 2 ] * T.get( i_el, i_beta ) );
      }
    }
    // compute D = Q * T_curl and then the moment mu = L * D
    aux_matrix.resize( n_time_elements, n_multi_indices );
    aux_matrix.multiply( sources, T_curl );
    moment.multiply( L, aux_matrix );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_s2m_operations_p1_normal_hs( block_vector_type const &
                                                       x,
  const lo dim ) const {
  lo max_elem_time_cluster
    = _spacetime_tree->get_time_cluster_tree( )->get_n_max_elems_leaf( );
  lo max_elem_space_cluster
    = _spacetime_tree->get_space_cluster_tree( )->get_n_max_elems_leaf( );
  lo n_multi_indices
    = ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
  // allocate large enough matrix for sources to avoid reallocation
  // (3 * nr_elem is used as bound for the number of vertices in a cluster)
  full_matrix_type sources(
    max_elem_time_cluster, 3 * max_elem_space_cluster, false );
  full_matrix_type aux_matrix( max_elem_time_cluster, n_multi_indices, false );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( );
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of current moment and all required matrices
    full_matrix_type & moment = *( ( *it )->get_moment_contribution( ) );
    full_matrix_type & T_normal
      = ( *it )->get_space_cluster( ).get_cheb_times_normal_quad( dim );
    full_matrix_type & L_drv
      = ( *it )->get_time_cluster( ).get_lagrange_drv_int( );

    // get the relevant entries of the block vector x and store them in sources
    lo n_space_nodes = ( *it )->get_space_cluster( ).get_n_nodes( );
    const std::vector< lo > & local_2_global_nodes
      = ( *it )->get_space_cluster( ).get_local_2_global_nodes( );
    lo n_time_elements = ( *it )->get_time_cluster( ).get_n_elements( );
    const std::vector< lo > & time_elements
      = ( *it )->get_time_cluster( ).get_all_elements( );
    sources.resize( n_time_elements, n_space_nodes );
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
        sources( i_time, i_space )
          = x.get( time_elements[ i_time ], local_2_global_nodes[ i_space ] );
      }
    }
    // compute D = Q * T_normal and then the moment mu = L * D
    aux_matrix.resize( n_time_elements, n_multi_indices );
    aux_matrix.multiply( sources, T_normal );
    moment.multiply( L_drv, aux_matrix );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::call_m2m_operations( spacetime_cluster * root,
  std::vector< full_matrix_type > & buffer_matrices ) const {
  if ( root->get_n_children( ) > 0 ) {
    std::vector< spacetime_cluster * > * children = root->get_children( );
    // compute the moment contributions for all children recursively
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      call_m2m_operations( *it, buffer_matrices );
    }
    lo temporal_level = root->get_time_cluster( ).get_level( );
    full_matrix_type * parent_moment = root->get_moment_contribution( );
    // decide whether to use only temporal m2m or spacetime m2m
    space_cluster & root_space_cluster = root->get_space_cluster( );
    space_cluster & child_space_cluster
      = ( *children )[ 0 ]->get_space_cluster( );
    bool temporal_only = ( &root_space_cluster == &child_space_cluster );
    if ( temporal_only ) {
      // execute only temporal m2m operation
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        full_matrix_type * child_moment = ( *it )->get_moment_contribution( );
        // determine configuration in time for temporal m2m
        bool is_left_time_child = ( *it )->get_time_cluster( ).is_left_child( );
        apply_temporal_m2m(
          *child_moment, temporal_level, is_left_time_child, *parent_moment );
      }
    } else {
      bool octant_used[ 8 ];
      // apply temporal m2m to all child moments
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        full_matrix_type * child_moment = ( *it )->get_moment_contribution( );
        // determine configuration in space to determine auxiliary target matrix
        // to store result of temporal m2m
        slou octant = ( *it )->get_space_cluster( ).get_octant( );
        // determine configuration in time for temporal m2m
        bool is_left_time_child = ( *it )->get_time_cluster( ).is_left_child( );
        octant_used[ octant ] = true;  // remember that octant was used
        apply_temporal_m2m( *child_moment, temporal_level, is_left_time_child,
          buffer_matrices[ octant ] );
      }
      // compute parent moment with spatial m2m using the auxiliary matrices
      lo spatial_level = root_space_cluster.get_level( );
      for ( slou octant = 0; octant < 8; ++octant ) {
        if ( octant_used[ octant ] ) {
          apply_spatial_m2m(
            buffer_matrices[ octant ], spatial_level, octant, *parent_moment );
        }
      }
      // reset buffer matrices to zero if they were used
      for ( slou octant = 0; octant < 8; ++octant ) {
        if ( octant_used[ octant ] ) {
          buffer_matrices[ octant ].fill( 0.0 );
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_m2l_operation( spacetime_cluster * target_cluster,
  spacetime_cluster * source_cluster, vector_type & buffer_for_gaussians,
  vector_type & buffer_for_coeffs, full_matrix_type & aux_buffer_0,
  full_matrix_type & aux_buffer_1 ) const {
  // initialize temporal interpolation nodes in source and target cluster
  vector_type src_time_nodes( _temp_order + 1, false );
  vector_type tar_time_nodes( _temp_order + 1, false );
  vector_type time_nodes( _temp_order + 1, false );
  sc padding_time
    = _spacetime_tree->get_time_tree( )
        ->get_paddings( )[ source_cluster->get_time_cluster( ).get_level( ) ];
  sc half_size_time = source_cluster->get_time_cluster( ).get_half_size( );
  half_size_time += padding_time;
  sc src_center_time = source_cluster->get_time_cluster( ).get_center( );
  sc tar_center_time = target_cluster->get_time_cluster( ).get_center( );

  // TODO store and get time_nodes in [-1,1] from somewhere
  // (to guarantee that the same interpolation nodes are used everywhere)
  for ( lo i = 0; i <= _temp_order; ++i ) {
    time_nodes[ i ]
      = std::cos( M_PI * ( 2 * i + 1 ) / ( 2 * ( _temp_order + 1 ) ) );
    tar_time_nodes[ i ] = tar_center_time + half_size_time * time_nodes[ i ];
    src_time_nodes[ i ] = src_center_time + half_size_time * time_nodes[ i ];
  }

  // initialize Chebyshev nodes for numerical integration
  vector_type cheb_nodes( _m2l_integration_order + 1, false );
  for ( lo i = 0; i <= _m2l_integration_order; ++i ) {
    cheb_nodes[ i ] = std::cos(
      M_PI * ( 2 * i + 1 ) / ( 2 * ( _m2l_integration_order + 1 ) ) );
  }
  // evaluate Chebyshev polynomials for all degrees <= _spat_order for integrals
  vector_type all_poly_vals(
    ( _m2l_integration_order + 1 ) * ( _spat_order + 1 ), false );
  _chebyshev.evaluate( cheb_nodes, all_poly_vals );

  // get spatial properties ( difference of cluster, half length )
  vector_type source_center_space( 3, false );
  source_cluster->get_space_cluster( ).get_center( source_center_space );
  vector_type center_diff_space( 3, false );
  target_cluster->get_space_cluster( ).get_center( center_diff_space );
  for ( lo i = 0; i < 3; ++i ) {
    center_diff_space[ i ] -= source_center_space[ i ];
  }
  vector_type half_size_space( 3, false );
  target_cluster->get_space_cluster( ).get_half_size( half_size_space );
  sc padding_space
    = _spacetime_tree->get_space_tree( )
        ->get_paddings( )[ source_cluster->get_space_cluster( ).get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // compute coupling coefficients for dimension 2
  compute_coupling_coeffs( src_time_nodes, tar_time_nodes, cheb_nodes,
    all_poly_vals, half_size_space[ 2 ], center_diff_space[ 2 ],
    buffer_for_gaussians, buffer_for_coeffs );

  // set entries of auxiliary vectors to zero
  aux_buffer_0.fill( 0.0 );
  aux_buffer_1.fill( 0.0 );

  full_matrix_type & src_moment
    = *( source_cluster->get_moment_contribution( ) );
  full_matrix_type & tar_local = *( target_cluster->get_local_contribution( ) );
  // efficient m2l operation similar to Tausch, 2009, p. 3558
  // help variables for accessing right values in coefficient buffer
  lo hlp_acs_alpha
    = ( _spat_order + 1 ) * ( _temp_order + 1 ) * ( _temp_order + 1 );
  lo hlp_acs_beta = ( _temp_order + 1 ) * ( _temp_order + 1 );
  lo hlp_acs_a = ( _temp_order + 1 );
  // compute first intermediate product and store it in aux_buffer_0
  lo buffer_0_index = 0;
  for ( lo alpha2 = 0; alpha2 <= _spat_order; ++alpha2 ) {
    lo moment_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order - alpha2; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - alpha2 - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            for ( lo b = 0; b <= _temp_order; ++b ) {
              aux_buffer_0( hlp_acs_a * a + b, buffer_0_index )
                += buffer_for_coeffs[ alpha2 * hlp_acs_alpha
                     + beta2 * hlp_acs_beta + a * hlp_acs_a + b ]
                * src_moment( b, moment_index );
            }
          }
          ++moment_index;
        }
        ++buffer_0_index;
      }
      // correction for moment index; this is necessary since beta1 does not run
      // until _spat_order - beta0 as it does in src_moment;
      moment_index += ( ( alpha2 + 1 ) * alpha2 ) / 2;
    }
  }
  // update coefficients and compute 2nd intermediate product in aux_buffer_1
  compute_coupling_coeffs( src_time_nodes, tar_time_nodes, cheb_nodes,
    all_poly_vals, half_size_space[ 1 ], center_diff_space[ 1 ],
    buffer_for_gaussians, buffer_for_coeffs );
  lo buffer_1_index = 0;
  for ( lo alpha1 = 0; alpha1 <= _spat_order; ++alpha1 ) {
    buffer_0_index = 0;
    for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha1; ++alpha2 ) {
      for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
        for ( lo beta1 = 0; beta1 <= _spat_order - beta0 - alpha2; ++beta1 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            for ( lo b = 0; b <= _temp_order; ++b ) {
              aux_buffer_1( hlp_acs_a * a + b, buffer_1_index )
                += buffer_for_coeffs[ alpha1 * hlp_acs_alpha
                     + beta1 * hlp_acs_beta + a * hlp_acs_a + b ]
                * aux_buffer_0( hlp_acs_a * a + b, buffer_0_index );
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
  // update coefficients and update targets local coefficient with m2l result
  compute_coupling_coeffs( src_time_nodes, tar_time_nodes, cheb_nodes,
    all_poly_vals, half_size_space[ 0 ], center_diff_space[ 0 ],
    buffer_for_gaussians, buffer_for_coeffs );

  // C(b0,b1,b2) = sum_{a0=0}^{p-b2-b1} E_{b0,a0}*lambda^(1)(b1,b2,a0)
  int local_index = 0;
  for ( lo alpha0 = 0; alpha0 <= _spat_order; ++alpha0 ) {
    buffer_1_index = 0;
    for ( lo alpha1 = 0; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
      for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha0 - alpha1; ++alpha2 ) {
        for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            for ( lo b = 0; b <= _temp_order; ++b ) {
              tar_local( a, local_index )
                += buffer_for_coeffs[ alpha0 * hlp_acs_alpha
                     + beta0 * hlp_acs_beta + a * hlp_acs_a + b ]
                * aux_buffer_1( hlp_acs_a * a + b, buffer_1_index );
            }
          }
          ++buffer_1_index;
        }
        ++local_index;
      }
      // correction for buffer_1 index; this is necessary since alpha0 does not
      // run until _spat_order - alpha1 as it does in aux_buffer_1;
      buffer_1_index += ( ( alpha0 + 1 ) * alpha0 ) / 2;
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::call_m2l_operations( spacetime_cluster * root,
  vector_type & buffer_for_gaussians, vector_type & buffer_for_coeffs,
  full_matrix_type & aux_buffer_0, full_matrix_type & aux_buffer_1 ) const {
  std::vector< spacetime_cluster * > * interaction_list
    = root->get_interaction_list( );
  // apply m2l operations for all clusters in the interaction list
  if ( interaction_list != nullptr ) {
    for ( auto it = interaction_list->begin( ); it != interaction_list->end( );
          ++it ) {
      apply_m2l_operation( root, *it, buffer_for_gaussians, buffer_for_coeffs,
        aux_buffer_0, aux_buffer_1 );
    }
  }
  // call the routine recursively for all children
  if ( root->get_n_children( ) > 0 ) {
    std::vector< spacetime_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      call_m2l_operations( *it, buffer_for_gaussians, buffer_for_coeffs,
        aux_buffer_0, aux_buffer_1 );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::call_l2l_operations( spacetime_cluster * root,
  std::vector< full_matrix_type > & buffer_matrices ) const {
  if ( root->get_n_children( ) > 0 ) {
    std::vector< spacetime_cluster * > * children = root->get_children( );
    lo temporal_level = root->get_time_cluster( ).get_level( );
    full_matrix_type * parent_local = root->get_local_contribution( );
    // decide whether to use only temporal m2m or spacetime m2m
    space_cluster & root_space_cluster = root->get_space_cluster( );
    space_cluster & child_space_cluster
      = ( *children )[ 0 ]->get_space_cluster( );
    bool temporal_only = ( &root_space_cluster == &child_space_cluster );
    if ( temporal_only ) {
      // execute only temporal l2l operation
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        full_matrix_type * child_local = ( *it )->get_local_contribution( );
        // determine configuration in time for temporal m2m
        bool is_left_time_child = ( *it )->get_time_cluster( ).is_left_child( );
        apply_temporal_l2l(
          *parent_local, temporal_level, is_left_time_child, *child_local );
      }
    } else {
      bool octant_used[ 8 ];
      // check which of the 8 configurations in space are active
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        octant_used[ ( *it )->get_space_cluster( ).get_octant( ) ] = true;
      }
      lo spatial_level = root_space_cluster.get_level( );
      for ( slou octant = 0; octant < 8; ++octant ) {
        if ( octant_used[ octant ] ) {
          apply_spatial_l2l(
            *parent_local, spatial_level, octant, buffer_matrices[ octant ] );
        }
      }
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        full_matrix_type * child_local = ( *it )->get_local_contribution( );
        // determine configuration in space to determine auxiliary source matrix
        // from which the temporal l2l is computed
        slou octant = ( *it )->get_space_cluster( ).get_octant( );
        // determine configuration in time for temporal m2m
        bool is_left_time_child = ( *it )->get_time_cluster( ).is_left_child( );
        apply_temporal_l2l( buffer_matrices[ octant ], temporal_level,
          is_left_time_child, *child_local );
      }
      // reset buffer matrices to zero if they were used
      for ( slou octant = 0; octant < 8; ++octant ) {
        if ( octant_used[ octant ] ) {
          buffer_matrices[ octant ].fill( 0.0 );
        }
      }
    }
    // let the children pass the local contributions down recursively
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      call_l2l_operations( *it, buffer_matrices );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_l2t_operations( block_vector_type & y,
  bool trans ) const {
  std::cout << "apply_l2t_operations: NOT IMPLEMENTED" << std::endl;
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_l2t_operations( block_vector_type & y, bool trans ) const {
  this->apply_l2t_operations_p0( y );
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::basis_tri_p1,
  besthea::bem::basis_tri_p1 >::apply_l2t_operations( block_vector_type & y,
  bool trans ) const {
  this->apply_l2t_operations_p1( y );
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_l2t_operations( block_vector_type & y, bool trans ) const {
  if ( trans ) {
    this->apply_l2t_operations_p1( y );
  } else {
    this->apply_l2t_operations_p0( y );
  }
}

template<>
void besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_l2t_operations( block_vector_type & y, bool trans ) const {
  this->apply_l2t_operations_p0( y );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_l2t_operations_p0( block_vector_type & y ) const {
  lo max_elem_time_cluster
    = _spacetime_tree->get_time_cluster_tree( )->get_n_max_elems_leaf( );
  lo max_elem_space_cluster
    = _spacetime_tree->get_space_cluster_tree( )->get_n_max_elems_leaf( );
  full_matrix_type targets(
    max_elem_time_cluster, max_elem_space_cluster, false );
  full_matrix_type aux_matrix( max_elem_time_cluster,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( );
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of local moment and all required matrices
    full_matrix_type & local = *( ( *it )->get_local_contribution( ) );
    full_matrix_type & T = ( *it )->get_space_cluster( ).get_chebyshev_quad( );
    full_matrix_type & L = ( *it )->get_time_cluster( ).get_lagrange_quad( );
    lo n_space_elements = ( *it )->get_space_cluster( ).get_n_elements( );
    lo n_time_elements = ( *it )->get_time_cluster( ).get_n_elements( );
    const std::vector< lo > & space_elements
      = ( *it )->get_space_cluster( ).get_all_elements( );
    const std::vector< lo > & time_elements
      = ( *it )->get_time_cluster( ).get_all_elements( );
    // resize auxiliary matrix and matrix for targets appropriately
    aux_matrix.resize( n_time_elements,
      ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
    targets.resize( n_time_elements, n_space_elements );
    // compute D = trans(L) * lambda and then the result Y = D * trans(T)
    aux_matrix.multiply( L, local, true, false );
    targets.multiply( aux_matrix, T, false, true );
    // add the results to the correct position in the result vector
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
        y.add( time_elements[ i_time ], space_elements[ i_space ],
          targets( i_time, i_space ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_l2t_operations_p1( block_vector_type & y ) const {
  lo max_elem_time_cluster
    = _spacetime_tree->get_time_cluster_tree( )->get_n_max_elems_leaf( );
  lo max_elem_space_cluster
    = _spacetime_tree->get_space_cluster_tree( )->get_n_max_elems_leaf( );
  // allocate large enough matrix for sources to avoid reallocation
  // (3 * nr_elem is used as bound for the number of vertices in a cluster)
  full_matrix_type targets(
    max_elem_time_cluster, 3 * max_elem_space_cluster, false );
  full_matrix_type aux_matrix( max_elem_time_cluster,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6,
    false );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( );
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of local moment and all required matrices
    full_matrix_type & local = *( ( *it )->get_local_contribution( ) );
    full_matrix_type & T_drv
      = ( *it )->get_space_cluster( ).get_normal_drv_chebyshev_quad( );
    full_matrix_type & L = ( *it )->get_time_cluster( ).get_lagrange_quad( );
    lo n_space_nodes = ( *it )->get_space_cluster( ).get_n_nodes( );
    const std::vector< lo > & local_2_global_nodes
      = ( *it )->get_space_cluster( ).get_local_2_global_nodes( );
    lo n_time_elements = ( *it )->get_time_cluster( ).get_n_elements( );
    const std::vector< lo > & time_elements
      = ( *it )->get_time_cluster( ).get_all_elements( );
    // resize auxiliary matrix and matrix for targets appropriately
    aux_matrix.resize( n_time_elements,
      ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
    targets.resize( n_time_elements, n_space_nodes );
    // compute D = trans(L) * lambda and then the result Y = D * trans(T)
    aux_matrix.multiply( L, local, true, false );
    targets.multiply( aux_matrix, T_drv, false, true );
    // add the results to the correct position in the result vector
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
        y.add( time_elements[ i_time ], local_2_global_nodes[ i_space ],
          targets( i_time, i_space ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_l2t_operations_curl_p1_hs( block_vector_type & y,
  const lo dim ) const {
  lo max_elem_time_cluster
    = _spacetime_tree->get_time_cluster_tree( )->get_n_max_elems_leaf( );
  lo max_elem_space_cluster
    = _spacetime_tree->get_space_cluster_tree( )->get_n_max_elems_leaf( );
  lo n_multi_indices
    = ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
  // allocate large enough matrix for sources to avoid reallocation
  // (3 * nr_elem is used as bound for the number of vertices in a cluster)
  full_matrix_type targets(
    max_elem_time_cluster, 3 * max_elem_space_cluster, false );
  full_matrix_type T_curl( 3 * max_elem_space_cluster, n_multi_indices, false );
  full_matrix_type aux_matrix( max_elem_time_cluster, n_multi_indices, false );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( );
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of local moment and all required matrices and information
    full_matrix_type & local = *( ( *it )->get_local_contribution( ) );
    full_matrix_type & T = ( *it )->get_space_cluster( ).get_chebyshev_quad( );
    full_matrix_type & L = ( *it )->get_time_cluster( ).get_lagrange_quad( );

    // get the relevant entries of the block vector x and store them in sources
    lo n_space_nodes = ( *it )->get_space_cluster( ).get_n_nodes( );
    lo n_space_elements = ( *it )->get_space_cluster( ).get_n_elements( );
    const std::vector< lo > & local_2_global_nodes
      = ( *it )->get_space_cluster( ).get_local_2_global_nodes( );
    lo n_time_elements = ( *it )->get_time_cluster( ).get_n_elements( );
    const std::vector< lo > & time_elements
      = ( *it )->get_time_cluster( ).get_all_elements( );
    T_curl.resize( n_space_nodes, n_multi_indices );
    T_curl.fill( 0.0 );
    const std::vector< lo > & elems_2_local_nodes
      = ( *it )->get_space_cluster( ).get_elems_2_local_nodes( );
    const std::vector< sc > & surf_curls_curr_dim
      = ( *it )->get_space_cluster( ).get_surf_curls( dim );
    // use quadrature of plain Chebyshev polynomials to compute quadrature of
    // Chebyshev polynomials times surface curls of p1 basis functions.
    for ( lo i_beta = 0; i_beta < n_multi_indices; ++i_beta ) {
      for ( lo i_el = 0; i_el < n_space_elements; ++i_el ) {
        T_curl.add_atomic( elems_2_local_nodes[ 3 * i_el ], i_beta,
          surf_curls_curr_dim[ 3 * i_el ] * T.get( i_el, i_beta ) );
        T_curl.add_atomic( elems_2_local_nodes[ 3 * i_el + 1 ], i_beta,
          surf_curls_curr_dim[ 3 * i_el + 1 ] * T.get( i_el, i_beta ) );
        T_curl.add_atomic( elems_2_local_nodes[ 3 * i_el + 2 ], i_beta,
          surf_curls_curr_dim[ 3 * i_el + 2 ] * T.get( i_el, i_beta ) );
      }
    }
    aux_matrix.resize( n_time_elements, n_multi_indices );
    targets.resize( n_time_elements, n_space_nodes );
    // compute D = trans(L) * lambda and then the result Y = D * trans(T_curl)
    aux_matrix.multiply( L, local, true, false );
    targets.multiply( aux_matrix, T_curl, false, true );
    // add the results to the correct position in the result vector
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
        y.add( time_elements[ i_time ], local_2_global_nodes[ i_space ],
          targets( i_time, i_space ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::apply_l2t_operations_p1_normal_hs( block_vector_type & y,
  const lo dim ) const {
  lo max_elem_time_cluster
    = _spacetime_tree->get_time_cluster_tree( )->get_n_max_elems_leaf( );
  lo max_elem_space_cluster
    = _spacetime_tree->get_space_cluster_tree( )->get_n_max_elems_leaf( );
  lo n_multi_indices
    = ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
  // allocate large enough matrix for sources to avoid reallocation
  // (3 * nr_elem is used as bound for the number of vertices in a cluster)
  full_matrix_type targets(
    max_elem_time_cluster, 3 * max_elem_space_cluster, false );
  full_matrix_type aux_matrix( max_elem_time_cluster, n_multi_indices, false );
  for ( auto it = _spacetime_tree->get_leaves( ).begin( );
        it != _spacetime_tree->get_leaves( ).end( ); ++it ) {
    // get references of local moment and all required matrices and information
    full_matrix_type & local = *( ( *it )->get_local_contribution( ) );
    full_matrix_type & T_normal
      = ( *it )->get_space_cluster( ).get_cheb_times_normal_quad( dim );
    full_matrix_type & L = ( *it )->get_time_cluster( ).get_lagrange_quad( );

    // get the relevant entries of the block vector x and store them in sources
    lo n_space_nodes = ( *it )->get_space_cluster( ).get_n_nodes( );
    const std::vector< lo > & local_2_global_nodes
      = ( *it )->get_space_cluster( ).get_local_2_global_nodes( );
    lo n_time_elements = ( *it )->get_time_cluster( ).get_n_elements( );
    const std::vector< lo > & time_elements
      = ( *it )->get_time_cluster( ).get_all_elements( );

    aux_matrix.resize( n_time_elements, n_multi_indices );
    targets.resize( n_time_elements, n_space_nodes );
    // compute D = trans(L) * lambda and then the result Y = D * trans(T_curl)
    aux_matrix.multiply( L, local, true, false );
    targets.multiply( aux_matrix, T_normal, false, true );
    // add the results to the correct position in the result vector
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
        y.add( time_elements[ i_time ], local_2_global_nodes[ i_space ],
          targets( i_time, i_space ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::pFMM_matrix< kernel_type, target_space,
  source_space >::compute_coupling_coeffs( const vector_type & src_time_nodes,
  const vector_type & tar_time_nodes, const vector_type & cheb_nodes,
  const vector_type & evaluated_chebyshev, const sc half_size,
  const sc center_diff, vector_type & buffer_for_gaussians,
  vector_type & coupling_coeffs ) const {
  coupling_coeffs.fill( 0.0 );
  // evaluate the gaussian kernel for the numerical integration
  sc h_alpha = half_size * half_size / ( 4.0 * _alpha );
  sc scaled_center_diff = center_diff / half_size;
  lou index_gaussian = 0;
  for ( lo a = 0; a <= _temp_order; ++a ) {
    for ( lo b = 0; b <= _temp_order; ++b ) {
      sc h_delta_ab = h_alpha / ( tar_time_nodes[ a ] - src_time_nodes[ b ] );
      for ( lo mu = 0; mu < cheb_nodes.size( ); ++mu ) {
        for ( lo nu = 0; nu < cheb_nodes.size( ); ++nu ) {
          buffer_for_gaussians[ index_gaussian ] = std::exp( -h_delta_ab
            * ( scaled_center_diff + cheb_nodes[ mu ] - cheb_nodes[ nu ] )
            * ( scaled_center_diff + cheb_nodes[ mu ] - cheb_nodes[ nu ] ) );
          ++index_gaussian;
        }
      }
    }
  }

  // compute the numerical integrals
  lou index_integral = 0;
  sc mul_factor = 4.0 / ( cheb_nodes.size( ) * cheb_nodes.size( ) );
  for ( lo alpha = 0; alpha <= _spat_order; ++alpha ) {
    for ( lo beta = 0; beta <= _spat_order; ++beta ) {
      index_gaussian = 0;
      for ( lo a = 0; a <= _temp_order; ++a ) {
        for ( lo b = 0; b <= _temp_order; ++b ) {
          for ( lo mu = 0; mu < cheb_nodes.size( ); ++mu ) {
            for ( lo nu = 0; nu < cheb_nodes.size( ); ++nu ) {
              coupling_coeffs[ index_integral ]
                += buffer_for_gaussians[ index_gaussian ]
                * evaluated_chebyshev[ alpha * cheb_nodes.size( ) + mu ]
                * evaluated_chebyshev[ beta * cheb_nodes.size( ) + nu ];
              ++index_gaussian;
            }
          }
          sc mul_factor_ab = mul_factor
            / std::sqrt( 4.0 * M_PI * _alpha
              * ( tar_time_nodes[ a ] - src_time_nodes[ b ] ) );
          // gamma = 2 for all alpha and beta ( wrong, correction later )
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

template class besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
template class besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
template class besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
template class besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
