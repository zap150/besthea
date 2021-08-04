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
  target_space,
  source_space >::set_trees( mesh::distributed_spacetime_cluster_tree *
                               spacetime_target_tree,
  mesh::space_cluster_tree * space_source_tree ) {
  _distributed_spacetime_target_tree = spacetime_target_tree;
  _space_source_tree = space_source_tree;
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
  target_space, source_space >::compute_spatial_m2m_coeffs( ) {
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
