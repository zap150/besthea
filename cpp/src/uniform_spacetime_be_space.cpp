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

#include "besthea/uniform_spacetime_be_space.h"

#include "besthea/quadrature.h"
#include "besthea/sparse_matrix.h"
#include "besthea/uniform_spacetime_be_identity.h"

template< class basis >
besthea::bem::uniform_spacetime_be_space< basis >::uniform_spacetime_be_space(
  st_mesh_type & spacetime_mesh )
  : _basis( spacetime_mesh ) {
  _spacetime_mesh = &spacetime_mesh;
}

template< class basis >
besthea::bem::uniform_spacetime_be_space<
  basis >::~uniform_spacetime_be_space( ) {
}

template< class basis >
void besthea::bem::uniform_spacetime_be_space< basis >::l2_projection(
  sc ( *f )( sc, sc, sc, sc *, sc ), block_vector_type & projection,
  int order_matrix, int order_rhs_spatial, int order_rhs_temporal ) {
  besthea::linear_algebra::sparse_matrix M;
  besthea::bem::uniform_spacetime_be_identity identity(
    *this, *this, order_matrix );
  identity.assemble( M );

  lo global_dim = _basis.dimension_global( );
  besthea::linear_algebra::vector rhs( global_dim, true );

  lo n_timesteps = _spacetime_mesh->get_n_temporal_elements( );
  sc timestep = _spacetime_mesh->get_timestep( );
  lo n_elements = _spacetime_mesh->get_n_spatial_elements( );

  projection.resize( n_timesteps );
  projection.resize_blocks( global_dim );

  lo local_dim = _basis.dimension_local( );
  std::vector< lo > l2g( local_dim );

  sc x1[ 3 ], x2[ 3 ], x3[ 3 ], n[ 3 ];
  sc area_xt, basis_val, fun_val;
  sc cg_eps;
  lo n_iter;

  quadrature_wrapper my_quadrature;
  init_quadrature( order_rhs_spatial, order_rhs_temporal, my_quadrature );
  lo size_t = my_quadrature._wt.size( );
  lo size_x = my_quadrature._wx.size( );
  sc * x1_ref = my_quadrature._x1_ref.data( );
  sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * wx = my_quadrature._wx.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * wt = my_quadrature._wt.data( );
  sc * t_mapped = my_quadrature._t.data( );
  sc * rhs_data = rhs.data( );
  lo * l2g_data = l2g.data( );

  for ( lo d = 0; d < n_timesteps; ++d ) {
    line_to_time( d, timestep, my_quadrature );
    for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
      _spacetime_mesh->get_spatial_nodes( i_elem, x1, x2, x3 );
      triangle_to_geometry( x1, x2, x3, my_quadrature );
      _basis.local_to_global( i_elem, l2g );
      _spacetime_mesh->get_spatial_normal( i_elem, n );
      area_xt = _spacetime_mesh->spatial_area( i_elem ) * timestep;

      for ( lo i_t = 0; i_t < size_t; ++i_t ) {
        for ( lo i_x = 0; i_x < size_x; ++i_x ) {
          fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ], x3_mapped[ i_x ], n,
                      t_mapped[ i_t ] )
            * wx[ i_x ] * wt[ i_t ] * area_xt;
          for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
            basis_val = _basis.evaluate(
              i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], n );
            rhs_data[ l2g_data[ i_loc ] ] += basis_val * fun_val;
          }
        }
      }
    }
    cg_eps = 1e-6;
    n_iter = 200;
    M.eigen_cg_solve( rhs, projection.get_block( d ), cg_eps, n_iter );
    rhs.fill( 0.0 );
  }
}

template< class basis >
void besthea::bem::uniform_spacetime_be_space< basis >::init_quadrature(
  int order_rhs_spatial, int order_rhs_temporal,
  quadrature_wrapper & my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._x1_ref = quadrature::triangle_x1( order_rhs_spatial );
  my_quadrature._x2_ref = quadrature::triangle_x2( order_rhs_spatial );
  my_quadrature._wx = quadrature::triangle_w( order_rhs_spatial );

  lo size = my_quadrature._wx.size( );
  my_quadrature._x1.resize( size );
  my_quadrature._x2.resize( size );
  my_quadrature._x3.resize( size );

  // calling copy constructor of std::vector
  my_quadrature._t_ref = quadrature::line_x( order_rhs_temporal );
  my_quadrature._wt = quadrature::line_w( order_rhs_temporal );

  size = my_quadrature._wt.size( );
  my_quadrature._t.resize( size );
}

template< class basis >
void besthea::bem::uniform_spacetime_be_space< basis >::triangle_to_geometry(
  const sc * x1, const sc * x2, const sc * x3,
  quadrature_wrapper & my_quadrature ) const {
  const sc * x1_ref = my_quadrature._x1_ref.data( );
  const sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );

  lo size = my_quadrature._wx.size( );

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref \
                          : data_align ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ];
    x2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ];
    x3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ];
  }
}

template< class basis >
void besthea::bem::uniform_spacetime_be_space< basis >::line_to_time(
  lo d, sc timestep, quadrature_wrapper & my_quadrature ) const {
  const sc * t_ref = my_quadrature._t_ref.data( );
  sc * t_mapped = my_quadrature._t.data( );

  lo size = my_quadrature._wt.size( );

#pragma omp simd aligned( t_mapped, t_ref ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    t_mapped[ i ] = ( t_ref[ i ] + d ) * timestep;
  }
}

template class besthea::bem::uniform_spacetime_be_space<
  besthea::bem::basis_tri_p0 >;
template class besthea::bem::uniform_spacetime_be_space<
  besthea::bem::basis_tri_p1 >;
