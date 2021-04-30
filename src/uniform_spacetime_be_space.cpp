/*
Copyright (c) 2020, VSB - Technical University of Ostrava and Graz University of
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

#include "besthea/uniform_spacetime_be_space.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/sparse_matrix.h"
#include "besthea/uniform_spacetime_be_identity.h"

template< class basis_type >
besthea::bem::uniform_spacetime_be_space<
  basis_type >::uniform_spacetime_be_space( st_mesh_type & spacetime_mesh )
  : spacetime_be_space< basis_type >( spacetime_mesh ) {
  _spacetime_mesh = &spacetime_mesh;
}

template< class basis_type >
besthea::bem::uniform_spacetime_be_space<
  basis_type >::~uniform_spacetime_be_space( ) {
}

template< class basis_type >
void besthea::bem::uniform_spacetime_be_space< basis_type >::L2_projection(
  sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  block_vector_type & projection, int order_matrix, int order_rhs_spatial,
  int order_rhs_temporal ) const {
  besthea::linear_algebra::sparse_matrix M;
  besthea::bem::uniform_spacetime_be_identity identity(
    *this, *this, order_matrix );
  identity.assemble( M );

  lo global_dim = this->_basis.dimension_global( );
  besthea::linear_algebra::vector rhs( global_dim, true );

  lo n_timesteps = _spacetime_mesh->get_n_temporal_elements( );
  sc timestep = _spacetime_mesh->get_timestep( );
  lo n_elements = _spacetime_mesh->get_n_spatial_elements( );

  projection.resize( n_timesteps );
  projection.resize_blocks( global_dim );

  lo local_dim = this->_basis.dimension_local( );
  std::vector< lo > l2g( local_dim );

  linear_algebra::coordinates< 3 > x1, x2, x3, n;
  sc area_xt, basis_val, fun_val;
  sc cg_eps;
  lo n_iter;

  typename spacetime_be_space< basis_type >::quadrature_wrapper my_quadrature;
  this->init_quadrature( order_rhs_spatial, order_rhs_temporal, my_quadrature );
  lo size_wt = my_quadrature._wt.size( );
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
    this->line_to_time( d, timestep, my_quadrature );
    for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
      _spacetime_mesh->get_spatial_nodes( i_elem, x1, x2, x3 );
      this->triangle_to_geometry( x1, x2, x3, my_quadrature );
      this->_basis.local_to_global( i_elem, l2g );
      _spacetime_mesh->get_spatial_normal( i_elem, n );
      area_xt = _spacetime_mesh->spatial_area( i_elem ) * timestep;

      for ( lo i_t = 0; i_t < size_wt; ++i_t ) {
        for ( lo i_x = 0; i_x < size_x; ++i_x ) {
          fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ], x3_mapped[ i_x ], n,
                      t_mapped[ i_t ] )
            * wx[ i_x ] * wt[ i_t ] * area_xt;
          for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
            basis_val = this->_basis.evaluate(
              i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], n.data( ) );
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

template< class basis_type >
sc besthea::bem::uniform_spacetime_be_space< basis_type >::L2_relative_error(
  sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  const block_vector_type & approximation, int order_rhs_spatial,
  int order_rhs_temporal ) const {
  lo n_timesteps = _spacetime_mesh->get_n_temporal_elements( );
  sc timestep = _spacetime_mesh->get_timestep( );
  lo n_elements = _spacetime_mesh->get_n_spatial_elements( );

  lo local_dim = this->_basis.dimension_local( );
  std::vector< lo > l2g( local_dim );

  linear_algebra::coordinates< 3 > x1, x2, x3, n;
  sc area_xt, basis_val, fun_val;
  sc l2_err = 0.0;
  sc l2_norm = 0.0;
  sc local_value;
  sc absdiff, absf;

  typename spacetime_be_space< basis_type >::quadrature_wrapper my_quadrature;
  this->init_quadrature( order_rhs_spatial, order_rhs_temporal, my_quadrature );
  lo size_wt = my_quadrature._wt.size( );
  lo size_x = my_quadrature._wx.size( );
  sc * x1_ref = my_quadrature._x1_ref.data( );
  sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * wx = my_quadrature._wx.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * wt = my_quadrature._wt.data( );
  sc * t_mapped = my_quadrature._t.data( );
  lo * l2g_data = l2g.data( );
  const sc * approximation_data = nullptr;

  for ( lo d = 0; d < n_timesteps; ++d ) {
    this->line_to_time( d, timestep, my_quadrature );
    approximation_data = approximation.get_block( d ).data( );
    for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
      _spacetime_mesh->get_spatial_nodes( i_elem, x1, x2, x3 );
      this->triangle_to_geometry( x1, x2, x3, my_quadrature );
      this->_basis.local_to_global( i_elem, l2g );
      _spacetime_mesh->get_spatial_normal( i_elem, n );
      area_xt = _spacetime_mesh->spatial_area( i_elem ) * timestep;
      for ( lo i_x = 0; i_x < size_x; ++i_x ) {
        local_value = 0.0;
        for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
          basis_val = this->_basis.evaluate(
            i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], n.data( ) );
          local_value += approximation_data[ l2g_data[ i_loc ] ] * basis_val;
        }
        for ( lo i_t = 0; i_t < size_wt; ++i_t ) {
          fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ], x3_mapped[ i_x ], n,
            t_mapped[ i_t ] );
          absdiff = std::abs( fun_val - local_value );
          absf = std::abs( fun_val );
          l2_err += absdiff * absdiff * wx[ i_x ] * wt[ i_t ] * area_xt;
          l2_norm += absf * absf * wx[ i_x ] * wt[ i_t ] * area_xt;
        }
      }
    }
  }
  sc result;
  if ( l2_norm < 1e-10 ) {
    if ( l2_err < 1e-10 ) {
      result = 0;
    } else {
      result = std::numeric_limits< sc >::infinity( );
    }
  } else {
    result = std::sqrt( l2_err / l2_norm );
  }
  return result;
}

/**
 * Projects a function to the piecewise constant boundary element space.
 * @param[in] f Function to be projected.
 * @param[out] interpolation Interpolation vector.
 */
template<>
void besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >::
  interpolation(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    block_vector_type & interpolation ) const {
  lo n_timesteps = _spacetime_mesh->get_n_temporal_elements( );
  lo n_elements = _spacetime_mesh->get_n_spatial_elements( );

  interpolation.resize( n_timesteps );
  interpolation.resize_blocks( n_elements );
  linear_algebra::coordinates< 3 > centroid, n;

  for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
    _spacetime_mesh->get_spatial_centroid( i_elem, centroid );
    _spacetime_mesh->get_spatial_normal( i_elem, n );
    for ( lo d = 0; d < n_timesteps; ++d ) {
      interpolation.set( d, i_elem,
        f( centroid[ 0 ], centroid[ 1 ], centroid[ 2 ], n,
          _spacetime_mesh->get_temporal_centroid( d ) ) );
    }
  }
}

/**
 * Projects a function to the piecewise linear boundary element space.
 * @param[in] f Function to be projected.
 * @param[out] interpolation Interpolation vector.
 */
template<>
void besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >::
  interpolation(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    block_vector_type & interpolation ) const {
  lo n_timesteps = _spacetime_mesh->get_n_temporal_elements( );
  lo n_nodes = _spacetime_mesh->get_n_spatial_nodes( );

  interpolation.resize( n_timesteps );
  interpolation.resize_blocks( n_nodes );
  linear_algebra::coordinates< 3 > x, n;

  for ( lo i_node = 0; i_node < n_nodes; ++i_node ) {
    _spacetime_mesh->get_spatial_node( i_node, x );
    _spacetime_mesh->get_spatial_nodal_normal( i_node, n );
    for ( lo d = 0; d < n_timesteps; ++d ) {
      interpolation.set( d, i_node,
        f( x[ 0 ], x[ 1 ], x[ 2 ], n,
          _spacetime_mesh->get_temporal_centroid( d ) ) );
    }
  }
}

template class besthea::bem::uniform_spacetime_be_space<
  besthea::bem::basis_tri_p0 >;
template class besthea::bem::uniform_spacetime_be_space<
  besthea::bem::basis_tri_p1 >;
