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

#include "besthea/tetrahedral_spacetime_be_space.h"

#include "besthea/quadrature.h"
#include "besthea/spacetime_basis_tetra_p0.h"
#include "besthea/spacetime_basis_tetra_p1.h"
#include "besthea/sparse_matrix.h"
#include "besthea/tetrahedral_spacetime_be_identity.h"

template< class basis_type >
besthea::bem::tetrahedral_spacetime_be_space<
  basis_type >::tetrahedral_spacetime_be_space( mesh_type & mesh )
  : _basis( mesh ), _mesh( &mesh ) {
}

template< class basis_type >
besthea::bem::tetrahedral_spacetime_be_space<
  basis_type >::~tetrahedral_spacetime_be_space( ) {
}

/**
 * Projects a function to the piecewise linear finite element space.
 * @param[in] f Function to be projected.
 * @param[out] interpolation Interpolation vector.
 */
template<>
void besthea::bem::tetrahedral_spacetime_be_space<
  besthea::bem::spacetime_basis_tetra_p1 >::
  interpolation(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    vector_type & interpolation ) const {
  lo n_nodes = _mesh->get_n_nodes( );

  interpolation.resize( n_nodes );
  linear_algebra::coordinates< 4 > x;
  linear_algebra::coordinates< 3 > n;

  for ( lo i_node = 0; i_node < n_nodes; ++i_node ) {
    _mesh->get_node( i_node, x );
    _mesh->get_spatial_nodal_normal( i_node, n );
    interpolation.set( i_node, f( x[ 0 ], x[ 1 ], x[ 2 ], n, x[ 3 ] ) );
  }
}

template< class basis_type >
sc besthea::bem::tetrahedral_spacetime_be_space<
  basis_type >::l2_relative_error( const vector_type & f,
  const vector_type & approximation ) const {
  lo size = f.size( );
  sc l2diffnorm = 0.0;
  sc l2norm = 0.0;
  sc aux;

  for ( lo i_elem = 0; i_elem < size; ++i_elem ) {
    aux = f.get( i_elem );
    l2norm += aux * aux;
    aux -= approximation.get( i_elem );
    l2diffnorm += aux * aux;
  }

  return std::sqrt( l2diffnorm / l2norm );
}

template< class basis_type >
void besthea::bem::tetrahedral_spacetime_be_space<
  basis_type >::init_quadrature( int order_rhs,
  quadrature_wrapper & my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._x1_ref = quadrature::tetrahedron_x1( order_rhs );
  my_quadrature._x2_ref = quadrature::tetrahedron_x2( order_rhs );
  my_quadrature._x3_ref = quadrature::tetrahedron_x3( order_rhs );
  my_quadrature._w = quadrature::tetrahedron_w( order_rhs );

  lo size = my_quadrature._w.size( );
  my_quadrature._x1.resize( size );
  my_quadrature._x2.resize( size );
  my_quadrature._x3.resize( size );
  my_quadrature._t.resize( size );
}

template< class basis_type >
void besthea::bem::tetrahedral_spacetime_be_space< basis_type >::L2_projection(
  sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  vector_type & projection, int order_matrix, int order_rhs ) const {
  besthea::linear_algebra::sparse_matrix M;
  besthea::bem::tetrahedral_spacetime_be_identity identity(
    *this, *this, order_matrix );
  identity.assemble( M );

  lo global_dim = this->_basis.dimension_global( );
  besthea::linear_algebra::vector rhs( global_dim, true );

  lo n_elements = _mesh->get_n_elements( );

  projection.resize( global_dim );

  lo local_dim = this->_basis.dimension_local( );
  std::vector< lo > l2g( local_dim );

  linear_algebra::coordinates< 4 > x1, x2, x3, x4;
  linear_algebra::coordinates< 3 > n;
  sc area, basis_val, fun_val;
  sc cg_eps;
  lo n_iter;

  typename tetrahedral_spacetime_be_space< basis_type >::quadrature_wrapper
    my_quadrature;
  this->init_quadrature( order_rhs, my_quadrature );
  lo size = my_quadrature._w.size( );
  sc * x1_ref = my_quadrature._x1_ref.data( );
  sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * x3_ref = my_quadrature._x3_ref.data( );
  sc * w = my_quadrature._w.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * t_mapped = my_quadrature._t.data( );
  sc * rhs_data = rhs.data( );
  lo * l2g_data = l2g.data( );

  for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
    _mesh->get_nodes( i_elem, x1, x2, x3, x4 );
    this->tetrahedron_to_geometry( x1, x2, x3, x4, my_quadrature );
    this->_basis.local_to_global( i_elem, l2g );
    _mesh->get_spatial_normal( i_elem, n );
    area = _mesh->area( i_elem );

    for ( lo i_x = 0; i_x < size; ++i_x ) {
      fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ], x3_mapped[ i_x ], n,
                  t_mapped[ i_x ] )
        * w[ i_x ] * area;
      for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
        basis_val = this->_basis.evaluate(
          i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], x3_ref[ i_x ] );
        rhs_data[ l2g_data[ i_loc ] ] += basis_val * fun_val;
      }
    }
  }
  cg_eps = 1e-6;
  n_iter = 200;
  M.eigen_cg_solve( rhs, projection, cg_eps, n_iter );
}

template< class basis_type >
sc besthea::bem::tetrahedral_spacetime_be_space<
  basis_type >::L2_relative_error( sc ( *f )( sc, sc, sc,
                                     const linear_algebra::coordinates< 3 > &,
                                     sc ),
  const vector_type & approximation, int order_rhs ) const {
  lo n_elements = _mesh->get_n_elements( );

  lo local_dim = _basis.dimension_local( );
  std::vector< lo > l2g( local_dim );

  linear_algebra::coordinates< 4 > x1, x2, x3, x4;
  linear_algebra::coordinates< 3 > n;
  sc area, basis_val, fun_val;
  sc l2_err = 0.0;
  sc l2_norm = 0.0;
  sc local_value;
  sc absdiff, absf;

  quadrature_wrapper my_quadrature;
  this->init_quadrature( order_rhs, my_quadrature );
  lo size_x = my_quadrature._w.size( );
  sc * x1_ref = my_quadrature._x1_ref.data( );
  sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * x3_ref = my_quadrature._x3_ref.data( );
  sc * w = my_quadrature._w.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * t_mapped = my_quadrature._t.data( );
  lo * l2g_data = l2g.data( );
  const sc * approximation_data = approximation.data( );

  for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
    _mesh->get_nodes( i_elem, x1, x2, x3, x4 );
    _mesh->get_spatial_normal( i_elem, n );
    this->tetrahedron_to_geometry( x1, x2, x3, x4, my_quadrature );
    this->_basis.local_to_global( i_elem, l2g );
    area = _mesh->area( i_elem );
    for ( lo i_x = 0; i_x < size_x; ++i_x ) {
      local_value = 0.0;
      for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
        basis_val = _basis.evaluate(
          i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], x3_ref[ i_x ] );
        local_value += approximation_data[ l2g_data[ i_loc ] ] * basis_val;
      }

      fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ], x3_mapped[ i_x ], n,
        t_mapped[ i_x ] );
      absdiff = std::abs( fun_val - local_value );
      absf = std::abs( fun_val );
      l2_err += absdiff * absdiff * w[ i_x ] * area;
      l2_norm += absf * absf * w[ i_x ] * area;
    }
  }

  sc result = std::sqrt( l2_err / l2_norm );
  return result;
}

template< class basis_type >
void besthea::bem::tetrahedral_spacetime_be_space< basis_type >::
  tetrahedron_to_geometry( const linear_algebra::coordinates< 4 > & x1,
    const linear_algebra::coordinates< 4 > & x2,
    const linear_algebra::coordinates< 4 > & x3,
    const linear_algebra::coordinates< 4 > & x4,
    quadrature_wrapper & my_quadrature ) const {
  const sc * x1_ref = my_quadrature._x1_ref.data( );
  const sc * x2_ref = my_quadrature._x2_ref.data( );
  const sc * x3_ref = my_quadrature._x3_ref.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * t_mapped = my_quadrature._t.data( );

  lo size = my_quadrature._w.size( );

#pragma omp simd aligned(                                           \
  x1_mapped, x2_mapped, x3_mapped, t_mapped, x1_ref, x2_ref, x3_ref \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ]
      + ( x4[ 0 ] - x1[ 0 ] ) * x3_ref[ i ];
    x2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ]
      + ( x4[ 1 ] - x1[ 1 ] ) * x3_ref[ i ];
    x3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ]
      + ( x4[ 2 ] - x1[ 2 ] ) * x3_ref[ i ];
    t_mapped[ i ] = x1[ 3 ] + ( x2[ 3 ] - x1[ 3 ] ) * x1_ref[ i ]
      + ( x3[ 3 ] - x1[ 3 ] ) * x2_ref[ i ]
      + ( x4[ 3 ] - x1[ 3 ] ) * x3_ref[ i ];
  }
}

template class besthea::bem::tetrahedral_spacetime_be_space<
  besthea::bem::spacetime_basis_tetra_p0 >;
template class besthea::bem::tetrahedral_spacetime_be_space<
  besthea::bem::spacetime_basis_tetra_p1 >;
