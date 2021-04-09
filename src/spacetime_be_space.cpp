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

#include "besthea/spacetime_be_space.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"

template< class basis_type >
besthea::bem::spacetime_be_space< basis_type >::spacetime_be_space(
  const mesh_type & mesh )
  : _basis( mesh ) {
}

template< class basis_type >
besthea::bem::spacetime_be_space< basis_type >::~spacetime_be_space( ) {
}

template< class basis_type >
void besthea::bem::spacetime_be_space< basis_type >::interpolation(
  [[maybe_unused]] sc ( *f )(
    sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  [[maybe_unused]] block_vector_type & interpolation ) const {
  std::cout << "Only use specialized templates in descendant classes!"
            << std::endl;
}

template< class basis_type >
sc besthea::bem::spacetime_be_space< basis_type >::l2_relative_error(
  const block_vector_type & f, const block_vector_type & approximation ) const {
  lo n_blocks = f.get_n_blocks( );
  lo size = f.get_size_of_block( );
  sc l2diffnorm = 0.0;
  sc l2norm = 0.0;
  sc aux;

  for ( lo i_block = 0; i_block < n_blocks; ++i_block ) {
    for ( lo i_elem = 0; i_elem < size; ++i_elem ) {
      aux = f.get( i_block, i_elem );
      l2norm += aux * aux;
      aux -= approximation.get( i_block, i_elem );
      l2diffnorm += aux * aux;
    }
  }

  return std::sqrt( l2diffnorm / l2norm );
}

template< class basis_type >
void besthea::bem::spacetime_be_space< basis_type >::init_quadrature(
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

template< class basis_type >
void besthea::bem::spacetime_be_space< basis_type >::triangle_to_geometry(
  const linear_algebra::coordinates< 3 > & x1,
  const linear_algebra::coordinates< 3 > & x2,
  const linear_algebra::coordinates< 3 > & x3,
  quadrature_wrapper & my_quadrature ) const {
  const sc * x1_ref = my_quadrature._x1_ref.data( );
  const sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );

  lo size = my_quadrature._wx.size( );

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ];
    x2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ];
    x3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ];
  }
}

template< class basis_type >
void besthea::bem::spacetime_be_space< basis_type >::line_to_time(
  lo d, sc timestep, quadrature_wrapper & my_quadrature ) const {
  const sc * t_ref = my_quadrature._t_ref.data( );
  sc * t_mapped = my_quadrature._t.data( );

  lo size = my_quadrature._wt.size( );

#pragma omp simd aligned( t_mapped, t_ref ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    t_mapped[ i ] = ( t_ref[ i ] + d ) * timestep;
  }
}

template class besthea::bem::spacetime_be_space< besthea::bem::basis_tri_p0 >;
template class besthea::bem::spacetime_be_space< besthea::bem::basis_tri_p1 >;
