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

#include "besthea/tetrahedral_spacetime_be_evaluator.h"

#include "besthea/quadrature.h"
#include "besthea/spacetime_basis_tetra_p0.h"
#include "besthea/spacetime_basis_tetra_p1.h"
#include "besthea/spacetime_heat_kernel.h"
#include "besthea/spacetime_heat_kernel_normal_derivative.h"
#include "besthea/tetrahedral_spacetime_be_space.h"

template< class kernel_type, class space_type >
besthea::bem::tetrahedral_spacetime_be_evaluator< kernel_type,
  space_type >::tetrahedral_spacetime_be_evaluator( kernel_type & kernel,
  space_type & space, int order_spatial )
  : _kernel( &kernel ), _space( &space ), _order_spatial( order_spatial ) {
}

template< class kernel_type, class space_type >
besthea::bem::tetrahedral_spacetime_be_evaluator< kernel_type,
  space_type >::~tetrahedral_spacetime_be_evaluator( ) {
}

template< class kernel_type, class space_type >
void besthea::bem::
  tetrahedral_spacetime_be_evaluator< kernel_type, space_type >::evaluate(
    const std::vector< linear_algebra::coordinates< 4 > > & xt,
    const vector_type & density, vector_type & result ) const {
  auto & basis = _space->get_basis( );
  auto mesh = _space->get_mesh( );

  lo n_points = xt.size( );
  lo n_elements = mesh->get_n_elements( );
  lo loc_dim = basis.dimension_local( );

  result.resize( n_points, false );

#pragma omp parallel
  {
    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    lo size_quad = my_quadrature._wy.size( );
    linear_algebra::coordinates< 4 > y1, y2, y3, y4;
    linear_algebra::coordinates< 3 > ny;
    std::vector< lo > l2g( loc_dim );

    sc * y1_ref = my_quadrature._y1_ref.data( );
    sc * y2_ref = my_quadrature._y2_ref.data( );
    sc * y3_ref = my_quadrature._y3_ref.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );
    sc * tau_mapped = my_quadrature._tau.data( );
    sc * w = my_quadrature._wy.data( );

#pragma omp for schedule( dynamic, 8 )
    for ( lo i_point = 0; i_point < n_points; ++i_point ) {
      const auto & point = xt[ i_point ];
      const auto & x1 = point[ 0 ];
      const auto & x2 = point[ 1 ];
      const auto & x3 = point[ 2 ];
      const auto & t = point[ 3 ];

      // init
      result[ i_point ] = 0.0;

      if ( t <= 0.0 ) {
        // potential vanishes
        continue;
      }

      sc res = 0.0;

      for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
        mesh->get_nodes( i_elem, y1, y2, y3, y4 );
        mesh->get_spatial_normal( i_elem, ny );
        sc area = mesh->area( i_elem );
        basis.local_to_global( i_elem, l2g );
        tetrahedron_to_geometry( y1, y2, y3, y4, my_quadrature );

#pragma omp simd aligned(                                                \
  y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, y3_ref, tau_mapped, w \
  : DATA_ALIGN ) reduction(+ : res) simdlen( BESTHEA_SIMD_WIDTH )
        for ( lo i_quad = 0; i_quad < size_quad; ++i_quad ) {
          sc kernel = _kernel->evaluate( x1 - y1_mapped[ i_quad ],
            x2 - y2_mapped[ i_quad ], x3 - y3_mapped[ i_quad ], nullptr,
            ny.data( ), t - tau_mapped[ i_quad ] );

          for ( lo i_loc = 0; i_loc < loc_dim; ++i_loc ) {
            sc basis_value = basis.evaluate( i_elem, i_loc, y1_ref[ i_quad ],
                               y2_ref[ i_quad ], y3_ref[ i_quad ] )
              * w[ i_quad ] * area * kernel;

            // adding value
            res += density.get( l2g[ i_loc ] ) * basis_value;
          }  // i_loc
        }    // i_quad
        // save result
        result[ i_point ] = res;
      }  // i_elem
    }    // for i_point
  }      // omp parallel
}

template< class kernel_type, class space_type >
void besthea::bem::tetrahedral_spacetime_be_evaluator< kernel_type,
  space_type >::init_quadrature( quadrature_wrapper & my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._y1_ref = quadrature::tetrahedron_x1( _order_spatial );
  my_quadrature._y2_ref = quadrature::tetrahedron_x2( _order_spatial );
  my_quadrature._y3_ref = quadrature::tetrahedron_x3( _order_spatial );
  my_quadrature._wy = quadrature::tetrahedron_w( _order_spatial );

  lo size = my_quadrature._wy.size( );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );
  my_quadrature._tau.resize( size );
}

template< class kernel_type, class space_type >
void besthea::bem::tetrahedral_spacetime_be_evaluator< kernel_type,
  space_type >::
  tetrahedron_to_geometry( const linear_algebra::coordinates< 4 > & x1,
    const linear_algebra::coordinates< 4 > & x2,
    const linear_algebra::coordinates< 4 > & x3,
    const linear_algebra::coordinates< 4 > & x4,
    quadrature_wrapper & my_quadrature ) const {
  const sc * y1_ref = my_quadrature._y1_ref.data( );
  const sc * y2_ref = my_quadrature._y2_ref.data( );
  const sc * y3_ref = my_quadrature._y3_ref.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );
  sc * tau = my_quadrature._tau.data( );

  lo size = my_quadrature._wy.size( );

#pragma omp simd aligned(                                      \
  y1_mapped, y2_mapped, y3_mapped, tau, y1_ref, y2_ref, y3_ref \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
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
    tau[ i ] = x1[ 3 ] + ( x2[ 3 ] - x1[ 3 ] ) * y1_ref[ i ]
      + ( x3[ 3 ] - x1[ 3 ] ) * y2_ref[ i ]
      + ( x4[ 3 ] - x1[ 3 ] ) * y3_ref[ i ];
  }
}

template class besthea::bem::tetrahedral_spacetime_be_evaluator<
  besthea::bem::spacetime_heat_kernel,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p0 > >;
template class besthea::bem::tetrahedral_spacetime_be_evaluator<
  besthea::bem::spacetime_heat_kernel,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p1 > >;

template class besthea::bem::tetrahedral_spacetime_be_evaluator<
  besthea::bem::spacetime_heat_kernel_normal_derivative,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p0 > >;
template class besthea::bem::tetrahedral_spacetime_be_evaluator<
  besthea::bem::spacetime_heat_kernel_normal_derivative,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p1 > >;
