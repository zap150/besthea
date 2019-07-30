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

#include "besthea/uniform_spacetime_be_evaluator.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/uniform_spacetime_be_space.h"
#include "besthea/uniform_spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_heat_sl_kernel_antiderivative.h"

template< class kernel_type, class space_type >
besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::uniform_spacetime_be_evaluator( kernel_type & kernel,
  space_type & space, int order_spatial )
  : _kernel( &kernel ), _space( &space ), _order_spatial( order_spatial ) {
}

template< class kernel_type, class space_type >
besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::~uniform_spacetime_be_evaluator( ) {
}

template< class kernel_type, class space_type >
void besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::evaluate( const std::vector< sc > & x,
  const block_vector_type & density, block_vector_type & result ) const {
  auto & basis = _space->get_basis( );
  auto mesh = _space->get_mesh( );

  lo n_timesteps = mesh->get_n_temporal_elements( );
  sc timestep = mesh->get_timestep( );
  lo n_points = x.size( ) / 3;

  // result[ 0 ] holds the initial condition
  result.resize( n_timesteps + 1 );
  result.resize_blocks( n_points, true );

#pragma omp parallel
  {
    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
  }
}

template< class kernel_type, class space_type >
void besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::init_quadrature( quadrature_wrapper & my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._x1_ref = quadrature::triangle_x1( _order_spatial );
  my_quadrature._x2_ref = quadrature::triangle_x2( _order_spatial );
  my_quadrature._wx = quadrature::triangle_w( _order_spatial );

  lo size = my_quadrature._wx.size( );
  my_quadrature._x1.resize( size );
  my_quadrature._x2.resize( size );
  my_quadrature._x3.resize( size );
}

template< class kernel_type, class space_type >
void besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::triangle_to_geometry( const sc * x1, const sc * x2,
  const sc * x3, quadrature_wrapper & my_quadrature ) const {
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

template class besthea::bem::uniform_spacetime_be_evaluator<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::uniform_spacetime_be_evaluator<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class besthea::bem::uniform_spacetime_be_evaluator<
  besthea::bem::uniform_spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::uniform_spacetime_be_evaluator<
  besthea::bem::uniform_spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
