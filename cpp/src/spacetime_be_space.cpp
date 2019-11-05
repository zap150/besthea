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

#include "besthea/spacetime_be_space.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"

template< class basis_type >
besthea::bem::spacetime_be_space< basis_type >::spacetime_be_space(
  mesh_type & mesh )
  : _basis( mesh ) {
}

template< class basis_type >
besthea::bem::spacetime_be_space< basis_type >::~spacetime_be_space( ) {
}

template< class basis_type >
void besthea::bem::spacetime_be_space< basis_type >::L2_projection(
  sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  block_vector_type & projection, int order_matrix, int order_rhs_spatial,
  int order_rhs_temporal ) const {
}

template< class basis_type >
sc besthea::bem::spacetime_be_space< basis_type >::L2_relative_error(
  sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  const block_vector_type & approximation, int order_rhs_spatial,
  int order_rhs_temporal ) const {
  return 0.0;
}

template< class basis_type >
sc besthea::bem::spacetime_be_space< basis_type >::l2_relative_error(
  const block_vector_type & f, const block_vector_type & approximation ) const {
  lo block_size = f.get_block_size( );
  lo size = f.get_size( );
  sc l2diffnorm = 0.0;
  sc l2norm = 0.0;
  sc aux;

  for ( lo i_block = 0; i_block < block_size; ++i_block ) {
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
void besthea::bem::spacetime_be_space< basis_type >::interpolation(
  sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  block_vector_type & interpolation ) const {
}

template class besthea::bem::spacetime_be_space< besthea::bem::basis_tri_p0 >;
template class besthea::bem::spacetime_be_space< besthea::bem::basis_tri_p1 >;
