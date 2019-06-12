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

#include "besthea/basis_tri_p1.h"

besthea::bem::basis_tri_p1::basis_tri_p1( mesh_type & mesh ) {
  _mesh = &mesh;
}

besthea::bem::basis_tri_p1::~basis_tri_p1( ) {
}

lo besthea::bem::basis_tri_p1::dimension_local( lo i_elem ) {
  return 3;
}

lo besthea::bem::basis_tri_p1::dimension_global( ) {
  return _mesh->get_spatial_mesh( )->get_n_nodes( );
}

void besthea::bem::basis_tri_p1::local_to_global( lo i_elem, adjacency type,
  int rotation, bool swap, std::vector< lo > indices ) {
  lo element[ 3 ];
  _mesh->get_spatial_mesh( )->get_element( i_elem, element );

  if ( type == adjacency::edge && swap ) {
    indices[ 0 ] = element[ map[ rotation + 1 ] ];
    indices[ 1 ] = element[ map[ rotation ] ];
  } else {
    indices[ 0 ] = element[ map[ rotation ] ];
    indices[ 1 ] = element[ map[ rotation + 1 ] ];
  }
  indices[ 2 ] = element[ map[ rotation + 2 ] ];
}

void besthea::bem::basis_tri_p1::evaluate( lo i_elem,
  const std::vector< sc > & x1_ref, const std::vector< sc > & x2_ref,
  const sc * n, adjacency type, int rotation, bool swap,
  std::vector< matrix_type > & values ) {
  std::vector< sc >::size_type size = x1_ref.size( );

  const sc * x1_ref_data = x1_ref.data( );
  const sc * x2_ref_data = x2_ref.data( );

  sc * values1_data = values[ 0 ].data( );
  sc * values2_data = values[ 1 ].data( );
  sc * values3_data = values[ 2 ].data( );

#pragma omp simd simdlen( DATA_WIDTH )
  for ( std::vector< sc >::size_type i = 0; i < size; ++i ) {
    values1_data[ i ] = 1.0 - x1_ref_data[ i ] - x2_ref_data[ i ];
    values2_data[ i ] = x1_ref_data[ i ];
    values3_data[ i ] = x2_ref_data[ i ];
  }
}
