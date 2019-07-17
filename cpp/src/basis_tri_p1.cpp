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

lo besthea::bem::basis_tri_p1::dimension_local( ) {
  return 3;
}

lo besthea::bem::basis_tri_p1::dimension_global( ) {
  return _mesh->get_spatial_mesh( )->get_n_nodes( );
}

void besthea::bem::basis_tri_p1::do_local_to_global( lo i_elem,
  int n_shared_vertices, int rotation, bool swap,
  std::vector< lo > & indices ) {
  lo element[ 3 ];
  _mesh->get_spatial_mesh( )->get_element( i_elem, element );

  if ( n_shared_vertices == 2 && swap ) {
    indices[ 0 ] = element[ _map[ rotation + 1 ] ];
    indices[ 1 ] = element[ _map[ rotation ] ];
  } else {
    indices[ 0 ] = element[ _map[ rotation ] ];
    indices[ 1 ] = element[ _map[ rotation + 1 ] ];
  }
  indices[ 2 ] = element[ _map[ rotation + 2 ] ];
}

#pragma omp declare simd uniform( \
  i_elem, n, n_shared_vertices, rotation, swap ) simdlen( DATA_WIDTH )
void besthea::bem::basis_tri_p1::do_evaluate( lo i_elem, sc x1_ref, sc x2_ref,
  const sc * n, int n_shared_vertices, int rotation, bool swap,
  std::vector< sc > & values ) {
  values[ 0 ] = 1 - x1_ref - x2_ref;
  values[ 1 ] = x1_ref;
  values[ 2 ] = x2_ref;
}

#pragma omp declare simd uniform( \
  i_elem, i_fun, n, n_shared_vertices, rotation, swap ) simdlen( DATA_WIDTH )
sc besthea::bem::basis_tri_p1::do_evaluate( lo i_elem, lo i_fun, sc x1_ref,
  sc x2_ref, const sc * n, int n_shared_vertices, int rotation, bool swap ) {
  sc value = 0.0;

  if ( i_fun == 0 ) {
    value = 1 - x1_ref - x2_ref;
  } else if ( i_fun == 1 ) {
    value = x1_ref;
  } else if ( i_fun == 2 ) {
    value = x2_ref;
  }

  return value;
}