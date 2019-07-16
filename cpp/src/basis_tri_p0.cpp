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

#include "besthea/basis_tri_p0.h"

#include <algorithm>

besthea::bem::basis_tri_p0::basis_tri_p0( mesh_type & mesh ) {
  _mesh = &mesh;
}

besthea::bem::basis_tri_p0::~basis_tri_p0( ) {
}

lo besthea::bem::basis_tri_p0::dimension_local( ) {
  return 1;
}

lo besthea::bem::basis_tri_p0::dimension_global( ) {
  return _mesh->get_spatial_mesh( )->get_n_elements( );
}

void besthea::bem::basis_tri_p0::do_local_to_global( lo i_elem,
  int n_shared_vertices, int rotation, bool swap,
  std::vector< lo > & indices ) {
  indices[ 0 ] = i_elem;
}

#pragma omp declare simd uniform( \
  i_elem, n, n_shared_vertices, rotation, swap ) simdlen( DATA_WIDTH )
void besthea::bem::basis_tri_p0::do_evaluate( lo i_elem, sc x1_ref, sc x2_ref,
  const sc * n, int n_shared_vertices, int rotation, bool swap,
  std::vector< sc > & values ) {
  values[ 0 ] = 1.0;
}

#pragma omp declare simd uniform( \
  i_elem, i_fun, n, n_shared_vertices, rotation, swap ) simdlen( DATA_WIDTH )
sc besthea::bem::basis_tri_p0::do_evaluate( lo i_elem, lo i_fun, sc x1_ref,
  sc x2_ref, const sc * n, int n_shared_vertices, int rotation, bool swap ) {
  return 1.0;
}
