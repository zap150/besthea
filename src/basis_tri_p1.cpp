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

besthea::bem::basis_tri_p1::basis_tri_p1( const mesh_type & mesh ) {
  _mesh = &mesh;
}

besthea::bem::basis_tri_p1::~basis_tri_p1( ) {
}

lo besthea::bem::basis_tri_p1::dimension_local( ) const {
  return 3;
}

lo besthea::bem::basis_tri_p1::dimension_global( ) const {
  return _mesh->get_spatial_surface_mesh( )->get_n_nodes( );
}

void besthea::bem::basis_tri_p1::do_local_to_global(
  lo i_elem, std::vector< lo > & indices ) const {
  linear_algebra::indices< 3 > element;
  _mesh->get_spatial_surface_mesh( )->get_element( i_elem, element );
  indices[ 0 ] = element[ 0 ];
  indices[ 1 ] = element[ 1 ];
  indices[ 2 ] = element[ 2 ];
}

void besthea::bem::basis_tri_p1::do_local_to_global( lo i_elem,
  int n_shared_vertices, int rotation, bool swap,
  std::vector< lo > & indices ) const {
  linear_algebra::indices< 3 > element;
  _mesh->get_spatial_surface_mesh( )->get_element( i_elem, element );

  if ( n_shared_vertices == 2 && swap ) {
    indices[ 0 ] = element[ _map[ rotation + 1 ] ];
    indices[ 1 ] = element[ _map[ rotation ] ];
  } else {
    indices[ 0 ] = element[ _map[ rotation ] ];
    indices[ 1 ] = element[ _map[ rotation + 1 ] ];
  }
  indices[ 2 ] = element[ _map[ rotation + 2 ] ];
}

void besthea::bem::basis_tri_p1::evaluate_curl( lo i_elem,
  const linear_algebra::coordinates< 3 > & n, int n_shared_vertices,
  int rotation, bool swap, sc * curls ) const {
  linear_algebra::indices< 3 > element;
  linear_algebra::coordinates< 3 > x1rot, x2rot, x3rot;

  _mesh->get_spatial_surface_mesh( )->get_element( i_elem, element );

  if ( n_shared_vertices == 2 && swap ) {
    _mesh->get_spatial_surface_mesh( )->get_node(
      element[ _map[ rotation + 1 ] ], x1rot );
    _mesh->get_spatial_surface_mesh( )->get_node(
      element[ _map[ rotation ] ], x2rot );
  } else {
    _mesh->get_spatial_surface_mesh( )->get_node(
      element[ _map[ rotation ] ], x1rot );
    _mesh->get_spatial_surface_mesh( )->get_node(
      element[ _map[ rotation + 1 ] ], x2rot );
  }
  _mesh->get_spatial_surface_mesh( )->get_node(
    element[ _map[ rotation + 2 ] ], x3rot );

  // first two rows of R^\trans, third is n
  sc a11 = x2rot[ 0 ] - x1rot[ 0 ];
  sc a12 = x2rot[ 1 ] - x1rot[ 1 ];
  sc a13 = x2rot[ 2 ] - x1rot[ 2 ];
  sc a21 = x3rot[ 0 ] - x1rot[ 0 ];
  sc a22 = x3rot[ 1 ] - x1rot[ 1 ];
  sc a23 = x3rot[ 2 ] - x1rot[ 2 ];

  // determinant to invert the matrix
  sc det = n[ 0 ] * ( a12 * a23 - a13 * a22 )
    + n[ 1 ] * ( a13 * a21 - a11 * a23 ) + n[ 2 ] * ( a11 * a22 - a21 * a12 );

  // gradients in actual triangle
  // R^{-\trans} * [1;0;0]
  sc g21 = n[ 2 ] * a22 - n[ 1 ] * a23;
  sc g22 = -n[ 2 ] * a21 + n[ 0 ] * a23;
  sc g23 = n[ 1 ] * a21 - n[ 0 ] * a22;
  // n x gradient
  curls[ 3 ] = ( n[ 1 ] * g23 - n[ 2 ] * g22 ) / det;
  curls[ 4 ] = ( n[ 2 ] * g21 - n[ 0 ] * g23 ) / det;
  curls[ 5 ] = ( n[ 0 ] * g22 - n[ 1 ] * g21 ) / det;

  // R^{-\trans} * [0;1;0]
  sc g31 = -n[ 2 ] * a12 + n[ 1 ] * a13;
  sc g32 = n[ 2 ] * a11 - n[ 0 ] * a13;
  sc g33 = -n[ 1 ] * a11 + n[ 0 ] * a12;
  // n x gradient
  curls[ 6 ] = ( n[ 1 ] * g33 - n[ 2 ] * g32 ) / det;
  curls[ 7 ] = ( n[ 2 ] * g31 - n[ 0 ] * g33 ) / det;
  curls[ 8 ] = ( n[ 0 ] * g32 - n[ 1 ] * g31 ) / det;

  // R^{-\trans} * [-1;-1;0]
  // n x gradient
  curls[ 0 ] = ( -n[ 1 ] * ( g23 + g33 ) + n[ 2 ] * ( g22 + g32 ) ) / det;
  curls[ 1 ] = ( -n[ 2 ] * ( g21 + g31 ) + n[ 0 ] * ( g23 + g33 ) ) / det;
  curls[ 2 ] = ( -n[ 0 ] * ( g22 + g32 ) + n[ 1 ] * ( g21 + g31 ) ) / det;
}
