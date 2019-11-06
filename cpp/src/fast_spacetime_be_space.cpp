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

#include "besthea/fast_spacetime_be_space.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"

template< class basis_type >
besthea::bem::fast_spacetime_be_space< basis_type >::fast_spacetime_be_space(
  mesh::spacetime_cluster_tree & tree )
  : spacetime_be_space< basis_type >( tree.get_mesh( ) ), _tree( &tree ) {
}

template< class basis_type >
besthea::bem::fast_spacetime_be_space<
  basis_type >::~fast_spacetime_be_space( ) {
}

/**
 * Projects a function to the piecewise constant boundary element space.
 * @param[in] f Function to be projected.
 * @param[out] interpolation Interpolation vector.
 */
template<>
void besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >::
  interpolation(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    block_vector_type & interpolation ) const {
  const s_mesh_type & s_mesh = _tree->get_spatial_mesh( );
  const t_mesh_type & t_mesh = _tree->get_temporal_mesh( );
  lo n_timesteps = t_mesh.get_n_elements( );
  lo n_elements = s_mesh.get_n_elements( );

  interpolation.resize( n_timesteps );
  interpolation.resize_blocks( n_elements );
  linear_algebra::coordinates< 3 > centroid, n;

  for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
    s_mesh.get_centroid( i_elem, centroid );
    s_mesh.get_normal( i_elem, n );
    for ( lo d = 0; d < n_timesteps; ++d ) {
      interpolation.set( d, i_elem,
        f( centroid[ 0 ], centroid[ 1 ], centroid[ 2 ], n,
          t_mesh.get_centroid( d ) ) );
    }
  }
}

/**
 * Projects a function to the piecewise linear boundary element space.
 * @param[in] f Function to be projected.
 * @param[out] interpolation Interpolation vector.
 */
template<>
void besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >::
  interpolation(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    block_vector_type & interpolation ) const {
  const s_mesh_type & s_mesh = _tree->get_spatial_mesh( );
  const t_mesh_type & t_mesh = _tree->get_temporal_mesh( );
  lo n_timesteps = t_mesh.get_n_elements( );
  lo n_nodes = s_mesh.get_n_nodes( );

  interpolation.resize( n_timesteps );
  interpolation.resize_blocks( n_nodes );
  linear_algebra::coordinates< 3 > x, n;

  for ( lo i_node = 0; i_node < n_nodes; ++i_node ) {
    s_mesh.get_node( i_node, x );
    s_mesh.get_nodal_normal( i_node, n );
    for ( lo d = 0; d < n_timesteps; ++d ) {
      interpolation.set(
        d, i_node, f( x[ 0 ], x[ 1 ], x[ 2 ], n, t_mesh.get_centroid( d ) ) );
    }
  }
}

template< class basis_type >
void besthea::bem::fast_spacetime_be_space< basis_type >::L2_projection(
  sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  block_vector_type & projection, int order_matrix, int order_rhs_spatial,
  int order_rhs_temporal ) const {
}

template< class basis_type >
sc besthea::bem::fast_spacetime_be_space< basis_type >::L2_relative_error(
  sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  const block_vector_type & approximation, int order_rhs_spatial,
  int order_rhs_temporal ) const {
  return 0.0;
}

template class besthea::bem::fast_spacetime_be_space<
  besthea::bem::basis_tri_p0 >;
template class besthea::bem::fast_spacetime_be_space<
  besthea::bem::basis_tri_p1 >;
