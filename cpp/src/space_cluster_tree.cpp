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

#include "besthea/space_cluster_tree.h"

#include <cstdlib>
#include <limits>

besthea::mesh::space_cluster_tree::space_cluster_tree(
  const triangular_surface_mesh & mesh, lo levels )
  : _mesh( mesh ), _levels( levels ), _non_empty_nodes( _levels ) {
  sc xmin, xmax, ymin, ymax, zmin, zmax;
  compute_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );

  vector_type center
    = { ( xmin + xmax ) / 2.0, ( ymin + ymax ) / 2.0, ( zmin + zmax ) / 2.0 };
  vector_type half_sizes = { std::abs( xmax - xmin ) / 2.0,
    std::abs( ymax - ymin ) / 2.0, std::abs( zmax - zmin ) / 2.0 };

  // create a root cluster and call recursive tree building
  _root = new space_cluster(
    center, half_sizes, _mesh.get_n_elements( ), nullptr, 0, _mesh );

  for ( lo i = 0; i < _mesh.get_n_elements( ); ++i ) {
    _root->add_element( i );
  }

  _non_empty_nodes[ 0 ].push_back( _root );
  this->build_tree( *_root, 1 );
}

void besthea::mesh::space_cluster_tree::build_tree(
  space_cluster & root, lo level ) {
  // stop recursion if maximum number of tree levels is reached
  if ( level > _levels - 1 ) {
    return;
  }

  // allocate children's and temporary data
  vector_type center( 3 );
  vector_type half_size( 3 );
  sc el_centroid[ 3 ];
  root.get_center( center );
  root.get_half_size( half_size );
  lo elem_idx = 0;
  lo oct_sizes[ 8 ] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  space_cluster * clusters[ 8 ];

  // first count the number of elements in octants for data preallocation
  for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
    elem_idx = root.get_element( i );

    _mesh.get_centroid( elem_idx, el_centroid );

    if ( el_centroid[ 0 ] >= center( 0 ) && el_centroid[ 1 ] >= center( 1 )
      && el_centroid[ 2 ] >= center( 2 ) ) {
      ++oct_sizes[ 0 ];
    } else if ( el_centroid[ 0 ] < center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] >= center( 2 ) ) {
      ++oct_sizes[ 1 ];
    } else if ( el_centroid[ 0 ] < center( 0 ) && el_centroid[ 1 ] < center( 1 )
      && el_centroid[ 2 ] >= center( 2 ) ) {
      ++oct_sizes[ 2 ];
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] < center( 1 ) && el_centroid[ 2 ] >= center( 2 ) ) {
      ++oct_sizes[ 3 ];
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      ++oct_sizes[ 4 ];
    } else if ( el_centroid[ 0 ] < center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      ++oct_sizes[ 5 ];
    } else if ( el_centroid[ 0 ] < center( 0 ) && el_centroid[ 1 ] < center( 1 )
      && el_centroid[ 2 ] < center( 2 ) ) {
      ++oct_sizes[ 6 ];
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] < center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      ++oct_sizes[ 7 ];
    }
  }

  lo n_clusters = 0;
  vector_type new_center( 3 );
  vector_type new_half_size( 3 );
  for ( lo i = 0; i < 8; ++i ) {
    if ( oct_sizes[ i ] > 0 ) {
      root.compute_suboctant( i, new_center, new_half_size );
      ++n_clusters;
      clusters[ i ] = new space_cluster(
        new_center, new_half_size, oct_sizes[ i ], &root, level, _mesh );
      _non_empty_nodes[ level ].push_back( clusters[ i ] );
    } else {
      clusters[ i ] = nullptr;
    }
  }

  // next assign elements to clusters
  for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
    elem_idx = root.get_element( i );
    _mesh.get_centroid( elem_idx, el_centroid );
    if ( el_centroid[ 0 ] >= center( 0 ) && el_centroid[ 1 ] >= center( 1 )
      && el_centroid[ 2 ] >= center( 2 ) ) {
      clusters[ 0 ]->add_element( elem_idx );
    } else if ( el_centroid[ 0 ] < center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] >= center( 2 ) ) {
      clusters[ 1 ]->add_element( elem_idx );
    } else if ( el_centroid[ 0 ] < center( 0 ) && el_centroid[ 1 ] < center( 1 )
      && el_centroid[ 2 ] >= center( 2 ) ) {
      clusters[ 2 ]->add_element( elem_idx );
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] < center( 1 ) && el_centroid[ 2 ] >= center( 2 ) ) {
      clusters[ 3 ]->add_element( elem_idx );
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      clusters[ 4 ]->add_element( elem_idx );
    } else if ( el_centroid[ 0 ] < center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      clusters[ 5 ]->add_element( elem_idx );
    } else if ( el_centroid[ 0 ] < center( 0 ) && el_centroid[ 1 ] < center( 1 )
      && el_centroid[ 2 ] < center( 2 ) ) {
      clusters[ 6 ]->add_element( elem_idx );
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] < center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      clusters[ 7 ]->add_element( elem_idx );
    }
  }

  root.set_n_children( n_clusters );

  for ( lo i = 0; i < 8; ++i ) {
    if ( clusters[ i ] != nullptr ) {
      root.add_child( clusters[ i ] );
      this->build_tree( *clusters[ i ], level + 1 );
    }
  }
}

void besthea::mesh::space_cluster_tree::compute_bounding_box(
  sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax ) {
  xmin = ymin = zmin = std::numeric_limits< sc >::max( );
  xmax = ymax = zmax = std::numeric_limits< sc >::min( );

  sc centroid[ 3 ];
  for ( lo i = 0; i < _mesh.get_n_elements( ); ++i ) {
    _mesh.get_centroid( i, centroid );

    if ( centroid[ 0 ] < xmin )
      xmin = centroid[ 0 ];
    if ( centroid[ 0 ] > xmax )
      xmax = centroid[ 0 ];

    if ( centroid[ 1 ] < ymin )
      ymin = centroid[ 1 ];
    if ( centroid[ 1 ] > ymax )
      ymax = centroid[ 1 ];

    if ( centroid[ 2 ] < zmin )
      zmin = centroid[ 2 ];
    if ( centroid[ 2 ] > zmax )
      zmax = centroid[ 2 ];
  }
}
