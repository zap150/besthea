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
  const triangular_surface_mesh & mesh, lo levels, lo n_min_elems )
  : _mesh( mesh ),
    _levels( levels ),
    _n_min_elems( n_min_elems ),
    _non_empty_nodes( _levels ),
    _paddings( _levels, 0.0 ) {
  sc xmin, xmax, ymin, ymax, zmin, zmax;
  compute_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );

  vector_type center
    = { ( xmin + xmax ) / 2.0, ( ymin + ymax ) / 2.0, ( zmin + zmax ) / 2.0 };
  vector_type half_sizes = { std::abs( xmax - xmin ) / 2.0,
    std::abs( ymax - ymin ) / 2.0, std::abs( zmax - zmin ) / 2.0 };

  // create a root cluster and call recursive tree building
  _root = new space_cluster(
    center, half_sizes, _mesh.get_n_elements( ), nullptr, 0, 0, _mesh );

  for ( lo i = 0; i < _mesh.get_n_elements( ); ++i ) {
    _root->add_element( i );
  }

  _non_empty_nodes[ 0 ].push_back( _root );
  this->build_tree( *_root, 1 );
  this->compute_padding( *_root );
  for ( auto it = _paddings.begin( ); it != _paddings.end( ); ++it ) {
    std::cout << *it << std::endl;
  }
}

void besthea::mesh::space_cluster_tree::build_tree(
  space_cluster & root, lo level ) {
  // stop recursion if maximum number of tree levels is reached
  if ( level > _levels - 1 || root.get_n_elements( ) < _n_min_elems ) {
    root.set_n_children( 0 );
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
  for ( short i = 0; i < 8; ++i ) {
    if ( oct_sizes[ i ] > 0 ) {
      root.compute_suboctant( i, new_center, new_half_size );
      ++n_clusters;
      clusters[ i ] = new space_cluster(
        new_center, new_half_size, oct_sizes[ i ], &root, level, i, _mesh );
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

  // finally, add new children to cluster's children list and call the method
  // recursively on them
  root.set_n_children( n_clusters );

  for ( lo i = 0; i < 8; ++i ) {
    if ( clusters[ i ] != nullptr ) {
      root.add_child( clusters[ i ] );
      this->build_tree( *clusters[ i ], level + 1 );
    }
  }

  // shrink internal data storage for child clusters
  root.shrink_children( );
}

void besthea::mesh::space_cluster_tree::compute_bounding_box(
  sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax ) {
  xmin = ymin = zmin = std::numeric_limits< sc >::max( );
  xmax = ymax = zmax = std::numeric_limits< sc >::min( );

  sc node[ 3 ];
  for ( lo i = 0; i < _mesh.get_n_nodes( ); ++i ) {
    _mesh.get_node( i, node );

    if ( node[ 0 ] < xmin )
      xmin = node[ 0 ];
    if ( node[ 0 ] > xmax )
      xmax = node[ 0 ];

    if ( node[ 1 ] < ymin )
      ymin = node[ 1 ];
    if ( node[ 1 ] > ymax )
      ymax = node[ 1 ];

    if ( node[ 2 ] < zmin )
      zmin = node[ 2 ];
    if ( node[ 2 ] > zmax )
      zmax = node[ 2 ];
  }
}

sc besthea::mesh::space_cluster_tree::compute_padding( space_cluster & root ) {
  std::vector< space_cluster * > * children = root.get_children( );
  sc padding = -1.0;
  sc tmp_padding;

  if ( children != nullptr ) {
    // for non-leaf clusters, find the largest padding of its descendants
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      tmp_padding = this->compute_padding( **it );
      if ( tmp_padding > padding ) {
        padding = tmp_padding;
      }
    }
    if ( padding > _paddings[ root.get_level( ) ] ) {
      _paddings[ root.get_level( ) ] = padding;
    }
  } else {
    // for leaf clusters, compute padding directly
    padding = root.compute_padding( );
    if ( padding > _paddings[ root.get_level( ) ] ) {
      _paddings[ root.get_level( ) ] = padding;
    }
  }
  return padding;
}
