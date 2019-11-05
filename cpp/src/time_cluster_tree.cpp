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

#include "besthea/time_cluster_tree.h"

besthea::mesh::time_cluster_tree::time_cluster_tree(
  const temporal_mesh & mesh, lo levels, lo n_min_elems )
  : _mesh( mesh ),
    _levels( levels ),
    _real_max_levels( 0 ),
    _n_min_elems( n_min_elems ),
    _paddings( _levels, -1.0 ) {
  sc center = ( _mesh.get_end( ) + _mesh.get_start( ) ) / 2;
  sc half_size = ( _mesh.get_end( ) - _mesh.get_start( ) ) / 2;

  _root = new time_cluster(
    center, half_size, _mesh.get_n_elements( ), nullptr, 0, _mesh );

  for ( lo i = 0; i < _mesh.get_n_elements( ); ++i ) {
    _root->add_element( i );
  }

  this->build_tree( *_root, 1 );
  this->compute_padding( *_root );
  _levels = std::min( _levels, _real_max_levels );

  _paddings.resize( _levels );
  _paddings.shrink_to_fit( );
  collect_leaves( *_root );
}

void besthea::mesh::time_cluster_tree::build_tree(
  time_cluster & root, lo level ) {
  // stop recursion if maximum number of levels is reached

  if ( level > _levels - 1 || root.get_n_elements( ) < _n_min_elems ) {
    root.set_n_children( 0 );

    if ( level > _real_max_levels ) {
      _real_max_levels = level;
    }
    return;
  }

  sc center = root.get_center( );
  sc half_size = root.get_half_size( );

  sc el_centroid;

  lo root_n_elems = root.get_n_elements( );
  lo n_left = 0;
  lo n_right = 0;
  lo elem_idx = 0;

  // count the number of elements in each subcluster
  for ( lo i = 0; i < root_n_elems; ++i ) {
    elem_idx = root.get_element( i );
    el_centroid = _mesh.get_centroid( elem_idx );
    if ( el_centroid <= center ) {
      ++n_left;
    } else {
      ++n_right;
    }
  }

  time_cluster * left_cluster = new time_cluster(
    center - half_size / 2, half_size / 2, n_left, &root, level, _mesh );

  time_cluster * right_cluster = new time_cluster(
    center + half_size / 2, half_size / 2, n_right, &root, level, _mesh );

  // add elements to each subcluster
  for ( lo i = 0; i < root_n_elems; ++i ) {
    elem_idx = root.get_element( i );
    el_centroid = _mesh.get_centroid( elem_idx );
    if ( el_centroid <= center ) {
      left_cluster->add_element( elem_idx );
    } else {
      right_cluster->add_element( elem_idx );
    }
  }

  root.set_n_children( 2 );
  root.add_child( left_cluster );
  this->build_tree( *left_cluster, level + 1 );
  root.add_child( right_cluster );
  this->build_tree( *right_cluster, level + 1 );
}

sc besthea::mesh::time_cluster_tree::compute_padding( time_cluster & root ) {
  std::vector< time_cluster * > * children = root.get_children( );
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

void besthea::mesh::time_cluster_tree::collect_leaves( time_cluster & root ) {
  if ( root.get_n_children( ) == 0 ) {
    _leaves.push_back( &root );
  } else {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      collect_leaves( **it );
    }
  }
}
