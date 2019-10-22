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

#include <besthea/spacetime_cluster_tree.h>

#include <algorithm>
#include <vector>

besthea::mesh::spacetime_cluster_tree::spacetime_cluster_tree(
  const triangular_surface_mesh & space_mesh, const temporal_mesh & time_mesh,
  lo time_levels, lo n_min_time_elems, lo n_min_space_elems )
  : _space_mesh( space_mesh ), _time_mesh( time_mesh ) {
  // first, we create the temporal and spatial trees
  _time_tree
    = new time_cluster_tree( _time_mesh, time_levels, n_min_time_elems );
  sc xmin, xmax, ymin, ymax, zmin, zmax;
  space_mesh.compute_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );

  sc max_half_size = std::max( { ( xmax - xmin ) / 2.0, ( ymax - ymin ) / 2.0,
                       ( zmax - zmin ) / 2.0 } )
    / 2.0;

  sc delta = _time_mesh.get_end( ) - _time_mesh.get_start( );

  _start_spatial_level = 0;

  // determine the number of initial octasections that has to be performed to
  // get to the level of the spatial tree satisfying the condition h_x^l \approx
  // sqrt(delta)
  while ( max_half_size > 1.5 * sqrt( delta ) ) {
    max_half_size *= 0.5;
    _start_spatial_level += 1;
  }
  std::cout << _start_spatial_level << std::endl;

  lo n_levels = _time_tree->get_levels( ) / 2;
  if ( _time_tree->get_levels( ) % 2 ) {
    ++n_levels;
  }
  _space_tree = new space_cluster_tree(
    _space_mesh, _start_spatial_level + n_levels, n_min_space_elems );

  // next, we assemble their combination into a space-time tree
  // the level -1 node is always the combination of the whole space & time
  _root = new spacetime_cluster(
    *_space_tree->get_root( ), *_time_tree->get_root( ), nullptr, -1 );

  // if the space has to be split to fullfil the condition, the individual roots
  // are stored in the space_roots vector
  std::vector< space_cluster * > space_roots;
  get_space_clusters_on_level(
    _space_tree->get_root( ), _start_spatial_level, space_roots );

  for ( auto it = space_roots.begin( ); it != space_roots.end( ); ++it ) {
    spacetime_cluster * cluster
      = new spacetime_cluster( **it, *_time_tree->get_root( ), _root, 0 );
    build_tree( cluster, 1 );
    // the roots of the subtrees are linked to the level -1 global root
    _root->add_child( cluster );
  }
  
  // initialize temporal m2m matrices
  set_temporal_m2m_matrices( );
}

void besthea::mesh::spacetime_cluster_tree::build_tree(
  spacetime_cluster * root, lo level ) {
  std::vector< space_cluster * > * space_children;
  bool split_space = false;
  if ( level % 2 == 0 ) {
    split_space = true;
    space_children = root->get_space_cluster( ).get_children( );

  } else {
    space_children = new std::vector< space_cluster * >;
    space_children->push_back( &root->get_space_cluster( ) );
  }

  std::vector< time_cluster * > * time_children
    = root->get_time_cluster( ).get_children( );

  if ( space_children != nullptr && time_children != nullptr ) {
    root->set_n_children( time_children->size( ) * space_children->size( ) );
    for ( auto it = time_children->begin( ); it != time_children->end( );
          ++it ) {
      for ( auto it2 = space_children->begin( ); it2 != space_children->end( );
            ++it2 ) {
        spacetime_cluster * cluster
          = new spacetime_cluster( **it2, **it, root, level + 1 );
        root->add_child( cluster );
        build_tree( cluster, level + 1 );
      }
    }
  }
  if ( !split_space ) {
    delete space_children;
  }
}

void besthea::mesh::spacetime_cluster_tree::get_space_clusters_on_level(
  space_cluster * root, lo level, std::vector< space_cluster * > & clusters ) {
  if ( root->get_level( ) < level ) {
    std::vector< space_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      get_space_clusters_on_level( *it, level, clusters );
    }
  } else {
    clusters.push_back( root );
  }
}

void besthea::mesh::spacetime_cluster_tree::set_temporal_m2m_matrices( ) {
  lo n_levels = _time_tree->get_levels( );
  // Declare the two structures containing matrices of appropriate size.
  // NOTE: For level 0 and 1 matrices are stored, but not needed. This allows
  //       for a direct access of the matrices via their level. The matrix for
  //       level n_levels is not needed and hence not allocated.
  _m2m_matrices_t_left = std::vector< full_matrix_type >( n_levels, 
                  full_matrix_type( _temp_order + 1, _temp_order + 1, false ) );
  _m2m_matrices_t_right = std::vector< full_matrix_type >( n_levels, 
                  full_matrix_type( _temp_order + 1, _temp_order + 1, false ) );
  std::vector< sc > paddings = _time_tree->get_paddings( );
  sc h_root_no_pad = _time_tree->get_root( )->get_half_size( );
  sc h_par_no_pad = h_root_no_pad / 4.0;
  // Initialize class for evaluation of lagrange polynomials and get
  // interpolation nodes in the interval [-1, 1].
  besthea::bem::lagrange_interpolant lagrange( _temp_order );
  vector_type nodes = lagrange.get_nodes( );
  vector_type nodes_l_child( _temp_order + 1, false );
  vector_type nodes_r_child( _temp_order + 1, false );
  vector_type values_lagrange( _temp_order + 1, false );
  for ( lo curr_level = 2; curr_level < n_levels; ++ curr_level ) {
    sc h_child_no_pad = h_par_no_pad / 2.0;
    // compute center of children (assuming that parent center is 0)
    sc padding_par = paddings[ curr_level ];
    sc padding_child = paddings[ curr_level + 1 ];
    // transform the nodes from [-1, 1] to the child interval and then back to 
    // [-1, 1] with the transformation of the parent interval:
    for ( lo j = 0; j <= _temp_order; ++ j ) {
      nodes_l_child[ j ] = 1.0 / ( h_par_no_pad + padding_par ) * 
        ( -h_child_no_pad + ( h_child_no_pad + padding_child ) * nodes [ j ] );
      nodes_r_child[ j ] = 1.0 / ( h_par_no_pad + padding_par ) * 
        ( h_child_no_pad + ( h_child_no_pad + padding_child ) * nodes [ j ] );
    }
    // compute left m2m matrix at current level
    for ( lo j = 0; j <= _temp_order; ++j ) {
      lagrange.evaluate( j, nodes_l_child, values_lagrange );
      for ( lo k = 0; k <= _temp_order; ++k )
        _m2m_matrices_t_left[curr_level].set( j, k, values_lagrange[ k ] );
    }
    // compute right m2m matrix at current level
    for ( lo j = 0; j <= _temp_order; ++j ) {
      lagrange.evaluate( j, nodes_r_child, values_lagrange );
      for ( lo k = 0; k <= _temp_order; ++k ) 
        _m2m_matrices_t_right[curr_level].set( j, k, values_lagrange[ k ] );
    }
    // TODO: The construction of the matrices is probably far from optimal: The
    // values are computed in row major order, but matrix memory is column major
    // Idea: Compute L2L matrices instead of M2M matrices?
    
    // update for next iteration
    h_par_no_pad = h_child_no_pad;
  }
}
