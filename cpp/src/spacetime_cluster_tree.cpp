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

#include "besthea/spacetime_cluster_tree.h"

#include <algorithm>
#include <cmath>
#include <vector>

besthea::mesh::spacetime_cluster_tree::spacetime_cluster_tree(
  const spacetime_tensor_mesh & spacetime_mesh, lo time_levels,
  lo n_min_time_elems, lo n_min_space_elems, sc st_coeff,
  slou spatial_nearfield_limit )
  : _levels( 0 ),
    _spacetime_mesh( spacetime_mesh ),
    _space_mesh( *( spacetime_mesh.get_spatial_surface_mesh( ) ) ),
    _time_mesh( *( spacetime_mesh.get_temporal_mesh( ) ) ),
    _s_t_coeff( st_coeff ),
    _spatial_nearfield_limit( spatial_nearfield_limit ) {
  // first, we create the temporal and spatial trees

  _time_tree
    = new time_cluster_tree( _time_mesh, time_levels, n_min_time_elems );
  sc xmin, xmax, ymin, ymax, zmin, zmax;
  _space_mesh.compute_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );

  sc max_half_size
    = std::max( { ( xmax - xmin ), ( ymax - ymin ), ( zmax - zmin ) } ) / 2.0;

  sc delta = _time_mesh.get_end( ) - _time_mesh.get_start( );

  _start_spatial_level = 0;

  // determine the number of initial octasections that has to be performed to
  // get to the level of the spatial tree satisfying the condition
  // h_x^l \approx st_coeff sqrt(delta)
  while ( max_half_size > st_coeff * sqrt( delta ) ) {
    max_half_size *= 0.5;
    _start_spatial_level += 1;
  }

  lo n_t_levels = _time_tree->get_levels( );
  lo n_s_levels = n_t_levels / 2;
  if ( n_t_levels % 2 ) {
    ++n_s_levels;
  }
  _space_tree = new space_cluster_tree(
    _space_mesh, _start_spatial_level + n_s_levels, n_min_space_elems );

  // determine for which temporal level the first spatial refinement is needed
  _start_temporal_level = 1;
  delta *= 0.5;
  if ( _start_spatial_level == 0 ) {
    while ( max_half_size <= st_coeff * sqrt( delta ) ) {
      delta *= 0.5;
      _start_temporal_level += 1;
    }
    // // shift _start_temporal_level if necessary to guarantee levels as in
    // // Messner's work
    // if ( ( n_t_levels - _start_temporal_level ) % 2 ) {
    //   _start_temporal_level -= 1;
    // }
  } else {
    _start_temporal_level = 2;
  }

  // next, we assemble their combination into a space-time tree
  // the level -1 node is always the combination of the whole space & time
  _root = new spacetime_cluster(
    *_space_tree->get_root( ), *_time_tree->get_root( ), nullptr, -1 );

  // _map_to_spacetime_clusters.insert( std::pair< std::pair< space_cluster *,
  //   time_cluster * >, spacetime_cluster * >( std::pair< space_cluster *,
  //   time_cluster * >( _space_tree->get_root( ), _time_tree->get_root( ) ),
  //   & _root ) );

  // if the space has to be split to fulfill the condition, the individual roots
  // are stored in the space_roots vector
  std::vector< space_cluster * > space_roots;
  get_space_clusters_on_level(
    _space_tree->get_root( ), _start_spatial_level, space_roots );

  // determine whether the individual roots are split in the first step of the
  // building of the cluster tree (guarantees levels as in Messner's work)
  // TODO: discuss if this should be kept in the final code (somehow haphazard)
  // if ( ( _start_temporal_level == 0 ) && ( n_t_levels % 2 ) ) {
  //   split_space = true;
  // }
  bool split_space = ( _start_temporal_level <= 1 );

  for ( auto it = space_roots.begin( ); it != space_roots.end( ); ++it ) {
    spacetime_cluster * cluster
      = new spacetime_cluster( **it, *_time_tree->get_root( ), _root, 0 );
    _map_to_spacetime_clusters.insert(
      { { *it, _time_tree->get_root( ) }, cluster } );
    build_tree( cluster, 1, split_space );

    // the roots of the subtrees are linked to the level -1 global root
    _root->add_child( cluster );
  }

  // collect all clusters without descendants
  collect_leaves( *_root );

  // fill interaction lists
  determine_interactions( *_root );

  // fill nearfield lists
  determine_nearfield( );
}

void besthea::mesh::spacetime_cluster_tree::build_tree(
  spacetime_cluster * root, lo level, bool split_space ) {
  std::vector< space_cluster * > * space_children;
  bool split_space_descendant;
  if ( !split_space ) {
    space_children = new std::vector< space_cluster * >;
    space_children->push_back( &root->get_space_cluster( ) );
    if ( level + 1 < _start_temporal_level ) {
      // no spatial refinement as long as next level < _start_temporal_level
      split_space_descendant = false;
    } else {
      // alternate between refinement and non refinement
      split_space_descendant = true;
    }
  } else {
    space_children = root->get_space_cluster( ).get_children( );
    split_space_descendant = false;
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
          = new spacetime_cluster( **it2, **it, root, level );
        _map_to_spacetime_clusters.insert(
          std::pair< std::pair< space_cluster *, time_cluster * >,
            spacetime_cluster * >(
            std::pair< space_cluster *, time_cluster * >( *it2, *it ),
            cluster ) );
        root->add_child( cluster );
        build_tree( cluster, level + 1, split_space_descendant );
      }
    }
  } else {
    // Root is a leaf. Update the number of levels if necessary.
    if ( level > _levels ) {
      _levels = level;
    }
  }
  // delete space children, if they have been allocated in this routine
  if ( !split_space ) {
    delete space_children;
  }
  // old code:
  // if ( ( split_space_descendant ) || ( level < _start_temporal_level ) ) {
  //   delete space_children;
  // }
}

void besthea::mesh::spacetime_cluster_tree::get_space_clusters_on_level(
  space_cluster * root, lo level, std::vector< space_cluster * > & clusters ) {
  if ( root->get_level( ) < level ) {
    std::vector< space_cluster * > * children = root->get_children( );
    if ( children == nullptr ) {
      // early spatial leaf is added to the cluster tree
      clusters.push_back( root );
    } else {
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        get_space_clusters_on_level( *it, level, clusters );
      }
    }
  } else {
    clusters.push_back( root );
  }
}

void besthea::mesh::spacetime_cluster_tree::collect_leaves(
  spacetime_cluster & root ) {
  if ( root.get_n_children( ) == 0 ) {
    _leaves.push_back( &root );
  } else {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      collect_leaves( **it );
    }
  }
}

void besthea::mesh::spacetime_cluster_tree::initialize_moment_contributions(
  spacetime_cluster * root, const lo & n_rows_contribution,
  const lo & n_columns_contribution ) {
  root->set_moment_contribution( n_rows_contribution, n_columns_contribution );
  if ( root->get_n_children( ) > 0 ) {
    std::vector< spacetime_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      initialize_moment_contributions(
        *it, n_rows_contribution, n_columns_contribution );
    }
  }
}

void besthea::mesh::spacetime_cluster_tree::initialize_local_contributions(
  spacetime_cluster * root, const lo & n_rows_contribution,
  const lo & n_columns_contribution ) {
  root->set_local_contribution( n_rows_contribution, n_columns_contribution );
  if ( root->get_n_children( ) > 0 ) {
    std::vector< spacetime_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      initialize_local_contributions(
        *it, n_rows_contribution, n_columns_contribution );
    }
  }
}

void besthea::mesh::spacetime_cluster_tree::clean_local_contributions(
  spacetime_cluster * root ) {
  root->clean_local_contribution( );
  if ( root->get_n_children( ) > 0 ) {
    std::vector< spacetime_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      clean_local_contributions( *it );
    }
  }
}

void besthea::mesh::spacetime_cluster_tree::clean_moment_contributions(
  spacetime_cluster * root ) {
  root->clean_moment_contribution( );
  if ( root->get_n_children( ) > 0 ) {
    std::vector< spacetime_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      clean_moment_contributions( *it );
    }
  }
}

void besthea::mesh::spacetime_cluster_tree::determine_interactions(
  spacetime_cluster & root ) {
  space_cluster & root_space_cluster = root.get_space_cluster( );
  time_cluster & root_time_cluster = root.get_time_cluster( );
  lo time_level = root_time_cluster.get_level( );
  // interacting clusters exist only if time level > 1
  if ( time_level > 1 ) {
    // compute the interacting time clusters
    std::vector< time_cluster * > interaction_list_time;
    bool is_left_child = root_time_cluster.is_left_child( );
    time_cluster * admissible_cluster;
    // the right child can have two time intervals in its interaction list
    // add the farthest first
    if ( !is_left_child ) {
      // check if there is a left left neighbour
      if ( root_time_cluster.get_left_neighbour( )->get_left_neighbour( )
        != nullptr ) {
        // add left left left neighbour to interaction list, if != nullptr
        admissible_cluster = root_time_cluster.get_left_neighbour( )
                               ->get_left_neighbour( )
                               ->get_left_neighbour( );
        if ( admissible_cluster != nullptr )
          interaction_list_time.push_back( admissible_cluster );
      }
    }
    // add the left left neighbour to the interaction list in all cases
    // check first if there is a left neighbour
    if ( root_time_cluster.get_left_neighbour( ) != nullptr ) {
      admissible_cluster
        = root_time_cluster.get_left_neighbour( )->get_left_neighbour( );
      if ( admissible_cluster != nullptr )
        interaction_list_time.push_back( admissible_cluster );
    }
    // compute the interacting space clusters
    std::vector< space_cluster * > interaction_list_space;
    _space_tree->find_neighbors(
      root_space_cluster, _spatial_nearfield_limit, interaction_list_space );

    // construct the interaction list of the spacetime cluster from the
    // interaction lists of its spatial and temporal component.
    for ( auto it_time = interaction_list_time.begin( );
          it_time != interaction_list_time.end( ); ++it_time ) {
      for ( auto it_space = interaction_list_space.begin( );
            it_space != interaction_list_space.end( ); ++it_space ) {
        // find space time cluster corresponding to the current spatial and
        // temporal clusters in the respective interaction lists
        spacetime_cluster * admissible_st_cluster
          = _map_to_spacetime_clusters[ std::pair< space_cluster *,
            time_cluster * >( *it_space, *it_time ) ];
        root.add_to_interaction_list( admissible_st_cluster );
      }
    }
  }
  // call the same routine for all children of root, if root has children
  std::vector< spacetime_cluster * > * root_children = root.get_children( );
  if ( root_children != nullptr ) {
    for ( auto it = root_children->begin( ); it != root_children->end( );
          ++it ) {
      determine_interactions( **it );
    }
  }
}

void besthea::mesh::spacetime_cluster_tree::determine_nearfield( ) {
  for ( auto it = _leaves.begin( ); it != _leaves.end( ); ++it ) {
    space_cluster & curr_space_cluster = ( *it )->get_space_cluster( );
    time_cluster & curr_time_cluster = ( *it )->get_time_cluster( );
    // construct nearfield lists in time and space separately and combine them
    // later to get the spacetime list
    std::vector< time_cluster * > nearfield_list_time;
    // if the left neighbor of the time cluster exists, add it to the list
    if ( curr_time_cluster.get_left_neighbour( ) != nullptr ) {
      nearfield_list_time.push_back( curr_time_cluster.get_left_neighbour( ) );
    }
    // in any case, add the cluster itself to the temporal nearfield list
    nearfield_list_time.push_back( &curr_time_cluster );

    std::vector< space_cluster * > nearfield_list_space;
    _space_tree->find_neighbors(
      curr_space_cluster, _spatial_nearfield_limit, nearfield_list_space );

    // construct the nearfield list of the spacetime cluster from the
    // nearfield lists of its spatial and temporal component.
    for ( auto it_time = nearfield_list_time.begin( );
          it_time != nearfield_list_time.end( ); ++it_time ) {
      for ( auto it_space = nearfield_list_space.begin( );
            it_space != nearfield_list_space.end( ); ++it_space ) {
        // find space time cluster corresponding to the current spatial and
        // temporal clusters in the respective nearfield lists
        spacetime_cluster * nearfield_st_cluster
          = _map_to_spacetime_clusters[ std::pair< space_cluster *,
            time_cluster * >( *it_space, *it_time ) ];
        ( *it )->add_to_nearfield_list( nearfield_st_cluster );
      }
    }
  }
}
