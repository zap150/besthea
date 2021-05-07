/*
Copyright (c) 2020, VSB - Technical University of Ostrava and Graz University of
Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the names of VSB - Technical University of  Ostrava and Graz
  University of Technology nor the names of its contributors may be used to
  endorse or promote products derived from this software without specific prior
  written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL VSB - TECHNICAL UNIVERSITY OF OSTRAVA AND
GRAZ UNIVERSITY OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "besthea/distributed_spacetime_cluster_tree.h"

#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/tree_structure.h"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <vector>

besthea::mesh::distributed_spacetime_cluster_tree::
  distributed_spacetime_cluster_tree(
    distributed_spacetime_tensor_mesh & spacetime_mesh, lo levels,
    lo n_min_elems, sc st_coeff, slou spatial_nearfield_limit, MPI_Comm * comm )
  : _max_levels( levels ),
    _real_max_levels( 0 ),
    _spacetime_mesh( spacetime_mesh ),
    _local_max_space_level( 0 ),
    //_s_t_coeff( st_coeff ),
    _n_min_elems( n_min_elems ),
    _spatial_paddings( _max_levels, 0.0 ),
    _spatial_nearfield_limit( spatial_nearfield_limit ),
    _comm( comm ) {
  MPI_Comm_rank( *_comm, &_my_rank );
  MPI_Comm_size( *_comm, &_n_processes );

  sc xmin, xmax, ymin, ymax, zmin, zmax;
  // this is now a local computation since space mesh is replicated on all MPI
  // processes
  compute_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );
  _bounding_box_size.resize( 4 );
  _bounding_box_size[ 0 ] = ( xmax - xmin ) / 2.0;
  _bounding_box_size[ 1 ] = ( ymax - ymin ) / 2.0;
  _bounding_box_size[ 2 ] = ( zmax - zmin ) / 2.0;
  _bounding_box_size[ 3 ]
    = ( _spacetime_mesh.get_end( ) - _spacetime_mesh.get_start( ) ) / 2.0;

  vector_type space_center
    = { ( xmin + xmax ) / 2.0, ( ymin + ymax ) / 2.0, ( zmin + zmax ) / 2.0 };
  vector_type space_half_sizes = { std::abs( xmax - xmin ) / 2.0,
    std::abs( ymax - ymin ) / 2.0, std::abs( zmax - zmin ) / 2.0 };
  sc time_center = get_distribution_tree( )->get_root( )->get_center( );
  sc time_half_size = get_distribution_tree( )->get_root( )->get_half_size( );

  // determine _start_space_refinement, i.e. the level where space is refined
  // for the first time, and _initial_space_refinement, i.e. the number of
  // spatial refinements needed for level 0.
  _initial_space_refinement = 0;
  sc delta = 2 * time_half_size;
  sc max_half_size
    = std::max( { ( xmax - xmin ), ( ymax - ymin ), ( zmax - zmin ) } ) / 2.0;

  // determine the number of initial octasections that has to be performed to
  // get clusters whose spatial half size (or rather its largest component) h_x
  // satisfies the condition h_x \approx st_coeff sqrt(delta). this is
  // _initial_space_refinement
  // @todo the criterion should depend on the heat capacity constant alpha too:
  // i.e. delta should be replaced by delta * alpha (or st_coeff has to be
  // chosen accordingly)
  while ( max_half_size > st_coeff * sqrt( delta ) ) {
    max_half_size *= 0.5;
    _initial_space_refinement += 1;
  }
  // determine for which temporal level the first spatial refinement is needed
  _start_space_refinement = 1;
  delta *= 0.5;
  while ( max_half_size <= st_coeff * sqrt( delta ) ) {
    delta *= 0.5;
    _start_space_refinement += 1;
  }

  // like this it is guaranteed that max_halfsize <= st_coeff * sqrt( delta )
  // on all levels of the tree

  // // old version:
  // if ( _initial_space_refinement == 0 ) {
  //   while ( max_half_size <= st_coeff * sqrt( delta ) ) {
  //     delta *= 0.5;
  //     _start_space_refinement += 1;
  //   }
  // } else {
  //   _start_space_refinement = 2;
  // }

  // create root at level -1 as combination of whole space and time.
  // set first value of pseudoroot to a distinguished value to avoid problems.
  std::vector< slou > coordinates
    = { std::numeric_limits< slou >::max( ), 0, 0, 0, 0 };
  _root = new general_spacetime_cluster( space_center, time_center,
    space_half_sizes, time_half_size, _spacetime_mesh.get_n_elements( ),
    nullptr, -1, 0, coordinates, 0, 0, 0, 0, _spacetime_mesh, -1, false );

  std::vector< lo > elems_in_clusters;

  build_tree( _root );

  _max_levels = std::min( _max_levels, _real_max_levels );

  // note: _real_max_levels is global, since communication takes place in build
  // tree routine
  _spatial_paddings.resize( _max_levels );
  _spatial_paddings.shrink_to_fit( );

  compute_local_spatial_padding( *_root );

  // Correct the spatial padding to guarantee, that clusters which are refined
  // in space the same number of times are padded equally (a correction is
  // necessary in case of early space-time leaf clusters). It suffices to
  // identify levels where the number of spatial refinements is the same and set
  // the padding to the maximal value.
  lo space_level_group_begin
    = 0;  // used to keep track of the level where space was last refined
  lo next_space_ref_level = _start_space_refinement;
  sc max_padding_at_spatial_level = 0.0;
  for ( lo i = 0; i < _max_levels; ++i ) {
    if ( _spatial_paddings[ i ] > max_padding_at_spatial_level ) {
      max_padding_at_spatial_level = _spatial_paddings[ i ];
    }
    if ( i + 1 == next_space_ref_level ) {
      for ( lo j = space_level_group_begin; j <= i; ++j ) {
        _spatial_paddings[ j ] = max_padding_at_spatial_level;
      }
      space_level_group_begin = i + 1;
      next_space_ref_level += 2;
      max_padding_at_spatial_level = 0.0;
    }
  }

  // reduce padding data among processes
  MPI_Allreduce( MPI_IN_PLACE, _spatial_paddings.data( ),
    _spatial_paddings.size( ), get_scalar_type< sc >::MPI_SC( ), MPI_MAX,
    *_comm );

  // extend the locally essential distribution tree:
  // first determine clusters in the distribution tree for which the extension
  // cannot be done locally. in addition, determine scheduling time clusters
  // where the leaf information of the associated spacetime clusters is
  // required, but not available (for later)
  std::set< std::pair< lo, scheduling_time_cluster * >,
    compare_pairs_of_process_ids_and_scheduling_time_clusters >
    subtree_send_list;
  std::set< std::pair< lo, scheduling_time_cluster * >,
    compare_pairs_of_process_ids_and_scheduling_time_clusters >
    subtree_receive_list;
  std::set< std::pair< lo, scheduling_time_cluster * >,
    compare_pairs_of_process_ids_and_scheduling_time_clusters >
    leaf_info_send_list;
  std::set< std::pair< lo, scheduling_time_cluster * >,
    compare_pairs_of_process_ids_and_scheduling_time_clusters >
    leaf_info_receive_list;
  tree_structure * distribution_tree = get_distribution_tree( );
  distribution_tree->determine_cluster_communication_lists(
    distribution_tree->get_root( ), subtree_send_list, subtree_receive_list,
    leaf_info_send_list, leaf_info_receive_list );
  // secondly, expand the distribution tree locally
  expand_distribution_tree_locally( );
  // finally, expand the distribution tree communicatively and reduce it again
  // to a locally essential tree
  expand_distribution_tree_communicatively(
    subtree_send_list, subtree_receive_list );
  distribution_tree->reduce_2_essential( );

  collect_local_leaves( *_root );

  associate_scheduling_clusters_and_space_time_clusters( );
  // communicate necessary leaf information
  communicate_necessary_leaf_information(
    leaf_info_send_list, leaf_info_receive_list );

  std::vector< general_spacetime_cluster * > leaf_buffer;
  std::vector< general_spacetime_cluster * > non_leaf_buffer;
  sort_associated_space_time_clusters_recursively(
    distribution_tree->get_root( ), leaf_buffer, non_leaf_buffer );
  // clear the buffer vectors
  leaf_buffer.clear( );
  leaf_buffer.shrink_to_fit( );
  non_leaf_buffer.clear( );
  non_leaf_buffer.shrink_to_fit( );
  fill_nearfield_and_interaction_lists( *_root );
}

void besthea::mesh::distributed_spacetime_cluster_tree::build_tree(
  [[maybe_unused]] general_spacetime_cluster * pseudo_root ) {
  tree_structure * dist_tree = get_distribution_tree( );
  lo dist_tree_depth = dist_tree->get_levels( );
  lo dist_tree_depth_coll;

  MPI_Allreduce( &dist_tree_depth, &dist_tree_depth_coll, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );

  scheduling_time_cluster * temporal_root = dist_tree->get_root( );

  std::vector< lo > n_elems_per_subdivisioning;
  n_elems_per_subdivisioning.resize( 0 );
  std::vector<
    std::pair< general_spacetime_cluster *, scheduling_time_cluster * > >
    cluster_pairs;

  // create the space-time roots at level 0
  lo n_time_div = 0;
  lo n_space_div = _initial_space_refinement;

  // construct a vector which contains the time clusters of the distribution
  // tree of the next level which is considered in the tree construction.
  std::vector< scheduling_time_cluster * > time_clusters_on_level;
  time_clusters_on_level.push_back( dist_tree->get_root( ) );

  if ( _initial_space_refinement > 0 ) {
    // execute the initial spatial refinements
    get_n_elements_in_subdivisioning( *_root, n_space_div, 0,
      time_clusters_on_level, n_elems_per_subdivisioning );
    create_spacetime_roots( n_elems_per_subdivisioning, cluster_pairs );
  } else {
    // no initial spatial refinement is necessary. construct the root at level 0
    // directly (as copy of _root with different level)
    vector_type space_center( 3 );
    vector_type space_half_size( 3 );
    sc time_center, time_half_size;
    _root->get_center( space_center, time_center );
    _root->get_half_size( space_half_size, time_half_size );
    std::vector< slou > coordinates = { 0, 0, 0, 0, 0 };
    general_spacetime_cluster * spacetime_root = new general_spacetime_cluster(
      space_center, time_center, space_half_size, time_half_size,
      _spacetime_mesh.get_n_elements( ), _root, 0, 0, coordinates, 0, 0, 0, 0,
      _spacetime_mesh, _root->get_process_id( ), false );
    _root->add_child( spacetime_root );
    cluster_pairs.push_back(
      { spacetime_root, get_distribution_tree( )->get_root( ) } );
  }
  // remember whether space was split (value of split_space) for all levels
  // (to allow for a consistent local refinement in case of leaves at different
  // levels) and add the entry for level 0
  std::vector< bool > split_space_levelwise;
  split_space_levelwise.push_back( ( _initial_space_refinement > 0 ) );

  // construct clusters in the spacetime tree according to the distribution tree
  n_time_div += 1;
  bool split_space = ( _start_space_refinement <= 1 );
  if ( split_space ) {
    n_space_div += 1;
  }
  split_space_levelwise.push_back( split_space );

  // update time_clusters_on_level (to contain clusters at level 1)
  time_clusters_on_level = *( dist_tree->get_root( )->get_children( ) );

  // loop over levels of the clusters which are next to be constructed
  for ( lo child_level = 1; child_level < dist_tree_depth_coll;
        ++child_level ) {
    // get global number of elements per cluster
    get_n_elements_in_subdivisioning( *_root, n_space_div, child_level,
      time_clusters_on_level, n_elems_per_subdivisioning );
    split_clusters_levelwise( split_space, n_space_div, n_time_div,
      n_elems_per_subdivisioning, cluster_pairs );
    // replace time_cluster_on_level with the appropriate vector for the next
    // level
    std::vector< scheduling_time_cluster * > time_clusters_next_level;
    for ( auto time_cluster : time_clusters_on_level ) {
      if ( time_cluster->get_n_children( ) > 0 ) {
        for ( auto child_cluster : *time_cluster->get_children( ) ) {
          time_clusters_next_level.push_back( child_cluster );
        }
      }
    }
    time_clusters_on_level = std::move( time_clusters_next_level );

    n_time_div++;
    if ( !split_space && child_level + 1 >= _start_space_refinement ) {
      split_space = true;
      n_space_div++;
    } else {
      split_space = false;
    }
    split_space_levelwise.push_back( split_space );
  }

  std::vector< general_spacetime_cluster * > leaves;
  // collect the real leaves of the local spacetime cluster tree
  for ( auto spacetime_root : *_root->get_children( ) ) {
    collect_real_leaves( *spacetime_root, *temporal_root, leaves );
  }

  // const spacetime_tensor_mesh * current_mesh
  //  = _spacetime_mesh.get_local_mesh( );
  for ( auto it : leaves ) {
    it->reserve_elements( it->get_n_elements( ) );
  }

  //  std::cout << "Inserting local elements" << std::endl;
  //  vector_type space_center;
  //  space_center.resize( 3 );
  //  vector_type half_size;
  //  half_size.resize( 3 );
  //  linear_algebra::coordinates< 4 > centroid;
  //  std::vector< sc > boundary( 8 );
  //  besthea::tools::timer t;
  //  t.reset( "Filling recursively" );
  //  for ( lo i = 0; i < current_mesh->get_n_elements( ); ++i ) {
  //    insert_local_element(
  //      i, *_root, space_center, half_size, centroid, boundary );
  //  }
  //  t.measure( );
  //  std::cout << "Done" << std::endl;

  for ( auto it : leaves ) {
    // @todo Discuss: Inefficient way of filling in the elements? For each
    // leaf cluster the whole mesh is traversed once. If the depth of the tree
    // is reasonably high this takes a while!
    fill_elements( *it );

    build_subtree( *it, split_space_levelwise[ it->get_level( ) + 1 ] );
  }

  // exchange necessary data
  MPI_Allreduce( MPI_IN_PLACE, &_real_max_levels, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );

  if ( _real_max_levels < get_distribution_tree( )->get_levels( ) ) {
    std::cout << "Warning: Depth of local spacetime tree is less than depth of"
              << " local distribution tree!" << std::endl;
    assert( _real_max_levels >= get_distribution_tree( )->get_levels( ) );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  expand_distribution_tree_locally( ) {
  std::unordered_map< lo, bool > refine_map;
  tree_structure * distribution_tree = get_distribution_tree( );
  scheduling_time_cluster * time_root = distribution_tree->get_root( );
  distribution_tree->determine_clusters_to_refine( time_root, refine_map );
  if ( _root != nullptr ) {
    // expand the tree structure according to the spacetime tree, by traversing
    // the distribution tree and the spacetime tree (for each spacetime root)
    for ( auto spacetime_root : *( _root->get_children( ) ) ) {
      expand_tree_structure_recursively(
        distribution_tree, spacetime_root, time_root, refine_map );
    }
    // clear the nearfield, interaction and send list of each cluster and fill
    // them anew, to guarantee correctness.
    distribution_tree->clear_cluster_lists( time_root );
    distribution_tree->set_nearfield_interaction_and_send_list( *time_root );
    // determine activity of clusters in upward and downward path of FMM anew
    distribution_tree->determine_cluster_activity( *time_root );
    // reduce the tree to make it essential again
  } else {
    std::cout << "Error: Corrupted spacetime tree" << std::endl;
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  expand_distribution_tree_communicatively(
    const std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      subtree_send_list,
    const std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      subtree_receive_list ) {
  tree_structure * distribution_tree = get_distribution_tree( );
  // first communicate the maximal depth of the distribution tree.
  lo global_tree_levels = distribution_tree->get_levels( );
  MPI_Allreduce( MPI_IN_PLACE, &global_tree_levels, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );
  // the sets are sorted by default lexicographically, i.e. first in ascending
  // order with respect to the process ids. the code relies on that.
  lo max_offset = 0;
  if ( subtree_send_list.size( ) > 0 ) {
    max_offset = _my_rank - subtree_send_list.begin( )->first;
  }
  if ( subtree_receive_list.size( ) > 0 ) {
    max_offset = std::max(
      max_offset, subtree_receive_list.rbegin( )->first - _my_rank );
  }
  // execute the send and receive operations offsetwise
  auto send_list_it = subtree_send_list.rbegin( );
  auto receive_list_it = subtree_receive_list.begin( );
  for ( lo offset = 1; offset <= max_offset; ++offset ) {
    // depending on the rank decide whether to send or receive first
    if ( ( _my_rank / offset ) % 2 == 1 ) {
      // send first, receive later
      std::vector< scheduling_time_cluster * > current_send_clusters;
      while ( ( send_list_it != subtree_send_list.rend( ) )
        && ( _my_rank - send_list_it->first == offset ) ) {
        current_send_clusters.push_back( send_list_it->second );
        ++send_list_it;
      }
      send_subtree_data_of_distribution_tree(
        current_send_clusters, global_tree_levels, offset );
      // now receive
      std::vector< scheduling_time_cluster * > current_receive_clusters;
      while ( ( receive_list_it != subtree_receive_list.end( ) )
        && ( receive_list_it->first - _my_rank == offset ) ) {
        current_receive_clusters.push_back( receive_list_it->second );
        ++receive_list_it;
      }
      receive_subtree_data_of_distribution_tree(
        current_receive_clusters, global_tree_levels, offset );
    } else {
      // receive first
      std::vector< scheduling_time_cluster * > current_receive_clusters;
      while ( ( receive_list_it != subtree_receive_list.end( ) )
        && ( receive_list_it->first - _my_rank == offset ) ) {
        current_receive_clusters.push_back( receive_list_it->second );
        ++receive_list_it;
      }
      receive_subtree_data_of_distribution_tree(
        current_receive_clusters, global_tree_levels, offset );
      // now send
      std::vector< scheduling_time_cluster * > current_send_clusters;
      while ( ( send_list_it != subtree_send_list.rend( ) )
        && ( _my_rank - send_list_it->first == offset ) ) {
        current_send_clusters.push_back( send_list_it->second );
        ++send_list_it;
      }
      send_subtree_data_of_distribution_tree(
        current_send_clusters, global_tree_levels, offset );
    }
  }
  // clear the nearfield, interaction and send list of each cluster and fill
  // them anew, to guarantee correctness.
  distribution_tree->clear_cluster_lists( distribution_tree->get_root( ) );
  distribution_tree->set_nearfield_interaction_and_send_list(
    *distribution_tree->get_root( ) );
  // determine activity of clusters in upward and downward path of FMM anew
  distribution_tree->determine_cluster_activity(
    *distribution_tree->get_root( ) );
  // reduce the tree to make it essential again
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  communicate_necessary_leaf_information(
    const std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      leaf_info_send_list,
    const std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      leaf_info_receive_list ) {
  // the sets are sorted by default lexicographically, i.e. first in ascending
  // order with respect to the process ids. the code relies on that.
  lo max_offset = 0;
  if ( leaf_info_send_list.size( ) > 0 ) {
    max_offset = leaf_info_send_list.rbegin( )->first - _my_rank;
  }
  if ( leaf_info_receive_list.size( ) > 0 ) {
    max_offset = std::max(
      max_offset, _my_rank - leaf_info_receive_list.begin( )->first );
  }
  // execute the send and receive operations offsetwise
  auto send_list_it = leaf_info_send_list.begin( );
  auto receive_list_it = leaf_info_receive_list.rbegin( );
  for ( lo offset = 1; offset <= max_offset; ++offset ) {
    // depending on the rank decide whether to send or receive first
    if ( ( _my_rank / offset ) % 2 == 1 ) {
      // send first, receive later
      std::vector< scheduling_time_cluster * > current_send_clusters;
      while ( ( send_list_it != leaf_info_send_list.end( ) )
        && ( send_list_it->first - _my_rank == offset ) ) {
        current_send_clusters.push_back( send_list_it->second );
        ++send_list_it;
      }
      send_leaf_info( current_send_clusters, offset );
      // now receive
      std::vector< scheduling_time_cluster * > current_receive_clusters;
      while ( ( receive_list_it != leaf_info_receive_list.rend( ) )
        && ( _my_rank - receive_list_it->first == offset ) ) {
        current_receive_clusters.push_back( receive_list_it->second );
        ++receive_list_it;
      }
      receive_leaf_info( current_receive_clusters, offset );
    } else {
      // receive first
      std::vector< scheduling_time_cluster * > current_receive_clusters;
      while ( ( receive_list_it != leaf_info_receive_list.rend( ) )
        && ( _my_rank - receive_list_it->first == offset ) ) {
        current_receive_clusters.push_back( receive_list_it->second );
        ++receive_list_it;
      }
      receive_leaf_info( current_receive_clusters, offset );
      // now send
      std::vector< scheduling_time_cluster * > current_send_clusters;
      while ( ( send_list_it != leaf_info_send_list.end( ) )
        && ( send_list_it->first - _my_rank == offset ) ) {
        current_send_clusters.push_back( send_list_it->second );
        ++send_list_it;
      }
      send_leaf_info( current_send_clusters, offset );
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  expand_tree_structure_recursively( tree_structure * distribution_tree,
    general_spacetime_cluster * spacetime_root,
    scheduling_time_cluster * time_root,
    std::unordered_map< lo, bool > & refine_map ) {
  // if the current time cluster is a leaf handled by the process _my_process_id
  // and the current space-time cluster is not a leaf expand the temporal tree
  // structure
  if ( time_root->get_n_children( ) == 0
    && refine_map[ time_root->get_global_index( ) ]
    && spacetime_root->get_n_children( ) > 0 ) {
    std::vector< general_spacetime_cluster * > * spacetime_children
      = spacetime_root->get_children( );
    lo level_parent = time_root->get_level( );
    // determine whether the left and right children have to be added
    scheduling_time_cluster * left_cluster = nullptr;
    scheduling_time_cluster * right_cluster = nullptr;
    char child_count = 0;
    auto st_it = spacetime_children->begin( );
    // consider the temporal components of all space-time children and
    // create a new scheduling time cluster if a new one is encountered.
    while ( child_count < 2 && st_it != spacetime_children->end( ) ) {
      slou child_configuration = ( *st_it )->get_temporal_configuration( );
      if ( child_configuration == 0 && left_cluster == nullptr ) {
        // construct left cluster and set its process id and global index
        sc center_t_child = ( *st_it )->get_time_center( );
        sc half_size_child = ( *st_it )->get_time_half_size( );
        left_cluster
          = new scheduling_time_cluster( center_t_child, half_size_child,
            time_root, 0, level_parent + 1, time_root->get_process_id( ) );
        left_cluster->set_index( 2 * time_root->get_global_index( ) + 1 );
        refine_map[ 2 * time_root->get_global_index( ) + 1 ] = true;
        child_count += 1;
      } else if ( child_configuration == 1 && right_cluster == nullptr ) {
        // construct right cluster and set its process id and global index
        sc center_t_child = ( *st_it )->get_time_center( );
        sc half_size_child = ( *st_it )->get_time_half_size( );
        right_cluster
          = new scheduling_time_cluster( center_t_child, half_size_child,
            time_root, 1, level_parent + 1, time_root->get_process_id( ) );
        right_cluster->set_index( 2 * time_root->get_global_index( ) + 2 );
        refine_map[ 2 * time_root->get_global_index( ) + 2 ] = true;
        child_count += 1;
      }
      ++st_it;
    }
    // add the new children to the temporal cluster and update leaf status and
    // mesh availability
    time_root->set_n_children( child_count );
    if ( left_cluster != nullptr ) {
      time_root->add_child( left_cluster );
      left_cluster->set_global_leaf_status( true );
      left_cluster->set_mesh_availability( true );
    }
    if ( right_cluster != nullptr ) {
      time_root->add_child( right_cluster );
      right_cluster->set_global_leaf_status( true );
      right_cluster->set_mesh_availability( true );
    }
    // since time_root isn't a leaf anymore reset its global leaf status and
    // mesh availability to false (both is assumed to be false for
    // non-leaves).
    time_root->set_global_leaf_status( false );
    time_root->set_mesh_availability( false );
    // update the member _levels of the distribution tree, if it has
    // increased.
    if ( level_parent + 1 == distribution_tree->get_levels( ) ) {
      distribution_tree->set_levels( distribution_tree->get_levels( ) + 1 );
    }
    // remove the entry of time_root from the refine_map
    refine_map.erase( time_root->get_global_index( ) );
  }
  // call the routine recursively for non-leaf clusters (including the current
  // cluster if it has become a non-leaf in the previous step)
  if ( time_root->get_n_children( ) > 0 ) {
    if ( spacetime_root->get_n_children( ) > 0 ) {
      std::vector< scheduling_time_cluster * > * time_children
        = time_root->get_children( );
      std::vector< general_spacetime_cluster * > * spacetime_children
        = spacetime_root->get_children( );
      for ( auto time_child : *time_children ) {
        short time_child_configuration = time_child->get_configuration( );
        for ( auto spacetime_child : *spacetime_children ) {
          short spacetime_child_configuration
            = spacetime_child->get_temporal_configuration( );
          // check if the temporal component of the spacetime child is the
          // same as the current temporal child and call routine recursively
          // if yes
          if ( time_child_configuration == spacetime_child_configuration ) {
            expand_tree_structure_recursively(
              distribution_tree, spacetime_child, time_child, refine_map );
          }
        }
      }
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  get_n_elements_in_subdivisioning( general_spacetime_cluster & root,
    lo n_space_div, lo level_time,
    const std::vector< scheduling_time_cluster * > & time_clusters_on_level,
    std::vector< lo > & elems_in_clusters ) {
  lo n_space_clusters = 1;
  lo n_time_clusters
    = 1 << level_time;  // upper estimate on the number of levels
  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }
  lo n_clusters
    = n_space_clusters * n_space_clusters * n_space_clusters * n_time_clusters;

  elems_in_clusters.resize( n_clusters );
  std::vector< lo > loc_elems_in_clusters( n_clusters, 0 );
  lo pos_x, pos_y, pos_z, pos_t;

  vector_type space_center;
  space_center.resize( 3 );
  vector_type half_size;
  half_size.resize( 3 );
  sc time_center, time_half_size;
  root.get_center( space_center, time_center );
  root.get_half_size( half_size, time_half_size );

  // doing it this complicated to avoid inconsistencies with tree
  // assembly due to rounding errors
  std::vector< sc > steps_x( 0 );
  steps_x.reserve( n_space_clusters );
  std::vector< sc > steps_y( 0 );
  steps_y.reserve( n_space_clusters );
  std::vector< sc > steps_z( 0 );
  steps_z.reserve( n_space_clusters );

  decompose_line( space_center[ 0 ], half_size[ 0 ],
    space_center[ 0 ] - half_size[ 0 ], n_space_div, 0, steps_x );
  decompose_line( space_center[ 1 ], half_size[ 1 ],
    space_center[ 1 ] - half_size[ 1 ], n_space_div, 0, steps_y );
  decompose_line( space_center[ 2 ], half_size[ 2 ],
    space_center[ 2 ] - half_size[ 2 ], n_space_div, 0, steps_z );

  steps_x.push_back( space_center[ 0 ] + half_size[ 0 ] + 1.0 );
  steps_y.push_back( space_center[ 1 ] + half_size[ 1 ] + 1.0 );
  steps_z.push_back( space_center[ 2 ] + half_size[ 2 ] + 1.0 );

  // assign slices to tree nodes
  std::vector< sc > starts(
    n_time_clusters, std::numeric_limits< sc >::infinity( ) );
  std::vector< sc > ends(
    n_time_clusters, -std::numeric_limits< sc >::infinity( ) );
  // compute the level conversion factor to compute local levelwise cluster
  // indices from global cluster indices.
  lo level_conversion_factor = ( 1 << level_time ) - 1;
  for ( lou i = 0; i < time_clusters_on_level.size( ); ++i ) {
    sc center_time = time_clusters_on_level[ i ]->get_center( );
    sc half_size_time = time_clusters_on_level[ i ]->get_half_size( );
    lo access_index = time_clusters_on_level[ i ]->get_global_index( )
      - level_conversion_factor;
    starts[ access_index ] = center_time - half_size_time;
    ends[ access_index ] = center_time + half_size_time;
  }

  sc delta_x = ( 2 * half_size[ 0 ] ) / n_space_clusters;
  sc delta_y = ( 2 * half_size[ 1 ] ) / n_space_clusters;
  sc delta_z = ( 2 * half_size[ 2 ] ) / n_space_clusters;
  lo start, end;

#pragma omp parallel private( pos_x, pos_y, pos_z, pos_t, start, end ) \
  reduction( lo_vec_plus                                               \
             : loc_elems_in_clusters )
  {
    linear_algebra::coordinates< 4 > centroid;
#pragma omp for schedule( static )
    for ( lo i = 0; i < _spacetime_mesh.get_local_mesh( )->get_n_elements( );
          ++i ) {
      _spacetime_mesh.get_local_mesh( )->get_centroid( i, centroid );
      pos_x
        = ( centroid[ 0 ] - ( space_center[ 0 ] - half_size[ 0 ] ) ) / delta_x;
      pos_y
        = ( centroid[ 1 ] - ( space_center[ 1 ] - half_size[ 1 ] ) ) / delta_y;
      pos_z
        = ( centroid[ 2 ] - ( space_center[ 2 ] - half_size[ 2 ] ) ) / delta_z;

      pos_t = -1;
      for ( lo j = 0; j < n_time_clusters; ++j ) {
        if ( centroid[ 3 ] > starts[ j ] && centroid[ 3 ] <= ends[ j ] ) {
          pos_t = j;
          break;
        }
      }

      start = pos_x > 0 ? pos_x - 1 : pos_x;
      end
        = pos_x < static_cast< lo >( steps_x.size( ) ) - 2 ? pos_x + 1 : pos_x;
      for ( lo j = start; j <= end; ++j ) {
        if ( ( centroid[ 0 ] >= steps_x[ j ] )
          && ( centroid[ 0 ] < steps_x[ j + 1 ] ) ) {
          pos_x = j;
          break;
        }
      }

      start = pos_y > 0 ? pos_y - 1 : pos_y;
      end
        = pos_y < static_cast< lo >( steps_y.size( ) ) - 2 ? pos_y + 1 : pos_y;
      for ( lo j = start; j <= end; ++j ) {
        if ( ( centroid[ 1 ] >= steps_y[ j ] )
          && ( centroid[ 1 ] < steps_y[ j + 1 ] ) ) {
          pos_y = j;
          break;
        }
      }
      start = pos_z > 0 ? pos_z - 1 : pos_z;
      end
        = pos_z < static_cast< lo >( steps_z.size( ) ) - 2 ? pos_z + 1 : pos_z;
      for ( lo j = start; j <= end; ++j ) {
        if ( ( centroid[ 2 ] >= steps_z[ j ] )
          && ( centroid[ 2 ] < steps_z[ j + 1 ] ) ) {
          pos_z = j;
          break;
        }
      }
      // if the element was found in a time cluster update the counter
      if ( pos_t > -1 ) {
        lo pos = pos_t * n_space_clusters * n_space_clusters * n_space_clusters
          + pos_x * n_space_clusters * n_space_clusters
          + pos_y * n_space_clusters + pos_z;
        loc_elems_in_clusters.at( pos )++;
      }
    }
  }

  MPI_Allreduce( loc_elems_in_clusters.data( ), elems_in_clusters.data( ),
    loc_elems_in_clusters.size( ), get_index_type< lo >::MPI_LO( ), MPI_SUM,
    *_comm );
}

void besthea::mesh::distributed_spacetime_cluster_tree::decompose_line(
  sc center, sc half_size, sc left_bound, lo n_ref, lo curr_level,
  std::vector< sc > & steps ) {
  if ( curr_level == n_ref - 1 ) {
    steps.push_back( left_bound );
    steps.push_back( center );
    return;
  } else if ( n_ref == 0 ) {
    steps.push_back( left_bound );
    return;
  }

  decompose_line( center - half_size / 2.0, half_size / 2.0, left_bound, n_ref,
    curr_level + 1, steps );
  decompose_line( center + half_size / 2.0, half_size / 2.0, center, n_ref,
    curr_level + 1, steps );
}

void besthea::mesh::distributed_spacetime_cluster_tree::create_spacetime_roots(
  std::vector< lo > & elems_in_clusters,
  std::vector< std::pair< general_spacetime_cluster *,
    scheduling_time_cluster * > > & spacetime_root_pairs ) {
  if ( _root->get_n_elements( ) >= _n_min_elems ) {
    scheduling_time_cluster * time_root = get_distribution_tree( )->get_root( );
    // compute number of space and time clusters at the level of children
    lo n_space_clusters = 1 << _initial_space_refinement;

    // reserve a fair amount of entries
    spacetime_root_pairs.reserve(
      n_space_clusters * n_space_clusters * n_space_clusters );

    lo owner = time_root->get_process_id( );
    lo global_time_index = 0;

    sc time_center, time_half_size;
    vector_type spat_center( 3 );
    vector_type spat_half_size( 3 );
    _root->get_center( spat_center, time_center );
    _root->get_half_size( spat_half_size, time_half_size );
    // compute the new spatial half size, spatial step size to get from one
    // cluster center to the other and the center of the bottom front left
    // spatial cluster, which is formed during the refinement.
    vector_type new_spat_half_size = { spat_half_size[ 0 ] / n_space_clusters,
      spat_half_size[ 1 ] / n_space_clusters,
      spat_half_size[ 2 ] / n_space_clusters };
    vector_type spat_step_size = { 2 * spat_half_size[ 0 ] / n_space_clusters,
      2 * spat_half_size[ 1 ] / n_space_clusters,
      2 * spat_half_size[ 2 ] / n_space_clusters };
    vector_type spat_center_corner
      = { spat_center[ 0 ] - spat_half_size[ 0 ] + new_spat_half_size[ 0 ],
          spat_center[ 1 ] - spat_half_size[ 1 ] + new_spat_half_size[ 1 ],
          spat_center[ 2 ] - spat_half_size[ 2 ] + new_spat_half_size[ 2 ] };
    vector_type new_spat_center( 3 );
    for ( slou i_x = 0; i_x < n_space_clusters; ++i_x ) {
      new_spat_center[ 0 ]
        = spat_center_corner[ 0 ] + i_x * spat_step_size[ 0 ];
      for ( slou i_y = 0; i_y < n_space_clusters; ++i_y ) {
        new_spat_center[ 1 ]
          = spat_center_corner[ 1 ] + i_y * spat_step_size[ 1 ];
        for ( slou i_z = 0; i_z < n_space_clusters; ++i_z ) {
          new_spat_center[ 2 ]
            = spat_center_corner[ 2 ] + i_z * spat_step_size[ 2 ];
          // construct coordinates of cluster (level and temporal coordinate
          // 0)
          std::vector< slou > coordinates = { 0, i_x, i_y, i_z, 0 };
          lou pos = i_x * n_space_clusters * n_space_clusters
            + i_y * n_space_clusters + i_z;

          if ( elems_in_clusters[ pos ] > 0 ) {
            // construct spacetime root cluster if its not empty. octant and
            // left_right are meaningless, and set to 0.
            general_spacetime_cluster * new_child
              = new general_spacetime_cluster( new_spat_center, time_center,
                new_spat_half_size, time_half_size, elems_in_clusters[ pos ],
                _root, 0, 0, coordinates, 0, global_time_index,
                _initial_space_refinement, 0, _spacetime_mesh, owner, false );
            // add the spacetime root as child to the "pseudo"root at level -1
            _root->add_child( new_child );
            spacetime_root_pairs.push_back( { new_child, time_root } );
          }
        }
      }
    }
    // shrink the spacetime root pairs to the appropriate size
    spacetime_root_pairs.shrink_to_fit( );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  split_clusters_levelwise( bool split_space, lo n_space_div, lo n_time_div,
    std::vector< lo > & elems_in_clusters,
    std::vector< std::pair< general_spacetime_cluster *,
      scheduling_time_cluster * > > & cluster_pairs ) {
  // compute number of space clusters at the level of children
  lo n_space_clusters = 1;
  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }

  // vector to store the pairs of children which are constructed below.
  std::vector<
    std::pair< general_spacetime_cluster *, scheduling_time_cluster * > >
    new_cluster_pairs;
  // reserve a fair amount of entries
  // (assuming <~ 5 children in time, 2 in space due to surface mesh)
  new_cluster_pairs.reserve( cluster_pairs.size( ) * 10 );

  // refine all space-time clusters whose children are locally essential
  for ( lou i = 0; i < cluster_pairs.size( ); ++i ) {
    // get current spacetime cluster and time cluster
    general_spacetime_cluster * st_cluster = cluster_pairs[ i ].first;
    const std::vector< slou > parent_coord = st_cluster->get_box_coordinate( );
    scheduling_time_cluster * t_cluster = cluster_pairs[ i ].second;

    // split the cluster only if it contains enough elements and the temporal
    // component is a non-leaf
    if ( st_cluster->get_n_elements( ) >= _n_min_elems
      && t_cluster->get_n_children( ) > 0 ) {
      std::vector< scheduling_time_cluster * > * t_children
        = t_cluster->get_children( );
      for ( auto t_child : *t_children ) {
        // check if temporal child is locally essential with respect to the
        // space-time cluster tree.
        if ( t_child->get_essential_status( ) > 1 ) {
          lo owner = t_child->get_process_id( );
          // set coord_t and left_right appropriately distinguishing the left
          // and right children.
          short left_right = t_child->get_configuration( );
          lo global_time_index = t_child->get_global_index( );
          slou coord_t;
          if ( left_right == 0 ) {
            coord_t = ( slou )( 2 * parent_coord[ 4 ] );  // left child
          } else {
            coord_t = ( slou )( 2 * parent_coord[ 4 ] + 1 );  // right child
          }
          // compute the time index on the current level (n_time_div) by
          // substracting the correct conversion term.
          lo time_index_on_level = global_time_index - ( 1 << n_time_div ) + 1;

          sc new_time_center = t_child->get_center( );
          sc new_time_half_size = t_child->get_half_size( );
          if ( split_space ) {
            vector_type new_spat_center( 3 );
            vector_type new_spat_half_size( 3 );
            for ( short j = 0; j < 8; ++j ) {
              st_cluster->compute_spatial_suboctant(
                j, new_spat_center, new_spat_half_size );

              slou coord_x = 2 * st_cluster->get_box_coordinate( )[ 1 ]
                + _idx_2_coord[ j ][ 0 ];
              slou coord_y = 2 * st_cluster->get_box_coordinate( )[ 2 ]
                + _idx_2_coord[ j ][ 1 ];
              slou coord_z = 2 * st_cluster->get_box_coordinate( )[ 3 ]
                + _idx_2_coord[ j ][ 2 ];
              std::vector< slou > coordinates
                = { static_cast< slou >( st_cluster->get_level( ) + 1 ),
                    coord_x, coord_y, coord_z, coord_t };

              lou pos = time_index_on_level * n_space_clusters
                  * n_space_clusters * n_space_clusters
                + coord_x * n_space_clusters * n_space_clusters
                + coord_y * n_space_clusters + coord_z;

              if ( elems_in_clusters[ pos ] > 0 ) {
                general_spacetime_cluster * new_child
                  = new general_spacetime_cluster( new_spat_center,
                    new_time_center, new_spat_half_size, new_time_half_size,
                    elems_in_clusters[ pos ], st_cluster,
                    st_cluster->get_level( ) + 1, j, coordinates, left_right,
                    global_time_index, n_space_div, n_time_div, _spacetime_mesh,
                    owner, false );
                st_cluster->add_child( new_child );
                new_cluster_pairs.push_back( { new_child, t_child } );
              }
            }
          } else {
            slou coord_x = parent_coord[ 1 ];
            slou coord_y = parent_coord[ 2 ];
            slou coord_z = parent_coord[ 3 ];
            std::vector< slou > coordinates
              = { static_cast< slou >( st_cluster->get_level( ) + 1 ), coord_x,
                  coord_y, coord_z, coord_t };
            lou pos = time_index_on_level * n_space_clusters * n_space_clusters
                * n_space_clusters
              + coord_x * n_space_clusters * n_space_clusters
              + coord_y * n_space_clusters + coord_z;
            // get the spatial center and half size from the parent
            vector_type space_center( 3 );
            vector_type space_half_size( 3 );
            sc dummy_var;
            st_cluster->get_center( space_center, dummy_var );
            st_cluster->get_half_size( space_half_size, dummy_var );

            if ( elems_in_clusters[ pos ] > 0 ) {
              general_spacetime_cluster * new_child
                = new general_spacetime_cluster( space_center, new_time_center,
                  space_half_size, new_time_half_size, elems_in_clusters[ pos ],
                  st_cluster, st_cluster->get_level( ) + 1,
                  st_cluster->get_spatial_octant( ), coordinates, left_right,
                  global_time_index, n_space_div, n_time_div, _spacetime_mesh,
                  owner, false );
              st_cluster->add_child( new_child );
              new_cluster_pairs.push_back( { new_child, t_child } );
            }
          }
        }
      }
    } else if ( st_cluster->get_n_elements( ) < _n_min_elems ) {
      // mark st_cluster as a global leaf in the distributed tree.
      st_cluster->set_global_leaf_status( true );
    }
  }
  // replace the old vector of cluster pairs by the one which was newly
  // constructed
  new_cluster_pairs.shrink_to_fit( );
  cluster_pairs = std::move( new_cluster_pairs );
}

void besthea::mesh::distributed_spacetime_cluster_tree::compute_bounding_box(
  sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax ) {
  // only local computation since spatial mesh is now duplicated
  xmin = ymin = zmin = std::numeric_limits< sc >::max( );
  xmax = ymax = zmax = std::numeric_limits< sc >::min( );

  linear_algebra::coordinates< 4 > node;
  for ( lo i = 0; i < _spacetime_mesh.get_local_mesh( )->get_n_nodes( ); ++i ) {
    _spacetime_mesh.get_local_mesh( )->get_node( i, node );

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

void besthea::mesh::distributed_spacetime_cluster_tree::collect_local_leaves(
  general_spacetime_cluster & root ) {
  std::vector< general_spacetime_cluster * > * children = root.get_children( );
  if ( children != nullptr ) {
    for ( auto it : *children ) {
      collect_local_leaves( *it );
    }
  } else if ( _my_rank == root.get_process_id( ) ) {
    _local_leaves.push_back( &root );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::collect_real_leaves(
  general_spacetime_cluster & st_root, scheduling_time_cluster & t_root,
  std::vector< general_spacetime_cluster * > & leaves ) {
  std::vector< scheduling_time_cluster * > * t_children
    = t_root.get_children( );
  if ( t_children != nullptr ) {
    std::vector< general_spacetime_cluster * > * st_children
      = st_root.get_children( );
    if ( st_children != nullptr ) {
      lou t_idx = 0;
      scheduling_time_cluster * t_child = ( *t_children )[ t_idx ];
      short t_child_configuration = t_child->get_configuration( );
      for ( lou st_idx = 0; st_idx < st_children->size( ); ++st_idx ) {
        general_spacetime_cluster * st_child = ( *st_children )[ st_idx ];
        // check whether the temporal component of the st_child is the t_child
        // and if not update the t_child
        short st_child_configuration = st_child->get_temporal_configuration( );
        if ( t_child_configuration != st_child_configuration ) {
          ++t_idx;
          t_child = ( *t_children )[ t_idx ];
          t_child_configuration = t_child->get_configuration( );
        }
        // call the routine recursively for the appropriate pair of spacetime
        // cluster and scheduling time cluster
        collect_real_leaves( *st_child, *t_child, leaves );
      }
    }
    // else if ( st_root.get_n_elements( ) < _n_min_elems ) {
    else if ( st_root.is_global_leaf( ) ) {
      leaves.push_back( &st_root );
    }
  }
  // if t_root is a leaf in the global tree structure, the corresponding
  // space-time clusters are leaves and have to be refined if their meshes are
  // available. Clusters whose mesh is not available are not added to the vector
  // leaves.
  else if ( t_root.is_global_leaf( ) && t_root.mesh_is_available( ) ) {
    leaves.push_back( &st_root );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::fill_elements(
  general_spacetime_cluster & cluster ) {
  assert( cluster.get_level( ) > 1 );
  lo n_space_div, n_time_div;
  cluster.get_n_divs( n_space_div, n_time_div );
  lo n_space_clusters = 1;
  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }

  const std::vector< slou > coord = cluster.get_box_coordinate( );
  vector_type space_center;
  space_center.resize( 3 );
  vector_type half_size;
  half_size.resize( 3 );
  sc time_center, time_half_size;
  cluster.get_center( space_center, time_center );
  cluster.get_half_size( half_size, time_half_size );

  sc left = space_center[ 0 ] - half_size[ 0 ];
  sc right = space_center[ 0 ] + half_size[ 0 ];
  sc front = space_center[ 1 ] - half_size[ 1 ];
  sc back = space_center[ 1 ] + half_size[ 1 ];
  sc bottom = space_center[ 2 ] - half_size[ 2 ];
  sc top = space_center[ 2 ] + half_size[ 2 ];
  sc beginning = time_center - time_half_size;
  sc end = time_center + time_half_size;

  if ( coord[ 1 ] == n_space_clusters - 1 ) {
    right += 1.0;
  }
  if ( coord[ 2 ] == n_space_clusters - 1 ) {
    back += 1.0;
  }

  if ( coord[ 3 ] == n_space_clusters - 1 ) {
    top += 1.0;
  }
  if ( coord[ 4 ] == 0 ) {
    beginning -= 1.0;
  }

  cluster.reserve_elements( cluster.get_n_elements( ) );

  const spacetime_tensor_mesh * current_mesh;
  lo start_idx;
  if ( cluster.get_process_id( ) == _my_rank ) {
    current_mesh = _spacetime_mesh.get_local_mesh( );
    start_idx = _spacetime_mesh.get_local_start_idx( );
    cluster.set_elements_are_local( true );
  } else {
    current_mesh = _spacetime_mesh.get_nearfield_mesh( );
    start_idx = _spacetime_mesh.get_nearfield_start_idx( );
    // elements are local is set to false by default -> not set here
  }

  // while adding elements to the cluster count the number of different
  // time elements
  // lo n_time_elements = 0;
  // sc last_timestep = beginning - time_half_size;

  std::vector< std::vector< lo > > elems_thread( omp_get_max_threads( ) );
  std::vector< std::vector< sc > > timesteps_thread( omp_get_max_threads( ) );
  std::vector< sc > max_thread( omp_get_max_threads( ) );

  for ( auto & it : max_thread ) {
    it = beginning - time_half_size;
  }

#pragma omp parallel
  {
    elems_thread[ omp_get_thread_num( ) ].reserve(
      cluster.get_n_elements( ) / omp_get_num_threads( ) );
    linear_algebra::coordinates< 4 > centroid;
#pragma omp for
    for ( lo i = 0; i < current_mesh->get_n_elements( ); ++i ) {
      current_mesh->get_centroid( i, centroid );
      if ( ( centroid[ 0 ] >= left ) && ( centroid[ 0 ] < right )
        && ( centroid[ 1 ] >= front ) && ( centroid[ 1 ] < back )
        && ( centroid[ 2 ] >= bottom ) && ( centroid[ 2 ] < top )
        && ( centroid[ 3 ] > beginning ) && ( centroid[ 3 ] <= end ) ) {
        elems_thread[ omp_get_thread_num( ) ].push_back(
          _spacetime_mesh.local_2_global( start_idx, i ) );
        // cluster.add_element( _spacetime_mesh.local_2_global( start_idx, i )
        // );
        // check if the temporal component of the element is a new timestep
        // (the check with > is safe, due to the computation of the centroid in
        // get_centroid. since the elements are sorted with respect to time the
        // right number of time elements is determined.)
        if ( timesteps_thread[ omp_get_thread_num( ) ].size( ) == 0 ) {
          timesteps_thread[ omp_get_thread_num( ) ].push_back( centroid[ 3 ] );
        }
        // if ( centroid[ 3 ]
        //    > timesteps_thread[ omp_get_thread_num( ) ].back( ) ) {
        if ( centroid[ 3 ] > max_thread[ omp_get_thread_num( ) ] ) {
          timesteps_thread[ omp_get_thread_num( ) ].push_back( centroid[ 3 ] );
          max_thread[ omp_get_thread_num( ) ] = centroid[ 3 ];
        }
      }
    }
  }
  std::set< sc > timesteps_union;
  for ( auto it : timesteps_thread ) {
    timesteps_union.insert( it.begin( ), it.end( ) );
  }

  for ( auto it : elems_thread ) {
    for ( auto it2 : it ) {
      cluster.add_element( it2 );
    }
  }
  cluster.set_n_time_elements( timesteps_union.size( ) );
  cluster.compute_node_mapping( );
  cluster.set_n_space_nodes( );
}

// void besthea::mesh::distributed_spacetime_cluster_tree::fill_elements2(
//  std::vector< general_spacetime_cluster * > & leaves,
//  spacetime_tensor_mesh const * current_mesh ) {
//  lo n_space_div, n_time_div;
//
//  linear_algebra::coordinates< 4 > centroid;
//  vector_type space_center;
//  space_center.resize( 3 );
//  vector_type half_size;
//  half_size.resize( 3 );
//  sc time_center, time_half_size;
//  lo start_idx = _spacetime_mesh.get_local_start_idx( );
//
//  for ( lo i = 0; i < current_mesh->get_n_elements( ); ++i ) {
//
//    current_mesh->get_centroid( i, centroid );
//
//    for ( auto it : leaves ) {
//      it->get_center( space_center, time_center );
//      it->get_half_size( half_size, time_half_size );
//
//      sc left = space_center[ 0 ] - half_size[ 0 ];
//      sc right = space_center[ 0 ] + half_size[ 0 ];
//      sc front = space_center[ 1 ] - half_size[ 1 ];
//      sc back = space_center[ 1 ] + half_size[ 1 ];
//      sc bottom = space_center[ 2 ] - half_size[ 2 ];
//      sc top = space_center[ 2 ] + half_size[ 2 ];
//      sc beginning = time_center - time_half_size;
//      sc end = time_center + time_half_size;
//
//      it->get_n_divs( n_space_div, n_time_div );
//      lo n_space_clusters = 1;
//      for ( lo i = 0; i < n_space_div; ++i ) {
//        n_space_clusters *= 2;
//      }
//
//      const std::vector< slou > coord = it->get_box_coordinate( );
//
//      if ( coord[ 1 ] == n_space_clusters - 1 ) {
//        right += 1.0;
//      }
//      if ( coord[ 2 ] == n_space_clusters - 1 ) {
//        back += 1.0;
//      }
//
//      if ( coord[ 3 ] == n_space_clusters - 1 ) {
//        top += 1.0;
//      }
//      if ( coord[ 4 ] == 0 ) {
//        beginning -= 1.0;
//      }
//
//      if ( ( centroid[ 0 ] >= left ) && ( centroid[ 0 ] < right )
//        && ( centroid[ 1 ] >= front ) && ( centroid[ 1 ] < back )
//        && ( centroid[ 2 ] >= bottom ) && ( centroid[ 2 ] < top )
//        && ( centroid[ 3 ] > beginning ) && ( centroid[ 3 ] <= end ) ) {
//        it->add_element( _spacetime_mesh.local_2_global( start_idx, i ) );
//        break;
//      }
//    }
//  }
//}
//
// void besthea::mesh::distributed_spacetime_cluster_tree::insert_local_element(
//  lo elem_idx, general_spacetime_cluster & root, vector_type & space_center,
//  vector_type & half_size, linear_algebra::coordinates< 4 > & centroid,
//  std::vector< sc > & boundary ) {
//  const spacetime_tensor_mesh * current_mesh
//    = _spacetime_mesh.get_local_mesh( );
//  current_mesh->get_centroid( elem_idx, centroid );
//
//  sc time_center, time_half_size;
//
//  lo n_space_div, n_time_div;
//  root.get_n_divs( n_space_div, n_time_div );
//  lo n_space_clusters = 1;
//  n_space_clusters << n_space_div;
//
//  if ( root.get_n_children( ) > 0 ) {
//    for ( auto it = root.get_children( )->begin( );
//          it != root.get_children( )->end( ); ++it ) {
//      const std::vector< slou > coord = ( *it )->get_box_coordinate( );
//
//      ( *it )->get_center( space_center, time_center );
//      ( *it )->get_half_size( half_size, time_half_size );
//
//      // sc left = space_center[ 0 ] - half_size[ 0 ];
//      boundary[ 0 ] = space_center[ 0 ] - half_size[ 0 ];
//      // sc right = space_center[ 0 ] + half_size[ 0 ];
//      boundary[ 1 ] = space_center[ 0 ] + half_size[ 0 ];
//      // sc front = space_center[ 1 ] - half_size[ 1 ];
//      boundary[ 2 ] = space_center[ 1 ] - half_size[ 1 ];
//      // sc back = space_center[ 1 ] + half_size[ 1 ];
//      boundary[ 3 ] = space_center[ 1 ] + half_size[ 1 ];
//      // sc bottom = space_center[ 2 ] - half_size[ 2 ];
//      boundary[ 4 ] = space_center[ 2 ] - half_size[ 2 ];
//      // sc top = space_center[ 2 ] + half_size[ 2 ];
//      boundary[ 5 ] = space_center[ 2 ] + half_size[ 2 ];
//      // sc beginning = time_center - time_half_size;
//      boundary[ 6 ] = time_center - time_half_size;
//      // sc end = time_center + time_half_size;
//      boundary[ 7 ] = time_center + time_half_size;
//
//      if ( coord[ 1 ] == n_space_clusters - 1 ) {
//        boundary[ 1 ] += 1.0;
//      }
//      if ( coord[ 2 ] == n_space_clusters - 1 ) {
//        boundary[ 3 ] += 1.0;
//      }
//
//      if ( coord[ 3 ] == n_space_clusters - 1 ) {
//        boundary[ 5 ] += 1.0;
//      }
//      if ( coord[ 4 ] == 0 ) {
//        boundary[ 6 ] -= 1.0;
//      }
//
//      if ( ( centroid[ 0 ] >= boundary[ 0 ] )
//        && ( centroid[ 0 ] < boundary[ 1 ] )
//        && ( centroid[ 1 ] >= boundary[ 2 ] )
//        && ( centroid[ 1 ] < boundary[ 3 ] )
//        && ( centroid[ 2 ] >= boundary[ 4 ] )
//        && ( centroid[ 2 ] < boundary[ 5 ] )
//        && ( centroid[ 3 ] > boundary[ 6 ] )
//        && ( centroid[ 3 ] <= boundary[ 7 ] ) ) {
//        insert_local_element(
//          elem_idx, **it, space_center, half_size, centroid, boundary );
//        break;
//      }
//      //
//    }
//  } else {
//    const spacetime_tensor_mesh * current_mesh;
//    lo start_idx;
//    if ( root.get_process_id( ) == _my_rank ) {
//      current_mesh = _spacetime_mesh.get_local_mesh( );
//      start_idx = _spacetime_mesh.get_local_start_idx( );
//      root.set_elements_are_local( true );
//    } else {
//      current_mesh = _spacetime_mesh.get_nearfield_mesh( );
//      start_idx = _spacetime_mesh.get_nearfield_start_idx( );
//      // elements are local is set to false by default -> not set here
//    }
//    root.add_element( _spacetime_mesh.local_2_global( start_idx, elem_idx ) );
//  }
//}

void besthea::mesh::distributed_spacetime_cluster_tree::build_subtree(
  general_spacetime_cluster & root, const bool split_space ) {
  if ( root.get_level( ) + 1 > _max_levels - 1
    || root.get_n_elements( ) < _n_min_elems
    || root.get_n_time_elements( ) == 1 ) {
    root.set_n_children( 0 );
    root.set_global_leaf_status( true );
    if ( root.get_level( ) + 1 > _real_max_levels ) {
      _real_max_levels = root.get_level( ) + 1;
    }
    lo n_space_div, n_time_div;
    root.get_n_divs( n_space_div, n_time_div );
    if ( n_space_div > _local_max_space_level ) {
      _local_max_space_level = n_space_div;
    }
    return;
  }

  const spacetime_tensor_mesh * current_mesh;
  lo start_idx;
  bool elements_are_local = root.get_elements_are_local( );
  if ( root.get_process_id( ) == _my_rank ) {
    current_mesh = _spacetime_mesh.get_local_mesh( );
    start_idx = _spacetime_mesh.get_local_start_idx( );
    elements_are_local = true;
  } else {
    current_mesh = _spacetime_mesh.get_nearfield_mesh( );
    start_idx = _spacetime_mesh.get_nearfield_start_idx( );
  }

  const std::vector< slou > parent_coord = root.get_box_coordinate( );

  vector_type space_center( 3 );
  vector_type space_half_size( 3 );
  sc time_center, time_half_size;
  root.get_center( space_center, time_center );
  root.get_half_size( space_half_size, time_half_size );

  lo n_space_div, n_time_div;
  root.get_n_divs( n_space_div, n_time_div );

  linear_algebra::coordinates< 4 > el_centroid;
  lo elem_idx = 0;
  lo oct_sizes[ 16 ];
  for ( lo i = 0; i < 16; ++i ) {
    oct_sizes[ i ] = 0;
  }

  general_spacetime_cluster * clusters[ 16 ];

  vector_type new_space_center( 3 );
  vector_type new_space_half_size( 3 );

  // decide whether to split space when refining the descendant
  bool split_space_descendant = false;
  if ( !split_space && root.get_level( ) + 2 >= _start_space_refinement ) {
    // root.get_level( ) + 2 is the correct level since the decision is always
    // made from the perspective of the children
    split_space_descendant = true;
  }

  slou coord_x, coord_y, coord_z, coord_t = 0;
  sc temporal_splitting_point = -std::numeric_limits< sc >::infinity( );
  bool set_temporal_splitting_point = false;
  lo first_time_element = current_mesh->get_time_element(
    _spacetime_mesh.global_2_local( start_idx, root.get_element( 0 ) ) );
  lo last_time_element
    = current_mesh->get_time_element( _spacetime_mesh.global_2_local(
      start_idx, root.get_element( root.get_n_elements( ) - 1 ) ) );
  lo n_time_elements_left = 0;
  lo n_time_elements_right = 0;

  if ( split_space ) {
    for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
      elem_idx
        = _spacetime_mesh.global_2_local( start_idx, root.get_element( i ) );
      current_mesh->get_centroid( elem_idx, el_centroid );

      if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 0 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 1 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 2 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 3 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 4 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 5 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 6 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 7 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 8 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 9 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 10 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 11 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 12 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 13 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 14 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 15 ];
      }
      // check if the temporal splitting point has been set or has to be set
      // NOTE: this check is only reasonable for pure spacetime tensor meshes
      // (the intersection of the temporal intervals of two elements is either
      // empty, a point, or the whole time interval)
      if ( !set_temporal_splitting_point ) {
        sc temp_node_1, temp_node_2;
        lo current_time_element = current_mesh->get_time_element( elem_idx );
        current_mesh->get_temporal_nodes(
          current_time_element, &temp_node_1, &temp_node_2 );
        if ( ( el_centroid[ 3 ] < time_center )
          && ( temp_node_2 >= time_center ) ) {
          // the element is assigned to the left half, but its right bound
          // is greater or equal than the center of the cluster.
          set_temporal_splitting_point = true;
          temporal_splitting_point = temp_node_2;
          // set the number of time elements in the left and right clusters
          n_time_elements_left = current_time_element - first_time_element + 1;
          n_time_elements_right = last_time_element - current_time_element;
        } else if ( el_centroid[ 3 ] >= time_center
          && ( temp_node_1 <= time_center ) ) {
          // the element is assigned to the right half, but its left bound
          // is less or equal than the center of the cluster.
          set_temporal_splitting_point = true;
          temporal_splitting_point = temp_node_1;
          // set the number of time elements in the left and right clusters
          n_time_elements_left = current_time_element - first_time_element;
          n_time_elements_right = last_time_element - current_time_element + 1;
        }
      }
    }
    lo n_clusters = 0;

    sc time_half_size_right
      = ( time_center + time_half_size - temporal_splitting_point ) * 0.5;
    sc time_center_right = temporal_splitting_point + time_half_size_right;
    sc time_half_size_left
      = ( temporal_splitting_point - ( time_center - time_half_size ) ) * 0.5;
    sc time_center_left = temporal_splitting_point - time_half_size_left;

    for ( short i = 0; i < 8; ++i ) {
      root.compute_spatial_suboctant(
        i, new_space_center, new_space_half_size );

      coord_x = 2 * root.get_box_coordinate( )[ 1 ] + _idx_2_coord[ i ][ 0 ];
      coord_y = 2 * root.get_box_coordinate( )[ 2 ] + _idx_2_coord[ i ][ 1 ];
      coord_z = 2 * root.get_box_coordinate( )[ 3 ] + _idx_2_coord[ i ][ 2 ];

      // std::cout << oct_sizes[ i ] << " " << oct_sizes[ i + 8 ] <<
      // std::endl;
      if ( oct_sizes[ i ] > 0 ) {
        ++n_clusters;
        coord_t = ( slou )( 2 * parent_coord[ 4 ] );
        std::vector< slou > coordinates
          = { static_cast< slou >( root.get_level( ) + 1 ), coord_x, coord_y,
              coord_z, coord_t };
        clusters[ i ] = new general_spacetime_cluster( new_space_center,
          time_center_left, new_space_half_size, time_half_size_left,
          oct_sizes[ i ], &root, root.get_level( ) + 1, i, coordinates, 0,
          2 * root.get_global_time_index( ) + 1, n_space_div + 1,
          n_time_div + 1, _spacetime_mesh, root.get_process_id( ), true );
        clusters[ i ]->set_n_time_elements( n_time_elements_left );
        clusters[ i ]->set_elements_are_local( elements_are_local );
      } else {
        clusters[ i ] = nullptr;
      }
      if ( oct_sizes[ i + 8 ] > 0 ) {
        ++n_clusters;
        coord_t = ( slou )( 2 * parent_coord[ 4 ] + 1 );
        std::vector< slou > coordinates
          = { static_cast< slou >( root.get_level( ) + 1 ), coord_x, coord_y,
              coord_z, coord_t };
        clusters[ i + 8 ] = new general_spacetime_cluster( new_space_center,
          time_center_right, new_space_half_size, time_half_size_right,
          oct_sizes[ i + 8 ], &root, root.get_level( ) + 1, i, coordinates, 1,
          2 * root.get_global_time_index( ) + 2, n_space_div + 1,
          n_time_div + 1, _spacetime_mesh, root.get_process_id( ), true );
        clusters[ i + 8 ]->set_n_time_elements( n_time_elements_right );
        clusters[ i + 8 ]->set_elements_are_local( elements_are_local );
      } else {
        clusters[ i + 8 ] = nullptr;
      }
    }

    // finally, assign elements to clusters
    for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
      elem_idx
        = _spacetime_mesh.global_2_local( start_idx, root.get_element( i ) );
      current_mesh->get_centroid( elem_idx, el_centroid );

      if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 0 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 1 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 2 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 3 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 4 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 5 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 6 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 7 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 8 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 9 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 10 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 11 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 12 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 13 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 14 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 15 ]->add_element( root.get_element( i ) );
      }
    }

    root.set_n_children( n_clusters );

    for ( lo i = 0; i < 16; ++i ) {
      if ( clusters[ i ] != nullptr ) {
        root.add_child( clusters[ i ] );
        clusters[ i ]->compute_node_mapping( );
        clusters[ i ]->set_n_space_nodes( );
        build_subtree( *clusters[ i ], split_space_descendant );
      }
    }
  } else {
    for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
      // get elem idx in local mesh indexing
      elem_idx
        = _spacetime_mesh.global_2_local( start_idx, root.get_element( i ) );
      current_mesh->get_centroid( elem_idx, el_centroid );
      if ( el_centroid[ 3 ] >= time_center ) {
        oct_sizes[ 1 ] += 1;
      } else {
        oct_sizes[ 0 ] += 1;
      }
      // set the temporal splitting point as in the previous case
      if ( !set_temporal_splitting_point ) {
        sc temp_node_1, temp_node_2;
        lo current_time_element = current_mesh->get_time_element( elem_idx );
        current_mesh->get_temporal_nodes(
          current_time_element, &temp_node_1, &temp_node_2 );
        if ( ( el_centroid[ 3 ] < time_center )
          && ( temp_node_2 >= time_center ) ) {
          set_temporal_splitting_point = true;
          temporal_splitting_point = temp_node_2;
          n_time_elements_left = current_time_element - first_time_element + 1;
          n_time_elements_right = last_time_element - current_time_element;
        } else if ( el_centroid[ 3 ] >= time_center
          && ( temp_node_1 <= time_center ) ) {
          set_temporal_splitting_point = true;
          temporal_splitting_point = temp_node_1;
          n_time_elements_left = current_time_element - first_time_element;
          n_time_elements_right = last_time_element - current_time_element + 1;
        }
      }
    }

    lo n_clusters = 0;
    coord_x = parent_coord[ 1 ];
    coord_y = parent_coord[ 2 ];
    coord_z = parent_coord[ 3 ];
    coord_t = ( slou )( 2 * parent_coord[ 4 ] );
    std::vector< slou > coordinates
      = { static_cast< slou >( root.get_level( ) + 1 ), coord_x, coord_y,
          coord_z, coord_t };

    sc time_half_size_right
      = ( time_center + time_half_size - temporal_splitting_point ) * 0.5;
    sc time_center_right = temporal_splitting_point + time_half_size_right;
    sc time_half_size_left
      = ( temporal_splitting_point - ( time_center - time_half_size ) ) * 0.5;
    sc time_center_left = temporal_splitting_point - time_half_size_left;
    general_spacetime_cluster * left_child = nullptr;
    general_spacetime_cluster * right_child = nullptr;

    // left temporal cluster
    if ( oct_sizes[ 0 ] > 0 ) {
      n_clusters++;
      left_child = new general_spacetime_cluster( space_center,
        time_center_left, space_half_size, time_half_size_left, oct_sizes[ 0 ],
        &root, root.get_level( ) + 1, root.get_spatial_octant( ), coordinates,
        0, 2 * root.get_global_time_index( ) + 1, n_space_div, n_time_div + 1,
        _spacetime_mesh, root.get_process_id( ), true );
      left_child->set_n_time_elements( n_time_elements_left );
      left_child->set_elements_are_local( elements_are_local );
    }

    // right temporal cluster
    coord_t = ( slou )( 2 * parent_coord[ 4 ] + 1 );
    coordinates[ 4 ] = coord_t;
    if ( oct_sizes[ 1 ] > 0 ) {
      n_clusters++;
      right_child
        = new general_spacetime_cluster( space_center, time_center_right,
          space_half_size, time_half_size_right, oct_sizes[ 1 ], &root,
          root.get_level( ) + 1, root.get_spatial_octant( ), coordinates, 1,
          2 * root.get_global_time_index( ) + 2, n_space_div, n_time_div + 1,
          _spacetime_mesh, root.get_process_id( ), true );
      right_child->set_n_time_elements( n_time_elements_right );
      right_child->set_elements_are_local( elements_are_local );
    }

    for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
      // get elem idx in local mesh indexing
      elem_idx
        = _spacetime_mesh.global_2_local( start_idx, root.get_element( i ) );
      current_mesh->get_centroid( elem_idx, el_centroid );
      if ( el_centroid[ 3 ] >= time_center ) {
        right_child->add_element( root.get_element( i ) );
      } else {
        left_child->add_element( root.get_element( i ) );
      }
    }
    root.set_n_children( n_clusters );

    if ( left_child != nullptr ) {
      root.add_child( left_child );
      left_child->compute_node_mapping( );
      left_child->set_n_space_nodes( );
      build_subtree( *left_child, split_space_descendant );
    }
    if ( right_child != nullptr ) {
      root.add_child( right_child );
      right_child->compute_node_mapping( );
      right_child->set_n_space_nodes( );
      build_subtree( *right_child, split_space_descendant );
    }
  }
  root.shrink_children( );
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  associate_scheduling_clusters_and_space_time_clusters( ) {
  // Traverse the two trees to determine all associated clusters. It is not
  // distinguished whether space time clusters are leaves or not.
  if ( _root != nullptr ) {
    scheduling_time_cluster * time_root = get_distribution_tree( )->get_root( );
    // traverse the tree recursively for all spacetime roots
    for ( auto spacetime_root : *_root->get_children( ) ) {
      associate_scheduling_clusters_and_space_time_clusters_recursively(
        time_root, spacetime_root );
    }
  } else {
    std::cout << "Error: Corrupted spacetime tree; _root is nullptr"
              << std::endl;
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  associate_scheduling_clusters_and_space_time_clusters_recursively(
    scheduling_time_cluster * t_root, general_spacetime_cluster * st_root ) {
  t_root->add_associated_spacetime_cluster( st_root );
  if ( st_root->get_n_children( ) > 0 && t_root->get_n_children( ) > 0 ) {
    // if t_root is not a leaf traverse the two trees further to find the
    // associated spacetime clusters of the descendants.
    std::vector< scheduling_time_cluster * > * time_children
      = t_root->get_children( );
    std::vector< general_spacetime_cluster * > * spacetime_children
      = st_root->get_children( );
    for ( auto time_child : *time_children ) {
      short time_child_configuration = time_child->get_configuration( );
      for ( auto spacetime_child : *spacetime_children ) {
        short spacetime_child_configuration
          = spacetime_child->get_temporal_configuration( );
        // check if the temporal component of the spacetime child is the same
        // as the current temporal child and call routine recursively if yes
        if ( time_child_configuration == spacetime_child_configuration ) {
          associate_scheduling_clusters_and_space_time_clusters_recursively(
            time_child, spacetime_child );
        }
      }
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  sort_associated_space_time_clusters_recursively(
    scheduling_time_cluster * t_root,
    std::vector< general_spacetime_cluster * > & leaf_buffer,
    std::vector< general_spacetime_cluster * > & non_leaf_buffer ) {
  // execute the routine first for all children
  if ( t_root->get_n_children( ) > 0 ) {
    for ( auto t_child : *t_root->get_children( ) ) {
      sort_associated_space_time_clusters_recursively(
        t_child, leaf_buffer, non_leaf_buffer );
    }
  }
  std::vector< general_spacetime_cluster * > * associated_st_clusters
    = t_root->get_associated_spacetime_clusters( );
  if ( associated_st_clusters != nullptr ) {
    // construct vectors to temporarily store the leaf and non-leaf clusters.
    leaf_buffer.resize( 0 );
    non_leaf_buffer.resize( 0 );
    leaf_buffer.reserve( associated_st_clusters->size( ) );
    non_leaf_buffer.reserve( associated_st_clusters->size( ) );
    // go through the associated spacetime clusters and separate leaf and
    // non-leaf clusters
    for ( auto st_cluster : *associated_st_clusters ) {
      if ( st_cluster->is_global_leaf( ) ) {
        leaf_buffer.push_back( st_cluster );
      } else {
        non_leaf_buffer.push_back( st_cluster );
      }
    }
    // overwrite the associated_st_clusters by adding first the clusters of
    // leaf_buffer and then those of non_leaf_buffer
    lou n_leaves = leaf_buffer.size( );
    t_root->set_n_associated_leaves( n_leaves );
    for ( lou i = 0; i < n_leaves; ++i ) {
      ( *associated_st_clusters )[ i ] = leaf_buffer[ i ];
    }
    for ( lou i = 0; i < non_leaf_buffer.size( ); ++i ) {
      ( *associated_st_clusters )[ n_leaves + i ] = non_leaf_buffer[ i ];
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  fill_nearfield_and_interaction_lists( general_spacetime_cluster & root ) {
  if ( root.get_parent( ) == nullptr ) {
    root.add_to_nearfield_list( &root );
  } else {
    // get information about the temporal part of root
    sc root_temporal_center = root.get_time_center( );
    sc root_temporal_half_size = root.get_time_half_size( );
    // go through parent's nearfield list to determine root's lists
    std::vector< general_spacetime_cluster * > * parent_nearfield
      = root.get_parent( )->get_nearfield_list( );
    for ( auto parent_nearfield_cluster : *parent_nearfield ) {
      // check if the parent's nearfield cluster is a leaf
      if ( parent_nearfield_cluster->get_n_children( ) == 0 ) {
        // add leaves to the nearfield of root
        root.add_to_nearfield_list( parent_nearfield_cluster );
      } else {
        // check admissibility of all children
        std::vector< general_spacetime_cluster * > * relevant_clusters
          = parent_nearfield_cluster->get_children( );
        for ( auto current_cluster : *relevant_clusters ) {
          sc current_temporal_center = current_cluster->get_time_center( );
          // check if current cluster is in the spatial vicinity of root and
          // if it is not in the future
          // (if one of these conditions is violated the current cluster is
          // not added to any list)
          if ( ( current_temporal_center
                 < root_temporal_center + root_temporal_half_size )
            && ( root.is_in_spatial_vicinity(
              current_cluster, _spatial_nearfield_limit ) ) ) {
            // add the current cluster to the appropriate list
            if ( root.determine_temporal_admissibility( current_cluster ) ) {
              root.add_to_interaction_list( current_cluster );
            } else {
              if ( root.get_n_children( ) > 0 ) {
                root.add_to_nearfield_list( current_cluster );
              } else {
                add_leaves_to_nearfield_list( *current_cluster, root );
              }
            }
          }
        }
      }
    }
  }
  if ( root.get_n_children( ) > 0 ) {
    for ( auto child : *root.get_children( ) ) {
      fill_nearfield_and_interaction_lists( *child );
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  add_leaves_to_nearfield_list( general_spacetime_cluster & current_cluster,
    general_spacetime_cluster & target_cluster ) {
  if ( current_cluster.get_n_children( ) == 0 ) {
    target_cluster.add_to_nearfield_list( &current_cluster );
  } else {
    for ( auto child : *current_cluster.get_children( ) ) {
      add_leaves_to_nearfield_list( *child, target_cluster );
    }
  }
}

sc besthea::mesh::distributed_spacetime_cluster_tree::
  compute_local_spatial_padding( general_spacetime_cluster & root ) {
  std::vector< general_spacetime_cluster * > * children = root.get_children( );
  sc padding = -1.0;
  sc t_padding;
  sc tmp_padding;

  if ( children != nullptr ) {
    // for non-leaf clusters, find the largest padding of its descendants
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      tmp_padding = this->compute_local_spatial_padding( **it );
      if ( tmp_padding > padding ) {
        padding = tmp_padding;
      }
    }
    if ( root.get_level( ) > -1
      && padding > _spatial_paddings[ root.get_level( ) ] ) {
      _spatial_paddings[ root.get_level( ) ] = padding;
    }
  } else {
    // for leaf clusters, compute padding directly
    if ( _my_rank == root.get_process_id( ) ) {
      root.compute_padding( padding, t_padding );
    }

    if ( padding > _spatial_paddings[ root.get_level( ) ] ) {
      _spatial_paddings[ root.get_level( ) ] = padding;
    }
  }
  return padding;
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  send_subtree_data_of_distribution_tree(
    const std::vector< scheduling_time_cluster * > & send_cluster_vector,
    const lo global_tree_levels, const lo communication_offset ) const {
  if ( send_cluster_vector.size( ) > 0 ) {
    // prepare the array, which is sent, by determining first its size
    lou send_array_size = 0;
    for ( lou i = 0; i < send_cluster_vector.size( ); ++i ) {
      lo send_cluster_level = send_cluster_vector[ i ]->get_level( );
      lo send_cluster_vec_size
        = ( 1 << ( global_tree_levels - send_cluster_level ) ) - 1;
      send_array_size += send_cluster_vec_size;
    }
    std::vector< char > send_structure_array( send_array_size );
    std::vector< sc > send_cluster_bounds_array( 2 * send_array_size );
    for ( lou i = 0; i < send_array_size; ++i ) {
      send_structure_array[ i ] = 0;
    }
    for ( lou i = 0; i < 2 * send_array_size; ++i ) {
      send_cluster_bounds_array[ i ] = 0;
    }
    lou send_array_pos = 0;
    for ( lou i = 0; i < send_cluster_vector.size( ); ++i ) {
      // compute the tree structure of the subtree and copy it to
      // send_structure_array
      std::vector< char > subtree_structure_vector
        = send_cluster_vector[ i ]->determine_subtree_structure( );
      for ( lou j = 0; j < subtree_structure_vector.size( ); ++j ) {
        send_structure_array[ send_array_pos + j ]
          = subtree_structure_vector[ j ];
      }
      // compute the cluster bounds of the cluster in the subtree and
      // copy them to send_cluster_bounds_array
      std::vector< sc > subtree_cluster_bounds_vector
        = send_cluster_vector[ i ]->determine_subtree_cluster_bounds( );
      for ( lou j = 0; j < 2 * subtree_structure_vector.size( ); ++j ) {
        send_cluster_bounds_array[ 2 * send_array_pos + j ]
          = subtree_cluster_bounds_vector[ j ];
      }
      // jump to the position of the next subtree in the
      // send_structure_array
      lo send_cluster_level = send_cluster_vector[ i ]->get_level( );
      lo send_cluster_vec_size
        = ( 1 << ( global_tree_levels - send_cluster_level ) ) - 1;
      send_array_pos += send_cluster_vec_size;
    }
    // send first the tree structure
    MPI_Send( send_structure_array.data( ), send_array_size, MPI_CHAR,
      _my_rank - communication_offset, communication_offset, *_comm );
    // next, send the cluster bounds (tag increased by 1 to distinguish)
    MPI_Send( send_cluster_bounds_array.data( ), 2 * send_array_size,
      get_scalar_type< sc >::MPI_SC( ), _my_rank - communication_offset,
      communication_offset + 1, *_comm );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  receive_subtree_data_of_distribution_tree(
    const std::vector< scheduling_time_cluster * > & receive_clusters_vector,
    const lo global_tree_levels, const lo communication_offset ) {
  if ( receive_clusters_vector.size( ) > 0 ) {
    // determine the size of the arrays which are used for receiving data
    lou receive_array_size = 0;
    for ( lou i = 0; i < receive_clusters_vector.size( ); ++i ) {
      lo receive_cluster_level = receive_clusters_vector[ i ]->get_level( );
      lo receive_cluster_vec_size
        = ( 1 << ( global_tree_levels - receive_cluster_level ) ) - 1;
      receive_array_size += receive_cluster_vec_size;
    }
    std::vector< char > receive_structure_array( receive_array_size );
    std::vector< sc > receive_cluster_bounds_array( 2 * receive_array_size );
    // call the appropriate receive operations for the tree structure data and
    // the cluster bounds data
    MPI_Status status_1, status_2;
    MPI_Recv( receive_structure_array.data( ), receive_array_size, MPI_CHAR,
      _my_rank + communication_offset, communication_offset, *_comm,
      &status_1 );
    MPI_Recv( receive_cluster_bounds_array.data( ), 2 * receive_array_size,
      get_scalar_type< sc >::MPI_SC( ), _my_rank + communication_offset,
      communication_offset + 1, *_comm, &status_2 );

    // extend the distribution tree according to the received data
    tree_structure * distribution_tree = get_distribution_tree( );
    lou receive_array_pos = 0;
    for ( lou i = 0; i < receive_clusters_vector.size( ); ++i ) {
      lou local_pos = receive_array_pos;
      // check whether the cluster is a non-leaf in the local tree of the
      // sending process
      if ( receive_structure_array[ local_pos ] == 1 ) {
        local_pos += 1;
        // refine the tree structure uniformly at the given cluster.
        receive_clusters_vector[ i ]->set_global_leaf_status( false );
        distribution_tree->create_tree_from_arrays(
          receive_structure_array.data( ), receive_cluster_bounds_array.data( ),
          *( receive_clusters_vector[ i ] ), local_pos );
      }
      // find the starting position of the entries corresponding to the
      // subtree of the next cluster
      lo receive_cluster_level = receive_clusters_vector[ i ]->get_level( );
      lo receive_cluster_vec_size
        = ( 1 << ( global_tree_levels - receive_cluster_level ) ) - 1;
      receive_array_pos += receive_cluster_vec_size;
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::send_leaf_info(
  const std::vector< scheduling_time_cluster * > & send_cluster_vector,
  const lo communication_offset ) const {
  if ( send_cluster_vector.size( ) > 0 ) {
    // determine first the size of the array of leaf info which is sent
    lou array_size = 0;
    for ( auto send_cluster : send_cluster_vector ) {
      array_size += send_cluster->get_associated_spacetime_clusters( )->size( );
    }
    bool * leaf_info_array = new bool[ array_size ];
    // fill the array appropriately
    lo pos = 0;
    for ( auto send_cluster : send_cluster_vector ) {
      for ( auto st_cluster :
        *send_cluster->get_associated_spacetime_clusters( ) ) {
        leaf_info_array[ pos ] = st_cluster->is_global_leaf( );
        pos++;
      }
    }
    // send the whole array at once to the appropriate process
    MPI_Send( leaf_info_array, array_size, MPI_CXX_BOOL,
      _my_rank + communication_offset, communication_offset, *_comm );
    delete[] leaf_info_array;
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::receive_leaf_info(
  const std::vector< scheduling_time_cluster * > & receive_cluster_vector,
  const lo communication_offset ) const {
  if ( receive_cluster_vector.size( ) > 0 ) {
    // determine first the size of the array of leaf info which is received
    lou array_size = 0;
    for ( auto receive_cluster : receive_cluster_vector ) {
      array_size
        += receive_cluster->get_associated_spacetime_clusters( )->size( );
    }
    bool * leaf_info_array = new bool[ array_size ];

    // start a blocking receive operation to receive the information
    MPI_Status status;
    MPI_Recv( leaf_info_array, array_size, MPI_CXX_BOOL,
      _my_rank - communication_offset, communication_offset, *_comm, &status );
    // update the global leaf status of all spacetime clusters associated with
    // scheduling time clusters in the receive cluster vector
    lo pos = 0;
    for ( auto receive_cluster : receive_cluster_vector ) {
      for ( auto st_cluster :
        *receive_cluster->get_associated_spacetime_clusters( ) ) {
        st_cluster->set_global_leaf_status( leaf_info_array[ pos ] );
        pos++;
      }
    }
    delete[] leaf_info_array;
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::print_information(
  const int root_process ) {
  if ( _my_rank == root_process ) {
    std::cout << "#############################################################"
              << "###########################" << std::endl;
    std::cout << "number of levels = " << _max_levels << std::endl;
    std::cout << "initial space level = " << _initial_space_refinement
              << std::endl;
    std::cout << "first space refinement level = " << _start_space_refinement
              << std::endl;
  }
  lo global_max_space_level;
  MPI_Reduce( &_local_max_space_level, &global_max_space_level, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, root_process, *_comm );
  if ( _my_rank == root_process ) {
    std::cout << "maximal space level = " << global_max_space_level
              << std::endl;
  }
  // determine levelwise number of leaves:
  std::vector< lou > n_leaves_levelwise( _max_levels, 0 );
  std::vector< sc > n_time_elems_levelwise( _max_levels, 0.0 );
  std::vector< sc > n_space_elems_levelwise( _max_levels, 0.0 );
  for ( auto leaf : _local_leaves ) {
    n_leaves_levelwise[ leaf->get_level( ) ] += 1;
    n_time_elems_levelwise[ leaf->get_level( ) ]
      += leaf->get_n_time_elements( );
    n_space_elems_levelwise[ leaf->get_level( ) ]
      += leaf->get_n_space_elements( );
  }
  if ( _my_rank == root_process ) {
    MPI_Reduce( MPI_IN_PLACE, n_leaves_levelwise.data( ), _max_levels,
      get_index_type< lo >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_time_elems_levelwise.data( ), _max_levels,
      get_scalar_type< sc >::MPI_SC( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_space_elems_levelwise.data( ), _max_levels,
      get_scalar_type< sc >::MPI_SC( ), MPI_SUM, root_process, *_comm );
  } else {
    MPI_Reduce( n_leaves_levelwise.data( ), nullptr, _max_levels,
      get_index_type< lo >::MPI_LO( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_time_elems_levelwise.data( ), nullptr, _max_levels,
      get_scalar_type< sc >::MPI_SC( ), MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_space_elems_levelwise.data( ), nullptr, _max_levels,
      get_scalar_type< sc >::MPI_SC( ), MPI_SUM, root_process, *_comm );
  }
  if ( _my_rank == root_process ) {
    std::cout << "#############################################################"
              << "###########################" << std::endl;
    lou n_global_leaves = 0;
    for ( lo i = 0; i < _max_levels; ++i ) {
      n_global_leaves += n_leaves_levelwise[ i ];
    }
    std::cout << "leaf information:" << std::endl;
    std::cout << "global number of leaves: " << n_global_leaves << std::endl;
    std::cout << "levelwise information:" << std::endl;
    for ( lo i = 0; i < _max_levels; ++i ) {
      std::cout << "level " << i << ": "
                << " leaves: " << n_leaves_levelwise[ i ];
      if ( n_leaves_levelwise[ i ] > 0 ) {
        n_time_elems_levelwise[ i ] /= n_leaves_levelwise[ i ];
        n_space_elems_levelwise[ i ] /= n_leaves_levelwise[ i ];
        std::cout << ", mean_n_elems_time: " << n_time_elems_levelwise[ i ]
                  << ", mean_n_elems_space: " << n_space_elems_levelwise[ i ];
      }
      std::cout << std::endl;
    }
    std::cout << "#############################################################"
              << "###########################" << std::endl;
  }
}
