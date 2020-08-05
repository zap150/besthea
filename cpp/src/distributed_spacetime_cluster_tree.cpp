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

#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/distributed_spacetime_cluster_tree.h"
#include "besthea/tree_structure.h"

#include <algorithm>
#include <cmath>
#include <vector>

besthea::mesh::distributed_spacetime_cluster_tree::
  distributed_spacetime_cluster_tree(
    distributed_spacetime_tensor_mesh & spacetime_mesh, lo levels,
    lo n_min_elems, sc st_coeff, slou spatial_nearfield_limit, MPI_Comm * comm )
  : _max_levels( levels ),
    _real_max_levels( 0 ),
    _spacetime_mesh( spacetime_mesh ),
    _s_t_coeff( st_coeff ),
    _n_min_elems( n_min_elems ),
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
  sc time_center =  get_distribution_tree( )->get_root( )->get_center( );
  sc time_half_size = get_distribution_tree( )->get_root( )->get_half_size( );
  std::vector< slou > coordinates = { 0, 0, 0, 0, 0 };
  _root = new general_spacetime_cluster( space_center, time_center,
    space_half_sizes, time_half_size, spacetime_mesh.get_n_elements( ), nullptr,
    0, 0, coordinates, 0, 0, 0, _spacetime_mesh, -1, false );

  std::vector< lo > elems_in_clusters;

  build_tree( _root );

  // extend the locally essential distribution tree:
  // first determine clusters in the distribution tree for which the extension
  // cannot be done locally
  std::set< std::pair< lo, scheduling_time_cluster* > > cluster_send_list;
  std::set< std::pair< lo, scheduling_time_cluster* > > cluster_receive_list;
  tree_structure* distribution_tree = get_distribution_tree( );
  distribution_tree->determine_refinement_communication_lists(
    distribution_tree->get_root( ), cluster_send_list, cluster_receive_list );
  if ( _my_rank == 1 ) {
    std::cout << "receive list is " << std::endl;
    for ( auto it : cluster_receive_list ) {
      std::cout << it.second->get_global_index( ) << " ";
    }
    std::cout << std::endl;
  }
  MPI_Barrier( *_comm );
  if ( _my_rank == 2 ) {
    std::cout << "send list is " << std::endl;
    for ( auto it : cluster_send_list ) {
      std::cout << it.second->get_global_index( ) << " ";
    }
    std::cout << std::endl;
  }
  MPI_Barrier( *_comm );
  // secondly, expand the distribution tree locally
  expand_distribution_tree_locally( );
  // finally, expand the distribution tree communicatively and reduce it again
  // to a locally essential tree
  expand_distribution_tree_communicatively( 
    cluster_send_list, cluster_receive_list );
  distribution_tree->reduce_2_essential( );


  associate_scheduling_clusters_and_space_time_clusters( );
  fill_nearfield_and_interaction_lists( *_root );
}


void besthea::mesh::distributed_spacetime_cluster_tree::build_tree(
  general_spacetime_cluster * root ) {
  tree_structure * dist_tree = get_distribution_tree( );
  lo dist_tree_depth = dist_tree->get_levels( );
  lo dist_tree_depth_coll;

  MPI_Allreduce( &dist_tree_depth, &dist_tree_depth_coll, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );
  //  if ( _my_rank == 1 ) {
  //    dist_tree->print_tree_human_readable( 2, true );
  //    dist_tree->print( );
  //  }
  std::vector< lo > n_elems_per_subdivisioning;
  n_elems_per_subdivisioning.resize( 0 );

  lo n_space_div = 0;
  lo n_time_div = 1;
  std::vector< std::pair< general_spacetime_cluster*, 
    scheduling_time_cluster* > > cluster_pairs;
  scheduling_time_cluster* temporal_root = dist_tree->get_root( );
  cluster_pairs.push_back( { root, temporal_root } );
  lo level;
  for ( level = 0; level < dist_tree_depth_coll - 1; ++level ) {
    // get global number of elements per cluster 
    get_n_elements_in_subdivisioning(
      *_root, n_space_div, n_time_div, n_elems_per_subdivisioning );
    split_clusters_levelwise( level % 2, n_space_div, n_time_div,   
      n_elems_per_subdivisioning, cluster_pairs );

    n_time_div++;
    if ( level % 2 == 0 ) {
      n_space_div++;
    }
  }
  std::vector< general_spacetime_cluster * > leaves;
  collect_real_leaves( *_root, *temporal_root, leaves );
  for ( auto it : leaves ) {
    // @todo Discuss: Inefficient way of filling in the elements? For each leaf
    // cluster the whole mesh is traversed once. If the depth of the tree is 
    // reasonably high this takes a while!
    fill_elements( *it );

    build_subtree( *it, it->get_level( ) % 2 );
  }

  // exchange necessary data
  MPI_Allreduce( MPI_IN_PLACE, &_real_max_levels, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );
}


void besthea::mesh::distributed_spacetime_cluster_tree::
  expand_distribution_tree_locally( ) {
  // @todo use the various spacetime roots at level 0 instead of a single 
  //spacetime root at level -1, when this is implemented for the distributed
  //spacetime cluster tree
  std::unordered_map< lo, bool > refine_map;
  tree_structure* distribution_tree = get_distribution_tree( );
  scheduling_time_cluster* time_root = distribution_tree->get_root( );
  distribution_tree->determine_clusters_to_refine( time_root, refine_map );
  if ( _root != nullptr ) {
    // expand the tree structure according to the spacetime tree
    expand_tree_structure_recursively( 
      distribution_tree, _root, time_root, refine_map );
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
  const std::set< std::pair< lo, scheduling_time_cluster* > > 
    cluster_send_list,
  const std::set< std::pair< lo, scheduling_time_cluster* > > 
    cluster_receive_list ) {
  tree_structure* distribution_tree = get_distribution_tree( );
  // first communicate the maximal depth of the distribution tree.
  lo global_tree_levels = distribution_tree->get_levels( );
  MPI_Allreduce( MPI_IN_PLACE, &global_tree_levels, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );
  // the sets are sorted by default lexicographically, i.e. first in ascending 
  // order with respect to the process ids. the code relies on that.
  lo max_offset = 0;
  if ( cluster_send_list.size( ) > 0 ) {
    max_offset = _my_rank - cluster_send_list.begin( )->first;
  }
  if ( cluster_receive_list.size( ) > 0 ) {
    max_offset 
      = std::max( max_offset, 
                  cluster_receive_list.rbegin( )->first - _my_rank );
  }
  // execute the send and receive operations offsetwise
  auto send_list_it = cluster_send_list.rbegin( );
  auto receive_list_it = cluster_receive_list.begin( );
  for ( lo offset = 1; offset <= max_offset; ++ offset ) {
    // depending on the rank decide whether to send or receive first
    if ( ( _my_rank / offset ) % 2 == 1 ) {
      // send first, receive later
      std::vector< scheduling_time_cluster* > current_send_clusters;
      while ( ( send_list_it != cluster_send_list.rend( ) ) &&
              ( _my_rank - send_list_it->first == offset ) ) {
        current_send_clusters.push_back( send_list_it->second );
        ++send_list_it;
      }
      if ( current_send_clusters.size( ) > 0 ) {
        // prepare the array, which is sent, by determining first its size
        lou send_array_size = 0;
        for ( lou i = 0; i < current_send_clusters.size( ); ++i ) {
          lo send_cluster_level = current_send_clusters[ i ]->get_level( );
          lo send_cluster_vec_size 
            = ( 1 << ( global_tree_levels - send_cluster_level) ) - 1;
          send_array_size += send_cluster_vec_size; 
        }
        char send_array[ send_array_size ];
        for ( lou i = 0; i < send_array_size; ++i ) {
          send_array[ i ] = 0;
        }
        lou send_array_pos = 0;
        for ( lou i = 0; i < current_send_clusters.size( ); ++i ) {
          // compute the tree structure of the subtree and copy it to send_array
          std::vector< char > subtree_structure_vector = 
            current_send_clusters[ i ]->determine_subtree_structure( );
          for ( lou j = 0; j < subtree_structure_vector.size( ); ++j ) {
            send_array[ send_array_pos + j ] = subtree_structure_vector[ j ];
          }
          // jump to the position of the next subtree in the send_array
          lo send_cluster_level = current_send_clusters[ i ]->get_level( );
          lo send_cluster_vec_size 
            = ( 1 << ( global_tree_levels - send_cluster_level) ) - 1;
          send_array_pos += send_cluster_vec_size;
        }
        MPI_Send( send_array, send_array_size, MPI_CHAR, _my_rank - offset,
                  offset, *_comm);
      }
      // now receive
      std::vector< scheduling_time_cluster* > current_receive_clusters;
      while ( ( receive_list_it != cluster_receive_list.end( ) ) &&
              ( receive_list_it->first - _my_rank == offset ) ) {
        current_receive_clusters.push_back( receive_list_it->second );
        ++receive_list_it;
      }
      if ( current_receive_clusters.size( ) > 0 ) {
        lou receive_array_size = 0;
        for ( lou i = 0; i < current_receive_clusters.size( ); ++i ) {
          lo receive_cluster_level 
            = current_receive_clusters[ i ]->get_level( );
          lo receive_cluster_vec_size 
            = ( 1 << ( global_tree_levels - receive_cluster_level) ) - 1;
          receive_array_size += receive_cluster_vec_size; 
        }
        char receive_array[ receive_array_size ];
        MPI_Status status;
        MPI_Recv( receive_array, receive_array_size, MPI_CHAR, 
          _my_rank + offset, offset, *_comm, &status);
        lou receive_array_pos = 0;
        for ( lou i = 0; i < current_receive_clusters.size( ); ++i ) {
          lou local_pos = receive_array_pos;
          // check whether the cluster is a non-leaf in the local tree of the
          // sending process
          if ( receive_array[ local_pos ] == 1 ) {
            local_pos += 1;
            // refine the tree structure uniformly at the given cluster.
            current_receive_clusters[ i ]->set_global_leaf_status( false );
            distribution_tree->array_2_tree( 
              receive_array, *current_receive_clusters[ i ], local_pos );
          }
          // find the starting position of the entries corresponding to the 
          // subtree of the next cluster
          lo receive_cluster_level 
            = current_receive_clusters[ i ]->get_level( );
          lo receive_cluster_vec_size 
            = ( 1 << ( global_tree_levels - receive_cluster_level) ) - 1;
          receive_array_pos += receive_cluster_vec_size;
        }
      }
    } else {
      //receive first, send later
      std::vector< scheduling_time_cluster* > current_receive_clusters;
      while ( ( receive_list_it != cluster_receive_list.end( ) ) &&
              ( receive_list_it->first - _my_rank == offset ) ) {
        current_receive_clusters.push_back( receive_list_it->second );
        ++receive_list_it;
      }
      if ( current_receive_clusters.size( ) > 0 ) {
        lou receive_array_size = 0;
        for ( lou i = 0; i < current_receive_clusters.size( ); ++i ) {
          lo receive_cluster_level 
            = current_receive_clusters[ i ]->get_level( );
          lo receive_cluster_vec_size 
            = ( 1 << ( global_tree_levels - receive_cluster_level) ) - 1;
          receive_array_size += receive_cluster_vec_size; 
        }
        char receive_array[ receive_array_size ];
        MPI_Status status;
        MPI_Recv( receive_array, receive_array_size, MPI_CHAR, 
          _my_rank + offset, offset, *_comm, &status);
        lou receive_array_pos = 0;
        for ( lou i = 0; i < current_receive_clusters.size( ); ++i ) {
          lou local_pos = receive_array_pos;
          // check whether the cluster is a non-leaf in the local tree of the
          // sending process
          if ( receive_array[ local_pos ] == 1 ) {
            local_pos += 1;
            // refine the tree structure uniformly at the given cluster.
            current_receive_clusters[ i ]->set_global_leaf_status( false );
            distribution_tree->array_2_tree( 
              receive_array, *current_receive_clusters[ i ], local_pos );
          }
          // find the starting position of the entries corresponding to the 
          // subtree of the next cluster
          lo receive_cluster_level 
            = current_receive_clusters[ i ]->get_level( );
          lo receive_cluster_vec_size 
            = ( 1 << ( global_tree_levels - receive_cluster_level) ) - 1;
          receive_array_pos += receive_cluster_vec_size;
        }
      }
      // now send
      std::vector< scheduling_time_cluster* > current_send_clusters;
      while ( ( send_list_it != cluster_send_list.rend( ) ) &&
              ( _my_rank - send_list_it->first == offset ) ) {
        current_send_clusters.push_back( send_list_it->second );
        ++send_list_it;
      }
      if ( current_send_clusters.size( ) > 0 ) {
        // prepare the array, which is sent, by determining first its size
        lou send_array_size = 0;
        for ( lou i = 0; i < current_send_clusters.size( ); ++i ) {
          lo send_cluster_level = current_send_clusters[ i ]->get_level( );
          lo send_cluster_vec_size 
            = ( 1 << ( global_tree_levels - send_cluster_level) ) - 1;
          send_array_size += send_cluster_vec_size; 
        }
        char send_array[ send_array_size ];
        for ( lou i = 0; i < send_array_size; ++i ) {
          send_array[ i ] = 0;
        }
        lou send_array_pos = 0;
        for ( lou i = 0; i < current_send_clusters.size( ); ++i ) {
          // compute the tree structure of the subtree and copy it to send_array
          std::vector< char > subtree_structure_vector = 
            current_send_clusters[ i ]->determine_subtree_structure( );
          for ( lou j = 0; j < subtree_structure_vector.size( ); ++j ) {
            send_array[ send_array_pos + j ] = subtree_structure_vector[ j ];
          }
          // jump to the position of the next subtree in the send_array
          lo send_cluster_level = current_send_clusters[ i ]->get_level( );
          lo send_cluster_vec_size 
            = ( 1 << ( global_tree_levels - send_cluster_level) ) - 1;
          send_array_pos += send_cluster_vec_size;
        }
        MPI_Send( send_array, send_array_size, MPI_CHAR, _my_rank - offset,
                  offset, *_comm);
      }
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
  expand_tree_structure_recursively( tree_structure* distribution_tree, 
  general_spacetime_cluster* spacetime_root, 
  scheduling_time_cluster* time_root, 
  std::unordered_map< lo, bool > & refine_map ) {
  // if the current time cluster is a leaf handled by the process _my_process_id 
  // and the current space-time cluster is not a leaf expand the temporal tree
  // structure
  if ( time_root->get_n_children( ) == 0 
        && refine_map[ time_root->get_global_index( ) ]
        && spacetime_root->get_n_children( ) > 0 ) {
    std::vector< general_spacetime_cluster* > * spacetime_children =
        spacetime_root->get_children( );
    sc center_t_parent = time_root->get_center( );
    sc level_parent = time_root->get_level( );
    // determine whether the left and right children have to be added
    scheduling_time_cluster * left_cluster = nullptr;
    scheduling_time_cluster * right_cluster = nullptr;
    char child_count = 0;
    auto st_it = spacetime_children->begin( );
    // consider the temporal components of all space-time children and
    // create a new scheduling time cluster if a new one is encountered.
    while ( child_count < 2 && st_it != spacetime_children->end( ) ) {
      sc center_t_child = ( *st_it )->get_time_center( );
      if ( center_t_child < center_t_parent && left_cluster == nullptr ) {
        // construct left cluster and set its process id and global index
        sc half_size_child = ( *st_it )->get_time_half_size( );
        left_cluster = new scheduling_time_cluster(
          center_t_child, half_size_child, time_root, level_parent + 1 );
        left_cluster->set_process_id( time_root->get_process_id( ) );
        left_cluster->set_index( 2 * time_root->get_global_index( ) + 1 );
        refine_map[ 2 * time_root->get_global_index( ) + 1 ] = true;
        child_count += 1;
      }
      else if ( center_t_child > center_t_parent && right_cluster == nullptr ) {
        // construct right cluster and set its process id and global index
        sc half_size_child = ( *st_it )->get_time_half_size( );
        right_cluster = new scheduling_time_cluster(
          center_t_child, half_size_child, time_root, level_parent + 1 );
        right_cluster->set_process_id( time_root->get_process_id( ) );
        right_cluster->set_index( 2 * time_root->get_global_index( ) + 2 );
        refine_map[ 2 * time_root->get_global_index( ) + 2 ] = true;
        child_count += 1;
      }
      ++ st_it;
    }
    // add the new children to the temporal cluster and complete their data
    // (nearfield, interaction list, send list + determine activity)
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
    // since time_root is no leaf anymore reset its global leaf status and
    // mesh availability to false (both is false for non-leaves).
    time_root->set_global_leaf_status( false );
    time_root->set_mesh_availability( false );
    // update the member _levels of the distribution tree, if it has increased.
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
      std::vector< scheduling_time_cluster* > * time_children = 
        time_root->get_children( );
      std::vector< general_spacetime_cluster* > * spacetime_children =
        spacetime_root->get_children( );
      for ( auto it_time = time_children->begin( ); 
            it_time != time_children->end( ); ++it_time ) { 
        sc temporal_center = ( *it_time )->get_center( );
        sc half_size = ( *it_time )->get_half_size( );
        for ( auto it_st = spacetime_children->begin( ); 
                it_st != spacetime_children->end( ); ++it_st ) {
          sc st_temporal_center = ( *it_st )->get_time_center( );
          // check if the temporal component of the spacetime child is the same 
          // as the current temporal child and call routine recursively if yes
          if ( std::abs( st_temporal_center - temporal_center ) < half_size ) {
            expand_tree_structure_recursively( 
              distribution_tree, *it_st, *it_time, refine_map );
          } 
        }
      }
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  get_n_elements_in_subdivisioning( general_spacetime_cluster & root,
    lo n_space_div, lo n_time_div, std::vector< lo > & elems_in_clusters ) {
  lo n_space_clusters = 1;
  lo n_time_clusters = 1;

  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }
  for ( lo i = 0; i < n_time_div; ++i ) {
    n_time_clusters *= 2;
  }
  lo n_clusters
    = n_space_clusters * n_space_clusters * n_space_clusters * n_time_clusters;

  elems_in_clusters.resize( n_clusters );
  std::vector< lo > loc_elems_in_clusters( n_clusters, 0 );
  linear_algebra::coordinates< 4 > centroid;
  lo pos_x, pos_y, pos_z, pos_t = 0;

  vector_type space_center;
  space_center.resize( 3 );
  vector_type half_size;
  half_size.resize( 3 );
  sc time_center, time_half_size;
  root.get_center( space_center, time_center );
  root.get_half_size( half_size, time_half_size );

  // doing it this complicated to avoid inconsistencies with tree
  // assembly due to rounding errors
  std::vector< sc > timesteps( 0 );
  timesteps.reserve( n_time_clusters );
  std::vector< sc > steps_x( 0 );
  steps_x.reserve( n_space_clusters );
  std::vector< sc > steps_y( 0 );
  steps_y.reserve( n_space_clusters );
  std::vector< sc > steps_z( 0 );
  steps_z.reserve( n_space_clusters );

  decompose_line( time_center, time_half_size, time_center - time_half_size,
    n_time_div, 0, timesteps );

  decompose_line( space_center[ 0 ], half_size[ 0 ],
    space_center[ 0 ] - half_size[ 0 ], n_space_div, 0, steps_x );
  decompose_line( space_center[ 1 ], half_size[ 1 ],
    space_center[ 1 ] - half_size[ 1 ], n_space_div, 0, steps_y );
  decompose_line( space_center[ 2 ], half_size[ 2 ],
    space_center[ 2 ] - half_size[ 2 ], n_space_div, 0, steps_z );

  timesteps.push_back( time_center + time_half_size );
  steps_x.push_back( centroid[ 0 ] + half_size[ 0 ] + 1.0 );
  steps_y.push_back( centroid[ 1 ] + half_size[ 1 ] + 1.0 );
  steps_z.push_back( centroid[ 2 ] + half_size[ 2 ] + 1.0 );

  // assign slices to tree nodes
  const std::vector< sc > & slices = _spacetime_mesh.get_slices( );
  std::vector< sc > starts(
    timesteps.size( ) - 1, std::numeric_limits< sc >::infinity( ) );
  std::vector< sc > ends( 
    timesteps.size( ) - 1, -std::numeric_limits< sc >::infinity( ) );
  sc center;

  for ( lo i = 0; i < static_cast< lo >( timesteps.size( ) ) - 1; ++i ) {
    for ( lo j = 0; j < static_cast< lo >( slices.size( ) ) - 1; ++j ) {
      center = ( slices[ j ] + slices[ j + 1 ] ) / 2.0;
      if ( center > timesteps[ i ] && center <= timesteps[ i + 1 ] ) {
        if ( slices[ j ] < starts[ i ] ) {
          starts[ i ] = slices[ j ];
        }
        if ( slices[ j + 1 ] > ends[ i ] ) {
          ends[ i ] = slices[ j + 1 ];
        }
      }
    }
  }
 
  starts[ 0 ] -= 1.0;
  // before timesteps[ 0 ] was reduced by 1, but only starts is used below
  sc delta_x = ( 2 * half_size[ 0 ] ) / n_space_clusters;
  sc delta_y = ( 2 * half_size[ 1 ] ) / n_space_clusters;
  sc delta_z = ( 2 * half_size[ 2 ] ) / n_space_clusters;

  for ( lo i = 0; i < _spacetime_mesh.get_local_mesh( )->get_n_elements( ); 
        ++i ) {
    _spacetime_mesh.get_local_mesh( )->get_centroid( i, centroid );
    pos_x
      = ( centroid[ 0 ] - ( space_center[ 0 ] - half_size[ 0 ] ) ) / delta_x;
    pos_y
      = ( centroid[ 1 ] - ( space_center[ 1 ] - half_size[ 1 ] ) ) / delta_y;
    pos_z
      = ( centroid[ 2 ] - ( space_center[ 2 ] - half_size[ 2 ] ) ) / delta_z;

    lo start, end;

    for ( lo i = 0; i < static_cast< lo >( timesteps.size( ) ) - 1; ++i ) {
      if ( centroid[ 3 ] > starts[ i ] && centroid[ 3 ] <= ends[ i ] ) {
        pos_t = i;
        break;
      }
    }

    start = pos_x > 0 ? pos_x - 1 : pos_x;
    end = pos_x < static_cast< lo >( steps_x.size( ) ) - 2 ? pos_x + 1 : pos_x;
    for ( lo i = start; i <= end; ++i ) {
      if ( ( centroid[ 0 ] >= steps_x[ i ] )
        && ( centroid[ 0 ] < steps_x[ i + 1 ] ) ) {
        pos_x = i;
        break;
      }
    }

    start = pos_y > 0 ? pos_y - 1 : pos_y;
    end = pos_y < static_cast< lo >( steps_y.size( ) ) - 2 ? pos_y + 1 : pos_y;
    for ( lo i = start; i <= end; ++i ) {
      if ( ( centroid[ 1 ] >= steps_y[ i ] )
        && ( centroid[ 1 ] < steps_y[ i + 1 ] ) ) {
        pos_y = i;
        break;
      }
    }
    start = pos_z > 0 ? pos_z - 1 : pos_z;
    end = pos_z < static_cast< lo >( steps_z.size( ) ) - 2 ? pos_z + 1 : pos_z;
    for ( lo i = start; i <= end; ++i ) {
      if ( ( centroid[ 2 ] >= steps_z[ i ] )
        && ( centroid[ 2 ] < steps_z[ i + 1 ] ) ) {
        pos_z = i;
        break;
      }
    }

    lo pos = pos_t * n_space_clusters * n_space_clusters * n_space_clusters
      + pos_x * n_space_clusters * n_space_clusters + pos_y * n_space_clusters
      + pos_z;
    loc_elems_in_clusters.at( pos )++;
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

void besthea::mesh::distributed_spacetime_cluster_tree::
  collect_clusters_on_level( general_spacetime_cluster & root, lo level,
    std::vector< general_spacetime_cluster * > & clusters ) {
  if ( root.get_level( ) == level ) {
    clusters.push_back( &root );
    return;
  }

  std::vector< general_spacetime_cluster * > * children = root.get_children( );
  if ( children != nullptr ) {
    for ( auto it : *children ) {
      collect_clusters_on_level( *it, level, clusters );
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  collect_scheduling_clusters_on_level( scheduling_time_cluster & root,
    lo level, std::vector< scheduling_time_cluster * > & clusters ) {
  if ( root.get_level( ) == level ) {
    clusters.push_back( &root );
    return;
  }

  std::vector< scheduling_time_cluster * > * children = root.get_children( );
  if ( children != nullptr ) {
    for ( auto it : *children ) {
      collect_scheduling_clusters_on_level( *it, level, clusters );
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::split_cluster(
  general_spacetime_cluster & cluster,
  std::vector< scheduling_time_cluster * > & my_clusters_on_level,
  bool split_space, lo n_space_div, lo n_time_div,
  std::vector< lo > & elems_in_clusters ) {
  lo n_space_clusters = 1;
  lo n_time_clusters = 1;
  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }
  for ( lo i = 0; i < n_time_div; ++i ) {
    n_time_clusters *= 2;
  }

  const std::vector< slou > parent_coord = cluster.get_box_coordinate( );

  vector_type space_center( 3 );
  vector_type space_half_size( 3 );
  sc time_center, time_half_size;
  cluster.get_center( space_center, time_center );
  cluster.get_half_size( space_half_size, time_half_size );

  sc new_time_center, new_time_half_size;

  lo pos;

  // first, create left temporal cluster
  new_time_half_size = time_half_size / 2.0;
  new_time_center = time_center - new_time_half_size;
  slou coord_t = 2 * parent_coord[ 4 ];

  bool is_my_cluster = false;
  lo owner = -1;
  for ( auto it : my_clusters_on_level ) {
    if ( ( std::abs( new_time_center - it->get_center( ) )
           < it->get_half_size( ) )
      && ( it->get_essential_status( ) > 1 ) ) {
      is_my_cluster = true;
      owner = it->get_process_id( );
    }
  }

  if ( is_my_cluster ) {
    if ( split_space ) {
      vector_type new_spat_center( 3 );
      vector_type new_spat_half_size( 3 );
      for ( short i = 0; i < 8; ++i ) {
        cluster.compute_spatial_suboctant(
          i, new_spat_center, new_spat_half_size );

        slou coord_x
          = 2 * cluster.get_box_coordinate( )[ 1 ] + _idx_2_coord[ i ][ 0 ];
        slou coord_y
          = 2 * cluster.get_box_coordinate( )[ 2 ] + _idx_2_coord[ i ][ 1 ];
        slou coord_z
          = 2 * cluster.get_box_coordinate( )[ 3 ] + _idx_2_coord[ i ][ 2 ];
        std::vector< slou > coordinates
          = { static_cast< slou >( cluster.get_level( ) ), coord_x, coord_y,
              coord_z, coord_t };

        pos = coord_t * n_space_clusters * n_space_clusters * n_space_clusters
          + coord_x * n_space_clusters * n_space_clusters
          + coord_y * n_space_clusters + coord_z;

        if ( elems_in_clusters[ pos ] > 0 ) {
          general_spacetime_cluster * left_child
            = new general_spacetime_cluster( new_spat_center, new_time_center,
              new_spat_half_size, new_time_half_size, elems_in_clusters[ pos ],
              &cluster, cluster.get_level( ) + 1, i, coordinates, 0,
              n_space_div, n_time_div, _spacetime_mesh, owner, false );
          cluster.add_child( left_child );
        }
      }
    } else {
      slou coord_x = parent_coord[ 1 ];
      slou coord_y = parent_coord[ 2 ];
      slou coord_z = parent_coord[ 3 ];
      std::vector< slou > coordinates
        = { static_cast< slou >( cluster.get_level( ) ), coord_x, coord_y,
            coord_z, coord_t };
      pos = coord_t * n_space_clusters * n_space_clusters * n_space_clusters
        + coord_x * n_space_clusters * n_space_clusters
        + coord_y * n_space_clusters + coord_z;

      if ( elems_in_clusters[ pos ] > 0 ) {
        // todo: shouldn't left_right be 0 here?
        general_spacetime_cluster * left_child = new general_spacetime_cluster(
          space_center, new_time_center, space_half_size, new_time_half_size,
          elems_in_clusters[ pos ], &cluster, cluster.get_level( ) + 1,
          cluster.get_spatial_octant( ), coordinates, 1, n_space_div,
          n_time_div, _spacetime_mesh, owner, false );
        cluster.add_child( left_child );
      }
    }
  }

  // next, right temporal cluster
  new_time_center = time_center + new_time_half_size;
  coord_t = 2 * parent_coord[ 4 ] + 1;

  is_my_cluster = false;
  for ( auto it : my_clusters_on_level ) {
    if ( ( std::abs( new_time_center - it->get_center( ) )
           < it->get_half_size( ) )
      && ( it->get_essential_status( ) > 1 ) ) {
      is_my_cluster = true;
      owner = it->get_process_id( );
    }
  }

  if ( is_my_cluster ) {
    if ( split_space ) {
      vector_type new_spat_center( 3 );
      vector_type new_spat_half_size( 3 );
      for ( short i = 0; i < 8; ++i ) {
        cluster.compute_spatial_suboctant(
          i, new_spat_center, new_spat_half_size );

        slou coord_x
          = 2 * cluster.get_box_coordinate( )[ 1 ] + _idx_2_coord[ i ][ 0 ];
        slou coord_y
          = 2 * cluster.get_box_coordinate( )[ 2 ] + _idx_2_coord[ i ][ 1 ];
        slou coord_z
          = 2 * cluster.get_box_coordinate( )[ 3 ] + _idx_2_coord[ i ][ 2 ];
        std::vector< slou > coordinates
          = { static_cast< slou >( cluster.get_level( ) ), coord_x, coord_y,
              coord_z, coord_t };

        pos = coord_t * n_space_clusters * n_space_clusters * n_space_clusters
          + coord_x * n_space_clusters * n_space_clusters
          + coord_y * n_space_clusters + coord_z;

        if ( elems_in_clusters[ pos ] > 0 ) {
          general_spacetime_cluster * right_child
            = new general_spacetime_cluster( new_spat_center, new_time_center,
              new_spat_half_size, new_time_half_size, elems_in_clusters[ pos ],
              &cluster, cluster.get_level( ) + 1, i, coordinates, 1,
              n_space_div, n_time_div, _spacetime_mesh, owner, false );
          cluster.add_child( right_child );
        }
      }
    } else {
      slou coord_x = parent_coord[ 1 ];
      slou coord_y = parent_coord[ 2 ];
      slou coord_z = parent_coord[ 3 ];
      std::vector< slou > coordinates
        = { static_cast< slou >( cluster.get_level( ) ), coord_x, coord_y,
            coord_z, coord_t };

      pos = coord_t * n_space_clusters * n_space_clusters * n_space_clusters
        + coord_x * n_space_clusters * n_space_clusters
        + coord_y * n_space_clusters + coord_z;

      if ( elems_in_clusters[ pos ] > 0 ) {
        // todo shouldn't left_right be 1 here, and not 0?
        general_spacetime_cluster * right_child = new general_spacetime_cluster(
          space_center, new_time_center, space_half_size, new_time_half_size,
          elems_in_clusters[ pos ], &cluster, cluster.get_level( ) + 1,
          cluster.get_spatial_octant( ), coordinates, 0, n_space_div,
          n_time_div, _spacetime_mesh, owner, false );
        cluster.add_child( right_child );
      }
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  split_clusters_levelwise( bool split_space, lo n_space_div, lo n_time_div, 
  std::vector< lo > & elems_in_clusters,
  std::vector< std::pair< general_spacetime_cluster*, 
    scheduling_time_cluster * > > & cluster_pairs ) {
  // compute number of space and time clusters at the level of children
  lo n_space_clusters = 1;
  lo n_time_clusters = 1;
  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }
  for ( lo i = 0; i < n_time_div; ++i ) {
    n_time_clusters *= 2;
  }

  // vector to store the pairs of children which are constructed below.
  std::vector< std::pair< general_spacetime_cluster*, 
    scheduling_time_cluster * > > new_cluster_pairs;
  // reserve a fair amount of entries 
  // (assuming <~ 5 children in time, 2 in space due to surface mesh)
  new_cluster_pairs.reserve( cluster_pairs.size( ) * 10 );
  
  // refine all space-time clusters whose children are locally essential
  for ( lou i = 0; i < cluster_pairs.size( ); ++i ) {
    // get current spacetime cluster and time cluster
    general_spacetime_cluster* st_cluster = cluster_pairs[ i ].first;
    const std::vector< slou > parent_coord = st_cluster->get_box_coordinate( );
    scheduling_time_cluster* t_cluster = cluster_pairs[ i ].second;
    
    // split the cluster only if it contains enough elements and the temporal
    // component is a non-leaf
    if ( st_cluster->get_n_elements( ) >= _n_min_elems &&  
          t_cluster->get_n_children( ) > 0 ) {
      std::vector< scheduling_time_cluster* > * t_children =
        t_cluster->get_children( );
      for ( auto t_child : *t_children ) {
        // check if temporal child is locally essential with respect to the 
        // space-time cluster tree.
        if ( t_child->get_essential_status( ) > 1 ) {
          // set coord_t and left_right appropriately distinguishing the left 
          // and right children.
          lo owner = t_child->get_process_id( );
          slou coord_t;
          short left_right;
          if ( t_child->get_center( ) < t_cluster->get_center( ) ) {
            coord_t = 2 * parent_coord[ 4 ]; // left child
            left_right = 0;
          } else {
            coord_t = 2 * parent_coord[ 4 ] + 1; // right child
            left_right = 1;
          }

          sc new_time_center = t_child->get_center( );
          sc new_time_half_size = t_child->get_half_size( );
          if ( split_space ) {
            vector_type new_spat_center( 3 );
            vector_type new_spat_half_size( 3 );
            for ( short i = 0; i < 8; ++i ) {
              st_cluster->compute_spatial_suboctant(
                i, new_spat_center, new_spat_half_size );

              slou coord_x = 2 * st_cluster->get_box_coordinate( )[ 1 ] 
                            + _idx_2_coord[ i ][ 0 ];
              slou coord_y = 2 * st_cluster->get_box_coordinate( )[ 2 ] 
                            + _idx_2_coord[ i ][ 1 ];
              slou coord_z = 2 * st_cluster->get_box_coordinate( )[ 3 ] 
                            + _idx_2_coord[ i ][ 2 ];
              std::vector< slou > coordinates 
                = { static_cast< slou >( st_cluster->get_level( ) ), 
                    coord_x, coord_y, coord_z, coord_t };

              lou pos = coord_t * n_space_clusters * n_space_clusters 
                * n_space_clusters 
                + coord_x * n_space_clusters * n_space_clusters
                + coord_y * n_space_clusters + coord_z;

              if ( elems_in_clusters[ pos ] > 0 ) {
                general_spacetime_cluster * new_child
                  = new general_spacetime_cluster( new_spat_center, 
                    new_time_center, new_spat_half_size, new_time_half_size, 
                    elems_in_clusters[ pos ], st_cluster, 
                    st_cluster->get_level( ) + 1, i, coordinates, left_right,
                    n_space_div, n_time_div, _spacetime_mesh, owner, false );
                st_cluster->add_child( new_child );
                new_cluster_pairs.push_back( { new_child, t_child } );
              }
            }
          } else {
            slou coord_x = parent_coord[ 1 ];
            slou coord_y = parent_coord[ 2 ];
            slou coord_z = parent_coord[ 3 ];
            std::vector< slou > coordinates
              = { static_cast< slou >( st_cluster->get_level( ) ), coord_x, 
                  coord_y, coord_z, coord_t };
            lou pos = coord_t * n_space_clusters * n_space_clusters 
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
                = new general_spacetime_cluster(
                  space_center, new_time_center, space_half_size, 
                  new_time_half_size, elems_in_clusters[ pos ], st_cluster, 
                  st_cluster->get_level( ) + 1, 
                  st_cluster->get_spatial_octant( ), coordinates, 1, 
                  n_space_div, n_time_div, _spacetime_mesh, owner, false );
              st_cluster->add_child( new_child );
              new_cluster_pairs.push_back( { new_child, t_child } );
            }
          }
        }
      }
    }
  }
  // replace the old vector of cluster pairs by the one which was newly constructed
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

void besthea::mesh::distributed_spacetime_cluster_tree::collect_my_leaves(
  general_spacetime_cluster & root,
  std::vector< general_spacetime_cluster * > & leaves ) {
  std::vector< general_spacetime_cluster * > * children = root.get_children( );

  if ( children != nullptr ) {
    for ( auto it : *children ) {
      collect_my_leaves( *it, leaves );
    }
  } else if ( _my_rank == root.get_process_id( ) ) {
    leaves.push_back( &root );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  collect_real_leaves( general_spacetime_cluster & st_root, 
  scheduling_time_cluster & t_root,
  std::vector< general_spacetime_cluster * > & leaves ) {
  std::vector< scheduling_time_cluster * > * t_children 
    = t_root.get_children( );
  if ( t_children != nullptr ) {
    std::vector< general_spacetime_cluster * > * st_children 
      = st_root.get_children( );
    if ( st_children != nullptr ) {
      lou t_idx = 0;
      scheduling_time_cluster* t_child = ( *t_children )[ t_idx ];
      for ( lou st_idx = 0; st_idx < st_children->size( ); ++st_idx ) {
        general_spacetime_cluster* st_child = ( *st_children )[ st_idx ];
        // check whether the temporal component of the st_child is the t_child
        // and if not update the t_child
        sc st_child_time_center = st_child->get_time_center( );
        sc t_child_time_center = t_child->get_center( );
        sc time_half_size = t_child->get_half_size( );
        if ( std::abs( st_child_time_center - t_child_time_center ) 
              > time_half_size ) {
          ++t_idx;
          t_child = ( *t_children )[ t_idx ];
        }
        // call the routine recursively for the appropriate pair of spacetime
        // cluster and scheduling time cluster
        collect_real_leaves( *st_child, *t_child, leaves );
      }
    } 
    else if ( st_root.get_n_elements( ) < _n_min_elems ) {
      leaves.push_back( &st_root );
    }
  }
  // if t_root is a leaf in the global tree structure, the corresponding 
  // space-time cluster are leaves and have to be refined. By construction,
  // these clusters are either leaf clusters in the nearfield (whose meshes are
  // available ) or local leaf clusters.
  else if ( t_root.get_global_leaf_status( ) && t_root.mesh_is_available( ) ) { 
    leaves.push_back( &st_root );
  }
}


void besthea::mesh::distributed_spacetime_cluster_tree::fill_elements(
  general_spacetime_cluster & cluster ) {
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
  } else {
    current_mesh = _spacetime_mesh.get_nearfield_mesh( );
    start_idx = _spacetime_mesh.get_nearfield_start_idx( );
  }
  linear_algebra::coordinates< 4 > centroid;

  for ( lo i = 0; i < current_mesh->get_n_elements( ); ++i ) {
    current_mesh->get_centroid( i, centroid );

    if ( ( centroid[ 0 ] >= left ) && ( centroid[ 0 ] < right )
      && ( centroid[ 1 ] >= front ) && ( centroid[ 1 ] < back )
      && ( centroid[ 2 ] >= bottom ) && ( centroid[ 2 ] < top )
      && ( centroid[ 3 ] > beginning ) && ( centroid[ 3 ] <= end ) ) {
      cluster.add_element( _spacetime_mesh.local_2_global( start_idx, i ) );
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::build_subtree(
  general_spacetime_cluster & root, bool split_space ) {
  if ( root.get_level( ) + 1 > _max_levels - 1
    || root.get_n_elements( ) < _n_min_elems ) {
    root.set_n_children( 0 );

    if ( root.get_level( ) + 1 > _real_max_levels ) {
      _real_max_levels = root.get_level( ) + 1;
    }
    return;
  }

  const spacetime_tensor_mesh * current_mesh;
  lo start_idx;
  if ( root.get_process_id( ) == _my_rank ) {
    current_mesh = _spacetime_mesh.get_local_mesh( );
    start_idx = _spacetime_mesh.get_local_start_idx( );
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

  sc new_time_center, new_time_half_size;

  vector_type new_space_center( 3 );
  vector_type new_space_half_size( 3 );

  slou coord_x, coord_y, coord_z, coord_t = 0;

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
    }
    lo n_clusters = 0;

    for ( short i = 0; i < 8; ++i ) {
      root.compute_spatial_suboctant(
        i, new_space_center, new_space_half_size );

      coord_x = 2 * root.get_box_coordinate( )[ 1 ] + _idx_2_coord[ i ][ 0 ];
      coord_y = 2 * root.get_box_coordinate( )[ 2 ] + _idx_2_coord[ i ][ 1 ];
      coord_z = 2 * root.get_box_coordinate( )[ 3 ] + _idx_2_coord[ i ][ 2 ];

      std::vector< slou > coordinates
        = { static_cast< slou >( root.get_level( ) + 1 ), coord_x, coord_y,
            coord_z, coord_t };
      // std::cout << oct_sizes[ i ] << " " << oct_sizes[ i + 8 ] << std::endl;
      if ( oct_sizes[ i ] > 0 ) {
        ++n_clusters;
        new_time_half_size = time_half_size / 2.0;
        new_time_center = time_center - new_time_half_size;
        coord_t = 2 * parent_coord[ 4 ];
        clusters[ i ]
          = new general_spacetime_cluster( new_space_center, new_time_center,
            new_space_half_size, new_time_half_size, oct_sizes[ i ], &root,
            root.get_level( ) + 1, i, coordinates, 0, n_space_div + 1,
            n_time_div + 1, _spacetime_mesh, root.get_process_id( ), true );
      } else {
        clusters[ i ] = nullptr;
      }
      if ( oct_sizes[ i + 8 ] > 0 ) {
        ++n_clusters;
        new_time_half_size = time_half_size / 2.0;
        new_time_center = time_center + new_time_half_size;
        coord_t = 2 * parent_coord[ 4 ] + 1;
        clusters[ i + 8 ]
          = new general_spacetime_cluster( new_space_center, new_time_center,
            new_space_half_size, new_time_half_size, oct_sizes[ i ], &root,
            root.get_level( ) + 1, i, coordinates, 1, n_space_div + 1,
            n_time_div + 1, _spacetime_mesh, root.get_process_id( ), true );
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

        build_subtree( *clusters[ i ], !split_space );
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
    }

    lo n_clusters = 0;
    coord_x = parent_coord[ 1 ];
    coord_y = parent_coord[ 2 ];
    coord_z = parent_coord[ 3 ];
    coord_t = 2 * parent_coord[ 4 ];
    std::vector< slou > coordinates
      = { static_cast< slou >( root.get_level( ) + 1 ), coord_x, coord_y,
          coord_z, coord_t };

    new_time_half_size = time_half_size / 2.0;
    new_time_center = time_center - new_time_half_size;
    general_spacetime_cluster *left_child, *right_child;

    // left temporal cluster
    if ( oct_sizes[ 0 ] > 0 ) {
      n_clusters++;
      left_child = new general_spacetime_cluster( space_center, new_time_center,
        space_half_size, new_time_half_size, oct_sizes[ 0 ], &root,
        root.get_level( ) + 1, root.get_spatial_octant( ), coordinates, 0,
        n_space_div, n_time_div + 1, _spacetime_mesh, root.get_process_id( ),
        true );
    }

    // right temporal cluster
    new_time_center = time_center + new_time_half_size;
    coord_t = 2 * parent_coord[ 4 ] + 1;
    coordinates[ 4 ] = coord_t;
    if ( oct_sizes[ 1 ] > 0 ) {
      n_clusters++;
      right_child = new general_spacetime_cluster( space_center,
        new_time_center, space_half_size, new_time_half_size, oct_sizes[ 1 ],
        &root, root.get_level( ) + 1, root.get_spatial_octant( ), coordinates,
        1, n_space_div, n_time_div + 1, _spacetime_mesh, root.get_process_id( ),
        true );
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

    root.add_child( left_child );
    build_subtree( *left_child, !split_space );
    root.add_child( right_child );
    build_subtree( *right_child, !split_space );
  }
  root.shrink_children( );
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  associate_scheduling_clusters_and_space_time_clusters( ) {
  // Traverse the two trees twice to determine all associated clusters, first
  // the leaves and then the non-leaves. This ensures that first the spacetime 
  // leaves are added to the lists.
  // @todo Currently there is only one root in the tree with level 0. When the
  // code is changed to include several roots this code here has to be adapted
  // (add loop over spacetime roots)
  if ( _root != nullptr ) {
    scheduling_time_cluster* time_root 
      = get_distribution_tree( )->get_root( );
    associate_scheduling_clusters_and_space_time_leaves( time_root, _root );
    associate_scheduling_clusters_and_space_time_non_leaves( time_root, _root );
  } else {
    std::cout << "Error: Corrupted spacetime tree; _root is nullptr" 
              << std::endl;
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  associate_scheduling_clusters_and_space_time_leaves( 
  scheduling_time_cluster* t_root, general_spacetime_cluster * st_root ) {
  if ( st_root->get_n_children( ) == 0 ) {
    // check if st_root is a leaf in the global space time cluster tree
    if ( t_root->get_global_leaf_status( ) || 
         st_root->get_n_elements( ) < _n_min_elems ) {
      // add spacetime leaf to the list of associated clusters.
      t_root->add_associated_spacetime_cluster( st_root );
      t_root->increase_n_associated_leaves( );
    }
  } else {
    // if t_root is not a leaf traverse the two trees further to find the 
    // associated spacetime leaf clusters of the descendants.
    if ( t_root->get_n_children( ) > 0 ) {
      std::vector< scheduling_time_cluster* > * time_children = 
        t_root->get_children( );
      std::vector< general_spacetime_cluster* > * spacetime_children =
        st_root->get_children( );
      for ( auto it_t = time_children->begin( ); it_t != time_children->end( ); 
            ++it_t ) {
        sc temporal_center = ( *it_t )->get_center( );
        sc half_size = ( *it_t )->get_half_size( );
        for ( auto it_st = spacetime_children->begin( );
              it_st != spacetime_children->end( ); ++it_st ) {
          sc st_temporal_center = ( *it_st )->get_time_center( );
          // check if the temporal component of the spacetime child is the same 
          // as the current temporal child and call routine recursively if yes
          if ( std::abs( st_temporal_center - temporal_center ) < half_size ) {
            associate_scheduling_clusters_and_space_time_leaves( 
              *it_t, *it_st );
          } 
        }
      }
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  associate_scheduling_clusters_and_space_time_non_leaves( 
  scheduling_time_cluster* t_root, general_spacetime_cluster * st_root ) {
  if ( st_root->get_n_children( ) > 0 ) {
    // add spacetime non-leaf to the list of associated clusters.
    t_root->add_associated_spacetime_cluster( st_root );
    // if root is not a leaf traverse the two trees further to find the 
    // associated spacetime non-leaf clusters of the descendants.
    if ( t_root->get_n_children( ) > 0 ) {
      std::vector< scheduling_time_cluster* > * time_children = 
        t_root->get_children( );
      std::vector< general_spacetime_cluster* > * spacetime_children =
        st_root->get_children( );
      for ( auto it_t = time_children->begin( ); it_t != time_children->end( ); 
            ++it_t ) {
        sc temporal_center = ( *it_t )->get_center( );
        sc half_size = ( *it_t )->get_half_size( );
        for ( auto it_st = spacetime_children->begin( );
              it_st != spacetime_children->end( ); ++it_st ) {
          sc st_temporal_center = ( *it_st )->get_time_center( );
          // check if the temporal component of the spacetime child is the same 
          // as the current temporal child and call routine recursively if yes
          if ( std::abs( st_temporal_center - temporal_center ) < half_size ) {
            associate_scheduling_clusters_and_space_time_non_leaves(
              *it_t, *it_st );
          } 
        }
      }
    }
  }
  else if ( ( !t_root->get_global_leaf_status( ) ) &&
            ( st_root->get_n_elements( ) >= _n_min_elems ) ) {
    // spacetime leaf is a non-leaf in the global space-time cluster tree.
    t_root->add_associated_spacetime_cluster( st_root );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  fill_nearfield_and_interaction_lists( general_spacetime_cluster& root ) {
  if ( root.get_parent( ) == nullptr ) {
    root.add_to_nearfield_list( &root );
  } else {
    // get information about the temporal part of root
    sc root_temporal_center = root.get_time_center( );
    sc root_temporal_half_size = root.get_time_half_size( );
    // go through parent's nearfield list to determine root's lists 
    std::vector< general_spacetime_cluster* >* parent_nearfield 
      = root.get_parent( )->get_nearfield_list( );
    for ( auto parent_nearfield_cluster : *parent_nearfield ) {
      // check if the parent's nearfield cluster is a leaf
      if ( parent_nearfield_cluster->get_n_children( ) == 0 ) {
        // add leaves to the nearfield of root
        root.add_to_nearfield_list( parent_nearfield_cluster );
      } else {
        // check admissibility of all children
        std::vector< general_spacetime_cluster* >* relevant_clusters
          = parent_nearfield_cluster->get_children( );
        for ( auto current_cluster : *relevant_clusters ) {
          if ( current_cluster == &root ) {
            root.add_to_nearfield_list( current_cluster );
          } else {
            sc current_temporal_center = current_cluster->get_time_center( );
            std::vector< slou > current_box_coordinate 
              = current_cluster->get_box_coordinate( );
            // check if current cluster is in the spatial vicinity of root and
            // if it is not in the future 
            // (if one of these conditions is violated the current cluster is 
            // not added to any list)
            if ( ( root.is_in_spatial_vicinity( 
                      current_cluster, _spatial_nearfield_limit ) )  
                 && ( current_temporal_center 
                      < root_temporal_center + root_temporal_half_size ) ) {
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
    // std::cout << "called this " << std::endl;
    // target_cluster.print( );
    // current_cluster.print( );
  } else {
    for ( auto child : *current_cluster.get_children( ) ) {
      add_leaves_to_nearfield_list( *child, target_cluster );
    }
  }
}