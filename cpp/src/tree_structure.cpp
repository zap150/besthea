/*
 * Copyright 2020, VSB - Technical University of Ostrava and Graz University of
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

#include "besthea/tree_structure.h"

#include <algorithm>

besthea::mesh::tree_structure::tree_structure( const std::string & filename, 
  const sc start_time, const sc end_time )
  : _levels( 0 ), _my_process_id( -1 ) {
  // load tree structure from file
  std::vector< char > tree_vector = read_vector_from_bin_file< char >( 
    filename );
  // create tree structure from vector 
  if ( ( tree_vector.size( ) > 0 ) && ( tree_vector[ 0 ] != 0 ) ) {
    sc center = 0.5 * ( start_time + end_time );
    sc half_size = 0.5 * ( end_time - start_time );
    _root = new scheduling_time_cluster( center, half_size, nullptr, 0 );
      _levels = 1;
    if ( tree_vector[ 0 ] == 1 ) {
      lou position = 1;
      vector_2_tree( tree_vector, *_root, position );
    }
  } else {
    _root = nullptr;
  }
  lo next_index = 0;
  // set the global indices of all the clusters in the structure
  set_indices( *_root, next_index );
  set_nearfield_interaction_and_send_list( *_root );
  // determine activity of clusters in upward and downward path of FMM
  determine_cluster_activity( *_root );
  collect_leaves( *_root );
  lo next_id = 0;
  set_leaf_ids( *_root, next_id );
}

void besthea::mesh::tree_structure::load_process_assignments( 
  const std::string & filename ) {
  std::vector< lo > process_assignments = 
    read_vector_from_bin_file< lo >( filename );
  _root->set_process_id( process_assignments[ 0 ] );
  lou position = 1;
  set_process_assignments( process_assignments, *_root, position );
}

void besthea::mesh::tree_structure::
  reduce_2_essential( const lo my_process_id ) {
  _my_process_id = my_process_id;
  std::vector< char > status_vector;
  determine_essential_clusters( my_process_id, *_root, status_vector );
  // if only the leaves should be kept, which are leaves in the original tree
  // structure the following code can be used
  // ###########################################################################
  // remove all leaves with status 0 from _leaves
  // std::vector< scheduling_time_cluster* > new_leaves;
  // for ( lou i = 0; i < _leaves.size( ); ++ i ) {
  //   lo leaf_index = _leaves[ i ]->get_global_index( );
  //   if ( status_vector[ leaf_index ] > 0 ) {
  //     new_leaves.push_back( _leaves[ i ] );
  //   }
  // }
  // new_leaves.shrink_to_fit( );
  // _leaves = std::move( new_leaves );
  // ###########################################################################

  // traverse the tree structure and reduce it by eliminating all 
  // non-essential clusters. In addition, correct the nearfield, interaction 
  // lists and children of all clusters and reset _levels and _leaves.
  _levels = 0;
  prepare_essential_reduction( *_root, status_vector );
  execute_essential_reduction( *_root, status_vector );
  // reset leaves of tree structure appropriately
  _leaves.clear( );
  collect_leaves( *_root );
}

void besthea::mesh::tree_structure::
  prepare_fmm( std::list< scheduling_time_cluster* > & m_list,
  std::list< scheduling_time_cluster* > & m2l_list,
  std::list< scheduling_time_cluster* > & l_list,
  std::list< scheduling_time_cluster* > & n_list, 
  std::vector< std::pair< scheduling_time_cluster*, lo > > & receive_vector,
  lou & n_moments_upward, lou & n_moments_m2l ) const {
  m_list.clear( );
  init_fmm_lists_and_dependency_data( 
    *_root, m_list, m2l_list, l_list, n_list );
  // sort the m_list from bottom up, right to left
  m_list.sort( compare_clusters_bottom_up_right_2_left );
  // sort the m2l_list and l_list from top down, right to left and create 
  // n_list as copy of l_list
  m2l_list.sort( compare_clusters_top_down_right_2_left );
  l_list.sort( compare_clusters_top_down_right_2_left );

  // fill the receive list by determining all incoming data.
  // check for receive operations in the upward path
  for ( auto it = m_list.begin( ); it != m_list.end( ); ++it ) {
    std::vector< scheduling_time_cluster* > * children
      = ( *it )->get_children( );
    if ( children != nullptr ) {
      for ( auto it_child = children->begin( ); it_child != children->end( );
            ++it_child ) {
        // if the current cluster has a child which is handled by a different 
        // process p the current process has to receive the processed moments 
        // from p.
        if ( ( *it_child )->get_process_id( ) != _my_process_id ) {
          receive_vector.push_back( std::pair< scheduling_time_cluster*, lo >( 
                                      *it, ( *it_child )->get_process_id( ) ) );
          ( *it )->add_receive_buffer( ( *it_child )->get_process_id( ) );
        }
      }
    } 
  }
  n_moments_upward = receive_vector.size( );
  // check for receive operations in the interaction phase
  for ( auto it = m2l_list.begin( ); it != m2l_list.end( ); 
        ++it ) {
    std::vector< scheduling_time_cluster* > * interaction_list 
      = ( *it )->get_interaction_list( );
    // interaction list is never empty for clusters in the m2l_list, so it is
    // not checked, whether the pointer is null
    for ( auto it_src = interaction_list->begin( ); 
          it_src != interaction_list->end( ); ++it_src ) {
      // if the source cluster is handled by a different process p the current
      // process has to receive its moments from p.
      if ( ( *it_src )->get_process_id( ) != _my_process_id ) {
        receive_vector.push_back( std::pair< scheduling_time_cluster*, lo >(
                                    *it_src, ( *it_src )->get_process_id( ) ) ); 
      }
    }
  }
  // if two clusters have the same source cluster in their interaction list 
  // its moments have to be received only once -> find and eliminate double
  // entries in the second part of the receive vector
  std::sort( receive_vector.begin( ) + n_moments_upward, receive_vector.end( ),
    []( const std::pair< scheduling_time_cluster*, lo > pair_one,
        const std::pair< scheduling_time_cluster*, lo > pair_two ) {
          return compare_clusters_top_down_right_2_left(
            pair_one.first, pair_two.first );
        } );
  auto new_end 
    = std::unique( receive_vector.begin( ) + n_moments_upward, 
      receive_vector.end( ),
      []( const std::pair< scheduling_time_cluster*, lo > pair_one,
          const std::pair< scheduling_time_cluster*, lo > pair_two ) {
            return pair_one.first == pair_two.first;
          } );
  receive_vector.resize( std::distance( receive_vector.begin(), new_end ) );
  n_moments_m2l = receive_vector.size( ) - n_moments_upward;
  // check for receive operations in the downward path
  for ( auto it = l_list.begin( ); it != l_list.end( ); ++it ) {
    scheduling_time_cluster * parent = ( *it )->get_parent( );
    // if the parent cluster is handled by a different process p the current 
    // process has to receive its local contributions from p.
    if ( parent->get_process_id( ) != _my_process_id && 
         parent->get_process_id( ) != -1 ) {
      receive_vector.push_back( std::pair< scheduling_time_cluster*, lo >( 
                                  parent, parent->get_process_id( ) ) );
    }
  }
}

std::vector< char > besthea::mesh::tree_structure::
  compute_tree_structure( ) const {
  std::vector< char > tree_vector;
  if ( _root == nullptr ) {
    tree_vector.push_back( 0 );
  }
  else if ( _root->get_n_children( ) == 0 ) {
    tree_vector.push_back( 2 );
  }
  else {
    tree_vector.push_back( 1 );
    tree_2_vector( *_root, tree_vector );
  }
  return tree_vector;
}

void besthea::mesh::tree_structure::
  print_tree_structure( const std::string & filename ) const {
  write_vector_to_bin_file( compute_tree_structure( ), filename );
}

void besthea::mesh::tree_structure::print_tree_human_readable( 
  const lo digits, const bool print_process_ids ) const {
  if ( _levels > 0 ) {
    std::vector< std::string > print_strings;
    print_strings.resize( _levels );
    determine_levelwise_output_string( 
      digits, print_process_ids, _root, print_strings );
    for ( lou i = 0; i < print_strings.size( ); ++i ) {
      std::cout << print_strings[ i ] << std::endl;
    }
  } else {
    std::cout << "number of levels is " << _levels << std::endl;
  }
} 

void besthea::mesh::tree_structure::tree_2_vector( 
  const scheduling_time_cluster & root, 
  std::vector<char> & tree_vector ) const {
  // get the children of root and determine if they are leaves or not 
  // WARNING: it is assumed that root always has two children; this assumption
  // is reasonable if the method is called for a non-leaf cluster in the tree,
  // since the tree is a full binary tree by construction (in build tree)
  const std::vector< scheduling_time_cluster * > * children 
    = root.get_children( );
  char left_child_status = 
    ( ( *children )[ 0 ]->get_n_children( ) > 0 ) ? 1 : 2;
  char right_child_status = 
    ( ( *children )[ 1 ]->get_n_children( ) > 0 ) ? 1 : 2;
  tree_vector.push_back( left_child_status );
  tree_vector.push_back( right_child_status );
  if ( left_child_status == 1 ) {
    tree_2_vector( *( *children )[ 0 ], tree_vector );
  } 
  if ( right_child_status == 1 ) {
    tree_2_vector( *( *children )[ 1 ], tree_vector );
  }
}

void besthea::mesh::tree_structure::vector_2_tree( 
  const std::vector< char > & tree_vector, scheduling_time_cluster & root, 
  lou & position ) {
  // get the cluster data of root
  lo level = root.get_level( );
  sc center = root.get_center( );
  sc half_size = root.get_half_size( );
  // determine status of the children of root and create them accordingly
  char left_child_status = tree_vector[ position++ ];
  char right_child_status = tree_vector[ position++ ];
  lo child_counter = 0;
  scheduling_time_cluster * left_cluster = nullptr;
  scheduling_time_cluster * right_cluster = nullptr;
  if ( left_child_status > 0 ) {
    child_counter++;
    left_cluster = new scheduling_time_cluster(
    center - half_size / 2.0, half_size / 2.0, &root, level + 1 );
  }
  if ( right_child_status > 0 ) {
    child_counter++;
    right_cluster = new scheduling_time_cluster(
    center + half_size / 2.0, half_size / 2.0, &root, level + 1 );
  }
  // add the newly created clusters to the root
  root.set_n_children( child_counter );
  if ( left_cluster != nullptr ) {
    root.add_child( left_cluster );
  }
  if ( right_cluster != nullptr ) {
    root.add_child( right_cluster );
  }
  // call the routine recursively for non-leaf children or update the depth of
  // the cluster tree if a leaf is encountered
  if ( left_child_status == 1 ) {
    vector_2_tree( tree_vector, *left_cluster, position );
  } else {
    if ( level + 2 > _levels ) {
      _levels = level + 2;
    }
  }
  if ( right_child_status == 1 ) {
    vector_2_tree( tree_vector, *right_cluster, position );
  } else {
    if ( level + 2 > _levels ) {
      _levels = level + 2;
    }
  }
}

void besthea::mesh::tree_structure::set_process_assignments( 
  const std::vector< lo > process_assignments, scheduling_time_cluster & root, 
  lou & position ) {
  lo left_child_process_id = process_assignments[ position++ ];
  lo right_child_process_id = process_assignments[ position++ ];
  scheduling_time_cluster* left_child = ( *root.get_children( ) )[ 0 ];
  scheduling_time_cluster* right_child = ( *root.get_children( ) )[ 1 ];
  left_child->set_process_id( left_child_process_id );
  right_child->set_process_id( right_child_process_id );
  if ( left_child->get_n_children( ) > 0 ) {
    set_process_assignments( process_assignments, *left_child, position );
  }
  if ( right_child->get_n_children( ) > 0 ) {
    set_process_assignments( process_assignments, *right_child, position );
  }
}

void besthea::mesh::tree_structure::collect_leaves( 
  scheduling_time_cluster & root ) {
  if ( root.get_n_children( ) == 0 ) {
    _leaves.push_back( &root );
  } else {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      collect_leaves( **it );
    }
  }
}

void besthea::mesh::tree_structure::set_leaf_ids( 
  scheduling_time_cluster & root, lo & next_id ) {
  if ( root.get_n_children( ) == 0 ) {
    root.set_leaf_index( next_id );
    // std::cout << "setting id to " << root.get_leaf_index( next_id ) << std::endl;
    next_id++;
  } else {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      set_leaf_ids( **it, next_id );
    }
  }
}

void besthea::mesh::tree_structure::set_indices( 
  scheduling_time_cluster & root, lo & next_index ) {
  if ( root.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster* >* children = root.get_children( );
    for ( lou i = 0; i < children->size( ); ++i ) {
      set_indices( *( *children )[ i ], next_index );
    }
  }
  root.set_index( next_index );
  next_index += 1;
}

void besthea::mesh::tree_structure::set_nearfield_interaction_and_send_list( 
  scheduling_time_cluster & root ) {
  if ( root.get_parent( ) == nullptr ) {
    root.add_to_nearfield( &root );
  } else {
    // traverse parents nearfield to determine nearfield and interaction list
    sc current_center = root.get_center( );
    std::vector< scheduling_time_cluster* >* parent_nearfield 
      = root.get_parent( )->get_nearfield( );
    for ( lou i = 0; i < parent_nearfield->size( ); ++i ) {
      // check if neighbor of parent is a leaf cluster
      if ( ( *parent_nearfield )[ i ]->get_n_children( ) == 0 ) {
        // add a leaf in the nearfield of parent to the nearfield of root
        root.add_to_nearfield( ( *parent_nearfield )[ i ] );
      } else {
        // check admissibility of all children
        std::vector< scheduling_time_cluster* >* relevant_clusters
          = ( *parent_nearfield )[ i ]->get_children( );
        for ( lou j = 0; j < relevant_clusters->size( ); ++j ) {
          scheduling_time_cluster* src_cluster = ( *relevant_clusters )[ j ];
          if ( src_cluster == &root ) {
            root.add_to_nearfield( src_cluster );
          }
          else if ( src_cluster->get_center( ) < current_center ) {
            if ( root.determine_admissibility( src_cluster ) ) {
              root.add_to_interaction_list( src_cluster );
              src_cluster->add_to_send_list( &root );
            } else {
              // if root is not a leaf, add src_cluster directly, else add all
              // of the leaves of src_cluster to the nearfield
              if ( root.get_n_children( ) > 0 ) {
                root.add_to_nearfield( src_cluster );
              } else {
                add_leaves_to_nearfield( *src_cluster, root );
              }
            }
          }
        }
      }
    }
  }
  if ( root.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster* >* children = root.get_children( ); 
    for ( lou i = 0; i < children->size( ); ++i ){
      set_nearfield_interaction_and_send_list( *( *children )[ i ] );
    }
  }
}

void besthea::mesh::tree_structure::add_leaves_to_nearfield( 
  scheduling_time_cluster & current_cluster, 
  scheduling_time_cluster & target_cluster ) {
  if ( current_cluster.get_n_children( ) == 0 ) {
    target_cluster.add_to_nearfield( &current_cluster );
    // std::cout << "called this " << std::endl;
    // target_cluster.print( );
    // current_cluster.print( );
  } else {
    std::vector< scheduling_time_cluster* > * children 
      = current_cluster.get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      add_leaves_to_nearfield( **it, target_cluster );
    }
  }
}

void besthea::mesh::tree_structure::determine_cluster_activity( 
  scheduling_time_cluster & root ) {
  // check if cluster is active in upward path
  if ( ( root.get_send_list( ) != nullptr ) || 
       ( root.get_parent( ) != nullptr && 
         root.get_parent( )->get_active_upward_path( ) ) ) {
    root.set_active_upward_path( true );
  }
  // check if cluster is active in downward path
  if ( ( root.get_interaction_list( ) != nullptr ) ||
       ( root.get_parent( ) != nullptr &&
         root.get_parent( )->get_active_downward_path( ) ) ) {
    root.set_active_downward_path( true );
  }
  // check if cluster is a leaf and call the routine recursively if not
  if ( root.get_n_children( ) > 0 ) {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      determine_cluster_activity( **it );
    }
  }
}

void besthea::mesh::tree_structure::init_fmm_lists_and_dependency_data( 
  scheduling_time_cluster & root, 
  std::list< scheduling_time_cluster* > & m_list,
  std::list< scheduling_time_cluster* > & m2l_list,
  std::list< scheduling_time_cluster* > & l_list,
  std::list< scheduling_time_cluster* > & n_list ) const {
  // if the cluster is local and active in the upward path add it to the 
  // m-list and initialize the appropriate dependency data
  if ( root.get_process_id( ) == _my_process_id &&
       root.get_active_upward_path( ) ) {
    m_list.push_back( &root );
    root.set_upward_path_counter( root.get_n_children( ) );
  }
  // if the current cluster is local and its parent is active in the downward 
  // path add the current cluster to the l-list
  if ( root.get_process_id( ) == _my_process_id &&
       root.get_parent( )->get_active_downward_path( ) ) {
    l_list.push_back( &root );
  }

  // if the current cluster is local, active in the downward path and its parent
  // is inactive in the downward path change the downward path status of the 
  // current cluster (to signal that no l2l operation has to be done anymore)
  if ( root.get_active_downward_path( ) && 
        !( root.get_parent( )->get_active_downward_path( ) ) ) {
    root.set_downward_path_status( 1 );
  }

  // add the cluster to the m2l-list, if it is local and has a non-empty
  // interaction list
  if ( root.get_process_id( ) == _my_process_id && 
       root.get_interaction_list( ) != nullptr ) {
    m2l_list.push_back( &root );
  }

  // add the cluster to the n-list, if it is a leaf
  // TODO: change this later: A cluster is in the n-list if there is a
  // space-time leaf cluster associated to it.
  if ( root.get_process_id( ) == _my_process_id && 
       root.get_n_children( ) == 0 ){
    n_list.push_back( &root );
  }

  // recursive call for all children
  if ( root.get_n_children( ) > 0 ) {
    const std::vector< scheduling_time_cluster* >* children 
      = root.get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      init_fmm_lists_and_dependency_data( 
        **it, m_list, m2l_list, l_list, n_list );
    }
  }
}

void besthea::mesh::tree_structure::prepare_essential_reduction( 
  scheduling_time_cluster & root, 
  std::vector< char > & status_vector ) {
  lo current_index = root.get_global_index( );
  // call routine recursively
  if ( root.get_n_children( ) > 0 ) {
    bool child_included = false;
    const std::vector< scheduling_time_cluster* >* children = 
      root.get_children( );
    for ( lou i = 0; i < children->size( ); ++i ) {
      prepare_essential_reduction( *( *children )[ i ], status_vector );
      char child_status 
        = status_vector[ ( *children )[ i ]->get_global_index( ) ];
      if ( child_status > 0 ) {
        child_included = true;
      }
    }
    // change the status of the current cluster if at least one of its children
    // is essential
    if ( child_included ) {
      if ( status_vector[ current_index ] == 0 ) {
        status_vector[ current_index ] = 2;
      } 
    }
  }
  if ( status_vector[ current_index ] > 0 ) {
    // update _levels if necessary
    lo current_level = root.get_level( );
    if ( current_level + 1 > _levels ) {
      _levels = current_level + 1;
    }
    // check if the nearfield contains clusters which are not essential and 
    // remove them in case. If the resulting nearfield is empty delete the
    // container
    std::vector< scheduling_time_cluster * > * nearfield 
      = root.get_nearfield( );
    auto it = nearfield->begin( );
    while ( it != nearfield->end( ) ) {
      lo src_index = ( *it )->get_global_index( );
      if ( status_vector[ src_index ] == 0 ) {
        it = nearfield->erase( it );
      } else {
        ++it;
      }
    }
    if ( nearfield->size( ) == 0 ) {
      root.delete_nearfield( );
    }
    // same for the send list
    std::vector< scheduling_time_cluster * > * send_list 
      = root.get_send_list( );
    if ( send_list != nullptr ) {
      it = send_list->begin( );
      while ( it != send_list->end( ) ) {
        lo tar_index = ( *it )->get_global_index( );
        if ( status_vector[ tar_index ] == 0 ) {
          it = send_list->erase( it );
        } else {
          ++it;
        }
      }
      if ( send_list->size( ) == 0 ) {
        root.delete_send_list( );
      }
    }

    // same for the interaction list 
    std::vector< scheduling_time_cluster * > * interaction_list
      = root.get_interaction_list( );
    if ( interaction_list != nullptr ) {
      it = interaction_list->begin( );
      while ( it != interaction_list->end( ) ) {
        lo src_index = ( *it )->get_global_index( );
        if ( status_vector[ src_index ] == 0 ) {
          it = interaction_list->erase( it );
        } else {
          ++it;
        }
      }
      if ( interaction_list->size( ) == 0 ) {
        root.delete_interaction_list( );
      }
    }
  }
}

void besthea::mesh::tree_structure::execute_essential_reduction( 
  scheduling_time_cluster & root, 
  const std::vector< char > & status_vector ) {
  // Recursively traverse the tree structure and delete non-essential clusters
  if ( root.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children 
      = root.get_children( );
    auto it = children->begin( );
    while ( it != children->end( ) ) {
      lo child_index = ( *it )->get_global_index( );
      if ( status_vector[ child_index ] == 0 ) {
        delete ( *it );
        it = children->erase( it );
      } else {
        execute_essential_reduction( *( *it ), status_vector );
        ++it;
      }
    }
    if ( children->size( ) == 0 ) {
      root.delete_children( );
    }
  }  
}

void besthea::mesh::tree_structure::determine_essential_clusters( 
  const lo my_process_id, const scheduling_time_cluster & root, 
  std::vector< char > & status_vector ) const {
  // traverse the tree and add the status of each cluster to the vector in the
  // order of traversal, which corresponds to the order of the indices
  if ( root.get_n_children( ) > 0 ) {
    const std::vector< scheduling_time_cluster* >* children = 
      root.get_children( );
    for ( lou i = 0; i < children->size( ); ++i ) {
      determine_essential_clusters( my_process_id, *( *children )[ i ], 
        status_vector );
    }
  }
  lo current_process_id = root.get_process_id( );
  char cluster_status = ( current_process_id == my_process_id ) ? 1 : 0;
  if ( cluster_status == 1 ) {
    status_vector.push_back( cluster_status );
    //set status of each child to 2 if it is 0
    if ( root.get_n_children( ) > 0 ) {
      const std::vector< scheduling_time_cluster* >* children 
        = root.get_children( );
      for ( lou i = 0; i < children->size( ); ++i ) {
        lo child_index = ( *children )[ i ]->get_global_index( );
        if ( status_vector[ child_index ] == 0 ) {
          status_vector[ child_index ] = 2;
        }
      }
    }
    //set status of each cluster in the interaction list to 2 if it is 0
    if ( root.get_interaction_list( ) != nullptr) {
      const std::vector< scheduling_time_cluster* >* interaction_list 
        = root.get_interaction_list( );
      for ( lou i = 0; i < interaction_list->size( ); ++i ) {
        lo src_index = ( *interaction_list )[ i ]->get_global_index( );
        if ( status_vector[ src_index ] == 0 ) {
          status_vector[ src_index ] = 2;
        }
      }
    }
    // if root is a leaf set status of all clusters in the nearfield from 0 to 2
    if ( root.get_n_children( ) == 0 ) {
      const std::vector< scheduling_time_cluster* >* nearfield 
        = root.get_nearfield( );
      for ( lou i = 0; i < nearfield->size( ); ++i ) {
        lo src_index = ( *nearfield )[ i ]->get_global_index( );
        if ( status_vector[ src_index ] == 0 ) {
          status_vector[ src_index ] = 2;
        }
      }
    }
  } else {
    // if the status of a cluster in the interaction list is 1 set status to 2
    if ( root.get_interaction_list( ) != nullptr) {
      const std::vector< scheduling_time_cluster* >* interaction_list 
        = root.get_interaction_list( );
      for ( lou i = 0; i < interaction_list->size( ); ++i ) {
        lo src_index = ( *interaction_list )[ i ]->get_global_index( );
        if ( status_vector[ src_index ] == 1 ) {
          cluster_status = 2;
        }
      }
    }
    status_vector.push_back( cluster_status );
  }
}

void besthea::mesh::tree_structure::
  determine_levelwise_output_string( const lo digits, 
  const bool print_process_ids, scheduling_time_cluster * root, 
  std::vector< std::string > & levelwise_output_strings ) const {
  if ( root != nullptr ) {
    lo current_level = root->get_level( );
    lo output_val;
    if ( print_process_ids ) {
      output_val = root->get_process_id( );
    } else {
      output_val = root->get_global_index( );
    }
    // compute the number of digits the output_val needs
    lo id_digits = 1;
    if ( output_val > 0 ) {
      id_digits = ( lo ) ceil( log10( output_val + 1 ) );
    } else if ( output_val < 0 ) {
      id_digits = 2;
    }
    // construct the string for the cluster and append it to the output string
    // at the appropriate level
    lo n_digits_level = ( 1 << ( _levels - 1 - current_level ) ) * digits;
    lo n_trailing_whitespace = n_digits_level - id_digits;
    std::string next_string = std::to_string( output_val )
      + std::string( n_trailing_whitespace, ' ');
    levelwise_output_strings[ current_level ] += next_string;
    // proceed for the children
    if ( root->get_n_children( ) == 2 ) {
      // call the routine for each child
      auto children = root->get_children( );
      for ( auto child : *children ) {
        determine_levelwise_output_string( digits, print_process_ids, child, 
          levelwise_output_strings );
      }
    }
    else if ( root->get_n_children( ) == 1 ) {
      // call the routine for the existing child and add / and whitespaces for
      // the non-existing child
      auto child = ( * root->get_children( ) )[ 0 ];
      sc parent_center = root->get_center( );
      sc child_center = child->get_center( );
      std::vector< bool > child_exists( 2, false );
      if ( child_center < parent_center ) {
        child_exists[ 0 ] = true;
      } else {
        child_exists[ 1 ] = true;
      }
      for ( lou i = 0; i < 2; ++i ) {
        if ( child_exists[ i ] == true ) {
          determine_levelwise_output_string( digits, print_process_ids, child, 
            levelwise_output_strings );
        } else {
          // add / and whitespaces for non-existing clusters starting from
          // the non-existing leaf to the bottom of the tree
          lo n_children = 1;
          lo n_digits_level_mod = n_digits_level;
          for ( lo level = current_level + 1; level < _levels; ++level ) {
            n_digits_level_mod /= 2;
            std::string child_string 
              = '/' + std::string(n_digits_level_mod - 1, ' ');
            for ( lo child = 0; child < n_children; ++child ) {
              levelwise_output_strings[ level ] += child_string; 
            }
            n_children *= 2;
          }
        }
      }
    } else {
      // add / and whitespaces for non-existing clusters starting from the
      // non-existing leaves to the bottom of the tree
      lo n_children = 1;
      for ( lo level = current_level + 1; level < _levels; ++level ) {
        n_children *= 2;
        n_digits_level /= 2;
        std::string child_string = '/' + std::string(n_digits_level - 1, ' ');
        for ( lo child = 0; child < n_children; ++child ) {
          levelwise_output_strings[ level ] += child_string; 
        }
      }
    }
  }
}

bool besthea::mesh::tree_structure::compare_clusters_bottom_up_right_2_left( 
  scheduling_time_cluster* first, scheduling_time_cluster* second ) {
  bool ret_val = false;
  if ( first->get_level( ) > second->get_level( ) ) {
    ret_val = true;
  } 
  else if ( first->get_level( ) == second->get_level( ) &&
            first->get_global_index( ) > second->get_global_index( ) ) {
    ret_val = true;
  }
  return ret_val;
}

bool besthea::mesh::tree_structure::compare_clusters_top_down_right_2_left( 
  scheduling_time_cluster* first, scheduling_time_cluster* second ) {
  bool ret_val = false;
  if ( first->get_level( ) < second->get_level( ) ) {
    ret_val = true;
  } 
  else if ( first->get_level( ) == second->get_level( ) &&
            first->get_global_index( ) > second->get_global_index( ) ) {
    ret_val = true;
  }
  return ret_val;
}