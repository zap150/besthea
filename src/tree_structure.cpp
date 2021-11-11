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

#include "besthea/tree_structure.h"

#include <algorithm>
#include <unordered_set>

besthea::mesh::tree_structure::tree_structure( const std::string & filename,
  const sc start_time, const sc end_time, const lo process_id,
  bool enable_m2t_and_s2l )
  : _levels( 0 ),
    _supports_m2t_and_s2l( enable_m2t_and_s2l ),
    _my_process_id( process_id ) {
  // load tree structure from file
  std::vector< char > tree_vector
    = read_vector_from_bin_file< char >( filename );
  // create tree structure from vector
  if ( ( tree_vector.size( ) > 0 ) && ( tree_vector[ 0 ] != 0 ) ) {
    sc center = 0.5 * ( start_time + end_time );
    sc half_size = 0.5 * ( end_time - start_time );
    _root = new scheduling_time_cluster( center, half_size, nullptr, -1, 0 );
    _levels = 1;
    if ( tree_vector[ 0 ] == 1 ) {
      lou position = 1;
      array_2_tree( tree_vector.data( ), *_root, position );
    }
  } else {
    _root = nullptr;
  }
  // set the global indices of all the clusters in the structure
  _root->set_index( 0 );
  set_indices( *_root );
  // construct the operation lists
  set_cluster_operation_lists( *_root );
  if ( !_supports_m2t_and_s2l ) {
    // m2t lists are always needed for simpler subtree communication in the
    // construction of a distributed space-time cluster tree.
    set_m2t_lists_for_subtree_communication( *_root );
  }

  // determine activity of clusters in upward and downward path of FMM
  determine_cluster_activity( *_root );
  collect_leaves( *_root );
}

besthea::mesh::tree_structure::tree_structure(
  const std::string & structure_file, const std::string & cluster_bounds_file,
  const lo process_id, bool enable_m2t_and_s2l )
  : _levels( 0 ),
    _supports_m2t_and_s2l( enable_m2t_and_s2l ),
    _my_process_id( process_id ) {
  // load tree structure from file
  std::vector< char > tree_vector
    = read_vector_from_bin_file< char >( structure_file );
  // load cluster bounds from file
  std::vector< sc > cluster_bounds
    = read_vector_from_bin_file< sc >( cluster_bounds_file );
  // create tree structure from vector
  if ( ( tree_vector.size( ) > 0 ) && ( tree_vector[ 0 ] != 0 ) ) {
    sc center = 0.5 * ( cluster_bounds[ 0 ] + cluster_bounds[ 1 ] );
    sc half_size = 0.5 * ( cluster_bounds[ 1 ] - cluster_bounds[ 0 ] );
    _root = new scheduling_time_cluster( center, half_size, nullptr, -1, 0 );
    _levels = 1;
    if ( tree_vector[ 0 ] == 1 ) {
      lou position = 1;
      create_tree_from_arrays(
        tree_vector.data( ), cluster_bounds.data( ), *_root, position );
    }
  } else {
    _root = nullptr;
  }
  // set the global indices of all the clusters in the structure
  _root->set_index( 0 );
  set_indices( *_root );
  // construct the operation lists
  set_cluster_operation_lists( *_root );
  if ( !_supports_m2t_and_s2l ) {
    // m2t lists are always needed for simpler subtree communication in the
    // construction of a distributed space-time cluster tree.
    set_m2t_lists_for_subtree_communication( *_root );
  }

  // determine activity of clusters in upward and downward path of FMM
  determine_cluster_activity( *_root );
  collect_leaves( *_root );
}

void besthea::mesh::tree_structure::load_process_assignments(
  const std::string & filename ) {
  std::vector< lo > process_assignments
    = read_vector_from_bin_file< lo >( filename );
  _root->set_process_id( process_assignments[ 0 ] );
  lou position = 1;
  set_process_assignments( process_assignments, *_root, position );
}

void besthea::mesh::tree_structure::assign_slices_to_clusters(
  const std::vector< sc > & slice_nodes ) {
  lou i_start = 0, i_end;
  for ( auto leaf_cluster : _leaves ) {
    i_end = i_start;
    sc right_cluster_bound
      = leaf_cluster->get_center( ) + leaf_cluster->get_half_size( );
    while ( ( i_end < slice_nodes.size( ) - 1 )
      && ( ( slice_nodes[ i_end ] + slice_nodes[ i_end + 1 ] ) * 0.5
        < right_cluster_bound ) ) {
      ++i_end;
    }
    leaf_cluster->set_n_time_slices( i_end - i_start );
    for ( lou i = i_start; i < i_end; ++i ) {
      // todo: use vector< lou > for time slices?
      // (otherwise conversion from lou to lo here, which should be not critical
      // in our application)
      leaf_cluster->add_time_slice( i );
    }
    // set starting index for next cluster
    i_start = i_end;
  }
}

void besthea::mesh::tree_structure::reduce_2_essential( ) {
  determine_essential_clusters( );
  // traverse the tree structure and reduce it by eliminating all
  // non-essential clusters. In addition, correct the nearfield, interaction
  // lists and children of all clusters and reset _levels and _leaves.
  _levels = 0;
  prepare_essential_reduction( *_root );
  execute_essential_reduction( *_root );
  // reset leaves of tree structure appropriately. the global leaf status is
  // not changed.
  _leaves.clear( );
  collect_leaves( *_root );
}

char besthea::mesh::tree_structure::
  determine_downward_path_initial_op_status_recursively(
    scheduling_time_cluster & current_cluster ) {
  char return_value = 0;
  if ( current_cluster.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children
      = current_cluster.get_children( );
    for ( auto child_it = children->begin( ); child_it != children->end( );
          ++child_it ) {
      char child_return_value
        = determine_downward_path_initial_op_status_recursively( **child_it );
      if ( child_return_value > return_value ) {
        return_value = child_return_value;
      }
    }
  } else {
    if ( current_cluster.get_process_id( ) == _my_process_id
      && current_cluster.get_n_associated_leaves( ) > 0 ) {
      return_value = 1;
      // Check if the cluster's left end point is the starting point of the time
      // interval (i.e. the cluster is the leftmost cluster on its level). If
      // this is the case return_value is changed to 2
      lo level = current_cluster.get_level( );
      lo global_index = current_cluster.get_global_index( );
      if ( global_index == ( 1 << level ) - 1 ) {
        return_value = 2;
      }
    }
  }
  current_cluster.set_status_initial_op_downward_path( return_value );
  return return_value;
}

void besthea::mesh::tree_structure::initialize_moment_contributions(
  scheduling_time_cluster & root, lou contribution_size ) {
  if ( root.is_active_in_upward_path( ) ) {
    root.allocate_associated_moments( contribution_size );
  }
  if ( root.get_n_children( ) > 0 ) {
    for ( auto child : *root.get_children( ) ) {
      initialize_moment_contributions( *child, contribution_size );
    }
  }
}

void besthea::mesh::tree_structure::clear_moment_contributions(
  scheduling_time_cluster & root ) {
  root.clear_associated_moments( );
  if ( root.get_n_children( ) > 0 ) {
    for ( auto child : *root.get_children( ) ) {
      clear_moment_contributions( *child );
    }
  }
}

void besthea::mesh::tree_structure::initialize_local_contributions(
  scheduling_time_cluster & root, lou contribution_size ) {
  if ( root.is_active_in_downward_path( ) ) {
    root.allocate_associated_local_contributions( contribution_size );
  }
  if ( root.get_n_children( ) > 0 ) {
    for ( auto child : *root.get_children( ) ) {
      initialize_local_contributions( *child, contribution_size );
    }
  }
}

void besthea::mesh::tree_structure::clear_local_contributions(
  scheduling_time_cluster & root ) {
  root.clear_associated_local_contributions( );
  if ( root.get_n_children( ) > 0 ) {
    for ( auto child : *root.get_children( ) ) {
      clear_local_contributions( *child );
    }
  }
}

void besthea::mesh::tree_structure::initialize_local_contributions_initial_op(
  scheduling_time_cluster & root, lou contribution_size ) {
  if ( root.get_status_in_initial_op_downward_path( ) == 1 ) {
    root.allocate_associated_local_contributions( contribution_size );
  }
  if ( root.get_n_children( ) > 0 ) {
    for ( auto child : *root.get_children( ) ) {
      initialize_local_contributions_initial_op( *child, contribution_size );
    }
  }
}

void besthea::mesh::tree_structure::set_m2t_lists_for_subtree_communication(
  scheduling_time_cluster & current_cluster ) {
  if ( current_cluster.get_n_children( ) == 0 ) {
    if ( current_cluster.get_parent( ) != nullptr ) {
      sc current_center = current_cluster.get_center( );
      // go through parent's nearfield list to determine current_cluster's lists
      std::vector< scheduling_time_cluster * > * parent_nearfield
        = current_cluster.get_parent( )->get_nearfield_list( );
      for ( auto parent_nearfield_cluster : *parent_nearfield ) {
        // ensure that parent's nearfield cluster is not a leaf
        if ( parent_nearfield_cluster->get_n_children( ) > 0 ) {
          // check for potential m2t operations in all relevant child-subtrees
          std::vector< scheduling_time_cluster * > * relevant_clusters
            = parent_nearfield_cluster->get_children( );
          for ( auto current_source : *relevant_clusters ) {
            if ( current_source->get_center( ) < current_center
              && !current_cluster.determine_admissibility( current_source )
              && current_source->get_n_children( ) > 0 ) {
              determine_m2t_list_in_subtree( *current_source, current_cluster );
            }
          }
        }
      }
    }
  } else {
    for ( auto child : *current_cluster.get_children( ) ) {
      set_m2t_lists_for_subtree_communication( *child );
    }
  }
}

void besthea::mesh::tree_structure::determine_m2t_list_in_subtree(
  scheduling_time_cluster & current_source,
  scheduling_time_cluster & target_cluster ) {
  if ( target_cluster.determine_admissibility( &current_source ) ) {
    target_cluster.add_to_m2t_list( &current_source );
  } else {
    // no separation in time; continue tree traversal
    if ( current_source.get_n_children( ) > 0 ) {
      for ( auto child : *current_source.get_children( ) ) {
        determine_m2t_list_in_subtree( *child, target_cluster );
      }
    }
  }
}

void besthea::mesh::tree_structure::update_m2t_and_s2l_lists( ) {
  // first construct an auxiliary map from global indices to cluster pointers
  std::unordered_map< lo, scheduling_time_cluster * >
    global_indices_to_clusters;
  initialize_map_global_index_to_cluster( *_root, global_indices_to_clusters );
  // determine m2t and s2l lists for all clusters in the tree recursively by
  // using the proper routine.
  update_m2t_and_s2l_lists_recursively( *_root, global_indices_to_clusters );
}

void besthea::mesh::tree_structure::init_fmm_lists(
  scheduling_time_cluster & root,
  std::list< scheduling_time_cluster * > & m_list,
  std::list< scheduling_time_cluster * > & m2l_list,
  std::list< scheduling_time_cluster * > & l_list,
  std::list< scheduling_time_cluster * > & m2t_task_list,
  std::list< scheduling_time_cluster * > & s2l_task_list,
  std::list< scheduling_time_cluster * > & n_list ) const {
  // if the cluster is local and active in the upward path add it to the
  // m-list and initialize the appropriate dependency data
  if ( root.get_process_id( ) == _my_process_id
    && root.is_active_in_upward_path( ) ) {
    m_list.push_back( &root );
  }
  // if the current cluster is local and its parent is active in the downward
  // path add the current cluster to the l-list
  if ( root.get_process_id( ) == _my_process_id
    && root.get_parent( )->is_active_in_downward_path( ) ) {
    l_list.push_back( &root );
  }

  // add the cluster to the m2l-list, if it is local and has a non-empty
  // interaction list
  if ( root.get_process_id( ) == _my_process_id
    && root.get_interaction_list( ) != nullptr ) {
    m2l_list.push_back( &root );
  }

  // add the cluster to the m2t-list, if it is local and has a non-empty
  // m2t-list
  if ( root.get_process_id( ) == _my_process_id
    && root.get_m2t_list( ) != nullptr ) {
    m2t_task_list.push_back( &root );
  }

  // add the cluster to the s2l-list, if it is local and has a non-empty
  // s2l-list
  if ( root.get_process_id( ) == _my_process_id
    && root.get_s2l_list( ) != nullptr ) {
    s2l_task_list.push_back( &root );
  }

  // add the cluster to the n-list, if it is associated with spacetime leaf
  // clusters
  if ( root.get_process_id( ) == _my_process_id
    && root.get_n_associated_leaves( ) > 0 ) {
    n_list.push_back( &root );
  }

  // recursive call for all children
  if ( root.get_n_children( ) > 0 ) {
    const std::vector< scheduling_time_cluster * > * children
      = root.get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      init_fmm_lists(
        **it, m_list, m2l_list, l_list, m2t_task_list, s2l_task_list, n_list );
    }
  }
}

std::vector< char > besthea::mesh::tree_structure::compute_tree_structure( )
  const {
  std::vector< char > tree_vector;
  if ( _root == nullptr ) {
    tree_vector.push_back( 0 );
  } else if ( _root->get_n_children( ) == 0 ) {
    tree_vector.push_back( 2 );
  } else {
    tree_vector.push_back( 1 );
    _root->append_tree_structure_vector_recursively( tree_vector );
  }
  return tree_vector;
}

void besthea::mesh::tree_structure::print_tree_structure(
  const std::string & filename ) const {
  write_vector_to_bin_file( compute_tree_structure( ), filename );
}

void besthea::mesh::tree_structure::print_tree_human_readable(
  const lo digits, const bool print_process_ids ) const {
  if ( _levels > 0 ) {
    std::vector< std::string > print_strings;
    print_strings.reserve( _levels );
    // start each line with the level
    for ( lo i = 0; i < _levels; ++i ) {
      print_strings.push_back( std::to_string( i ) + ": " );
    }
    // determine the strings for the levelwise output
    determine_levelwise_output_string(
      digits, print_process_ids, _root, print_strings );
    for ( lou i = 0; i < print_strings.size( ); ++i ) {
      std::cout << print_strings[ i ] << std::endl;
    }
  } else {
    std::cout << "number of levels is " << _levels << std::endl;
  }
}

void besthea::mesh::tree_structure::array_2_tree(
  const char * tree_array, scheduling_time_cluster & root, lou & position ) {
  // get the cluster data of root
  lo level = root.get_level( );
  sc center = root.get_center( );
  sc half_size = root.get_half_size( );
  // determine status of the children of root and create them accordingly
  char left_child_status = tree_array[ position++ ];
  char right_child_status = tree_array[ position++ ];
  lo child_counter = 0;
  scheduling_time_cluster * left_cluster = nullptr;
  scheduling_time_cluster * right_cluster = nullptr;
  if ( left_child_status > 0 ) {
    child_counter++;
    left_cluster = new scheduling_time_cluster( center - half_size / 2.0,
      half_size / 2.0, &root, 0, level + 1, root.get_process_id( ) );
    left_cluster->set_index( 2 * root.get_global_index( ) + 1 );
  }
  if ( right_child_status > 0 ) {
    child_counter++;
    right_cluster = new scheduling_time_cluster( center + half_size / 2.0,
      half_size / 2.0, &root, 1, level + 1, root.get_process_id( ) );
    right_cluster->set_index( 2 * root.get_global_index( ) + 2 );
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
    array_2_tree( tree_array, *left_cluster, position );
  } else if ( left_child_status == 2 ) {
    left_cluster->set_global_leaf_status( true );
    if ( level + 2 > _levels ) {
      _levels = level + 2;
    }
  }
  if ( right_child_status == 1 ) {
    array_2_tree( tree_array, *right_cluster, position );
  } else if ( right_child_status == 2 ) {
    right_cluster->set_global_leaf_status( true );
    if ( level + 2 > _levels ) {
      _levels = level + 2;
    }
  }
}

void besthea::mesh::tree_structure::create_tree_from_arrays(
  const char * tree_array, const sc * cluster_bounds,
  scheduling_time_cluster & root, lou & position ) {
  // get the cluster data of root
  lo level = root.get_level( );
  // determine status of the children of root and create them accordingly
  char left_child_status = tree_array[ position ];
  sc left_child_left_bound = cluster_bounds[ 2 * position ];
  sc left_child_right_bound = cluster_bounds[ 2 * position + 1 ];
  position++;
  char right_child_status = tree_array[ position ];
  sc right_child_left_bound = cluster_bounds[ 2 * position ];
  sc right_child_right_bound = cluster_bounds[ 2 * position + 1 ];
  position++;
  lo child_counter = 0;
  scheduling_time_cluster * left_cluster = nullptr;
  scheduling_time_cluster * right_cluster = nullptr;
  if ( left_child_status > 0 ) {
    child_counter++;
    sc center = ( left_child_right_bound + left_child_left_bound ) * 0.5;
    sc half_size = ( left_child_right_bound - left_child_left_bound ) * 0.5;
    left_cluster = new scheduling_time_cluster(
      center, half_size, &root, 0, level + 1, root.get_process_id( ) );
  }
  if ( right_child_status > 0 ) {
    child_counter++;
    sc center = ( right_child_right_bound + right_child_left_bound ) * 0.5;
    sc half_size = ( right_child_right_bound - right_child_left_bound ) * 0.5;
    right_cluster = new scheduling_time_cluster(
      center, half_size, &root, 1, level + 1, root.get_process_id( ) );
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
    create_tree_from_arrays(
      tree_array, cluster_bounds, *left_cluster, position );
  } else if ( left_child_status == 2 ) {
    left_cluster->set_global_leaf_status( true );
    if ( level + 2 > _levels ) {
      _levels = level + 2;
    }
  }
  if ( right_child_status == 1 ) {
    create_tree_from_arrays(
      tree_array, cluster_bounds, *right_cluster, position );
  } else if ( right_child_status == 2 ) {
    right_cluster->set_global_leaf_status( true );
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
  scheduling_time_cluster * left_child = ( *root.get_children( ) )[ 0 ];
  scheduling_time_cluster * right_child = ( *root.get_children( ) )[ 1 ];
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

void besthea::mesh::tree_structure::set_indices(
  scheduling_time_cluster & root ) {
  if ( root.get_n_children( ) > 0 ) {
    lo parent_index = root.get_global_index( );
    std::vector< scheduling_time_cluster * > * children = root.get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      short child_configuration = ( *it )->get_configuration( );
      if ( child_configuration == 0 ) {
        ( *it )->set_index( 2 * parent_index + 1 );
      } else {
        ( *it )->set_index( 2 * parent_index + 2 );
      }
      set_indices( **it );
    }
  }
}

void besthea::mesh::tree_structure::set_cluster_operation_lists(
  scheduling_time_cluster & root ) {
  if ( root.get_parent( ) == nullptr ) {
    root.add_to_nearfield_list( &root );
  } else {
    // traverse parents nearfield to determine nearfield and interaction list
    sc current_center = root.get_center( );
    std::vector< scheduling_time_cluster * > * parent_nearfield
      = root.get_parent( )->get_nearfield_list( );
    for ( lou i = 0; i < parent_nearfield->size( ); ++i ) {
      scheduling_time_cluster * parent_nf_cluster = ( *parent_nearfield )[ i ];
      // check if neighbor of parent is a leaf cluster
      if ( parent_nf_cluster->get_n_children( ) == 0 ) {
        if ( _supports_m2t_and_s2l ) {
          // check whether to add the cluster to the s2l or nearfield list
          if ( root.determine_admissibility( parent_nf_cluster ) ) {
            if ( root.get_process_id( ) == _my_process_id
              && parent_nf_cluster->is_global_leaf( ) ) {
              // s2l lists are only needed and constructed for local clusters.
              // Only global leaves are added to s2l lists.
              root.add_to_s2l_list( parent_nf_cluster );
            }
          } else {
            root.add_to_nearfield_list( parent_nf_cluster );
          }
        } else {
          // in case that s2l operations are not used, add the cluster directly
          // to the nearfield.
          root.add_to_nearfield_list( parent_nf_cluster );
        }
      } else {
        // check admissibility of all children
        std::vector< scheduling_time_cluster * > * relevant_clusters
          = ( *parent_nearfield )[ i ]->get_children( );
        for ( lou j = 0; j < relevant_clusters->size( ); ++j ) {
          scheduling_time_cluster * src_cluster = ( *relevant_clusters )[ j ];
          if ( src_cluster == &root ) {
            root.add_to_nearfield_list( src_cluster );
          } else if ( src_cluster->get_center( ) < current_center ) {
            if ( root.determine_admissibility( src_cluster ) ) {
              root.add_to_interaction_list( src_cluster );
              src_cluster->add_to_send_list( &root );
            } else {
              // If root is not a leaf in the global tree structure, add
              // src_cluster to the nearfield. Otherwise traverse src_cluster's
              // subtree to update the nearfield and m2t list of root.
              if ( !root.is_global_leaf( ) ) {
                root.add_to_nearfield_list( src_cluster );
              } else {
                // only for global leaves we have to determine the correct
                // operation lists by traversing the subtree of src_cluster
                determine_operation_lists_in_subtree( *src_cluster, root );
              }
            }
          }
        }
      }
    }
  }
  if ( root.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children = root.get_children( );
    for ( lou i = 0; i < children->size( ); ++i ) {
      set_cluster_operation_lists( *( *children )[ i ] );
    }
  }
}

void besthea::mesh::tree_structure::determine_operation_lists_in_subtree(
  scheduling_time_cluster & current_cluster,
  scheduling_time_cluster & target_cluster ) {
  bool continue_search = true;
  if ( _supports_m2t_and_s2l ) {
    if ( target_cluster.determine_admissibility( &current_cluster ) ) {
      target_cluster.add_to_m2t_list( &current_cluster );
      if ( current_cluster.get_process_id( ) == _my_process_id ) {
        current_cluster.add_to_diagonal_send_list( &target_cluster );
      }
      continue_search = false;
    }
  }
  if ( continue_search ) {
    if ( current_cluster.get_n_children( ) == 0 ) {
      target_cluster.add_to_nearfield_list( &current_cluster );
    } else {
      std::vector< scheduling_time_cluster * > * children
        = current_cluster.get_children( );
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        determine_operation_lists_in_subtree( **it, target_cluster );
      }
    }
  }
}

void besthea::mesh::tree_structure::initialize_map_global_index_to_cluster(
  scheduling_time_cluster & current_cluster,
  std::unordered_map< lo, scheduling_time_cluster * > &
    global_index_to_cluster ) const {
  lo current_global_index = current_cluster.get_global_index( );
  global_index_to_cluster.insert( { current_global_index, &current_cluster } );
  if ( current_cluster.get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster.get_children( ) ) {
      initialize_map_global_index_to_cluster( *child, global_index_to_cluster );
    }
  }
}

void besthea::mesh::tree_structure::update_m2t_and_s2l_lists_recursively(
  scheduling_time_cluster & current_cluster,
  const std::unordered_map< lo, scheduling_time_cluster * > &
    global_index_to_cluster ) {
  // m2t and s2l lists have to be updated only for local non-leaf clusters
  if ( current_cluster.get_n_children( ) > 0 ) {
    if ( current_cluster.get_process_id( ) == _my_process_id ) {
      const std::vector< general_spacetime_cluster * > *
        associated_spacetime_clusters
        = current_cluster.get_associated_spacetime_clusters( );
      if ( associated_spacetime_clusters != nullptr ) {
        std::unordered_set< lo > m2t_list_indices;
        std::unordered_set< lo > s2l_list_indices;
        // for all associated space-time clusters check if they have a non-empty
        // m2t- or s2l-list; if yes, update the time clusters list accordingly.
        for ( auto st_cluster : *associated_spacetime_clusters ) {
          // consider the m2t list first
          std::vector< general_spacetime_cluster * > * current_m2t_list
            = st_cluster->get_m2t_list( );
          if ( current_m2t_list != nullptr ) {
            for ( auto m2t_source_cluster : *current_m2t_list ) {
              m2t_list_indices.insert(
                m2t_source_cluster->get_global_time_index( ) );
            }
          }
          // now consider the s2l list
          std::vector< general_spacetime_cluster * > * current_s2l_list
            = st_cluster->get_s2l_list( );
          if ( current_s2l_list != nullptr ) {
            for ( auto s2l_source_cluster : *current_s2l_list ) {
              s2l_list_indices.insert(
                s2l_source_cluster->get_global_time_index( ) );
            }
          }
        }
        // use the sets of m2t and s2l cluster indices and the map from global
        // indices to temporal clusters to fill the m2t and s2l list of the
        // current time cluster.
        if ( !m2t_list_indices.empty( ) ) {
          for ( auto m2t_source_cluster_index : m2t_list_indices ) {
            scheduling_time_cluster * current_source
              = global_index_to_cluster.at( m2t_source_cluster_index );
            current_cluster.add_to_m2t_list( current_source );
            current_source->add_to_diagonal_send_list( &current_cluster );
          }
        }
        if ( !s2l_list_indices.empty( ) ) {
          for ( auto s2l_source_cluster_index : s2l_list_indices ) {
            current_cluster.add_to_s2l_list(
              global_index_to_cluster.at( s2l_source_cluster_index ) );
          }
        }
      }
    }
    // call the routine recursively for all children.
    for ( auto child : *current_cluster.get_children( ) ) {
      update_m2t_and_s2l_lists_recursively( *child, global_index_to_cluster );
    }
  }
}

void besthea::mesh::tree_structure::determine_clusters_to_refine(
  scheduling_time_cluster * root,
  std::unordered_map< lo, bool > & refine_map ) const {
  if ( root->get_n_children( ) > 0 ) {
    // if root is handled by process _my_process_id go through its nearfield
    // and mark leaf clusters as clusters which have to be refined.
    if ( root->get_process_id( ) == _my_process_id ) {
      std::vector< scheduling_time_cluster * > * nearfield
        = root->get_nearfield_list( );
      for ( auto it = nearfield->begin( ); it != nearfield->end( ); ++it ) {
        if ( ( *it )->get_n_children( ) == 0 ) {
          refine_map[ ( *it )->get_global_index( ) ] = true;
        }
      }
    }
    std::vector< scheduling_time_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      determine_clusters_to_refine( *it, refine_map );
    }
  } else {
    if ( root->get_process_id( ) == _my_process_id ) {
      // mark root and all leaf clusters in its nearfield as clusters which
      // have to be refined. note: by construction all clusters in the
      // nearfield are leaves, and root itself is contained in its own
      // nearfield.
      std::vector< scheduling_time_cluster * > * nearfield
        = root->get_nearfield_list( );
      for ( auto it = nearfield->begin( ); it != nearfield->end( ); ++it ) {
        // updates the value of the cluster in the refine_map,
        // or creates a new entry
        refine_map[ ( *it )->get_global_index( ) ] = true;
      }
    } else {
      // if clusters in the nearfield of root at root's level or their
      // descendants are local, mark root as cluster which has to be refined.
      std::vector< scheduling_time_cluster * > nearfield;
      // determine the nearfield on the level of root (or lower levels) by
      // considering the children of the clusters in the nearfield of its
      // parent
      std::vector< scheduling_time_cluster * > * parent_nearfield
        = root->get_parent( )->get_nearfield_list( );
      for ( auto it_nf_par = parent_nearfield->begin( );
            it_nf_par != parent_nearfield->end( ); ++it_nf_par ) {
        if ( ( *it_nf_par )->get_n_children( ) == 0 ) {
          nearfield.push_back( *it_nf_par );
        } else {
          std::vector< scheduling_time_cluster * > * relevant_clusters
            = ( *it_nf_par )->get_children( );
          for ( auto it_rel = relevant_clusters->begin( );
                it_rel != relevant_clusters->end( ); ++it_rel ) {
            if ( ( *it_rel )->get_center( ) < root->get_center( )
              && ( !root->determine_admissibility( *it_rel ) ) ) {
              nearfield.push_back( *it_rel );
            }
          }
        }
      }
      bool nearfield_contains_local = false;
      auto it = nearfield.begin( );
      while ( !nearfield_contains_local && it != nearfield.end( ) ) {
        nearfield_contains_local = subtree_contains_local_cluster( *it );
        ++it;
      }
      if ( nearfield_contains_local ) {
        refine_map[ root->get_global_index( ) ] = true;
      } else {
        refine_map[ root->get_global_index( ) ] = false;
      }
    }
  }
}

bool besthea::mesh::tree_structure::subtree_contains_local_cluster(
  const scheduling_time_cluster * root ) const {
  bool is_local = ( root->get_process_id( ) == _my_process_id );
  if ( !is_local && root->get_n_children( ) > 0 ) {
    const std::vector< scheduling_time_cluster * > * children
      = root->get_children( );
    auto it = children->begin( );
    while ( !is_local && it != children->end( ) ) {
      is_local = subtree_contains_local_cluster( *it );
      ++it;
    }
  }
  return is_local;
}

void besthea::mesh::tree_structure::determine_cluster_communication_lists(
  scheduling_time_cluster * root,
  std::set< std::pair< lo, scheduling_time_cluster * >,
    compare_pairs_of_process_ids_and_scheduling_time_clusters > &
    subtree_send_list,
  std::set< std::pair< lo, scheduling_time_cluster * >,
    compare_pairs_of_process_ids_and_scheduling_time_clusters > &
    subtree_receive_list,
  std::set< std::pair< lo, scheduling_time_cluster * >,
    compare_pairs_of_process_ids_and_scheduling_time_clusters > &
    leaf_info_send_list,
  std::set< std::pair< lo, scheduling_time_cluster * >,
    compare_pairs_of_process_ids_and_scheduling_time_clusters > &
    leaf_info_receive_list ) const {
  if ( root->get_n_children( ) > 0 ) {
    for ( auto it_child : *( root->get_children( ) ) ) {
      determine_cluster_communication_lists( it_child, subtree_send_list,
        subtree_receive_list, leaf_info_send_list, leaf_info_receive_list );
    }
  }
  // only leaf clusters which are leaves in the global tree structure are
  // relevant.
  if ( root->is_global_leaf( ) ) {
    if ( root->get_process_id( ) == _my_process_id ) {
      // go through the nearfield of the cluster. If it contains a non-local
      // cluster add root together with the process id of the non-local
      // cluster to the subtree send list.
      for ( auto it_nf : *( root->get_nearfield_list( ) ) ) {
        lo nf_process_id = it_nf->get_process_id( );
        if ( nf_process_id != _my_process_id ) {
          subtree_send_list.insert( { nf_process_id, root } );
        }
      }
      // go through the m2t list and do the same
      if ( root->get_m2t_list( ) != nullptr ) {
        for ( auto it_m2t : *( root->get_m2t_list( ) ) ) {
          lo m2t_process_id = it_m2t->get_process_id( );
          if ( m2t_process_id != _my_process_id ) {
            subtree_send_list.insert( { m2t_process_id, root } );
          }
        }
      }

      // go through the send list of the cluster. If it contains a non-local
      // cluster add root together with the process id of the non-local
      // cluster to the leaf info send list NOTE: if root is falsely added
      // (because its mesh is available for the receiving process, it is
      // removed from the list below)
      if ( root->get_send_list( ) != nullptr ) {
        for ( auto it_sl : *( root->get_send_list( ) ) ) {
          lo sl_process_id = it_sl->get_process_id( );
          if ( sl_process_id != _my_process_id ) {
            leaf_info_send_list.insert( { sl_process_id, root } );
          }
        }
      }
    } else {
      lo root_process_id = root->get_process_id( );
      // if root's mesh is not available, check whether there is a local
      // cluster in its send list. if yes, add root with its owning process to
      // the leaf info receive list
      if ( !root->mesh_is_available( ) && root->get_send_list( ) != nullptr ) {
        for ( auto it_sl : *( root->get_send_list( ) ) ) {
          lo sl_process_id = it_sl->get_process_id( );
          if ( sl_process_id == _my_process_id ) {
            leaf_info_receive_list.insert( { root_process_id, root } );
          }
        }
      }
      // go through the nearfield of the cluster. If it contains a local
      // cluster add root together with its owning process to the subtree
      // receive list.
      std::vector< scheduling_time_cluster * > * nearfield
        = root->get_nearfield_list( );
      bool added_cluster = false;
      if ( nearfield != nullptr ) {
        auto it_nf = nearfield->begin( );
        while ( it_nf != nearfield->end( ) ) {
          lo nf_process_id = ( *it_nf )->get_process_id( );
          if ( nf_process_id == _my_process_id ) {
            if ( !added_cluster ) {
              subtree_receive_list.insert( { root_process_id, root } );
              added_cluster = true;
            }
            // If the nearfield contains a cluster from the leaf info send
            // list remove this cluster from the list.
            leaf_info_send_list.erase( { root_process_id, *it_nf } );
          }
          ++it_nf;
        }
      }
      // if root's m2t list is not empty and contains a local cluster do the
      // same, if root has not yet been added
      if ( !added_cluster ) {
        std::vector< scheduling_time_cluster * > * m2t_list
          = root->get_m2t_list( );
        if ( m2t_list != nullptr ) {
        }
      }
    }
  }
}

void besthea::mesh::tree_structure::clear_cluster_operation_lists(
  scheduling_time_cluster * root ) {
  // call the routine recursively for non-leaf clusters which were in the tree
  // before the expansion
  if ( root->get_nearfield_list( ) != nullptr )
    root->get_nearfield_list( )->clear( );
  if ( root->get_interaction_list( ) != nullptr )
    root->get_interaction_list( )->clear( );
  if ( root->get_send_list( ) != nullptr )
    root->get_send_list( )->clear( );
  if ( root->get_m2t_list( ) != nullptr )
    root->delete_m2t_list( );
  if ( root->get_s2l_list( ) != nullptr )
    root->delete_s2l_list( );
  if ( root->get_diagonal_send_list( ) != nullptr )
    root->delete_diagonal_send_list( );
  if ( root->get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children = root->get_children( );
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      clear_cluster_operation_lists( *it );
    }
  }
}

void besthea::mesh::tree_structure::clear_lists_of_associated_clusters(
  scheduling_time_cluster & current_cluster ) {
  if ( current_cluster.get_associated_spacetime_clusters( ) != nullptr ) {
    current_cluster.get_associated_spacetime_clusters( )->clear( );
  }
  if ( current_cluster.get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster.get_children( ) ) {
      clear_lists_of_associated_clusters( *child );
    }
  }
}

void besthea::mesh::tree_structure::determine_cluster_activity(
  scheduling_time_cluster & root ) {
  // check if cluster is active in upward path
  if ( ( root.get_send_list( ) != nullptr )
    || ( root.get_diagonal_send_list( ) != nullptr )
    || ( root.get_parent( ) != nullptr
      && root.get_parent( )->is_active_in_upward_path( ) ) ) {
    root.set_active_upward_path_status( true );
  }
  // check if cluster is active in downward path
  if ( ( root.get_interaction_list( ) != nullptr )
    || ( root.get_s2l_list( ) != nullptr )
    || ( root.get_parent( ) != nullptr
      && root.get_parent( )->is_active_in_downward_path( ) ) ) {
    root.set_active_downward_path_status( true );
  }
  // check if cluster is a leaf and call the routine recursively if not
  if ( root.get_n_children( ) > 0 ) {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      determine_cluster_activity( **it );
    }
  }
}

void besthea::mesh::tree_structure::prepare_essential_reduction(
  scheduling_time_cluster & root ) {
  // call routine recursively
  if ( root.get_n_children( ) > 0 ) {
    const std::vector< scheduling_time_cluster * > * children
      = root.get_children( );
    for ( auto it : *children ) {
      prepare_essential_reduction( *it );
    }
  }
  if ( root.get_essential_status( ) > 0 ) {
    // update _levels if necessary
    lo current_level = root.get_level( );
    if ( current_level + 1 > _levels ) {
      _levels = current_level + 1;
    }
    // check if the nearfield contains clusters which are not essential and
    // remove them in the affirmative case.
    std::vector< scheduling_time_cluster * > * nearfield
      = root.get_nearfield_list( );
    auto it = nearfield->begin( );
    while ( it != nearfield->end( ) ) {
      if ( ( *it )->get_essential_status( ) == 0 ) {
        it = nearfield->erase( it );
      } else {
        ++it;
      }
    }
    // same for the send list. If the resulting send list is empty, delete it.
    std::vector< scheduling_time_cluster * > * send_list
      = root.get_send_list( );
    if ( send_list != nullptr ) {
      it = send_list->begin( );
      while ( it != send_list->end( ) ) {
        if ( ( *it )->get_essential_status( ) == 0 ) {
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
        if ( ( *it )->get_essential_status( ) == 0 ) {
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
  scheduling_time_cluster & root ) {
  // Recursively traverse the tree structure and delete non-essential clusters
  if ( root.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children = root.get_children( );
    auto it = children->begin( );
    while ( it != children->end( ) ) {
      if ( ( *it )->get_essential_status( ) == 0 ) {
        delete ( *it );
        it = children->erase( it );
      } else {
        execute_essential_reduction( *( *it ) );
        ++it;
      }
    }
    if ( children->size( ) == 0 ) {
      root.delete_children( );
    }
  }
}

void besthea::mesh::tree_structure::determine_essential_clusters( ) const {
  // traverse the tree twice to determine the essential clusters correctly.
  determine_essential_clusters_first_traversal( *_root );
  determine_essential_clusters_second_traversal( *_root );
}

void besthea::mesh::tree_structure::
  determine_essential_clusters_first_traversal(
    scheduling_time_cluster & current_cluster ) const {
  // traverse the tree and set the status of clusters starting from the leaves
  if ( current_cluster.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children
      = current_cluster.get_children( );
    for ( lou i = 0; i < children->size( ); ++i ) {
      determine_essential_clusters_first_traversal( *( *children )[ i ] );
    }
  }
  lo current_process_id = current_cluster.get_process_id( );
  char current_cluster_status
    = ( current_process_id == _my_process_id ) ? (char) 3 : (char) 0;
  if ( current_cluster_status == 3 ) {
    // set status of each child to 1 if it is 0 (child is essential in time
    // tree)
    if ( current_cluster.get_n_children( ) > 0 ) {
      const std::vector< scheduling_time_cluster * > * children
        = current_cluster.get_children( );
      for ( auto it : *children ) {
        if ( it->get_essential_status( ) == 0 ) {
          it->set_essential_status( 1 );
        }
      }
    }
    // set status of each cluster in the interaction list to 2 if it is smaller
    // (cluster in interaction list is essential in time and space-time trees)
    if ( current_cluster.get_interaction_list( ) != nullptr ) {
      const std::vector< scheduling_time_cluster * > * interaction_list
        = current_cluster.get_interaction_list( );
      for ( auto it : *interaction_list ) {
        if ( it->get_essential_status( ) < 2 ) {
          it->set_essential_status( 2 );
        }
      }
    }
    // set status of each cluster in the m2t list to 2 if it is smaller
    // (cluster in m2t list is essential in time and space-time trees)
    if ( current_cluster.get_m2t_list( ) != nullptr ) {
      const std::vector< scheduling_time_cluster * > * current_m2t_list
        = current_cluster.get_m2t_list( );
      for ( auto it : *current_m2t_list ) {
        if ( it->get_essential_status( ) < 2 ) {
          it->set_essential_status( 2 );
        }
      }
    }
    // set status of each cluster in the s2l list to 2 if it is smaller
    // (cluster in s2l list is essential in time and space-time trees)
    if ( current_cluster.get_s2l_list( ) != nullptr ) {
      const std::vector< scheduling_time_cluster * > * current_s2l_list
        = current_cluster.get_s2l_list( );
      for ( auto it : *current_s2l_list ) {
        if ( it->get_essential_status( ) < 2 ) {
          it->set_essential_status( 2 );
        }
      }
    }
    // go through all clusters J in the nearfield of the current cluster. if
    // either J or current_cluster is a leaf, set status of J to 2, if it is
    // smaller than that. (nearfield clusters are essential in time and
    // space-time tree)
    bool current_cluster_is_leaf = current_cluster.get_n_children( ) == 0;
    const std::vector< scheduling_time_cluster * > * nearfield
      = current_cluster.get_nearfield_list( );
    for ( auto it : *nearfield ) {
      if ( current_cluster_is_leaf || it->get_n_children( ) == 0 ) {
        if ( it->get_essential_status( ) < 2 ) {
          it->set_essential_status( 2 );
        }
      }
    }
  } else {
    // if the status of a cluster in the interaction list is 3 set status to 1
    // NOTE: By construction, clusters in the interaction list are visited
    // earlier during the tree traversal of determine_essential_clusters, so
    // their status is already set!
    if ( current_cluster.get_interaction_list( ) != nullptr ) {
      const std::vector< scheduling_time_cluster * > * interaction_list
        = current_cluster.get_interaction_list( );
      for ( auto it : *interaction_list ) {
        if ( it->get_essential_status( ) == 3 ) {
          current_cluster_status = 1;
        }
      }
    }
  }
  // set the status of current_cluster to the determined status
  current_cluster.set_essential_status( current_cluster_status );
}

void besthea::mesh::tree_structure::
  determine_essential_clusters_second_traversal(
    scheduling_time_cluster & current_cluster ) const {
  // traverse the tree and set the status of clusters starting from the leaves
  if ( current_cluster.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children
      = current_cluster.get_children( );
    char max_child_status = 0;
    for ( auto child : *children ) {
      determine_essential_clusters_second_traversal( *child );
      char child_status = child->get_essential_status( );
      if ( child_status > max_child_status ) {
        max_child_status = child_status;
      }
    }
    // Change the status of the current cluster if at least one of its
    // children is essential. The essential status is inherited from the
    // child with the maximal essential status.
    if ( max_child_status > 0
      && current_cluster.get_essential_status( ) < max_child_status ) {
      current_cluster.set_essential_status( max_child_status );
      // if the status of the cluster was set to 3, go through its interaction
      // and send list and update essential status of clusters if necessary.
      if ( max_child_status == 3 ) {
        if ( current_cluster.get_interaction_list( ) != nullptr ) {
          for ( auto source_cluster :
            *current_cluster.get_interaction_list( ) ) {
            if ( source_cluster->get_essential_status( ) == 0 ) {
              source_cluster->set_essential_status( 1 );
              // if we update the status of a cluster in the interaction list,
              // we have to guarantee that its ancestors are in the locally
              // essential tree too. Some of them might not be visited anymore
              // during the tree traversal, so we visit them directly and
              // update their essential status if necessary.
              scheduling_time_cluster * current_parent
                = source_cluster->get_parent( );
              scheduling_time_cluster * current_source = source_cluster;
              while ( current_parent != nullptr
                && current_parent->get_essential_status( )
                  < source_cluster->get_essential_status( ) ) {
                current_parent->set_essential_status(
                  current_source->get_essential_status( ) );
                current_source = current_parent;
                current_parent = current_source->get_parent( );
              }
            }
          }
        }
        if ( current_cluster.get_send_list( ) != nullptr ) {
          for ( auto target_cluster : *current_cluster.get_send_list( ) ) {
            if ( target_cluster->get_essential_status( ) == 0 ) {
              target_cluster->set_essential_status( 1 );
            }
            // the target cluster's ancestors essential status will still be
            // updated in this routine, so we do not have to do it manually.
          }
        }
      }
    }
  }

  // if current_cluster is a leaf and the status of a cluster in the physical
  // nearfield is 3 set current_cluster's status to 1.
  // NOTE: We use that the status of a cluster is set to 3 if one of its
  // descendants status is 3.
  char current_cluster_status = current_cluster.get_essential_status( );
  if ( ( current_cluster.get_n_children( ) == 0 )
    && ( current_cluster_status == 0 ) ) {
    // consider all clusters in the physical nearfield of current_cluster
    // whose level is at most that of current_cluster. If one of them has
    // essential_status 3, set current_cluster's status to 1.

    // first, find the relevant nearfield clusters
    std::vector< scheduling_time_cluster * > relevant_nearfield_clusters;
    std::vector< scheduling_time_cluster * > * parent_nearfield
      = current_cluster.get_parent( )->get_nearfield_list( );
    for ( lou i = 0; i < parent_nearfield->size( ); ++i ) {
      // check if neighbor of parent is a leaf cluster
      if ( ( *parent_nearfield )[ i ]->get_n_children( ) == 0 ) {
        // add a leaf in the nearfield of parent to the nearfield of root
        relevant_nearfield_clusters.push_back( ( *parent_nearfield )[ i ] );
      } else {
        // check admissibility of all children
        std::vector< scheduling_time_cluster * > * relevant_clusters
          = ( *parent_nearfield )[ i ]->get_children( );
        for ( lou j = 0; j < relevant_clusters->size( ); ++j ) {
          scheduling_time_cluster * src_cluster = ( *relevant_clusters )[ j ];
          if ( src_cluster->get_center( ) < current_cluster.get_center( ) ) {
            if ( !current_cluster.determine_admissibility( src_cluster ) ) {
              relevant_nearfield_clusters.push_back( src_cluster );
            }
          }
        }
      }
    }

    // now, update the status of current_cluster if necessary
    lo i = 0;
    if ( relevant_nearfield_clusters.size( ) > 0 ) {
      while ( current_cluster_status == 0
        && i < (lo) relevant_nearfield_clusters.size( ) ) {
        if ( relevant_nearfield_clusters[ i ]->get_essential_status( ) == 3 ) {
          current_cluster_status = 1;
        }
        ++i;
      }
    }
    if ( current_cluster_status != 0 ) {
      current_cluster.set_essential_status( current_cluster_status );
    }
  }
}

void besthea::mesh::tree_structure::determine_levelwise_output_string(
  const lo digits, bool print_process_ids, scheduling_time_cluster * root,
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
      id_digits = (lo) ceil( log10( (double) output_val + 1 ) );
    } else if ( output_val < 0 ) {
      id_digits = 2;
    }
    // construct the string for the cluster and append it to the output string
    // at the appropriate level
    lo n_digits_level = ( 1 << ( _levels - 1 - current_level ) ) * digits;
    lo n_trailing_whitespace = n_digits_level - id_digits;
    std::string next_string = std::to_string( output_val )
      + std::string( n_trailing_whitespace, ' ' );
    levelwise_output_strings[ current_level ] += next_string;
    // proceed for the children
    if ( root->get_n_children( ) == 2 ) {
      // call the routine for each child
      auto children = root->get_children( );
      for ( auto child : *children ) {
        determine_levelwise_output_string(
          digits, print_process_ids, child, levelwise_output_strings );
      }
    } else if ( root->get_n_children( ) == 1 ) {
      // call the routine for the existing child and add / and whitespaces for
      // the non-existing child
      auto child = ( *root->get_children( ) )[ 0 ];
      short child_configuration = child->get_configuration( );
      std::vector< bool > child_exists( 2, false );
      if ( child_configuration == 0 ) {
        child_exists[ 0 ] = true;
      } else {
        child_exists[ 1 ] = true;
      }
      for ( lou i = 0; i < 2; ++i ) {
        if ( child_exists[ i ] == true ) {
          determine_levelwise_output_string(
            digits, print_process_ids, child, levelwise_output_strings );
        } else {
          // add / and whitespaces for non-existing clusters starting from
          // the non-existing leaf to the bottom of the tree
          lo n_children = 1;
          lo n_digits_level_mod = n_digits_level;
          for ( lo level = current_level + 1; level < _levels; ++level ) {
            n_digits_level_mod /= 2;
            std::string child_string
              = '/' + std::string( n_digits_level_mod - 1, ' ' );
            for ( lo child_idx = 0; child_idx < n_children; ++child_idx ) {
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
        std::string child_string = '/' + std::string( n_digits_level - 1, ' ' );
        for ( lo child = 0; child < n_children; ++child ) {
          levelwise_output_strings[ level ] += child_string;
        }
      }
    }
  }
}

bool besthea::mesh::tree_structure::compare_clusters_bottom_up_right_2_left(
  scheduling_time_cluster * first, scheduling_time_cluster * second ) {
  bool ret_val = false;
  if ( first->get_level( ) > second->get_level( ) ) {
    ret_val = true;
  } else if ( first->get_level( ) == second->get_level( )
    && first->get_global_index( ) > second->get_global_index( ) ) {
    ret_val = true;
  }
  return ret_val;
}

bool besthea::mesh::tree_structure::compare_clusters_top_down_right_2_left(
  scheduling_time_cluster * first, scheduling_time_cluster * second ) {
  bool ret_val = false;
  if ( first->get_level( ) < second->get_level( ) ) {
    ret_val = true;
  } else if ( first->get_level( ) == second->get_level( )
    && first->get_global_index( ) > second->get_global_index( ) ) {
    ret_val = true;
  }
  return ret_val;
}

void besthea::mesh::tree_structure::remove_clusters_with_no_association(
  scheduling_time_cluster & current_cluster ) {
  // each scheduling cluster with essential status > 1 should be associated
  // with a group of space-time clusters. Otherwise we can delete it.
  if ( current_cluster.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children
      = current_cluster.get_children( );
    auto it = children->begin( );
    while ( it != children->end( ) ) {
      if ( ( *it )->get_essential_status( ) > 1
        && ( ( *it )->get_associated_spacetime_clusters( ) == nullptr ) ) {
        delete ( *it );
        it = children->erase( it );
      } else {
        remove_clusters_with_no_association( *( *it ) );
        ++it;
      }
    }
    if ( children->size( ) == 0 ) {
      current_cluster.delete_children( );
      current_cluster.set_global_leaf_status( true );
    }
  }
}

void besthea::mesh::tree_structure::count_number_of_contributions(
  scheduling_time_cluster * root, lo & n_moments, lo & n_moments_receive,
  lo & n_local_contributions ) {
  if ( root->is_active_in_upward_path( )
    && root->get_associated_spacetime_clusters( ) != nullptr ) {
    n_moments += root->get_associated_spacetime_clusters( )->size( );
  }
  if ( root->is_active_in_downward_path( )
    && root->get_associated_spacetime_clusters( ) != nullptr ) {
    n_local_contributions
      += root->get_associated_spacetime_clusters( )->size( );
  }
  if ( root->get_n_associated_moment_receive_buffers( ) > 0 ) {
    n_moments_receive += root->get_n_associated_moment_receive_buffers( )
      * root->get_associated_spacetime_clusters( )->size( );
  }
  if ( root->get_n_children( ) > 0 ) {
    for ( auto child : *root->get_children( ) ) {
      count_number_of_contributions(
        child, n_moments, n_moments_receive, n_local_contributions );
    }
  }
}
