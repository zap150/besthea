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

template < class cluster_type >
void besthea::mesh::tree_structure< cluster_type >::vector_2_tree( 
  const std::vector<char> & tree_vector, cluster_type & root, 
  lou & position ) {
  std::cout << "vector_2_tree: NOT IMPLEMENTED!" << std::endl;
}

//! template specialization
template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  vector_2_tree( const std::vector< char > & tree_vector, 
  besthea::mesh::scheduling_time_cluster & root, lou & position ) {
  using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;
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

//! template specialization
template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  set_process_assignments( const std::vector< lo > process_assignments, 
  besthea::mesh::scheduling_time_cluster & root, lou & position ) {
  using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;
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

//! template specialization
template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  set_indices( besthea::mesh::scheduling_time_cluster & root, 
  std::vector< lo > & index_counters ) {
  lo level = root.get_level( );
  root.set_index( index_counters[ level ] );
  index_counters[ level ] += 1;
  if ( root.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster* >* children = root.get_children( );
    for ( lou i = 0; i < children->size( ); ++i ) {
      set_indices( *( *children )[ i ], index_counters );
    }
  }
}

//! template specialization
template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  set_nearfield_and_interaction_list( 
  besthea::mesh::scheduling_time_cluster & root ) {
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
            } else {
              root.add_to_nearfield( src_cluster );
            }
          }
        }
      }
    }
  }
  if ( root.get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster* >* children = root.get_children( ); 
    for ( lou i = 0; i < children->size( ); ++i ){
      set_nearfield_and_interaction_list( *( *children )[ i ] );
    }
  }
}

template < class cluster_type >
besthea::mesh::tree_structure< cluster_type >::tree_structure( 
  const std::string & filename, const sc start_time, const sc end_time )
  : _levels( 0 ) {
    std::cout << "Constructor NOT IMPLEMENTED!" << std::endl;
}

//! template specialization for constructor
template <>
besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  tree_structure( const std::string & filename, const sc start_time, 
    const sc end_time )
  : _levels( 0 ) {
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
  std::vector< lo > index_counters( _levels + 1, 0 );
  set_indices( *_root, index_counters );
  set_nearfield_and_interaction_list( *_root );
  collect_leaves( *_root );
}

//! template specialization
template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  load_process_assignments( const std::string & filename ) {
  std::vector< lo > process_assignments = 
    read_vector_from_bin_file< lo >( filename );
  _root->set_process_id( process_assignments[ 0 ] );
  lou position = 1;
  set_process_assignments( process_assignments, *_root, position );
}

//! template specialization
template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  determine_essential_clusters( const lo my_id, 
  const besthea::mesh::scheduling_time_cluster & root, 
  std::vector< std::vector< char > > & levelwise_status ) {
  if ( root.get_n_children( ) > 0 ) {
    const std::vector< scheduling_time_cluster* >* children = 
      root.get_children( );
    for ( lou i = 0; i < children->size( ); ++i ) {
      determine_essential_clusters( my_id, *( *children )[ i ], 
        levelwise_status );
    }
  }
  lo current_level = root.get_level( );
  lo current_process_id = root.get_process_id( );
  char cluster_status = ( current_process_id == my_id ) ? 1 : 0;
  if ( cluster_status == 1 ) {
    levelwise_status[ current_level ].push_back( cluster_status );
    //set status of each child to 2 if it is 0
    if ( root.get_n_children( ) > 0 ) {
      const std::vector< scheduling_time_cluster* >* children 
        = root.get_children( );
      for ( lou i = 0; i < children->size( ); ++i ) {
        lo child_index = ( *children )[ i ]->get_index( );
        if ( levelwise_status[ current_level + 1 ][ child_index ] == 0 ) {
          levelwise_status[ current_level + 1 ][ child_index ] = 2;
        }
      }
    }
    //set status of each cluster in the interaction list to 2 if it is 0
    if ( root.get_interaction_list( ) != nullptr) {
      const std::vector< scheduling_time_cluster* >* interaction_list 
        = root.get_interaction_list( );
      for ( lou i = 0; i < interaction_list->size( ); ++i ) {
        lo src_index = ( *interaction_list )[ i ]->get_index( );
        if ( levelwise_status[ current_level ][ src_index ] == 0 ) {
          levelwise_status[ current_level ][ src_index ] = 2;
        }
      }
    }
    // if root is a leaf set status of all clusters in the nearfield from 0 to 2
    if ( root.get_n_children( ) == 0 ) {
      const std::vector< scheduling_time_cluster* >* nearfield 
        = root.get_nearfield( );
      for ( lou i = 0; i < nearfield->size( ); ++i ) {
        lo src_level = ( *nearfield )[ i ]->get_level( );
        lo src_index = ( *nearfield )[ i ]->get_index( );
        if ( levelwise_status[ src_level ][ src_index ] == 0 ) {
          levelwise_status[ src_level ][ src_index ] = 2;
        }
      }
    }
  } else {
    // if the status of a cluster in the interaction list is 1 set status to 2
    if ( root.get_interaction_list( ) != nullptr) {
      const std::vector< scheduling_time_cluster* >* interaction_list 
        = root.get_interaction_list( );
      for ( lou i = 0; i < interaction_list->size( ); ++i ) {
        lo src_index = ( *interaction_list )[ i ]->get_index( );
        if ( levelwise_status[ current_level ][ src_index ] == 1 ) {
          cluster_status = 2;
        }
      }
    }
    levelwise_status[ current_level ].push_back( cluster_status );
  }
}

//! template specialization
template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  execute_essential_reduction( besthea::mesh::scheduling_time_cluster & root,
  std::vector< std::vector< char > >& levelwise_status ) {
  lo current_level = root.get_level( );
  lo current_index = root.get_index( );
  // call routine recursively
  if ( root.get_n_children( ) > 0 ) {
    bool child_included = false;
    const std::vector< scheduling_time_cluster* >* children = 
      root.get_children( );
    for ( lou i = 0; i < children->size( ); ++i ) {
      execute_essential_reduction( *( *children )[ i ], levelwise_status );
      char child_status = levelwise_status[ current_level + 1 ]
        [ ( *children )[ i ]->get_index( ) ];
      if ( child_status > 0 ) {
        child_included = true;
      }
    }
    // change the status of the current cluster if at least one of its children
    // is essential
    if ( child_included ) {
      if ( levelwise_status[ current_level ][ current_index ] == 0 ) {
        levelwise_status[ current_level ][ current_index ] = 2;
      } 
    }
  }
  if ( levelwise_status[ current_level ][ current_index ] > 0 ) {
    // update _levels if necessary
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
      lo src_level = ( *it )->get_level( );
      lo src_index = ( *it )->get_index( );
      if ( levelwise_status[ src_level ][ src_index ] == 0 ) {
        it = nearfield->erase( it );
      } else {
        ++it;
      }
    }
    if ( nearfield->size( ) == 0 ) {
      root.delete_nearfield( );
    }
    // same for the interaction list 
    std::vector< scheduling_time_cluster * > * interaction_list
      = root.get_interaction_list( );
    if ( interaction_list != nullptr ) {
      it = interaction_list->begin( );
      while ( it != interaction_list->end( ) ) {
        lo src_index = ( *it )->get_index( );
        if ( levelwise_status[ current_level ][ src_index ] == 0 ) {
          it = interaction_list->erase( it );
        } else {
          ++it;
        }
      }
      if ( interaction_list->size( ) == 0 ) {
        root.delete_interaction_list( );
      }
    }
    // Same for the children. In addition, if a child is removed from the list
    // of children it is destroyed.
    if ( root.get_n_children( ) > 0 ) {
      std::vector< scheduling_time_cluster * > * children 
        = root.get_children( );
      it = children->begin( );
      while ( it != children->end( ) ) {
        lo child_index = ( *it )->get_index( );
        if ( levelwise_status[ current_level + 1 ][ child_index ] == 0 ) {
          delete ( *it );
          it = children->erase( it );
        } else {
          ++it;
        }
      }
      if ( children->size( ) == 0 ) {
        root.delete_children( );
      }
    }
  }
}

//! template specialization
template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  reduce_2_essential( const lo my_id ) {
  std::vector< std::vector< char > > levelwise_status;
  levelwise_status.resize( _levels );
  determine_essential_clusters( my_id, *_root, levelwise_status );

  // if only the leaves should be kept, which are leaves in the original tree
  // structure the following code can be used
  // ###########################################################################
  // remove all leaves with status 0 from _leaves
  // std::vector< scheduling_time_cluster* > new_leaves;
  // for ( lou i = 0; i < _leaves.size( ); ++ i ) {
  //   lo leaf_level = _leaves[ i ]->get_level( );
  //   lo leaf_index = _leaves[ i ]->get_index( );
  //   if ( levelwise_status[ leaf_level ][ leaf_index ] > 0 ) {
  //     new_leaves.push_back( _leaves[ i ] );
  //   }
  // }
  // new_leaves.shrink_to_fit( );
  // _leaves = std::move( new_leaves );
  // ###########################################################################

  // traverse the tree structure and clean up interaction lists and nearfields 
  // of essential clusters (by eliminating non-essential clusters in the lists)
  std::vector< lo > levelwise_counters( _levels, 0 );
  // set _levels to 0; the correct new value is set by 
  // execute_essential_reduction
  _levels = 0;
  execute_essential_reduction( *_root, levelwise_status );
  // execute_essential_reduction( _root, levelwise_status );
  // reset indices of clusters appropriately
  std::vector< lo > index_counters( _levels + 1, 0 );
  set_indices( *_root, index_counters );
  // reset leaves of tree structure appropriately
  _leaves.clear( );
  collect_leaves( *_root );
}

template < class cluster_type >
std::vector< char > besthea::mesh::tree_structure< cluster_type >::
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

template < class cluster_type >
void besthea::mesh::tree_structure< cluster_type >::print_tree_structure( 
  const std::string & filename ) const
{
  write_vector_to_bin_file( compute_tree_structure( ), filename );
}

template < class cluster_type >
void besthea::mesh::tree_structure< cluster_type >::tree_2_vector( 
  const cluster_type & root, std::vector<char> & tree_vector ) const {
  // get the children of root and determine if they are leaves or not 
  // WARNING: it is assumed that root always has two children; this assumption
  // is reasonable if the method is called for a non-leaf cluster in the tree,
  // since the tree is a full binary tree by construction (in build tree)
  const std::vector< cluster_type * > * children = root.get_children( );
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

template < class cluster_type >
void besthea::mesh::tree_structure< cluster_type >::collect_leaves( 
  cluster_type & root ) {
  if ( root.get_n_children( ) == 0 ) {
    _leaves.push_back( &root );
  } else {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      collect_leaves( **it );
    }
  }
}

template class besthea::mesh::tree_structure<
  besthea::mesh::scheduling_time_cluster >;