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

#include <algorithm>
#include <math.h>

besthea::mesh::time_cluster_tree::time_cluster_tree(
  const temporal_mesh & mesh, lo levels, lo n_min_elems )
  : _mesh( mesh ),
    _levels( levels ),
    _real_max_levels( 0 ),
    _n_min_elems( n_min_elems ),
    _n_max_elems_leaf( 0 ),
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

std::vector< char > besthea::mesh::time_cluster_tree::compute_tree_structure( )
  const {
  std::vector< char > tree_vector;
  if ( _root == nullptr ) {
    tree_vector.push_back( 0 );
  } else if ( _root->get_n_children( ) == 0 ) {
    tree_vector.push_back( 2 );
  } else {
    tree_vector.push_back( 1 );
    tree_2_vector( *_root, tree_vector );
  }
  return tree_vector;
}

std::vector< lo > besthea::mesh::time_cluster_tree::compute_process_assignments(
  const lo n_processes, const lo strategy ) const {
  // determine the minimal level of all leaf clusters
  // for the assignment of processes only the clusters up to this level are
  // considered
  lo trunc_level = _leaves[ 0 ]->get_level( );
  for ( lou i = 1; i < _leaves.size( ); ++i ) {
    if ( _leaves[ i ]->get_level( ) < trunc_level ) {
      trunc_level = _leaves[ i ]->get_level( );
    }
  }
  // determine threshold level, starting from which every process gets at least
  // one cluster
  lo thresh_level = (lo) ceil( log2( n_processes ) );
  thresh_level = ( thresh_level < 2 ) ? 2 : thresh_level;

  if ( trunc_level <= 1 ) {
    std::cout << "Error: Temporal cluster tree is too coarse!" << std::endl;
    return std::vector< lo >( 1, -1 );
  } else if ( thresh_level > trunc_level ) {
    std::cout << "Error: The number of processes is higher than the number of "
              << "clusters at the level of the earliest leaf!" << std::endl;
    return std::vector< lo >( 1, -1 );
  } else {
    // Create a vector to store the assignment in a levelwise format.
    // Let k = ceil(log_2(n_processes)). In the first 2 * n_processes
    // entries of the vector the processes assigned to the clusters at level 2
    // up to (k-1) are given, cluster after cluster (from left to right), level
    // after level. Starting at entry number 2 * n_processes there are
    // n_processes entries per level which indicate how many clusters are
    // assigned to each process (in a consecutive assignment from left to
    // right).
    std::vector< lo > levelwise_assignment(
      ( 3 + trunc_level - thresh_level ) * n_processes, 0 );
    // use a vector to keep track of the clusters assigned to each process
    // the vector contains pairs of the form (id, number of assigned clusters)
    std::vector< std::pair< lo, lo > > clusters_per_process;
    clusters_per_process.reserve( n_processes );

    // assign the clusters at trunc_level
    lo n_clusters = 1 << trunc_level;  // clusters at current level
    lo n_clusters_process = n_clusters / n_processes;
    lo n_clusters_remain = n_clusters % n_processes;
    // determine the index to acces the correct position in levelwise_assignment
    lo access_index = ( 2 + trunc_level - thresh_level ) * n_processes;
    if ( n_clusters_remain <= 1 ) {
      for ( lo i = 0; i < n_processes; ++i ) {
        levelwise_assignment[ access_index + i ] = n_clusters_process;
      }
      // if there is a remainder give the first process an additional cluster
      if ( n_clusters_remain == 1 ) {
        levelwise_assignment[ access_index ]++;
      }
    } else {
      // distribute the remaining clusters such that processes with different
      // cluster numbers are separated
      levelwise_assignment[ access_index ] = n_clusters_process + 1;
      sc quotient = ( n_processes - 1.0 ) / ( n_clusters_remain - 1.0 );
      sc counter = 0.0;
      for ( lo proc_id = 1; proc_id < n_processes; ++proc_id ) {
        counter += 1;
        if ( counter >= quotient - 1e-8 ) {  // correction due to precision
          levelwise_assignment[ access_index + proc_id ]
            = n_clusters_process + 1;
          counter -= quotient;
        } else {
          levelwise_assignment[ access_index + proc_id ] = n_clusters_process;
        }
      }
    }
    for ( lo proc_id = 0; proc_id < n_processes; ++proc_id ) {
      clusters_per_process.push_back( std::pair< lo, lo >(
        proc_id, levelwise_assignment[ access_index + proc_id ] ) );
    }

    // assign the clusters at levels >= thresh_level in such a way that on all
    // levels the numbers of clusters per process is balanced (vary at most by
    // 1) and that the parent of two clusters is always assigned to a process
    // who handles also one of its children
    for ( lo level = trunc_level - 1; level >= thresh_level; --level ) {
      lo old_n_clusters_process = n_clusters_process;
      lo old_access_index = access_index;
      // update the help variables
      n_clusters /= 2;
      n_clusters_process = n_clusters / n_processes;
      access_index -= n_processes;
      // distinguish two cases for the assignment of clusters
      bool case_1 = ( old_n_clusters_process == 2 * n_clusters_process + 1 );
      if ( case_1 ) {
        // assign n_cluster_process + 1 clusters to each process
        for ( lo proc_id = 0; proc_id < n_processes; ++proc_id ) {
          levelwise_assignment[ access_index + proc_id ]
            = n_clusters_process + 1;
        }
        // do the correction
        lo proc_id = 0;
        while ( proc_id < n_processes ) {
          if ( levelwise_assignment[ old_access_index + proc_id ]
            == old_n_clusters_process ) {
            lo id_start = proc_id++;
            while ( levelwise_assignment[ old_access_index + proc_id ]
              != old_n_clusters_process ) {
              proc_id++;
            }
            lo id_mid = ( id_start + proc_id ) / 2;
            levelwise_assignment[ access_index + id_mid ] -= 1;
          }
          proc_id++;
        }
      } else {  // old_n_clusters_process == 2 * n_clusters_process
        // assign n_cluster_process clusters to each process
        for ( lo proc_id = 0; proc_id < n_processes; ++proc_id ) {
          levelwise_assignment[ access_index + proc_id ] = n_clusters_process;
        }
        // distribute the remaining ones
        lo proc_id = 0;
        while ( proc_id < n_processes ) {
          if ( levelwise_assignment[ old_access_index + proc_id ]
            == old_n_clusters_process + 1 ) {
            lo id_start = proc_id++;
            while ( levelwise_assignment[ old_access_index + proc_id ]
              != old_n_clusters_process + 1 ) {
              proc_id++;
            }
            lo id_mid = ( id_start + proc_id ) / 2;
            levelwise_assignment[ access_index + id_mid ] += 1;
          }
          proc_id++;
        }
      }
      for ( lo proc_id = 0; proc_id < n_processes; ++proc_id ) {
        clusters_per_process[ proc_id ].second
          += levelwise_assignment[ access_index + proc_id ];
      }
    }

    // assign a cluster at level thresh_level - 1 ( if > 2) to the same process
    // as its left child
    if ( thresh_level > 2 ) {
      n_clusters /= 2;
      lo old_access_index = access_index;
      access_index = n_clusters - 4;
      // assign the first cluster to process 0
      levelwise_assignment[ access_index ] = 0;
      clusters_per_process[ 0 ].second += 1;
      // assign the other clusters by checking the assignment of their children
      // via the entries in proccess_assignment
      lo n_assigned_clusters = 1;
      lo counter = levelwise_assignment[ old_access_index ];
      for ( lo proc_id = 1; proc_id < n_processes; ++proc_id ) {
        counter += levelwise_assignment[ old_access_index + proc_id ];
        if ( counter > 2 ) {
          levelwise_assignment[ access_index + n_assigned_clusters ] = proc_id;
          clusters_per_process[ proc_id ].second += 1;
          counter -= 2;
          n_assigned_clusters++;
        }
      }
    }
    if ( strategy == 0 ) {
      // assign clusters at all levels l satisfying 2 < l < thresh_level - 1 to
      // processes, which handle less clusters than the others
      for ( lo level = thresh_level - 2; level > 2; --level ) {
        // sort the vector of clusters per process according to the numbers of
        // clusters in a stable manner, and assign clusters to the first few
        // processes
        stable_sort( clusters_per_process.begin( ), clusters_per_process.end( ),
          []( std::pair< lo, lo > a, std::pair< lo, lo > b ) {
            return ( a.second < b.second );
          } );
        n_clusters /= 2;
        access_index = n_clusters - 4;
        for ( lo cluster_id = 0; cluster_id < n_clusters; ++cluster_id ) {
          levelwise_assignment[ access_index + cluster_id ]
            = clusters_per_process[ cluster_id ].first;
          // update counter of the process
          clusters_per_process[ cluster_id ].second += 1;
        }
      }

      if ( thresh_level > 3 ) {
        // level 2: assign the first two clusters to the processes who handle
        // the left child. The remaining two clusters are assigned as above.
        levelwise_assignment[ 0 ] = levelwise_assignment[ access_index ];
        clusters_per_process[ 0 ].second += 1;
        levelwise_assignment[ 1 ] = levelwise_assignment[ access_index + 2 ];
        clusters_per_process[ 2 ].second += 1;
        stable_sort( clusters_per_process.begin( ), clusters_per_process.end( ),
          []( std::pair< lo, lo > a, std::pair< lo, lo > b ) {
            return ( a.second < b.second );
          } );
        levelwise_assignment[ 2 ] = clusters_per_process[ 0 ].first;
        clusters_per_process[ 0 ].second += 1;
        levelwise_assignment[ 3 ] = clusters_per_process[ 1 ].first;
        clusters_per_process[ 1 ].second += 1;
      }
    } else if ( strategy == 1 ) {
      // for the assignment of clusters at level l satisfying the condition
      // 2 < l < thresh_level - 1 split the processes into l almost equal
      // groups and assign cluster j to the process in group j which handles
      // the least clusters among all processes in group j
      for ( lo level = thresh_level - 2; level > 2; --level ) {
        n_clusters /= 2;
        access_index = n_clusters - 4;
        for ( lo cluster_id = 0; cluster_id < n_clusters; ++cluster_id ) {
          lo search_start = ( cluster_id * n_processes ) / n_clusters;
          lo search_end = ( ( cluster_id + 1 ) * n_processes ) / n_clusters;
          lo min_clusters = clusters_per_process[ search_start ].second;
          lo min_proc_id = search_start;
          for ( lo proc_id = search_start + 1; proc_id < search_end;
                ++proc_id ) {
            if ( clusters_per_process[ proc_id ].second < min_clusters ) {
              min_clusters = clusters_per_process[ proc_id ].second;
              min_proc_id = proc_id;
            }
          }
          levelwise_assignment[ access_index + cluster_id ] = min_proc_id;
          clusters_per_process[ min_proc_id ].second += 1;
        }
      }

      if ( thresh_level > 3 ) {
        // level 2: assign the first two clusters to the processes who handle
        // the left child. The remaining two clusters are assigned as above
        levelwise_assignment[ 0 ] = levelwise_assignment[ access_index ];
        clusters_per_process[ 0 ].second += 1;
        levelwise_assignment[ 1 ] = levelwise_assignment[ access_index + 2 ];
        clusters_per_process[ 2 ].second += 1;
        for ( lo cluster_id = 2; cluster_id < 4; ++cluster_id ) {
          lo search_start = ( cluster_id * n_processes ) / 4;
          lo search_end = ( ( cluster_id + 1 ) * n_processes ) / 4;
          lo min_clusters = clusters_per_process[ search_start ].second;
          lo min_proc_id = search_start;
          for ( lo proc_id = search_start + 1; proc_id < search_end;
                ++proc_id ) {
            if ( clusters_per_process[ proc_id ].second < min_clusters ) {
              min_clusters = clusters_per_process[ proc_id ].second;
              min_proc_id = proc_id;
            }
          }
          levelwise_assignment[ cluster_id ] = min_proc_id;
          clusters_per_process[ min_proc_id ].second += 1;
        }
      }
    } else {
      // assign a cluster at a level l < thresh_level - 1  to the same process
      // as its left child
      for ( lo level = thresh_level - 2; level >= 2; --level ) {
        n_clusters /= 2;
        lo old_access_index = access_index;
        access_index = n_clusters - 4;
        for ( lo cluster_id = 0; cluster_id < n_clusters; ++cluster_id ) {
          lo proc_id
            = levelwise_assignment[ old_access_index + 2 * cluster_id ];
          levelwise_assignment[ access_index + cluster_id ] = proc_id;
          clusters_per_process[ proc_id ].second += 1;
        }
      }
    }

    // Convert the assignment from the levelwise format into a format compatible
    // with the format used for representing the tree structure, assigning also
    // clusters below trunc_level to processes (to the same process which
    // handles the ancestor at trunc_level).
    std::vector< lo > process_assignment;
    process_assignment.reserve( 1 << ( trunc_level + 1 ) );
    process_assignment.push_back(
      -1 );  // the root is not assigned to a process
    // construct auxiliary vectors used for conversion
    std::vector< lo > assigned_clusters( trunc_level + 1, 0 );
    std::vector< lo > process_pointers( trunc_level - thresh_level + 1, 0 );
    // root is not a leaf, since trunc_level > 1
    convert_assignment_vector_2_tree_format( *_root, levelwise_assignment,
      thresh_level, trunc_level, n_processes, -1, assigned_clusters,
      process_pointers, process_assignment );
    return process_assignment;
  }
}

void besthea::mesh::time_cluster_tree::build_tree(
  time_cluster & root, lo level ) {
  // stop recursion if maximum number of levels is reached or root contains less
  // than _n_min_elems elements
  if ( level > _levels - 1 || root.get_n_elements( ) < _n_min_elems ) {
    root.set_n_children( 0 );
    if ( root.get_n_elements( ) > _n_max_elems_leaf ) {
      _n_max_elems_leaf = root.get_n_elements( );
    }
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
      // if ( el_centroid - center <= 1e-12 ) {
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
      // if ( el_centroid - center <= 1e-12 ) {
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

void besthea::mesh::time_cluster_tree::tree_2_vector(
  const time_cluster & root, std::vector< char > & tree_vector ) const {
  // get the children of root and determine if they are leaves or not
  // WARNING: it is assumed that root always has two children; this assumption
  // is reasonable if the method is called for a non-leaf cluster in the tree,
  // since the tree is a full binary tree by construction (in build tree)
  const std::vector< time_cluster * > * children = root.get_children( );
  char left_child_status
    = ( ( *children )[ 0 ]->get_n_children( ) > 0 ) ? 1 : 2;
  char right_child_status
    = ( ( *children )[ 1 ]->get_n_children( ) > 0 ) ? 1 : 2;
  tree_vector.push_back( left_child_status );
  tree_vector.push_back( right_child_status );
  if ( left_child_status == 1 ) {
    tree_2_vector( *( *children )[ 0 ], tree_vector );
  }
  if ( right_child_status == 1 ) {
    tree_2_vector( *( *children )[ 1 ], tree_vector );
  }
}

void besthea::mesh::time_cluster_tree::convert_assignment_vector_2_tree_format(
  const time_cluster & root, const std::vector< lo > & levelwise_assignment,
  const lo thresh_level, const lo trunc_level, const lo n_processes,
  const lo my_id, std::vector< lo > & assigned_clusters,
  std::vector< lo > & process_pointers,
  std::vector< lo > & process_assignment ) const {
  lo child_level = root.get_level( ) + 1;
  lo left_child_id, right_child_id;
  // add process ids to process_assignment vector according to the level
  if ( child_level < 2 ) {
    // the two children are at level 1, and are not assigned to any process
    left_child_id = -1;
    right_child_id = -1;
  } else if ( child_level < thresh_level ) {
    // Read the process ids from the vector levelwise_assignment and update how
    // many clusters have been assigned at child_level.
    lo access_index = ( 1 << child_level ) - 4;
    left_child_id
      = levelwise_assignment[ access_index + assigned_clusters[ child_level ] ];
    assigned_clusters[ child_level ] += 1;
    right_child_id
      = levelwise_assignment[ access_index + assigned_clusters[ child_level ] ];
    assigned_clusters[ child_level ] += 1;
  } else if ( child_level <= trunc_level ) {
    // Determine the process ids from the vector levelwise_assignment using the
    // information which process was last considered (process_pointers) and how
    // many clusters have been assigned for this process (assigned_clusters)
    lo rel_level = child_level - thresh_level;
    lo access_index = n_processes * ( 2 + rel_level );
    if ( assigned_clusters[ child_level ] >= levelwise_assignment[ access_index
           + process_pointers[ rel_level ] ] ) {
      // left child is handled by new process, update process_pointers
      process_pointers[ rel_level ] += 1;
      // reset counter for new process to 1
      assigned_clusters[ child_level ] = 1;
    } else {
      assigned_clusters[ child_level ] += 1;
    }
    // set left_child_id appropriately
    left_child_id = process_pointers[ rel_level ];
    // repeat for the right cluster
    if ( assigned_clusters[ child_level ] >= levelwise_assignment[ access_index
           + process_pointers[ rel_level ] ] ) {
      process_pointers[ rel_level ] += 1;
      assigned_clusters[ child_level ] = 1;
    } else {
      assigned_clusters[ child_level ] += 1;
    }
    // set right_child_id appropriately
    right_child_id = process_pointers[ rel_level ];
  } else if ( child_level > trunc_level ) {
    // children are assigned to the same process as their parent
    left_child_id = my_id;
    right_child_id = my_id;
  }
  // add process ids to process_assignment vector
  process_assignment.push_back( left_child_id );
  process_assignment.push_back( right_child_id );
  // determine whether children are leaf clusters or not for further recursion
  const std::vector< time_cluster * > * children = root.get_children( );
  bool left_child_non_leaf = ( ( *children )[ 0 ]->get_n_children( ) > 0 );
  bool right_child_non_leaf = ( ( *children )[ 1 ]->get_n_children( ) > 0 );
  if ( left_child_non_leaf ) {
    convert_assignment_vector_2_tree_format( *( *children )[ 0 ],
      levelwise_assignment, thresh_level, trunc_level, n_processes,
      left_child_id, assigned_clusters, process_pointers, process_assignment );
  }
  if ( right_child_non_leaf ) {
    convert_assignment_vector_2_tree_format( *( *children )[ 1 ],
      levelwise_assignment, thresh_level, trunc_level, n_processes,
      right_child_id, assigned_clusters, process_pointers, process_assignment );
  }
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
