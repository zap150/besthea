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

/** @file scheduling_time_cluster.h
 * @brief Reduced time cluster used for job scheduling.
 */

#ifndef INCLUDE_BESTHEA_SCHEDULING_TIME_CLUSTER_H_
#define INCLUDE_BESTHEA_SCHEDULING_TIME_CLUSTER_H_

#include "besthea/general_spacetime_cluster.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

#include <iostream>
#include <map>
#include <vector>

// forward declaration of general spacetime clusters
namespace besthea {
  namespace mesh {
    class general_spacetime_cluster;
  }
}

namespace besthea {
  namespace mesh {
    class scheduling_time_cluster;
  }
}

/**
 * Class representing 1D temporal cluster used for job scheduling in 3+1D FMM.
 */
class besthea::mesh::scheduling_time_cluster {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Block vector type.
  /**
   * Constructor.
   * @param[in] center  Center of the cluster.
   * @param[in] half_size Half size of the cluster.
   * @param[in] parent Pointer to the cluster's parent.
   * @param[in] level Level within the cluster tree structure.
   * @param[in] process_id  Id of the process which is responsible for the
   *                        cluster in an MPI parallelized FMM. Default value
   *                        is -1.
   */
  scheduling_time_cluster( sc center, sc half_size,
    scheduling_time_cluster * parent, short left_right, lo level,
    lo process_id = -1 )
    : _center( center ),
      _half_size ( half_size ),
      _parent( parent ),
      _left_right( left_right ),
      _children( nullptr ),
      _level( level ),
      _global_index( -1 ),
      _process_id( process_id ),
      _time_slices( nullptr ),
      _global_leaf_status( false ),
      _mesh_available( false ),
      _nearfield_list( nullptr ),
      _interaction_list( nullptr ),
      _send_list( nullptr ),
      _essential_status( -1 ),
      _active_upward_path( false ),
      _active_downward_path( false ),
      _upward_path_counter( -1 ),
      _ready_interaction_list( nullptr ),
      _m2l_counter( 0 ),
      _downward_path_status( 0 ),
      _associated_spacetime_clusters( nullptr ),
      _n_associated_leaves( 0 ),
      _associated_moments( nullptr ),
      _associated_local_contributions( nullptr ),
      _contribution_size( 0 ),
      _map_to_moment_receive_buffers( nullptr ) {
  }

  scheduling_time_cluster( const scheduling_time_cluster & that ) = delete;

  /**
   * Destructor.
   */
  virtual ~scheduling_time_cluster( ) {
    if ( _children != nullptr ) {
      for ( auto it = _children->begin( ); it != _children->end( ); ++it ) {
        if ( *it != nullptr ) {
          delete *it;
        }
      }
      delete _children;
    }
    if ( _time_slices != nullptr )
      delete _time_slices;
    if ( _nearfield_list != nullptr )
      delete _nearfield_list;
    if ( _interaction_list != nullptr )
      delete _interaction_list;
    if ( _send_list != nullptr )
      delete _send_list;
    if ( _ready_interaction_list != nullptr )
      delete _ready_interaction_list;
    if ( _associated_spacetime_clusters != nullptr )
      delete _associated_spacetime_clusters;
    if ( _associated_moments != nullptr )
      delete [] _associated_moments;
    if ( _associated_local_contributions != nullptr )
      delete [] _associated_local_contributions;
    for ( auto moment_buffer : _associated_moments_receive_buffers ) {
      delete [] moment_buffer;
    }
    if ( _map_to_moment_receive_buffers != nullptr )
      delete _map_to_moment_receive_buffers;
  }

  /**
   * Deletes the nearfield
   */
  void delete_nearfield( ) {
    delete _nearfield_list;
    _nearfield_list = nullptr;
  }

  /**
   * Deletes the interaction list
   */
  void delete_interaction_list( ) {
    delete _interaction_list;
    _interaction_list = nullptr;
  }

  /**
   * Deletes the ready interaction list
   */
  void delete_ready_interaction_list( ) {
    delete _ready_interaction_list;
    _ready_interaction_list = nullptr;
  }

  /**
   * Deletes the send list
   */
  void delete_send_list( ) {
    delete _send_list;
    _send_list = nullptr;
  }

  /**
   * Deletes the vector of children
   * @warning If _children still contains elements these are also deleted!
   */
  void delete_children( ) {
    if ( _children != nullptr ) {
      for ( auto it = _children->begin( ); it != _children->end( ); ++it ) {
        if ( *it != nullptr ) {
          delete *it;
        }
      }
      delete _children;
    }
    _children = nullptr;
  }

  /**
   * Adds a cluster to the list of children.
   * @param[in] child Child cluster.
   */
  void add_child( scheduling_time_cluster * child ) {
    if ( _children == nullptr ) {
      _children = new std::vector< scheduling_time_cluster * >( );
    }
    _children->push_back( child );
  }

  /**
   * Returns the center of the cluster.
   */
  sc get_center( ) const {
    return _center;
  }

  /**
   * Returns the half size of the cluster.
   */
  sc get_half_size( ) const {
    return _half_size;
  }

  /**
   * Sets a number of children and allocates the vector of pointers to children.
   * @param[in] n_children Number of cluster's children.
   */
  void set_n_children( lo n_children ) {
    if ( n_children > 0 ) {
      _children = new std::vector< scheduling_time_cluster * >( );
      _children->reserve( n_children );
    } else {
      _children = nullptr;
    }
  }

  /**
   * Returns the number of the cluster's children.
   */
  lo get_n_children( ) const {
    if ( _children != nullptr ) {
      return _children->size( );
    } else {
      return 0;
    }
  }

  /**
   * Returns a pointer to the children.
   */
  std::vector< scheduling_time_cluster * > * get_children( ) {
    return _children;
  }

  /**
   * Returns a const pointer to the children.
   */
  const std::vector< scheduling_time_cluster * > * get_children( ) const {
    return _children;
  }

  /**
   * Returns a pointer to the parent.
   */
  scheduling_time_cluster * get_parent( ) {
    return _parent;
  }

  /**
   * Returns the configuration of the cluster with respect to its parent, i.e.
   * the value of _left_right
   */
  short get_configuration( ) const {
    return _left_right;
  }

  /**
   * Returns the level of the cluster in the cluster tree it is contained in.
   */
  lo get_level( ) const {
    return _level;
  }

  /**
   * Returns id of the process to which the cluster is assigned.
   */
  lo get_process_id( ) const {
    return _process_id;
  }

  /**
   * Sets the process id of the cluster.
   * @param[in] process_id  Id of the process.
   */
  void set_process_id( lo process_id ) {
    _process_id = process_id;
  }

  /**
   * Returns the levelwise index of the cluster in the cluster tree.
   * @note The clusters are enumerated according to a tree traversal, see
   * @ref besthea::mesh::tree_structure::set_indices for more details.
   */
  lo get_global_index( ) const {
    return _global_index;
  }

  /**
   * Sets the index of the cluster.
   * @param[in] index Index which is set.
   */
  void set_index( lo index ) {
    _global_index = index;
  }

  /**
   * Sets the number of time slices and allocates the vector of indices of time
   * slices.
   * @param[in] n_slices Number of cluster's time slices.
   */
  void set_n_time_slices( lou n_slices ) {
    if ( n_slices > 0 ) {
      _time_slices = new std::vector< lo >( );
      _time_slices->reserve( n_slices );
    }
  }

  /**
   * Returns the number of time slices in the current cluster.
   */
  lo get_n_time_slices( ) const {
    if ( _time_slices != nullptr ) {
      return _time_slices->size( );
    } else {
      return 0;
    }
  }

  /**
   * Adds the index of a time slice to the vector of time slices.
   * @param[in] idx Global index of the time slices which is added.
   */
  void add_time_slice( lo idx ) {
    _time_slices->push_back( idx );
  }

  /**
   * Returns a pointer to the time slices (unmodifiable).
   */
  const std::vector< lo >* get_time_slices( ) const {
    return _time_slices;
  }

  /**
   * Sets global leaf status to a new value.
   * @param[in] new_status  New global leaf status.
   */
  void set_global_leaf_status( bool new_status ) {
    _global_leaf_status = new_status;
  }

  /**
   * Returns the global leaf status of the cluster.
   */
  bool get_global_leaf_status( ) const {
    return _global_leaf_status;
  }

  /**
   * Sets status of mesh availability.
   * @param[in] new_status New mesh availability status.
   */
  void set_mesh_availability( bool new_status ) {
    _mesh_available = new_status;
  }

  /**
   * Indicates whether a mesh  is available for the current cluster or not.
   */
  bool mesh_is_available( ) const {
    return _mesh_available;
  }

  /**
   * Adds a cluster to @p _nearfield_list.
   * @param[in] src_cluster Pointer to a nearfield cluster.
   * @note If @p _nearfield_list is not allocated this is done here.
   */
  void add_to_nearfield_list( scheduling_time_cluster * src_cluster ) {
    if ( _nearfield_list == nullptr ) {
      _nearfield_list = new std::vector< scheduling_time_cluster * >( );
    }
    _nearfield_list->push_back( src_cluster );
  }

  /**
   * Returns a pointer to the const nearfield.
   */
  const std::vector< scheduling_time_cluster * > * get_nearfield_list( ) const {
    return _nearfield_list;
  }

  /**
   * Returns a pointer to the nearfield.
   */
  std::vector< scheduling_time_cluster * > * get_nearfield_list( ) {
    return _nearfield_list;
  }

   /**
   * Returns a pointer to the const interaction list.
   */
  const std::vector< scheduling_time_cluster * > *
    get_interaction_list( ) const {
    return _interaction_list;
  }

  /**
   * Returns a pointer to the interaction list.
   */
  std::vector< scheduling_time_cluster * > * get_interaction_list( ) {
    return _interaction_list;
  }

  /**
   * Adds a cluster to @p _interaction_list.
   * @param[in] src_cluster Pointer to a farfield cluster.
   * @note If @p _interaction_list is not allocated this is done here.
   */
  void add_to_interaction_list( scheduling_time_cluster * src_cluster ) {
    if ( _interaction_list == nullptr ) {
      _interaction_list = new std::vector< scheduling_time_cluster * >( );
    }
    _interaction_list->push_back( src_cluster );
  }

  /**
   * Determines admissibility based on the "neighborhood criterion" (as in
   * Messner's work).
   * @param[in] src_cluster Source cluster whose admissibility is checked.
   * @warning This check of admissibility is only reasonable for clusters at the
   * same level of a tree.
   * @note The current criterion guarantees that the distance in time of two
   * admissible clusters is greater than the minimum of the half sizes of the
   * two clusters.
   */
  bool determine_admissibility( scheduling_time_cluster * src_cluster ) {
    bool admissibility = false;
    sc src_center = src_cluster->get_center( );
    sc src_half_size = src_cluster->get_half_size( );
    sc min_half_size
      = ( src_half_size < _half_size ) ? src_half_size : _half_size;
    if ( src_center < _center - _half_size - src_half_size - min_half_size ) {
      admissibility = true;
    }
    return admissibility;
  }

  /**
   * Adds a cluster to @p _send_list.
   * @param[in] tar_cluster Pointer to a target cluster.
   * @note If @p _send_list is not allocated this is done here.
   */
  void add_to_send_list( scheduling_time_cluster * tar_cluster ) {
    if ( _send_list == nullptr ) {
      _send_list = new std::vector< scheduling_time_cluster * >( );
    }
    _send_list->push_back( tar_cluster );
  }

  /**
   * Returns a pointer to the const send list.
   */
  const std::vector< scheduling_time_cluster * > * get_send_list( ) const {
    return _send_list;
  }

  /**
   * Returns a pointer to the send list.
   */
  std::vector< scheduling_time_cluster * > * get_send_list( ) {
    return _send_list;
  }

  /**
   * Adds a cluster to @p _ready_interaction_list.
   * @param[in] src_cluster Pointer to a target cluster.
   * @note If @p _ready_interaction_list is not allocated this is done here.
   */
  void add_to_ready_interaction_list( scheduling_time_cluster * src_cluster ) {
    if ( _ready_interaction_list == nullptr ) {
      _ready_interaction_list = new std::vector< scheduling_time_cluster * >( );
    }
    _ready_interaction_list->push_back( src_cluster );
  }

  /**
   * Clears the ready interaction list.
   * @note: The ready interaction list is only cleared, not deleted!
   */
  void clear_ready_interaction_list( ) {
    if ( _ready_interaction_list != nullptr ) {
      _ready_interaction_list->clear( );
    }
  }

  /**
   * Returns a pointer to the list of clusters which are ready for interactions.
   */
  std::vector< scheduling_time_cluster * > * get_ready_interaction_list( ) {
    return _ready_interaction_list;
  }

  /**
   * Returns a pointer to the (const!) list of clusters which are ready for
   * interactions.
   */
  const std::vector< scheduling_time_cluster * > *
    get_ready_interaction_list( ) const {
    return _ready_interaction_list;
  }

  void set_essential_status( const char status ) {
    _essential_status = status;
  }

  char get_essential_status( ) const {
    return _essential_status;
  }

  /**
   * Sets a flag which indicates if the cluster is active in the upward path of
   * the FMM.
   * \param[in] new_status  Value to set.
   */
  void set_active_upward_path_status( const bool new_status ) {
    _active_upward_path = new_status;
  }

  /**
   * Indicates if the cluster is active in the upward path, i.e. moments of
   * associated spacetime clusters have to be computed.
   */
  bool is_active_in_upward_path( ) const {
    return _active_upward_path;
  }

  /**
   * Sets a flag which indicates if the cluster is active in the downward path
   * of the FMM.
   * \param[in] new_status  Value to set.
   */
  void set_active_downward_path_status( const bool new_status ) {
    _active_downward_path = new_status;
  }

  /**
   * Indicates if the cluster is active in the downward path, i.e. local
   * contributions of associated spacetime clusters have to be computed.
   */
  bool is_active_in_downward_path( ) const {
    return _active_downward_path;
  }

  /**
   * Sets the upward path counter to the given value.
   * @param[in] upward_path_counter Value to set.
   */
  void set_upward_path_counter( const lo upward_path_counter ) {
    _upward_path_counter = upward_path_counter;
  }

  /**
   * Returns the value of @p _upward_path_counter .
   */
  lo get_upward_path_counter( ) const {
    return _upward_path_counter;
  }

  /**
   * Reduces @p _upward_path_counter by 1.
   */
  void reduce_upward_path_counter( ) {
    _upward_path_counter -= 1;
  }

  /**
   * Returns the value of @p _m2l_counter .
   */
  lou get_m2l_counter( ) const {
    return _m2l_counter;
  }

  /**
   * Sets the m2l counter to a new given value.
   * @param[in] new_value Value to which the counter is set.
   */
  void set_m2l_counter( const slou new_value ) {
    _m2l_counter = new_value;
  }

  /**
   * Sets the downward path status to the given value.
   * @param[in] new_status  Value to be set.
   */
  void set_downward_path_status( const char new_status ) {
    _downward_path_status = new_status;
  }

  /**
   * Returns the value of @p _downward_path_status .
   */
  char get_downward_path_status( ) const {
    return _downward_path_status;
  }

  /**
   * Adds a space-time cluster to @p _associated_spacetime_clusters.
   * @param[in] cluster Cluster which is added to the list.
   * @note If @p _associated_spacetime_clusters is not allocated this is done
   * here.
   */
  void add_associated_spacetime_cluster( general_spacetime_cluster* cluster ) {
    if ( _associated_spacetime_clusters == nullptr ) {
      _associated_spacetime_clusters
        = new std::vector< general_spacetime_cluster* >( );
    }
    _associated_spacetime_clusters->push_back( cluster );
  }

  /**
   * Returns a pointer to the const list of associated spacetime clusters.
   */
  const std::vector< general_spacetime_cluster* >*
    get_associated_spacetime_clusters( ) const {
    return _associated_spacetime_clusters;
  }

  /**
   * Returns a pointer to the list of associated spacetime clusters.
   */
  std::vector< general_spacetime_cluster* >*
    get_associated_spacetime_clusters( ) {
    return _associated_spacetime_clusters;
  }

  /**
   * Returns the number of associated spacetime leaf clusters.
   */
  lou get_n_associated_leaves( ) const {
    return _n_associated_leaves;
  }

  /**
   * Increases the number of associated spacetime leaf clusters by one.
   */
  void increase_n_associated_leaves( ) {
    _n_associated_leaves ++;
  }

  /**
   * Adds an entry to the map from process ids to moments.
   * @param[in] proc_id Key value of the element to be added
   * @note @p _map_to_moment_receive_buffers is allocated if it does not exist
   * allready.
   */
  void add_receive_buffer( const lo proc_id ) {
    if ( _map_to_moment_receive_buffers == nullptr ) {
      _map_to_moment_receive_buffers = new std::map< lo, lou >( );
    }
    sc * moment_buffer
      = new sc [ _contribution_size * _associated_spacetime_clusters->size( ) ];
    _associated_moments_receive_buffers.push_back( moment_buffer );
    _map_to_moment_receive_buffers->insert(
      { proc_id, _associated_moments_receive_buffers.size( ) - 1 } );
  }

  /**
   * Returns a pointer to the position where the moments of an extraneous
   * process are stored.
   * @param[in] proc_id Id of the extraneous process.
   */
  sc* get_extraneous_moment_pointer( lo proc_id ) {
    return _associated_moments_receive_buffers[
      _map_to_moment_receive_buffers->at( proc_id ) ];
  }

  /**
   * Determines the tree structure of the subtree whose root is the current
   * cluster.
   * @return  A vector of the tree structure in the usual tree format
   *          (see @ref time_cluster_tree::compute_tree_structure ).
   */
  std::vector< char > determine_subtree_structure( ) const {
    std::vector< char > tree_vector;
    if ( _children == nullptr ) {
      tree_vector.push_back( 2 );
    } else {
      tree_vector.push_back( 1 );
      append_tree_structure_vector_recursively( tree_vector );
    }
    return tree_vector;
  }

  /**
   * Determines the cluster bounds of the clusters in the subtree whose
   * root is the current cluster.
   * @return  A vector of the cluster bounds of the subtree in the usual tree
   *          format (see @ref time_cluster_tree::compute_tree_structure ).
   */
  std::vector< sc > determine_subtree_cluster_bounds( ) const {
    std::vector< sc > cluster_bounds_vector;
    cluster_bounds_vector.push_back( _center - _half_size );
    cluster_bounds_vector.push_back( _center + _half_size );
    if ( _children != nullptr ) {
      append_cluster_bound_vector_recursively( cluster_bounds_vector );
    }
    return cluster_bounds_vector;
  }

  /**
   * Adds the status of the two children of the current cluster to the end
   * of the tree structure vector.
   *
   * The status is:
   * - 0: if the child does not exist.
   * - 1: if the child is a non-leaf cluster.
   * - 2: if the child is a leaf cluster.
   *
   * The routine is called recursively for the children (first left child, then
   * right child)
   * @param[in,out] tree_vector Vector to which the status of the children is
   *                            added.
   * @warning This routine has to be called for a non-leaf cluster only,
   *          otherwise a segmentation fault occurs.
   */
  void append_tree_structure_vector_recursively(
    std::vector< char > & tree_vector ) const {
    char left_child_status = 0;
    char right_child_status = 0;
    scheduling_time_cluster* left_child = nullptr;
    scheduling_time_cluster* right_child = nullptr;
    for ( auto child : *_children ) {
      char* status_pointer;
      if ( child->get_configuration( ) == 0 ) {
        left_child = child;
        status_pointer = &left_child_status;
      } else {
        right_child = child;
        status_pointer = &right_child_status;
      }
      if ( child->get_n_children( ) > 0 ) {
        *status_pointer = 1;
      } else {
        *status_pointer = 2;
      }
    }
    tree_vector.push_back( left_child_status );
    tree_vector.push_back( right_child_status );
    if ( left_child_status == 1 ) {
      left_child->append_tree_structure_vector_recursively( tree_vector );
    }
    if ( right_child_status == 1 ) {
      right_child->append_tree_structure_vector_recursively( tree_vector );
    }
  }

  void append_cluster_bound_vector_recursively(
    std::vector< sc > & cluster_bounds_vector ) const {
    char left_child_status = 0;
    char right_child_status = 0;
    sc left_child_lower_bound( 0.0 ), left_child_upper_bound( 0.0 );
    sc right_child_lower_bound( 0.0 ), right_child_upper_bound( 0.0 );
    scheduling_time_cluster* left_child = nullptr;
    scheduling_time_cluster* right_child = nullptr;
    for ( auto child : *_children ) {
      // determine which child it is and remember its bounds and status.
      char* status_pointer;
      if ( child->get_configuration( ) == 0 ) {
        left_child = child;
        status_pointer = &left_child_status;
        left_child_lower_bound
          = child->get_center( ) - child->get_half_size( );
        left_child_upper_bound
          = child->get_center( ) + child->get_half_size( );
      } else {
        right_child = child;
        status_pointer = &right_child_status;
        right_child_lower_bound
          = child->get_center( ) - child->get_half_size( );
        right_child_upper_bound
          = child->get_center( ) + child->get_half_size( );
      }
      if ( child->get_n_children( ) > 0 ) {
        *status_pointer = 1;
      } else {
        *status_pointer = 2;
      }
    }
    // add the cluster bounds appropriately to the vector
    cluster_bounds_vector.push_back( left_child_lower_bound );
    cluster_bounds_vector.push_back( left_child_upper_bound );
    cluster_bounds_vector.push_back( right_child_lower_bound );
    cluster_bounds_vector.push_back( right_child_upper_bound );
    if ( left_child_status == 1 ) {
      left_child->append_cluster_bound_vector_recursively(
        cluster_bounds_vector );
    }
    if ( right_child_status == 1 ) {
      right_child->append_cluster_bound_vector_recursively(
        cluster_bounds_vector );
    }
  }

  /**
   * Returns a pointer to left neighbour.
   */
  scheduling_time_cluster * get_left_neighbour( ) {
    if ( _parent == nullptr ) {
      // for the root cluster
      return nullptr;
    }

    if ( this == _parent->_children->back( ) ) {
      return _parent->_children->front( );
    } else if ( ( _parent->get_left_neighbour( ) != nullptr )
      && ( _parent->get_left_neighbour( )->_children != nullptr ) ) {
      return _parent->get_left_neighbour( )->_children->back( );
    } else {
      return nullptr;
    }
  }

  /**
   * Determines whether the current cluster is the left child of its parent.
   * @note If the current cluster is the root of the tree \p false is returned.
   */
  bool is_left_child( ) const {
    if ( _parent == nullptr )
      return false;
    else
      return ( this == _parent->_children->front( ) );
  }

  /**
   * Allocates an array containing all the moments of the associated
   * spacetime clusters. In addition, for each associated spacetime cluster it
   * sets the pointer to the appropriate moment.
   * @param[in] moment_size Size of the moment of a single space-time cluster.
   * @note Before calling this routine the spacetime clusters have to be
   * associated with this cluster.
   */
  void allocate_associated_moments( const lou moment_size ) {
    if ( _associated_spacetime_clusters != nullptr ) {
      if ( _associated_moments == nullptr ) {
        _contribution_size = moment_size;
        _associated_moments
          = new sc [ moment_size * _associated_spacetime_clusters->size( ) ];
        for ( lou i = 0; i < _associated_spacetime_clusters->size( ); ++i ) {
          general_spacetime_cluster * current_spacetime_cluster
            = ( *_associated_spacetime_clusters )[ i ];
          current_spacetime_cluster->set_pointer_to_moment(
            & _associated_moments[ i * moment_size ] );
        }
      } else {
        std::cout << "warning: associated moments allocated already!"
                  << std::endl;
      }
    }
  }

  /**
   * Sets all associated moments to 0.
   */
  void clear_associated_moments( ) {
    if ( _associated_moments != nullptr ) {
      for ( lou i = 0;
            i < _contribution_size * _associated_spacetime_clusters->size( );
            ++i  ) {
        _associated_moments[ i ] = 0.0;
      }
    }
  }

  /**
   * Returns a pointer to the associated moments.
   */
  sc* get_associated_moments( ) {
    return _associated_moments;
  }

  /**
   * Allocates an array containing all the local contributions of the associated
   * spacetime clusters. In addition, for each associated spacetime cluster it
   * sets the pointer to the appropriate local contribution.
   * @param[in] contribution_size Size of the local contribution of a single
   *                              space-time cluster.
   * @note Before calling this routine the spacetime clusters have to be
   * associated with this cluster.
   */
  void allocate_associated_local_contributions( const lou contribution_size ) {
    if ( _associated_spacetime_clusters != nullptr ) {
      if ( _associated_local_contributions == nullptr ) {
        _contribution_size = contribution_size;
        _associated_local_contributions
          = new sc [ contribution_size
                    * _associated_spacetime_clusters->size( ) ];
        for ( lou i = 0; i < _associated_spacetime_clusters->size( ); ++i ) {
          general_spacetime_cluster * current_spacetime_cluster
            = ( *_associated_spacetime_clusters )[ i ];
          current_spacetime_cluster->set_pointer_to_local_contribution(
            & _associated_local_contributions[ i * contribution_size ] );
        }
      } else {
        std::cout << "warning: associated local contributions allocated already!"
                  << std::endl;
      }
    }
  }

  /**
   * Sets all associated local contributions to 0.
   */
  void clear_associated_local_contributions( ) {
    if ( _associated_local_contributions != nullptr ) {
      for ( lou i = 0;
           i < _contribution_size * _associated_spacetime_clusters->size( );
           ++i  ) {
        _associated_local_contributions[ i ] = 0.0;
      }
    }
  }

  /**
   * Returns a pointer to the associated local contributions.
   */
  sc* get_associated_local_contributions( ) {
    return _associated_local_contributions;
  }

  /**
   * Prints info of the object.
   */
  void print( lo executing_process_id = -1 ) {
    std::cout << "level: " << _level
              << ", center: " << _center << ", half size: " << _half_size
              << ", global_index: " << _global_index
              << ", proc_id: " << _process_id;
    if ( _global_leaf_status ) {
      std::cout << ", is global leaf";
    }
    // if ( _nearfield_list != nullptr ) {
    //   std::cout << ", nearfield: ";
    //   for ( lou i = 0; i < _nearfield_list->size( ); ++i ) {
    //     std::cout << "(" << ( *_nearfield_list )[ i ]->get_level( ) << ", "
    //               << ( *_nearfield_list )[ i ]->get_global_index( ) << "), ";
    //   }
    // }
    // if ( _interaction_list != nullptr ) {
    //   std::cout << "interaction_list: ";
    //   for ( lou i = 0; i < _interaction_list->size( ); ++i ) {
    //     std::cout << "(" << ( *_interaction_list )[ i ]->get_level( ) << ", "
    //               << ( *_interaction_list )[ i ]->get_global_index( )
    //               << "), ";
    //   }
    // }
    // if ( _send_list != nullptr ) {
    //   std::cout << "send_list: ";
    //   for ( lou i = 0; i < _send_list->size( ); ++i ) {
    //     std::cout << "(" << ( *_send_list )[ i ]->get_level( ) << ", "
    //               << ( *_send_list )[ i ]->get_global_index( )
    //               << "), ";
    //   }
    // }
    // std::cout << ", upward_path_counter: " << _upward_path_counter;
    // std::cout << ", downward_path_status: " << (lo) _downward_path_status;
    // std::cout << ", m2l counter: " << _m2l_counter;

    if ( _associated_spacetime_clusters != nullptr ) {
      if ( _process_id == executing_process_id ) {
        std::cout << ", number of associated leaves: " << _n_associated_leaves
                << ", number of associated non-leaves: "
                <<  _associated_spacetime_clusters->size( )
                    - _n_associated_leaves;
      }
      else {
        std::cout << ", number of associated clusters: "
                  << _associated_spacetime_clusters->size( );
      }
    }
    if ( _time_slices != nullptr ) {
      std::cout << ", time slices: ";
      for ( auto idx : *_time_slices ) {
        std::cout << idx << ", ";
      }
    }
    std::cout << std::endl;
  }

 private:
  sc _center;       //!< Center of the cluster.
  sc _half_size;    //!< Half size of the cluster.
  scheduling_time_cluster * _parent;  //!< Parent of the cluster.
  short _left_right;  //!< Indicates if the child is the left (0), or right (1)
                      //!< child of its parent.
  std::vector< scheduling_time_cluster * >
    * _children;    //!< Children of the cluster.
  lo _level;        //!< Level within the cluster tree.
  lo _global_index; //!< Global index of the cluster. The children of a cluster
                    //!< with index k have the indices 2k+1 and 2k+2.
  lo _process_id;   //!< Id of the process to which the cluster is assigned.
  std::vector< lo > *
    _time_slices; //!< global indices of the cluster's time slices (only set for
                  //!< leaf clusters)
  bool _global_leaf_status; //!< indicates whether the cluster is a leaf (1) or
                            //!< non-leaf in a global tree structure
  bool _mesh_available; //!< Indicates whether the process who owns the cluster
                        //!< has access to the corresponding mesh. Only relevant
                        //!< in a distribution tree in a distributed spacetime
                        //!< tensor mesh. It is set to true for leaf clusters
                        //!< which are either local or in the nearfield of local
                        //!< clusters. It is set in
                        //!< @ref distributed_spacetime_tensor_mesh::find_slices_to_load.
  std::vector< scheduling_time_cluster * >
    * _nearfield_list;   //!< Nearfield of the cluster.
  std::vector< scheduling_time_cluster * >
    * _interaction_list;  //!< Interaction list of the cluster.
  std::vector< scheduling_time_cluster * >
    * _send_list;   //!< Contains all clusters in whose interaction list the
                    //!< cluster is contained.
  char _essential_status; //!< Indicates the status of a cluster in a
                          //!< distributed tree. Possible status are:
                          //!< - 0: not essential
                          //!< - 1: essential for time cluster only
                          //!< - 2: essential for time and space-time cluster
                          //!< - 3: local, i.e. directly essential
                          //!< The status is assigned when the tree containing
                          //!< the cluster is reduced to the locally essential
                          //!< tree, see
                          //!< @ref tree_structure::reduce_2_essential .
  bool _active_upward_path; //!< Indicates if the cluster is active in the
                            //!< upward path of the FMM.
  bool _active_downward_path; //!< Indicates if the cluster is active in the
                              //!< downward path of the FMM.
  lo _upward_path_counter; //!< Used to keep track of the dependencies in the
                           //!< upward path. If it is 0, the dependencies are
                           //!< fulfilled.
  std::vector< scheduling_time_cluster * >
    * _ready_interaction_list; //!< Clusters from the interaction list are added
                               //!< to this list, when their moments are ready.
                               //!< It is used to manage the execution of M2L
                               //!< operations in the distributed FMM.
  slou _m2l_counter;  //!< Used to keep track of the completed m2l operations.
  char _downward_path_status; //!< Used to keep track of the status in the
                              //!< downward path. Three status are distinguished
                              //!< - 0: L2L not executed,
                              //!< - 1: L2L executed, local contributions not
                              //!<      ready,
                              //!< - 2: local contributions ready.
  std::vector< general_spacetime_cluster* >
    * _associated_spacetime_clusters; //!< List of space-time clusters
                                      //!< associated to the scheduling time
                                      //!< cluster.
  lou _n_associated_leaves;  //!< Number of associated space-time leaf clusters.
                             //!< These are first in the list of associated
                             //!< space-time clusters.
  sc * _associated_moments; //!< Array containing all the moments of the
                            //!< associated general spacetime clusters.
  sc * _associated_local_contributions; //!< Array containing all the local
                                        //!< contributions of the associated
                                        //!< general spacetime clusters.
  lou _contribution_size; //!< Size of a single contribution (moments or local
                          //!< contribution) in the array of associated
                          //!< contributions
                          //!< @todo Is there a better way to make this value
                          //!< accessible for all clusters without storing it
                          //!< in each instance separately?
  std::vector< sc * >
    _associated_moments_receive_buffers;  //!< In case that moments have to be
                                          //!< received from other processes
                                          //!< they are written into these
                                          //!< buffers.
  std::map< lo, lou >
    * _map_to_moment_receive_buffers; //!< Map to access the correct position in
                                //!< @p _associated_moments_receive_buffers for
                                //!< extraneous children.
};

#endif /* INCLUDE_BESTHEA_SCHEDULING_TIME_CLUSTER_H_ */
