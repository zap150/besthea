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
   * @param[in] left_right  Indicates if the cluster is the left (0) or
   *                        right (1) child of its parent.
   * @param[in] level Level within the cluster tree structure.
   * @param[in] process_id  Id of the process which is responsible for the
   *                        cluster in an MPI parallelized FMM. Default value
   *                        is -1.
   */
  scheduling_time_cluster( sc center, sc half_size,
    scheduling_time_cluster * parent, short left_right, lo level,
    lo process_id = -1 )
    : _center( center ),
      _half_size( half_size ),
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
      _m2t_list( nullptr ),
      _s2l_list( nullptr ),
      _send_list( nullptr ),
      _m2t_send_list( nullptr ),
      _s2l_send_list( nullptr ),
      _essential_status( -1 ),
      _active_upward_path( false ),
      _active_downward_path( false ),
      _status_initial_op_downward_path( 0 ),
      _upward_path_counter( -1 ),
      _m2l_counter( 0 ),
      _s2l_execution_status( 2 ),
      _m2t_execution_status( 1 ),
      _downward_path_status( 0 ),
      _associated_spacetime_clusters( nullptr ),
      _n_associated_leaves_and_aux_clusters_per_level( nullptr ),
      _max_relative_space_level_s2ms( -1 ),
      _max_relative_space_level_ls2t( -1 ),
      _associated_additional_spacetime_leaves( nullptr ),
      _n_associated_leaves( 0 ),
      _associated_moments( nullptr ),
      _n_st_clusters_w_moments( 0 ),
      _associated_local_contributions( nullptr ),
      _n_st_clusters_w_local_contributions( 0 ),
      _assoc_spatial_moments( nullptr ),
      _n_st_clusters_w_spatial_moments( 0 ),
      _assoc_spatial_local_contributions( nullptr ),
      _n_st_clusters_w_spatial_local_contributions( 0 ),
      _assoc_aux_spatial_moments( nullptr ),
      _n_st_clusters_w_aux_spatial_moments( 0 ),
      _assoc_nearfield_targets( nullptr ),
      _assoc_standard_m2t_targets( nullptr ),
      _assoc_hybrid_m2t_targets( nullptr ),
      _assoc_standard_s2l_targets( nullptr ),
      _assoc_hybrid_s2l_targets( nullptr ),
      _assoc_aux_s2ms_cluster_pairs( nullptr ),
      _contribution_size( 0 ),
      _map_to_moment_receive_buffers( nullptr ),
      _pos_in_m_list( -1 ),
      _pos_in_l_list( -1 ),
      _pos_in_m2l_list( -1 ),
      _pos_in_s2l_list( -1 ),
      _n_ready_m2l_sources( 0 ),
      _n_ready_m2t_sources( 0 ) {
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
    if ( _m2t_list != nullptr )
      delete _m2t_list;
    if ( _s2l_list != nullptr )
      delete _s2l_list;
    if ( _send_list != nullptr )
      delete _send_list;
    if ( _m2t_send_list != nullptr )
      delete _m2t_send_list;
    if ( _s2l_send_list != nullptr )
      delete _s2l_send_list;
    if ( _associated_spacetime_clusters != nullptr )
      delete _associated_spacetime_clusters;
    if ( _n_associated_leaves_and_aux_clusters_per_level != nullptr )
      delete _n_associated_leaves_and_aux_clusters_per_level;
    if ( _associated_additional_spacetime_leaves != nullptr )
      delete _associated_additional_spacetime_leaves;
    if ( _associated_moments != nullptr )
      delete[] _associated_moments;
    if ( _associated_local_contributions != nullptr )
      delete[] _associated_local_contributions;
    if ( _assoc_spatial_moments != nullptr )
      delete[] _assoc_spatial_moments;
    if ( _assoc_spatial_local_contributions != nullptr )
      delete[] _assoc_spatial_local_contributions;
    if ( _assoc_aux_spatial_moments != nullptr )
      delete[] _assoc_aux_spatial_moments;
    if ( _assoc_nearfield_targets != nullptr )
      delete _assoc_nearfield_targets;
    if ( _assoc_standard_m2t_targets != nullptr )
      delete _assoc_standard_m2t_targets;
    if ( _assoc_hybrid_m2t_targets != nullptr )
      delete _assoc_hybrid_m2t_targets;
    if ( _assoc_standard_s2l_targets != nullptr )
      delete _assoc_standard_s2l_targets;
    if ( _assoc_hybrid_s2l_targets != nullptr )
      delete _assoc_hybrid_s2l_targets;
    if ( _assoc_aux_s2ms_cluster_pairs != nullptr )
      delete _assoc_aux_s2ms_cluster_pairs;
    for ( auto moment_buffer : _associated_moments_receive_buffers ) {
      delete[] moment_buffer;
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
   * Deletes the send list
   */
  void delete_send_list( ) {
    delete _send_list;
    _send_list = nullptr;
  }

  /**
   * Deletes @ref _m2t_list of the cluster.
   */
  void delete_m2t_list( ) {
    delete _m2t_list;
    _m2t_list = nullptr;
  }

  /**
   * Deletes @ref _s2l_list of the cluster.
   */
  void delete_s2l_list( ) {
    delete _s2l_list;
    _s2l_list = nullptr;
  }

  /**
   * Delete @ref _m2t_send_list of the cluster.
   */
  void delete_m2t_send_list( ) {
    delete _m2t_send_list;
    _m2t_send_list = nullptr;
  }

  /**
   * Delete @ref _s2l_send_list of the cluster.
   */
  void delete_s2l_send_list( ) {
    delete _s2l_send_list;
    _s2l_send_list = nullptr;
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
   * Returns a pointer to the parent.
   */
  const scheduling_time_cluster * get_parent( ) const {
    return _parent;
  }

  /**
   * Returns the configuration of the cluster with respect to its parent, i.e.
   * the value of @ref _left_right
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
  const std::vector< lo > * get_time_slices( ) const {
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
  bool is_global_leaf( ) const {
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
   * Adds a cluster to @ref _nearfield_list.
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
   * Adds and index to @ref _assoc_nearfield_targets.
   * @param[in] new_index Index to be added to the list.
   */
  void add_index_to_assoc_nearfield_targets( lo new_index ) {
    if ( _assoc_nearfield_targets == nullptr ) {
      _assoc_nearfield_targets = new std::vector< lo >( );
    }
    _assoc_nearfield_targets->push_back( new_index );
  }

  /**
   * Returns a pointer to (const) @ref _assoc_nearfield_targets.
   */
  const std::vector< lo > * get_assoc_nearfield_targets( ) const {
    return _assoc_nearfield_targets;
  }

  /**
   * Returns a pointer to the const interaction list.
   */
  const std::vector< scheduling_time_cluster * > * get_interaction_list( )
    const {
    return _interaction_list;
  }

  /**
   * Returns a pointer to the interaction list.
   */
  std::vector< scheduling_time_cluster * > * get_interaction_list( ) {
    return _interaction_list;
  }

  /**
   * Adds a cluster to @ref _interaction_list.
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
   * Adds a cluster to @ref _m2t_list.
   * @param[in] src_cluster Pointer to a source cluster for which m2t operations
   * have to be executed.
   * @note If @p _m2t_list is not allocated this is done here.
   */
  void add_to_m2t_list( scheduling_time_cluster * src_cluster ) {
    if ( _m2t_list == nullptr ) {
      _m2t_list = new std::vector< scheduling_time_cluster * >( );
    }
    _m2t_list->push_back( src_cluster );
  }

  /**
   * Returns a pointer to the m2t list.
   */
  std::vector< scheduling_time_cluster * > * get_m2t_list( ) {
    return _m2t_list;
  }

  /**
   * Returns a pointer to the const m2t list.
   */
  const std::vector< scheduling_time_cluster * > * get_m2t_list( ) const {
    return _m2t_list;
  }

  /**
   * Adds and index to @ref _assoc_standard_m2t_targets.
   * @param[in] new_index Index to be added to the list.
   */
  void add_index_to_assoc_standard_m2t_targets( lo new_index ) {
    if ( _assoc_standard_m2t_targets == nullptr ) {
      _assoc_standard_m2t_targets = new std::vector< lo >( );
    }
    _assoc_standard_m2t_targets->push_back( new_index );
  }

  /**
   * Adds and index to @ref _assoc_hybrid_m2t_targets.
   * @param[in] new_index Index to be added to the list.
   */
  void add_index_to_assoc_hybrid_m2t_targets( lo new_index ) {
    if ( _assoc_hybrid_m2t_targets == nullptr ) {
      _assoc_hybrid_m2t_targets = new std::vector< lo >( );
    }
    _assoc_hybrid_m2t_targets->push_back( new_index );
  }

  /**
   * Returns a pointer to (const) @ref _assoc_standard_m2t_targets.
   */
  const std::vector< lo > * get_assoc_standard_m2t_targets( ) const {
    return _assoc_standard_m2t_targets;
  }

  /**
   * Returns a pointer to (const) @ref _assoc_hybrid_m2t_targets.
   */
  const std::vector< lo > * get_assoc_hybrid_m2t_targets( ) const {
    return _assoc_hybrid_m2t_targets;
  }
  /**
   * Adds a cluster to @ref _s2l_list.
   * @param[in] src_cluster Pointer to a source cluster for which s2l operations
   * have to be executed.
   * @note If @p _s2l_list is not allocated this is done here.
   */
  void add_to_s2l_list( scheduling_time_cluster * src_cluster ) {
    if ( _s2l_list == nullptr ) {
      _s2l_list = new std::vector< scheduling_time_cluster * >( );
    }
    _s2l_list->push_back( src_cluster );
  }

  /**
   * Returns a pointer to the s2l list.
   */
  std::vector< scheduling_time_cluster * > * get_s2l_list( ) {
    return _s2l_list;
  }

  /**
   * Returns a pointer to the const s2l list.
   */
  const std::vector< scheduling_time_cluster * > * get_s2l_list( ) const {
    return _s2l_list;
  }

  /**
   * Adds and index to @ref _assoc_standard_s2l_targets.
   * @param[in] new_index Index to be added to the list.
   */
  void add_index_to_assoc_standard_s2l_targets( lo new_index ) {
    if ( _assoc_standard_s2l_targets == nullptr ) {
      _assoc_standard_s2l_targets = new std::vector< lo >( );
    }
    _assoc_standard_s2l_targets->push_back( new_index );
  }

  /**
   * Adds and index to @ref _assoc_hybrid_s2l_targets.
   * @param[in] new_index Index to be added to the list.
   */
  void add_index_to_assoc_hybrid_s2l_targets( lo new_index ) {
    if ( _assoc_hybrid_s2l_targets == nullptr ) {
      _assoc_hybrid_s2l_targets = new std::vector< lo >( );
    }
    _assoc_hybrid_s2l_targets->push_back( new_index );
  }

  /**
   * Returns a pointer to (const) @ref _assoc_standard_s2l_targets.
   */
  const std::vector< lo > * get_assoc_standard_s2l_targets( ) const {
    return _assoc_standard_s2l_targets;
  }

  /**
   * Returns a pointer to (const) @ref _assoc_hybrid_s2l_targets.
   */
  const std::vector< lo > * get_assoc_hybrid_s2l_targets( ) const {
    return _assoc_hybrid_s2l_targets;
  }

  /**
   * Adds a new pair to @ref _assoc_aux_s2ms_cluster_pairs.
   * @param[in] new_pair Pair to be added to the list.
   */
  void add_pair_to_assoc_aux_s2ms_cluster_pairs(
    const std::pair< lo, general_spacetime_cluster * > & new_pair ) {
    if ( _assoc_aux_s2ms_cluster_pairs == nullptr ) {
      _assoc_aux_s2ms_cluster_pairs
        = new std::vector< std::pair< lo, general_spacetime_cluster * > >( );
    }
    _assoc_aux_s2ms_cluster_pairs->push_back( new_pair );
  }

  /**
   * Returns a pointer to (const) @ref _assoc_aux_s2ms_clusters.
   */
  const std::vector< std::pair< lo, general_spacetime_cluster * > > *
  get_assoc_aux_s2ms_clusters( ) const {
    return _assoc_aux_s2ms_cluster_pairs;
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
   * Adds a cluster to @p _m2t_send_list.
   * @param[in] tar_cluster Pointer to a target cluster.
   * @note If @p _m2t_send_list is not allocated this is done here.
   */
  void add_to_m2t_send_list( scheduling_time_cluster * tar_cluster ) {
    if ( _m2t_send_list == nullptr ) {
      _m2t_send_list = new std::vector< scheduling_time_cluster * >( );
    }
    _m2t_send_list->push_back( tar_cluster );
  }

  /**
   * Returns a pointer to the const diagonal send list.
   */
  const std::vector< scheduling_time_cluster * > * get_m2t_send_list( ) const {
    return _m2t_send_list;
  }

  /**
   * Returns a pointer to the diagonal send list.
   */
  std::vector< scheduling_time_cluster * > * get_m2t_send_list( ) {
    return _m2t_send_list;
  }

  /**
   * Adds a cluster to @p _s2l_send_list.
   * @param[in] tar_cluster Pointer to a target cluster.
   * @note If @p _s2l_send_list is not allocated this is done here.
   */
  void add_to_s2l_send_list( scheduling_time_cluster * tar_cluster ) {
    if ( _s2l_send_list == nullptr ) {
      _s2l_send_list = new std::vector< scheduling_time_cluster * >( );
    }
    _s2l_send_list->push_back( tar_cluster );
  }

  /**
   * Returns a pointer to the const diagonal send list.
   */
  const std::vector< scheduling_time_cluster * > * get_s2l_send_list( ) const {
    return _s2l_send_list;
  }

  /**
   * Returns a pointer to the diagonal send list.
   */
  std::vector< scheduling_time_cluster * > * get_s2l_send_list( ) {
    return _s2l_send_list;
  }

  /**
   * Resets @ref _n_ready_m2l_sources to 0.
   */
  void reset_n_ready_m2l_sources( ) {
#pragma omp atomic write
    _n_ready_m2l_sources = 0;
  }

  /**
   * Increases the value of @ref _n_ready_m2l_sources by 1.
   */
  void update_n_ready_m2l_sources( ) {
#pragma omp atomic update
    _n_ready_m2l_sources++;
  }

  /**
   * Returns the number of clusters ready for interaction, i.e. the value of
   * @ref _n_ready_m2l_sources.
   */
  lou get_n_ready_m2l_sources( ) {
    lou size;
#pragma omp atomic read
    size = _n_ready_m2l_sources;
    return size;
  }

  /**
   * Resets @ref _n_ready_m2t_sources to 0.
   */
  void reset_n_ready_m2t_sources( ) {
#pragma omp atomic write
    _n_ready_m2t_sources = 0;
  }

  /**
   * Increases the value of @ref _n_ready_m2t_sources by 1.
   */
  void update_n_ready_m2t_sources( ) {
#pragma omp atomic update
    _n_ready_m2t_sources++;
  }

  /**
   * Returns the number of clusters ready for interaction, i.e. the value of
   * @ref _n_ready_m2t_sources.
   */
  lou get_n_ready_m2t_sources( ) const {
    lou size;
#pragma omp atomic read
    size = _n_ready_m2t_sources;
    return size;
  }

  /**
   * Sets the essential status of the current cluster (in the setting of a
   * distributed tree) to a given value.
   * @param[in] status  Value to be set.
   */
  void set_essential_status( const char status ) {
    _essential_status = status;
  }

  /**
   * Returns the essential status of the current cluster (in the setting of a
   * distributed tree).
   */
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
   * Sets @ref _status_initial_op_downward_path.
   * \param[in] new_status  Value to set.
   */
  void set_status_initial_op_downward_path( const char new_status ) {
    _status_initial_op_downward_path = new_status;
  }

  /**
   * Returns the value of @ref _status_initial_op_downward_path
   */
  char get_status_in_initial_op_downward_path( ) const {
    return _status_initial_op_downward_path;
  }

  /**
   * Sets the upward path counter to the given value.
   * @param[in] upward_path_counter Value to set.
   */
  void set_upward_path_counter( const lo upward_path_counter ) {
#pragma omp atomic write
    _upward_path_counter = upward_path_counter;
  }

  /**
   * Returns the value of @p _upward_path_counter.
   */
  lo get_upward_path_counter( ) const {
    lo counter;
#pragma omp atomic read
    counter = _upward_path_counter;
    return counter;
  }

  /**
   * Reduces @p _upward_path_counter by 1.
   */
  void reduce_upward_path_counter( ) {
#pragma omp atomic update
    _upward_path_counter -= 1;
  }

  /**
   * Returns the value of @p _m2l_counter.
   */
  lou get_m2l_counter( ) const {
    lou counter;
#pragma omp atomic read
    counter = _m2l_counter;
    return counter;
  }

  /**
   * Sets the m2l counter to a new given value.
   * @param[in] new_value Value to which the counter is set.
   */
  void set_m2l_counter( const slou new_value ) {
#pragma omp atomic write
    _m2l_counter = new_value;
  }

  /**
   * Returns the value of @ref _s2l_execution_status.
   */
  char get_s2l_execution_status( ) const {
    char status;
#pragma omp atomic read
    status = _s2l_execution_status;
    return status;
  }

  /**
   * Sets @ref _s2l_execution_status to a new value (between 0 and 2)
   * @param[in] new_value Value to which the status is set.
   */
  void set_s2l_execution_status( const char new_value ) {
#pragma omp atomic write
    _s2l_execution_status = new_value;
  }

  /**
   * Returns the value of @ref _m2t_execution_status.
   */
  char get_m2t_execution_status( ) const {
    char status;
#pragma omp atomic read
    status = _m2t_execution_status;
    return status;
  }

  /**
   * Sets @ref _m2t_execution_status to a new value (either 0 or 1)
   * @param[in] new_value Value to which the status is set.
   */
  void set_m2t_execution_status( const char new_status ) {
#pragma omp atomic write
    _m2t_execution_status = new_status;
  }

  /**
   * Sets the downward path status to the given value.
   * @param[in] new_status  Value to be set.
   */
  void set_downward_path_status( const char new_status ) {
#pragma omp atomic write
    _downward_path_status = new_status;
  }

  /**
   * Returns the value of @p _downward_path_status.
   */
  char get_downward_path_status( ) const {
    char status;
#pragma omp atomic read
    status = _downward_path_status;
    return status;
  }

  /**
   * Adds a space-time cluster to @p _associated_spacetime_clusters.
   * @param[in] cluster Cluster which is added to the list.
   * @note If @p _associated_spacetime_clusters is not allocated this is done
   * here.
   */
  void add_associated_spacetime_cluster( general_spacetime_cluster * cluster ) {
    if ( _associated_spacetime_clusters == nullptr ) {
      _associated_spacetime_clusters
        = new std::vector< general_spacetime_cluster * >( );
    }
    _associated_spacetime_clusters->push_back( cluster );
  }

  /**
   * Updates the counter of associated leaves and auxiliary clusters for a given
   * level by one.
   * @param[in] relative_level  Relative spatial level for which the counter is
   * increased.
   *
   * @note @ref _n_associated_leaves_and_aux_clusters_per_level is modified or
   * allocated.
   */
  void update_n_associated_leaves_and_aux_clusters_per_level(
    lo relative_level ) {
    if ( _n_associated_leaves_and_aux_clusters_per_level == nullptr ) {
      _n_associated_leaves_and_aux_clusters_per_level
        = new std::vector< lo >( );
    }
    if ( (lo) _n_associated_leaves_and_aux_clusters_per_level->size( )
      <= relative_level ) {
      _n_associated_leaves_and_aux_clusters_per_level->resize(
        relative_level + 1 );
      ( *_n_associated_leaves_and_aux_clusters_per_level )[ relative_level ]
        = 1;
    } else {
      ( *_n_associated_leaves_and_aux_clusters_per_level )[ relative_level ]++;
    }
  }

  /**
   * Returns a pointer to (const)
   * @ref _n_associated_leaves_and_aux_clusters_per_level.
   */
  const std::vector< lo > *
  get_n_associated_leaves_and_aux_clusters_per_level( ) const {
    return _n_associated_leaves_and_aux_clusters_per_level;
  }

  /**
   * Returns the value of @ref _max_relative_space_level_s2ms.
   */
  lo get_max_relative_space_level_s2ms( ) const {
    return _max_relative_space_level_s2ms;
  }

  /**
   * Returns the value of @ref _max_relative_space_level_ls2t.
   */
  lo get_max_relative_space_level_ls2t( ) const {
    return _max_relative_space_level_ls2t;
  }

  /**
   * Adds a space-time cluster to @p _associated_additional_spacetime_clusters.
   * @param[in] cluster Cluster which is added to the list.
   * @note If @p _associated_additional_spacetime_clusters is not allocated this
   * is done here.
   */
  void add_associated_additional_spacetime_leaf(
    general_spacetime_cluster * cluster ) {
    if ( _associated_additional_spacetime_leaves == nullptr ) {
      _associated_additional_spacetime_leaves
        = new std::vector< general_spacetime_cluster * >( );
    }

    _associated_additional_spacetime_leaves->push_back( cluster );
  }

  /**
   * Returns a pointer to the const list of associated spacetime clusters.
   */
  const std::vector< general_spacetime_cluster * > *
  get_associated_spacetime_clusters( ) const {
    return _associated_spacetime_clusters;
  }

  /**
   * Returns a pointer to the list of associated spacetime clusters.
   */
  std::vector< general_spacetime_cluster * > *
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
   * Sets the number of associated spacetime leaf clusters to the given value.
   * @param[in] n_associated_leaves Value to be set.
   */
  void set_n_associated_leaves( lou n_associated_leaves ) {
    _n_associated_leaves = n_associated_leaves;
  }

  /**
   * Adds an entry to the map from process ids to moments.
   * @param[in] proc_id Key value of the element to be added
   * @note @p _map_to_moment_receive_buffers is allocated if it does not exist
   * allready.
   * @warning The actual moments for this clusters have to exist already.
   * Otherwise @ref _n_st_clusters_w_moments is 0 and no receive buffers are
   * allocated.
   */
  void add_receive_buffer( const lo proc_id ) {
    if ( _map_to_moment_receive_buffers == nullptr ) {
      _map_to_moment_receive_buffers = new std::map< lo, lou >( );
    }
    sc * moment_buffer
      = new sc[ _contribution_size * _n_st_clusters_w_moments ];
    _associated_moments_receive_buffers.push_back( moment_buffer );
    _map_to_moment_receive_buffers->insert(
      { proc_id, _associated_moments_receive_buffers.size( ) - 1 } );
  }

  /**
   * Returns a pointer to the position where the moments of an extraneous
   * process are stored.
   * @param[in] proc_id Id of the extraneous process.
   */
  sc * get_extraneous_moment_pointer( lo proc_id ) {
    return _associated_moments_receive_buffers[ _map_to_moment_receive_buffers
                                                  ->at( proc_id ) ];
  }

  /**
   * Returns the number of associated moment receive buffers. It corresponds
   * to the number of children of the current scheduling time cluster which are
   * handled by a different process than the current cluster.
   */
  lou get_n_associated_moment_receive_buffers( ) {
    return _associated_moments_receive_buffers.size( );
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
    scheduling_time_cluster * left_child = nullptr;
    scheduling_time_cluster * right_child = nullptr;
    for ( auto child : *_children ) {
      char * status_pointer;
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

  /**
   * Recursively traverses the implicitly given tree structure and constructs
   * a vector which contains the cluster bounds in the tree in the usual tree
   * format.
   * @param[out] cluster_bounds_vector  Vector in which the cluster bounds are
   *                                    stored.
   */
  void append_cluster_bound_vector_recursively(
    std::vector< sc > & cluster_bounds_vector ) const {
    char left_child_status = 0;
    char right_child_status = 0;
    sc left_child_lower_bound( 0.0 ), left_child_upper_bound( 0.0 );
    sc right_child_lower_bound( 0.0 ), right_child_upper_bound( 0.0 );
    scheduling_time_cluster * left_child = nullptr;
    scheduling_time_cluster * right_child = nullptr;
    for ( auto child : *_children ) {
      // determine which child it is and remember its bounds and status.
      char * status_pointer;
      if ( child->get_configuration( ) == 0 ) {
        left_child = child;
        status_pointer = &left_child_status;
        left_child_lower_bound = child->get_center( ) - child->get_half_size( );
        left_child_upper_bound = child->get_center( ) + child->get_half_size( );
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
        _n_st_clusters_w_moments = _associated_spacetime_clusters->size( );
        // we do not have to allocate space-time moments for auxiliary clusters
        // associated with the current time cluster, so we check if such
        // clusters exist, and allocate moments only for other clusters
        if ( _n_associated_leaves_and_aux_clusters_per_level != nullptr
          && _n_associated_leaves_and_aux_clusters_per_level->size( ) > 1 ) {
          _n_st_clusters_w_moments
            = ( *_n_associated_leaves_and_aux_clusters_per_level )[ 0 ];
        }
        _associated_moments = new sc[ moment_size * _n_st_clusters_w_moments ];
        for ( lo i = 0; i < _n_st_clusters_w_moments; ++i ) {
          general_spacetime_cluster * current_spacetime_cluster
            = ( *_associated_spacetime_clusters )[ i ];
          current_spacetime_cluster->set_pointer_to_moment(
            &_associated_moments[ i * moment_size ] );
        }
      }
    }
  }

  /**
   * Sets all associated moments to 0.
   */
  void clear_associated_moments( ) {
    if ( _associated_moments != nullptr ) {
      for ( lou i = 0; i < _contribution_size * _n_st_clusters_w_moments;
            ++i ) {
        _associated_moments[ i ] = 0.0;
      }
    }
  }

  /**
   * Returns a pointer to the associated moments.
   */
  sc * get_associated_moments( ) {
    return _associated_moments;
  }

  /**
   * Returns the value of @ref _n_st_clusters_w_moments.
   */
  lo get_n_st_clusters_w_moments( ) const {
    return _n_st_clusters_w_moments;
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
        _n_st_clusters_w_local_contributions
          = _associated_spacetime_clusters->size( );
        // we do not have to allocate space-time local contributions for
        // auxiliary clusters associated with the current time cluster, so we
        // check if such clusters exist, and allocate local contributions only
        // for other clusters
        if ( _n_associated_leaves_and_aux_clusters_per_level != nullptr
          && _n_associated_leaves_and_aux_clusters_per_level->size( ) > 1 ) {
          _n_st_clusters_w_local_contributions
            = ( *_n_associated_leaves_and_aux_clusters_per_level )[ 0 ];
        }
        _associated_local_contributions
          = new sc[ contribution_size * _n_st_clusters_w_local_contributions ];
        for ( lo i = 0; i < _n_st_clusters_w_local_contributions; ++i ) {
          general_spacetime_cluster * current_spacetime_cluster
            = ( *_associated_spacetime_clusters )[ i ];
          current_spacetime_cluster->set_pointer_to_local_contribution(
            &_associated_local_contributions[ i * contribution_size ] );
        }
      }
    }
  }

  /**
   * Sets all associated local contributions to 0.
   */
  void clear_associated_local_contributions( ) {
    if ( _associated_local_contributions != nullptr ) {
      for ( lou i = 0;
            i < _contribution_size * _n_st_clusters_w_local_contributions;
            ++i ) {
        _associated_local_contributions[ i ] = 0.0;
      }
    }
  }

  /**
   * Returns a pointer to the associated local contributions.
   */
  sc * get_associated_local_contributions( ) {
    return _associated_local_contributions;
  }

  /**
   * Returns the value of @ref _n_st_clusters_w_moments.
   */
  lo get_n_st_clusters_w_local_contributions( ) const {
    return _n_st_clusters_w_local_contributions;
  }

  /**
   * Allocates an array containing all the spatial moments of the associated
   * spacetime clusters for all necessary spatial levels.
   *
   * In addition, for each involved associated spacetime cluster it sets the
   * pointer to the appropriate spatial moment, and initializes the value of
   * @ref _max_relative_space_level_s2ms.
   * @param[in] spatial_contribution_size Size of the spatial moment of a single
   * space-time cluster.
   * @param[in] max_rel_space_level  Maximal relative spatial level for which
   * spatial moments have to be allocated.
   * @note Before calling this routine the spacetime clusters have to be
   * associated with this cluster.
   * @note The reference level for the relative spatial level is the spatial
   * level of the non-auxiliary clusters associated with the current time
   * cluster. (i.e. the "natural" spatial level)
   */
  void allocate_associated_spatial_moments(
    const lou spatial_contribution_size, const lo max_rel_space_level ) {
    if ( _associated_spacetime_clusters != nullptr ) {
      // allocate the spatial moments if they are not allocated already
      if ( _assoc_spatial_moments == nullptr ) {
        _spatial_contribution_size = spatial_contribution_size;
        _max_relative_space_level_s2ms = max_rel_space_level;
        _n_st_clusters_w_spatial_moments = 0;
        for ( lo i = 0; i <= max_rel_space_level; ++i ) {
          _n_st_clusters_w_spatial_moments
            += ( *_n_associated_leaves_and_aux_clusters_per_level )[ i ];
        }
        _assoc_spatial_moments = new sc[ spatial_contribution_size
          * _n_st_clusters_w_spatial_moments ];
        // assign the individual spatial moments to the corresponding
        // space-time clusters
        for ( lo i = 0; i < _n_st_clusters_w_spatial_moments; ++i ) {
          general_spacetime_cluster * current_spacetime_cluster
            = ( *_associated_spacetime_clusters )[ i ];
          current_spacetime_cluster->set_pointer_to_spatial_moments(
            &( _assoc_spatial_moments[ i * spatial_contribution_size ] ) );
        }
      }
    }
  }

  /**
   * Sets all associated spatial moments to 0.
   */
  void clear_associated_spatial_moments( ) {
    if ( _assoc_spatial_moments != nullptr ) {
      for ( lou j = 0;
            j < _spatial_contribution_size * _n_st_clusters_w_spatial_moments;
            ++j ) {
        _assoc_spatial_moments[ j ] = 0.0;
      }
    }
  }

  /**
   * Returns the spatial moments @ref _assoc_spatial_moments.
   */
  sc * get_associated_spatial_moments( ) {
    return _assoc_spatial_moments;
  }

  /**
   * Returns the (const) spatial moments @ref _assoc_spatial_moments.
   */
  const sc * get_associated_spatial_moments( ) const {
    return _assoc_spatial_moments;
  }

  /**
   * Returns the value of @ref _n_st_clusters_w_spatial_moments.
   */
  lo get_n_st_clusters_w_spatial_moments( ) const {
    return _n_st_clusters_w_spatial_moments;
  }

  /**
   * Determines the associated space-time clusters, for which auxiliary spatial
   * moments are needed. The spatial moments are allocated in a single large
   * array, and assigned to the correct space-time clusters (and their coarse
   * ancestors at level @p max_rel_space_level_s_moments ).
   *
   * In addition, the routine initializes @ref _assoc_aux_s2ms_cluster_pairs,
   * which is used to access the determined space-time clusters in the
   * application of a pFMM matrix.
   * @param[in] spatial_contribution_size Size of the auxiliary spatial moment
   * of a single space-time cluster.
   * @param[in] max_rel_space_level_s_moments  Maximal relative spatial level
   * for which spatial moments have already been allocated. Auxiliary spatial
   * moments are only allocated for clusters with higher levels.
   */
  void allocate_associated_aux_spatial_moments(
    const lou spatial_contribution_size,
    const lo max_rel_space_level_s_moments ) {
    // determine the space-time clusters for which auxiliary S2M operations
    // have to be executed.
    lo n_clusters_before_max_rel_lvl = 0;
    for ( lo i = 0; i < max_rel_space_level_s_moments; ++i ) {
      n_clusters_before_max_rel_lvl
        += ( *_n_associated_leaves_and_aux_clusters_per_level )[ i ];
    }
    lo idx_offset = n_clusters_before_max_rel_lvl
      + ( *_n_associated_leaves_and_aux_clusters_per_level )
        [ max_rel_space_level_s_moments ];
    for ( lou rel_space_level = max_rel_space_level_s_moments + 1;
          rel_space_level
          < _n_associated_leaves_and_aux_clusters_per_level->size( );
          ++rel_space_level ) {
      for ( lo cluster_idx = 0;
            cluster_idx < ( *_n_associated_leaves_and_aux_clusters_per_level )
              [ rel_space_level ];
            ++cluster_idx ) {
        general_spacetime_cluster * current_st_cluster
          = ( *_associated_spacetime_clusters )[ idx_offset + cluster_idx ];
        if ( current_st_cluster->get_n_children( ) == 0 ) {
          // determine the ancestor of the current_st_cluster which has the
          // relative space level max_rel_space_level_s_moments by ascending in
          // the tree.
          general_spacetime_cluster * current_ancestor_for_aux_s2ms
            = current_st_cluster;
          for ( lou lvl = 0;
                lvl < rel_space_level - max_rel_space_level_s_moments; ++lvl ) {
            current_ancestor_for_aux_s2ms = current_st_cluster->get_parent( );
          }
          add_pair_to_assoc_aux_s2ms_cluster_pairs(
            { idx_offset + cluster_idx, current_ancestor_for_aux_s2ms } );
        }
      }
      idx_offset += ( *_n_associated_leaves_and_aux_clusters_per_level )
        [ rel_space_level ];
    }

    _n_st_clusters_w_aux_spatial_moments
      = _assoc_aux_s2ms_cluster_pairs->size( );
    _assoc_aux_spatial_moments = new sc[ _n_st_clusters_w_aux_spatial_moments
      * spatial_contribution_size ];
    _spatial_contribution_size = spatial_contribution_size;

    // assign the individual auxiliary spatial moments to the appropriate
    // space-time clusters
    lo moment_counter = 0;
    for ( lo cluster_idx = 0;
          cluster_idx < ( *_n_associated_leaves_and_aux_clusters_per_level )
            [ max_rel_space_level_s_moments ];
          ++cluster_idx ) {
      general_spacetime_cluster * current_st_cluster
        = ( *_associated_spacetime_clusters )[ cluster_idx ];
      if ( current_st_cluster->get_n_children( ) > 0 ) {
        current_st_cluster->set_pointer_to_aux_spatial_moments(
          &( _assoc_aux_spatial_moments[ moment_counter
            * spatial_contribution_size ] ) );
        lo previous_moment_counter = moment_counter;
        set_pointers_to_aux_spatial_moments_recursively(
          current_st_cluster, moment_counter );
        current_st_cluster->set_n_aux_spatial_moments(
          moment_counter - previous_moment_counter );
      }
    }
  }

  /**
   * Auxiliary routine that is used to assign the appropriate parts of the large
   * array @ref _assoc_aux_spatial_moments to the auxiliary spatially refined
   * leaf clusters.
   *
   * The routine is based on a recursive subtree traversal.
   * @param[in] current_st_cluster  Current space-time cluster. If it is a leaf,
   * an auxiliary moment is assigned to it, otherwise its subtree is visited.
   * @param[in] current_moment_index  Indicates which part of the global array
   * of auxiliary spatial moments to assign next.
   */
  void set_pointers_to_aux_spatial_moments_recursively(
    general_spacetime_cluster * current_st_cluster,
    lo & current_moment_index ) {
    if ( current_st_cluster->get_n_children( ) > 0 ) {
      for ( auto st_child : *current_st_cluster->get_children( ) ) {
        set_pointers_to_aux_spatial_moments_recursively(
          st_child, current_moment_index );
      }
    } else {
      current_st_cluster->set_pointer_to_aux_spatial_moments(
        &( _assoc_aux_spatial_moments[ current_moment_index
          * _spatial_contribution_size ] ) );
      current_moment_index++;
      current_st_cluster->set_n_aux_spatial_moments( 1 );
    }
  }

  /**
   * Sets all associated auxiliary spatial moments to 0.
   */
  void clear_associated_aux_spatial_moments( ) {
    if ( _assoc_aux_spatial_moments != nullptr ) {
      for ( lou j = 0; j
            < _spatial_contribution_size * _n_st_clusters_w_aux_spatial_moments;
            ++j ) {
        _assoc_aux_spatial_moments[ j ] = 0.0;
      }
    }
  }

  /**
   * Allocates an array containing all the spatial local contributions of the
   * associated spacetime clusters for all necessary spatial levels.
   *
   * In addition, for each involved associated spacetime cluster it sets the
   * pointer to the appropriate spatial local contribution, and initializes
   * the value of @ref _max_relative_space_level_ls2t.
   * @param[in] spatial_contribution_size Size of the spatial local
   * contribution of a single space-time cluster.
   * @param[in] max_rel_space_level  Maximal relative spatial level for which
   * spatial moments have to be allocated.
   * @note Before calling this routine the spacetime clusters have to be
   * associated with this cluster.
   * @note The reference level for the relative spatial level is the spatial
   * level of the non-auxiliary clusters associated with the current time
   * cluster. (i.e. the "natural" spatial level)
   */
  void allocate_associated_spatial_local_contributions(
    const lou spatial_contribution_size, const lo max_rel_space_level ) {
    if ( _associated_spacetime_clusters != nullptr ) {
      // allocate the spatial local contributions if they are not allocated
      // already
      _spatial_contribution_size = spatial_contribution_size;
      _max_relative_space_level_ls2t = max_rel_space_level;
      _n_st_clusters_w_spatial_local_contributions = 0;
      for ( lo i = 0; i <= max_rel_space_level; ++i ) {
        _n_st_clusters_w_spatial_local_contributions
          += ( *_n_associated_leaves_and_aux_clusters_per_level )[ i ];
      }
      _assoc_spatial_local_contributions = new sc[ spatial_contribution_size
        * _n_st_clusters_w_spatial_local_contributions ];
      // assign the individual spatial local contributions to the
      // corresponding space-time clusters
      for ( lo i = 0; i < _n_st_clusters_w_spatial_local_contributions; ++i ) {
        general_spacetime_cluster * current_spacetime_cluster
          = ( *_associated_spacetime_clusters )[ i ];
        current_spacetime_cluster->set_pointer_to_spatial_local_contributions(
          &( _assoc_spatial_local_contributions[ i
            * spatial_contribution_size ] ) );
      }
    }
  }

  /**
   * Sets all associated spatial local contributions to 0.
   */
  void clear_associated_spatial_local_contributions( ) {
    if ( _assoc_spatial_local_contributions != nullptr ) {
      for ( lou j = 0; j < _spatial_contribution_size
              * _n_st_clusters_w_spatial_local_contributions;
            ++j ) {
        _assoc_spatial_local_contributions[ j ] = 0.0;
      }
    }
  }

  /**
   * Returns the spatial local contributions
   * @ref _assoc_spatial_local_contributions.
   */
  sc * get_associated_spatial_local_contributions( ) {
    return _assoc_spatial_local_contributions;
  }

  /**
   * Returns the (const) spatial local contributions
   * @ref _assoc_spatial_local_contributions.
   */
  const sc * get_associated_spatial_local_contributions( ) const {
    return _assoc_spatial_local_contributions;
  }

  /**
   * Returns the value of @ref _n_st_clusters_w_spatial_local_contributions.
   */
  lo get_n_st_clusters_w_spatial_local_contributions( ) const {
    return _n_st_clusters_w_spatial_local_contributions;
  }

  /**
   * Prints info of the object.
   */
  void print( [[maybe_unused]] lo executing_process_id = -1,
    std::ostream & stream = std::cout ) const {
    stream << "level: " << _level << ", center: " << _center
           << ", half size: " << _half_size
           << ", global_index: " << _global_index
           << ", proc_id: " << _process_id;
    if ( _global_leaf_status ) {
      stream << ", is global leaf";
    }
    // if ( _nearfield_list != nullptr ) {
    //   stream << ", nearfield: ";
    //   for ( lou i = 0; i < _nearfield_list->size( ); ++i ) {
    //     stream << "(" << ( *_nearfield_list )[ i ]->get_level( ) << ", "
    //               << ( *_nearfield_list )[ i ]->get_global_index( ) << "),
    //               ";
    //   }
    // }
    // if ( _m2t_list != nullptr ) {
    //   stream << "m2t list: ";
    //   for ( lou i = 0; i < _m2t_list->size( ); ++i ) {
    //     stream << "(" << ( *_m2t_list )[ i ]->get_level( ) << ", "
    //               << ( *_m2t_list )[ i ]->get_global_index( ) << "), ";
    //   }
    // }
    // if ( _s2l_list != nullptr ) {
    //   stream << "s2l list: ";
    //   for ( lou i = 0; i < _s2l_list->size( ); ++i ) {
    //     stream << "(" << ( *_s2l_list )[ i ]->get_level( ) << ", "
    //               << ( *_s2l_list )[ i ]->get_global_index( ) << "), ";
    //   }
    // }
    // if ( _interaction_list != nullptr ) {
    //   stream << "interaction_list: ";
    //   for ( lou i = 0; i < _interaction_list->size( ); ++i ) {
    //     stream << "(" << ( *_interaction_list )[ i ]->get_level( ) << ", "
    //               << ( *_interaction_list )[ i ]->get_global_index( )
    //               << "), ";
    //   }
    // }
    // if ( _send_list != nullptr ) {
    //   stream << "send_list: ";
    //   for ( lou i = 0; i < _send_list->size( ); ++i ) {
    //     stream << "(" << ( *_send_list )[ i ]->get_level( ) << ", "
    //               << ( *_send_list )[ i ]->get_global_index( )
    //               << "), ";
    //   }
    // }
    // if ( _s2l_send_list != nullptr ) {
    //   stream << ", s2l send list: ";
    //   for ( lou i = 0; i < _s2l_send_list->size( ); ++i ) {
    //     stream << "(" << ( *_s2l_send_list )[ i ]->get_level( ) << ", "
    //            << ( *_s2l_send_list )[ i ]->get_global_index( ) << "), ";
    //   }
    // }
    // stream << ", upward_path_counter: " << _upward_path_counter;
    // stream << ", downward_path_status: " << (lo) _downward_path_status;
    // stream << ", m2l counter: " << _m2l_counter;

    // if ( _associated_spacetime_clusters != nullptr ) {
    //   std::cout << ", associated clusters: ";
    //   for ( auto st_cluster : *_associated_spacetime_clusters ) {
    //     auto box_coordinates = st_cluster->get_box_coordinate( );
    //     std::cout << "(" << box_coordinates[ 0 ] << ", " <<
    //     box_coordinates[ 1 ]
    //               << ", " << box_coordinates[ 2 ] << ", "
    //               << box_coordinates[ 3 ] << ", " << box_coordinates[ 4 ]
    //               << "), ";
    //   }
    // if ( _process_id == executing_process_id ) {
    //   stream << ", number of associated leaves: " << _n_associated_leaves
    //          << ", number of associated non-leaves: "
    //          << _associated_spacetime_clusters->size( )
    //       - _n_associated_leaves;
    // } else {
    //   stream << ", number of associated clusters: "
    //          << _associated_spacetime_clusters->size( );
    // }
    // }
    // if ( _time_slices != nullptr ) {
    //   stream << ", time slices: ";
    //   for ( auto idx : *_time_slices ) {
    //     stream << idx << ", ";
    //   }
    // }
    stream << std::endl;
  }

  /**
   * Setter for the @ref _pos_in_m_list variable
   * @param[in] index Index of the cluster in
   * @ref besthea::linear_algebra::distributed_pFMM_matrix::_m_list
   */
  void set_pos_in_m_list( lo index ) {
    _pos_in_m_list = index;
  }

  /**
   * Returns position in
   * @ref besthea::linear_algebra::distributed_pFMM_matrix::_m_list
   */
  lo get_pos_in_m_list( ) const {
    return _pos_in_m_list;
  }

  /**
   * Setter for the @ref _pos_in_l_list variable
   * @param[in] index Index of the cluster in
   * @ref besthea::linear_algebra::distributed_pFMM_matrix::_l_list
   */
  void set_pos_in_l_list( lo index ) {
    _pos_in_l_list = index;
  }

  /**
   * Returns position in
   * @ref besthea::linear_algebra::distributed_pFMM_matrix::_l_list
   */
  lo get_pos_in_l_list( ) const {
    return _pos_in_l_list;
  }

  /**
   * Setter for the @ref _pos_in_m2l_list variable
   * @param[in] index Index of the cluster in
   * @ref besthea::linear_algebra::distributed_pFMM_matrix::_m2l_list
   */
  void set_pos_in_m2l_list( lo index ) {
    _pos_in_m2l_list = index;
  }

  /**
   * Returns position in
   * @ref besthea::linear_algebra::distributed_pFMM_matrix::_m2l_list
   */
  lo get_pos_in_m2l_list( ) const {
    return _pos_in_m2l_list;
  }

  /**
   * Setter for the @ref _pos_in_s2l_list variable
   * @param[in] index Index of the cluster in
   * @ref besthea::linear_algebra::distributed_pFMM_matrix::_s2l_list
   */
  void set_pos_in_s2l_list( lo index ) {
    _pos_in_s2l_list = index;
  }

  /**
   * Returns position in
   * @ref besthea::linear_algebra::distributed_pFMM_matrix::_s2l_list
   */
  lo get_pos_in_s2l_list( ) const {
    return _pos_in_s2l_list;
  }

 private:
  sc _center;                         //!< Center of the cluster.
  sc _half_size;                      //!< Half size of the cluster.
  scheduling_time_cluster * _parent;  //!< Parent of the cluster.
  short _left_right;  //!< Indicates if the child is the left (0), or right
                      //!< (1) child of its parent (-1 for root).
  std::vector< scheduling_time_cluster * > *
    _children;       //!< Children of the cluster.
  lo _level;         //!< Level within the cluster tree.
  lo _global_index;  //!< Global index of the cluster. The children of a
                     //!< cluster with index k have the indices 2k+1 and 2k+2.
  lo _process_id;    //!< Id of the process to which the cluster is assigned.
  std::vector< lo > * _time_slices;  //!< global indices of the cluster's time
                                     //!< slices (only set for leaf clusters)
  bool _global_leaf_status;  //!< indicates whether the cluster is a leaf (1)
                             //!< or non-leaf in a global tree structure
  bool
    _mesh_available;  //!< Indicates whether the process who owns the cluster
                      //!< has access to the corresponding mesh. Only relevant
                      //!< in a distribution tree in a distributed spacetime
                      //!< tensor mesh. It is set to true for leaf clusters
                      //!< which are either local or in the nearfield of local
                      //!< clusters. It is set in
                      //!< @ref
                      //!< distributed_spacetime_tensor_mesh::find_slices_to_load.
  std::vector< scheduling_time_cluster * > *
    _nearfield_list;  //!< Nearfield of the cluster.
  std::vector< scheduling_time_cluster * > *
    _interaction_list;  //!< Interaction list of the cluster.
  std::vector< scheduling_time_cluster * > *
    _m2t_list;  //!< List of source clusters for which m2t operations have to
                //!< be executed.
  std::vector< scheduling_time_cluster * > *
    _s2l_list;  //!< List of source clusters for which s2l operations have to
                //!< be executed.
  std::vector< scheduling_time_cluster * > *
    _send_list;  //!< Contains all clusters in whose interaction list the
                 //!< cluster is contained.
  std::vector< scheduling_time_cluster * > *
    _m2t_send_list;  //!< Contains all clusters in whose m2t list the
                     //!< cluster is contained.
  std::vector< scheduling_time_cluster * > *
    _s2l_send_list;  //!< Contains all clusters in whose s2l list the cluster
                     //!< is contained.
  char _essential_status;    //!< Indicates the status of a cluster in a
                             //!< distributed tree. Possible status are:
                             //!< - 0: not essential
                             //!< - 1: essential for time cluster only
                             //!< - 2: essential for time and space-time cluster
                             //!< - 3: local, i.e. directly essential
                             //!< The status is assigned when the tree
                             //!< containing the cluster is reduced to the
                             //!< locally essential tree, see
                             //!< @ref tree_structure::reduce_2_essential.
  bool _active_upward_path;  //!< Indicates if the cluster is active in the
                             //!< upward path of the FMM.
  bool _active_downward_path;  //!< Indicates if the cluster is active in the
                               //!< downward path of the FMM.
  char
    _status_initial_op_downward_path;  //!< Indicates if the cluster is active
                                       //!< in the downward path of the FMM
                                       //!< for initial potential operators.
                                       //!< This is the case if its subtree
                                       //!< contains clusters associated with
                                       //!< local space-time leaf clusters.
  lo _upward_path_counter;  //!< Used to keep track of the dependencies in the
                            //!< upward path. If it is 0, the dependencies are
                            //!< fulfilled.
  slou _m2l_counter;  //!< Used to keep track of the completed m2l operations
                      //!< in pFMM.
  char _s2l_execution_status;  //!< Used to check the status of s2l operations
                               //!< in pFMM (0: not ready, 1: ready for
                               //!< execution, 2: executed).
  char _m2t_execution_status;  //!< Used to check the status of m2t operations
                               //!< in pFMM (0: not executed, 1: executed).
  char
    _downward_path_status;  //!< Used to keep track of the status in the
                            //!< downward path. Three status are distinguished
                            //!< - 0: L2L not executed,
                            //!< - 1: L2L executed, local contributions not
                            //!<      ready,
                            //!< - 2: local contributions ready.
  std::vector< general_spacetime_cluster * > *
    _associated_spacetime_clusters;  //!< List of space-time clusters
                                     //!< associated to the scheduling time
                                     //!< cluster.
  std::vector< lo > *
    _n_associated_leaves_and_aux_clusters_per_level;  //!< List that contains
                                                      //!< the numbers of all
                                                      //!< associated leaves
                                                      //!< and auxiliary
                                                      //!< clusters per level
                                                      //!< (entry 0 contains
                                                      //!< the number of
                                                      //!< leaves, entry 1 the
                                                      //!< number of one time
                                                      //!< spatially refined
                                                      //!< clusters,
                                                      //!< ...).
  lo _max_relative_space_level_s2ms;  //!< Maximal relative spatial level for
                                      //!< which spatial moments have to be
                                      //!< computed (Relative to the "natural"
                                      //!< spatial level of the time cluster).
  lo _max_relative_space_level_ls2t;  //!< Maximal relative spatial level for
                                      //!< which spatial local contributions
                                      //!< have to be evaluated (Relative to
                                      //!< the "natural" spatial level of the
                                      //!< cluster).
  std::vector< general_spacetime_cluster * > *
    _associated_additional_spacetime_leaves;  //!< List of space-time clusters
                                              //!< associated to the
                                              //!< scheduling time cluster.
  lou _n_associated_leaves;  //!< Number of associated space-time leaf
                             //!< clusters. These are first in the list of
                             //!< associated space-time clusters.

  sc * _associated_moments;     //!< Array containing all the moments of the
                                //!< associated general spacetime clusters.
  lo _n_st_clusters_w_moments;  //!< Number of associated space-time
                                //!< clusters for which moments are stored.
  sc * _associated_local_contributions;     //!< Array containing all the local
                                            //!< contributions of the associated
                                            //!< general spacetime clusters.
  lo _n_st_clusters_w_local_contributions;  //!< Number of associated
                                            //!< space-time
                                            //!< clusters for which
                                            //!< local contributions are
                                            //!< stored.
  sc *
    _assoc_spatial_moments;  //!< Array containing all spatial moments of
                             //!< associated general space-time clusters,
                             //!< stored in a levelwise manner. (first spatial
                             //!< moments of associated leaves, then those of
                             //!< their spatially refined children, ...)
                             //!< Relevant only for hybrid s2l operations.
  lo _n_st_clusters_w_spatial_moments;      //!< Number of associated space-time
                                            //!< clusters for which spatial
                                            //!< moments are stored.
  sc * _assoc_spatial_local_contributions;  //!< Array containing all spatial
                                            //!< local contributions of
                                            //!< associated general space-time
                                            //!< clusters, stored in a levelwise
                                            //!< manner. Relevant only for
                                            //!< hybrid m2t operations.
  lo _n_st_clusters_w_spatial_local_contributions;  //!< Number of associated
                                                    //!< space-time clusters
                                                    //!< for which spatial
                                                    //!< local contributions
                                                    //!< are stored.
  sc * _assoc_aux_spatial_moments;  //!< Array containing all auxiliary
                                    //!< spatial moments of associated general
                                    //!< space-time clusters.
  lo _n_st_clusters_w_aux_spatial_moments;  //!< Number of associated
                                            //!< space-time clusters for which
                                            //!< auxiliary spatial moments are
                                            //!< stored.
  std::vector< lo > *
    _assoc_nearfield_targets;  //!< Vector containing the indices of all
                               //!< associated space-time clusters for which
                               //!< nearfield operations have to be executed.
  std::vector< lo > *
    _assoc_standard_m2t_targets;  //!< Vector containing the indices of all
                                  //!< associated space-time clusters for
                                  //!< which standard m2t operations have to
                                  //!< be executed.
  std::vector< lo > *
    _assoc_hybrid_m2t_targets;  //!< Vector containing the indices of all
                                //!< associated space-time clusters for which
                                //!< hybrid m2t operations have to be
                                //!< executed.
  std::vector< lo > *
    _assoc_standard_s2l_targets;  //!< Vector containing the indices of all
                                  //!< associated space-time clusters for
                                  //!< which standard s2l operations have to
                                  //!< be executed.
  std::vector< lo > *
    _assoc_hybrid_s2l_targets;  //!< Vector containing the indices of all
                                //!< associated space-time clusters for which
                                //!< hybrid s2l operations have to be
                                //!< executed.
  std::vector< std::pair< lo, general_spacetime_cluster * > > *
    _assoc_aux_s2ms_cluster_pairs;  //!< Vector containing pairs of indices of
                                    //!< all associated space-time clusters for
                                    //!< which auxiliary s2ms operations have to
                                    //!< be executed, and pointers to their
                                    //!< respective ancestors for which the
                                    //!< actual spatial moments are needed.

  lou _contribution_size;  //!< Size of a single contribution (moments or
                           //!< local contribution) in the array of
                           //!< associated contributions
                           //!< @todo Is there a better way to make this
                           //!< value accessible for all clusters without
                           //!< storing it in each instance separately?
  lou _spatial_contribution_size;  //!< Size of single spatial contribution
                                   //!< (moment or local contribution)
  std::vector< sc * >
    _associated_moments_receive_buffers;  //!< In case that moments have to be
                                          //!< received from other processes
                                          //!< they are written into these
                                          //!< buffers.
  std::map< lo, lou > *
    _map_to_moment_receive_buffers;  //!< Map to access the correct position
                                     //!< in
                                     //!< @p
                                     //!< _associated_moments_receive_buffers
                                     //!< for extraneous children.

  lo
    _pos_in_m_list;  //!< auxiliary variable storing position in the m list
                     //!< used in @ref
                     //!< besthea::linear_algebra::distributed_pFMM_matrix::apply
  lo
    _pos_in_l_list;  //!< auxiliary variable storing position in the l list
                     //!< used in @ref
                     //!< besthea::linear_algebra::distributed_pFMM_matrix::apply
  lo
    _pos_in_m2l_list;  //!< auxiliary variable storing position in the m2l
                       //!< list used in @ref
                       //!< besthea::linear_algebra::distributed_pFMM_matrix::apply
  lo
    _pos_in_s2l_list;  //!< auxiliary variable storing position in the s2l
                       //!< list used in @ref
                       //!< besthea::linear_algebra::distributed_pFMM_matrix::apply
  lou _n_ready_m2l_sources;  //!< Number of source clusters whose moments are
                             //!< available for m2l operations. This is an
                             //!< auxiliary variable used in pFMM. If it
                             //!< equals the size of the interaction list, m2l
                             //!< list operations can be started.
  lou _n_ready_m2t_sources;  //!< Number of source clusters whose moments are
                             //!< available for m2t operations. This is an
                             //!< auxiliary variable used in pFMM. If it
                             //!< equals the size of the interaction list, m2t
                             //!< list operations can be started.
};

/**
 * Struct that realizes a comparison (<) between pairs consisting of process
 * ids (lo) and scheduling time clusters.
 */
struct compare_pairs_of_process_ids_and_scheduling_time_clusters {
  /**
   * Operator realizing the comparison < between pairs consisting of process
   * ids (lo) and scheduling time clusters
   *
   * (a, I) < (b, J) if the process ids satisfy a < b or
   * (a == b) and I's global index < J's global index
   * @param[in] first_pair  First pair considered in the comparison
   * @param[in] second_pair Second pair considered in the comparison
   * @return True if first_pair is smaller than second pair.
   */
  bool operator( )(
    const std::pair< lo, besthea::mesh::scheduling_time_cluster * > first_pair,
    const std::pair< lo, besthea::mesh::scheduling_time_cluster * >
      second_pair ) const {
    bool compare_value = false;
    if ( first_pair.first < second_pair.first
      || ( first_pair.first == second_pair.first
        && first_pair.second->get_global_index( )
          < second_pair.second->get_global_index( ) ) ) {
      compare_value = true;
    }
    return compare_value;
  }
};

#endif /* INCLUDE_BESTHEA_SCHEDULING_TIME_CLUSTER_H_ */
