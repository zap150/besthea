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

/** @file general_spacetime_cluster.h
 * @brief Collection of spacetime elements.
 */

#ifndef INCLUDE_BESTHEA_GENERAL_SPACETIME_CLUSTER_H_
#define INCLUDE_BESTHEA_GENERAL_SPACETIME_CLUSTER_H_

#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/full_matrix.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

namespace besthea {
  namespace mesh {
    class general_spacetime_cluster;
  }
}

// forward declaration of distributed_fast_spacetime_be_space and basis
// functions
namespace besthea {
  namespace bem {
    template< class basis_type >
    class distributed_fast_spacetime_be_space;
    class basis_tri_p1;
    class basis_tri_p0;
  }
}

/**
 * Class representing a spacetime cluster (collection of spacetime elements)
 */
class besthea::mesh::general_spacetime_cluster {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< Vector type.

  /**
   * Constructor.
   * @param[in] space_center Spatial center of the cluster
   * @param[in] time_center Temporal center of the cluster
   * @param[in] space_half_size Radius in the spatial dimension
   * @param[in] time_half_size Radius in the temporal dimension
   * @param[in] n_elements Number of space-time elements in the cluster
   * @param[in] parent Pointer to the cluster's parent
   * @param[in] level Level within the cluster tree
   * @param[in] octant Index of the spatial octant within the parent cluster
   * @param[in] coordinate  Coordinates of the box within boxes on given level
   * @param[in] left_right 0 - left temp. child, 1 - right temp. child
   * @param[in] global_time_index Global time index of the temporal component
   *                              of the current general space-time cluster.
   * @param[in] n_space_div Number of splittings in space dimensions.
   * @param[in] n_time_div  Number of splittings in temporal dimension.
   * @param[in] mesh Reference to the underlying distributed spacetime tensor
   * mesh.
   * @param[in] process_id Rank of an MPI process owning the cluster.
   * @param[in] reserve_elements Whether to allocate data for cluster's
   * elements.
   */
  general_spacetime_cluster( const vector_type & space_center, sc time_center,
    const vector_type & space_half_size, sc time_half_size, lo n_elements,
    general_spacetime_cluster * parent, lo level, short octant,
    std::vector< slou > & coordinate, short left_right, lo global_time_index,
    lo n_space_div, lo n_time_div,
    const distributed_spacetime_tensor_mesh & mesh, lo process_id,
    bool reserve_elements = false )
    : _n_elements( n_elements ),
      _n_time_elements( -1 ),
      _n_space_elements( -1 ),
      _time_center( time_center ),
      _space_center( space_center ),
      _time_half_size( time_half_size ),
      _space_half_size( space_half_size ),
      _max_element_space_diameter( -1.0 ),
      _n_space_nodes( 0 ),
      _parent( parent ),
      _children( nullptr ),
      _global_leaf_status( false ),
      _mesh( mesh ),
      _is_mesh_available( false ),
      _elements_are_local( false ),
      _level( level ),
      _octant( octant ),
      _left_right( left_right ),
      _box_coordinate( coordinate ),
      _global_time_index( global_time_index ),
      _process_id( process_id ),
      _n_space_div( n_space_div ),
      _n_time_div( n_time_div ),
      _nearfield_list( nullptr ),
      _spatially_admissible_nearfield_list( nullptr ),
      _interaction_list( nullptr ),
      _m2t_list( nullptr ),
      _s2l_list( nullptr ),
      _n_hybrid_m2t_operations( -1 ),
      _n_hybrid_s2l_operations( -1 ),
      _moment( nullptr ),
      _local_contribution( nullptr ),
      _spatial_moments( nullptr ),
      _spatial_local_contributions( nullptr ),
      _aux_spatial_moments( nullptr ),
      _n_aux_spatial_moments( 0 ),
      _has_additional_spatial_children( false ),
      _is_auxiliary_ref_cluster( false ) {
    if ( reserve_elements ) {
      _elements.reserve( _n_elements );
    }
  }

  general_spacetime_cluster( const general_spacetime_cluster & that ) = delete;

  /**
   * Destructor
   */
  virtual ~general_spacetime_cluster( ) {
    if ( _children != nullptr ) {
      for ( auto it = _children->begin( ); it != _children->end( ); ++it ) {
        if ( *it != nullptr ) {
          delete *it;
        }
      }
      delete _children;
    }
    if ( _nearfield_list != nullptr )
      delete _nearfield_list;
    if ( _spatially_admissible_nearfield_list != nullptr )
      delete _spatially_admissible_nearfield_list;
    if ( _interaction_list != nullptr )
      delete _interaction_list;
    if ( _m2t_list != nullptr )
      delete _m2t_list;
    if ( _s2l_list != nullptr )
      delete _s2l_list;
  }

  /**
   * Returns the distributed spacetime mesh associated with the cluster.
   */
  const distributed_spacetime_tensor_mesh & get_mesh( ) const {
    return _mesh;
  }

  /**
   * Sets the value of @ref _is_mesh_available to a new value.
   * @param[in] new_value New value to set.
   */
  void set_is_mesh_available( const bool new_value ) {
    _is_mesh_available = new_value;
  }

  /**
   * Returns the value of @ref _is_mesh_available.
   */
  bool get_is_mesh_available( ) const {
    return _is_mesh_available;
  }

  /**
   * Sets @ref _max_element_space_diameter to a given value.
   * @param[in] diam  Diameter to be set.
   */
  void set_max_element_space_diameter( sc diam ) {
    _max_element_space_diameter = diam;
  }

  /**
   * Returns the value of @ref _max_element_space_diameter.
   */
  sc get_max_element_space_diameter( ) const {
    return _max_element_space_diameter;
  }

  /**
   * Sets the value of the variable @p _elements_are_local to the given value.
   */
  void set_elements_are_local( bool new_value ) {
    _elements_are_local = new_value;
  }

  /**
   * Returns the value of the variable @p _elements_are_local.
   */
  bool get_elements_are_local( ) const {
    return _elements_are_local;
  }

  /**
   * Adds spacetime element to the cluster.
   * @param[in] idx Global index of the spacetime element in the underlying
   * distributed mesh.
   */
  void add_element( lo idx ) {
    _elements.push_back( idx );
  }

  /**
   * Sets the number of different time steps in the cluster. Furthermore it
   * sets the number of different spatial elements accordingly.
   * @param[in] n_time_elements Number of time steps in the cluster.
   */
  void set_n_time_elements( lo n_time_elements ) {
    _n_time_elements = n_time_elements;
    _n_space_elements = _n_elements / _n_time_elements;
  }

  /**
   * Returns the number of different time steps in the cluster.
   */
  lo get_n_time_elements( ) const {
    return _n_time_elements;
  }

  /**
   * Returns the number of different spatial elements in the cluster.
   */
  lo get_n_space_elements( ) const {
    return _n_space_elements;
  }

  /**
   * Adds cluster's child to the list
   * @param[in] child Child cluster.
   */
  void add_child( general_spacetime_cluster * child ) {
    if ( _children == nullptr ) {
      _children = new std::vector< general_spacetime_cluster * >( );
    }
    _children->push_back( child );
  }

  /**
   * Returns list of cluster's children
   */
  std::vector< general_spacetime_cluster * > * get_children( ) {
    return _children;
  }

  /**
   * Returns number of cluster's children.
   */
  lo get_n_children( ) const {
    if ( _children != nullptr ) {
      return _children->size( );
    } else {
      return 0;
    }
  }

  /**
   * Returns the global leaf status of the current cluster.
   */
  bool is_global_leaf( ) const {
    return _global_leaf_status;
  }

  /**
   * Sets the global leaf status of the current cluster to a given value
   * @param[in] status  Value to be set
   */
  void set_global_leaf_status( bool status ) {
    _global_leaf_status = status;
  }

  /**
   * Returns the parent of the current cluster.
   */
  general_spacetime_cluster * get_parent( ) const {
    return _parent;
  }

  /**
   * Returns the global time index of the current cluster.
   */
  lo get_global_time_index( ) const {
    return _global_time_index;
  }

  /**
   * Adds @p cluster to the nearfield list.
   * @param[in] cluster Cluster to be added.
   */
  void add_to_nearfield_list( general_spacetime_cluster * cluster ) {
    if ( _nearfield_list == nullptr ) {
      _nearfield_list = new std::vector< general_spacetime_cluster * >( );
    }
    _nearfield_list->push_back( cluster );
  }

  /**
   * Adds @p cluster to the spatially admissible nearfield list.
   * @param[in] cluster Cluster to be added.
   */
  void add_to_spatially_admissible_nearfield_list(
    general_spacetime_cluster * cluster ) {
    if ( _spatially_admissible_nearfield_list == nullptr ) {
      _spatially_admissible_nearfield_list
        = new std::vector< general_spacetime_cluster * >( );
    }
    _spatially_admissible_nearfield_list->push_back( cluster );
  }

  /**
   * Adds @p cluster to the interaction list.
   * @param[in] cluster Cluster to be added.
   */
  void add_to_interaction_list( general_spacetime_cluster * cluster ) {
    if ( _interaction_list == nullptr ) {
      _interaction_list = new std::vector< general_spacetime_cluster * >( );
    }
    _interaction_list->push_back( cluster );
  }

  /**
   * Adds @p cluster to the s2l list.
   * @param[in] cluster Cluster to be added.
   */
  void add_to_s2l_list( general_spacetime_cluster * cluster ) {
    if ( _s2l_list == nullptr ) {
      _s2l_list = new std::vector< general_spacetime_cluster * >( );
    }
    _s2l_list->push_back( cluster );
  }

  /**
   * Adds @p cluster to the m2t list.
   * @param[in] cluster Cluster to be added.
   */
  void add_to_m2t_list( general_spacetime_cluster * cluster ) {
    if ( _m2t_list == nullptr ) {
      _m2t_list = new std::vector< general_spacetime_cluster * >( );
    }
    _m2t_list->push_back( cluster );
  }

  /**
   * Returns cluster's interaction list.
   */
  std::vector< general_spacetime_cluster * > * get_interaction_list( ) {
    return _interaction_list;
  }

  /**
   * Returns cluster's nearfield list.
   */
  std::vector< general_spacetime_cluster * > * get_nearfield_list( ) {
    return _nearfield_list;
  }

  /**
   * Returns cluster's spatially admissible nearfield list.
   */
  std::vector< general_spacetime_cluster * > *
  get_spatially_admissible_nearfield_list( ) {
    return _spatially_admissible_nearfield_list;
  }

  /**
   * Returns cluster's s2l list
   */
  std::vector< general_spacetime_cluster * > * get_s2l_list( ) {
    return _s2l_list;
  }

  /**
   * Returns cluster's m2t list
   */
  std::vector< general_spacetime_cluster * > * get_m2t_list( ) {
    return _m2t_list;
  }

  /**
   * Sets the value of @ref _n_hybrid_m2t_operations to a given value.
   * @param[in] value Value to be set.
   */
  void set_n_hybrid_m2t_operations( lo value ) {
    _n_hybrid_m2t_operations = value;
  }

  /**
   * Returns the value of @ref _n_hybrid_m2t_operations
   */
  lo get_n_hybrid_m2t_operations( ) const {
    return _n_hybrid_m2t_operations;
  }

  /**
   * Sets the value of @ref _n_hybrid_s2l_operations to a given value.
   * @param[in] value Value to be set.
   */
  void set_n_hybrid_s2l_operations( lo value ) {
    _n_hybrid_s2l_operations = value;
  }

  /**
   * Returns the value of @ref _n_hybrid_s2l_operations
   */
  lo get_n_hybrid_s2l_operations( ) const {
    return _n_hybrid_s2l_operations;
  }

  /**
   * * Is the leaf cluster additionally refined in space?
   */
  bool has_additional_spatial_children( ) {
    return _has_additional_spatial_children;
  }

  /**
   * @param[in] children Boolean indicating whether the cluster
   *                     is additionally refined in space only.
   */
  void set_has_additional_spatial_children( bool children ) {
    _has_additional_spatial_children = children;
  }

  /**
   * Indicates whether the cluster was created by additional spatial
   * refinement.
   */
  bool is_auxiliary_ref_cluster( ) {
    return _is_auxiliary_ref_cluster;
  }

  /**
   *
   * @param[in] is_auxiliary_ref_cluster Was the cluster created as
   * an auxiliary one during additional spatial subdivision?
   */
  void set_auxiliary_ref_cluster_status( bool is_auxiliary_ref_cluster ) {
    _is_auxiliary_ref_cluster = is_auxiliary_ref_cluster;
  }

  /**
   * Determines temporal admissibility for a given general_spacetime_cluster
   * based on the "neighborhood criterion" (as in Messner's work).
   * @param[in] src_cluster Source cluster whose admissibility is checked.
   * @warning This check of admissibility is only reasonable for clusters at the
   * same level of a tree.
   * @note The current criterion guarantees that the distance in time of two
   * admissible clusters is greater than the minimum of the temporal half sizes
   * of the two clusters.
   * @note The current criterion is also suitable to check for admissibility in
   * adaptive situations (single sided expansions)
   */
  bool determine_temporal_admissibility(
    general_spacetime_cluster * src_cluster ) const {
    bool admissibility = false;
    sc src_center = src_cluster->get_time_center( );
    sc src_half_size = src_cluster->get_time_half_size( );
    sc min_half_size
      = ( src_half_size < _time_half_size ) ? src_half_size : _time_half_size;
    if ( src_center
      < _time_center - _time_half_size - src_half_size - min_half_size ) {
      admissibility = true;
    }
    return admissibility;
  }

  /**
   * Determines if a given source cluster is in the spatial vicinity of the
   * current cluster. This is determined by considering the regular grid in
   * space: A cluster is not in the spatial vicinity if it is more than
   * @p spatial_nearfield_limit steps away from the current cluster along any
   * of the tree spatial directions.
   * @param[in] src_cluster Source cluster which is considered.
   * @param[in] spatial_nearfield_limit Parameter used to define the spatial
   *                                    vicinity.
   */
  bool is_in_spatial_vicinity( general_spacetime_cluster * src_cluster,
    slou spatial_nearfield_limit ) const {
    std::vector< slou > src_box_coordinate = src_cluster->get_box_coordinate( );
    slou x_diff = ( _box_coordinate[ 1 ] > src_box_coordinate[ 1 ] )
      ? (slou) ( _box_coordinate[ 1 ] - src_box_coordinate[ 1 ] )
      : (slou) ( src_box_coordinate[ 1 ] - _box_coordinate[ 1 ] );
    slou y_diff = ( _box_coordinate[ 2 ] > src_box_coordinate[ 2 ] )
      ? (slou) ( _box_coordinate[ 2 ] - src_box_coordinate[ 2 ] )
      : (slou) ( src_box_coordinate[ 2 ] - _box_coordinate[ 2 ] );
    slou z_diff = ( _box_coordinate[ 3 ] > src_box_coordinate[ 3 ] )
      ? (slou) ( _box_coordinate[ 3 ] - src_box_coordinate[ 3 ] )
      : (slou) ( src_box_coordinate[ 3 ] - _box_coordinate[ 3 ] );
    return ( x_diff <= spatial_nearfield_limit
      && y_diff <= spatial_nearfield_limit
      && z_diff <= spatial_nearfield_limit );
  }

  /**
   * Returns center of the cluster.
   * @param[out] space_center Coordinates of the cluster's spatial centroid.
   * @param[out] time_center Coordinate of the cluster's temporal center.
   */
  void get_center( vector_type & space_center, sc & time_center ) const {
    space_center[ 0 ] = _space_center[ 0 ];
    space_center[ 1 ] = _space_center[ 1 ];
    space_center[ 2 ] = _space_center[ 2 ];
    time_center = _time_center;
  }

  /**
   * Returns the temporal center of the cluster.
   */
  sc get_time_center( ) const {
    return _time_center;
  }

  /**
   * Returns half sizes of the cluster.
   * @param[out] space_half_size Spatial half-sizes in individual directions.
   * @param[out] time_half_size Temporal half-size.
   */
  void get_half_size(
    vector_type & space_half_size, sc & time_half_size ) const {
    space_half_size[ 0 ] = _space_half_size[ 0 ];
    space_half_size[ 1 ] = _space_half_size[ 1 ];
    space_half_size[ 2 ] = _space_half_size[ 2 ];
    time_half_size = _time_half_size;
  }

  /**
   * Returns the temporal half size of the cluster.
   */
  sc get_time_half_size( ) const {
    return _time_half_size;
  }

  /**
   * Returns number of elements in the cluster.
   */
  lo get_n_elements( ) const {
    return _n_elements;
  }

  /**
   * Returns global element index in the distributed spacetime mesh.
   * @param[in] idx Index of element in the cluster's internal storage.
   */
  lo get_element( lo idx ) const {
    return _elements[ idx ];
  }

  /**
   * Returns reference to vector of global element indices for elements in the
   * cluster
   */
  const std::vector< lo > & get_all_elements( ) const {
    return _elements;
  }

  /**
   * Sets a number of children and allocates vector of pointers to children.
   * @param[in] n_children Number of cluster's children clusters.
   */
  void set_n_children( lo n_children ) {
    if ( n_children > 0 ) {
      _children = new std::vector< general_spacetime_cluster * >( );
      _children->reserve( n_children );
    } else {
      _children = nullptr;
    }
  }

  /**
   * Shrinks data in children list.
   */
  void shrink_children( ) {
    if ( _children != nullptr ) {
      _children->shrink_to_fit( );
    }
  }

  /**
   * Returns position in a subdivisioning of cluster.
   * @param[inout] octant Spatial octant of the parent cluster
   * @param[inout] left_right Left or right temporal child of parent cluster
   *
   * For parent cluster with center at (0, 0, 0) the following octant ordering
   * holds:
   *
   * oct/coord	1 2 3 4 5 6 7 8
   * 	x		+ - - + + - - +
   * 	y		+ + - - + + - -
   * 	z		+ + + + - - - -
   *
   * For temporal children we have
   * 	0: left child
   * 	1: right child
   */
  void get_position( short & octant, short & left_right ) const {
    octant = _octant;
    left_right = _left_right;
  }

  /**
   * Returns the temporal configuration of the cluster with respect to its
   * parent, i.e. the value of _left_right
   */
  short get_temporal_configuration( ) const {
    return _left_right;
  }

  /**
   * Returns parent's octant of the cluster.
   *
   * For parent cluster with center at (0, 0, 0) the following octant ordering
   * holds:
   *
   * oct/coord	1 2 3 4 5 6 7 8
   * 	x		+ - - + + - - +
   * 	y		+ + - - + + - -
   * 	z		+ + + + - - - -
   */
  short get_spatial_octant( ) const {
    return _octant;
  }

  /**
   * Computes center and half-sizes of the child in a given octant
   * @param[in] octant Spatial suboctant of the cluster.
   * @param[out] new_center Center of the suboctant.
   * @param[out] new_half_size Half-size of the suboctant.
   */
  void compute_spatial_suboctant(
    lo octant, vector_type & new_center, vector_type & new_half_size ) {
    new_half_size[ 0 ] = _space_half_size[ 0 ] / 2;
    new_half_size[ 1 ] = _space_half_size[ 1 ] / 2;
    new_half_size[ 2 ] = _space_half_size[ 2 ] / 2;

    if ( octant == 0 ) {
      new_center[ 0 ] = _space_center[ 0 ] + new_half_size[ 0 ];
      new_center[ 1 ] = _space_center[ 1 ] + new_half_size[ 1 ];
      new_center[ 2 ] = _space_center[ 2 ] + new_half_size[ 2 ];
    } else if ( octant == 1 ) {
      new_center[ 0 ] = _space_center[ 0 ] - new_half_size[ 0 ];
      new_center[ 1 ] = _space_center[ 1 ] + new_half_size[ 1 ];
      new_center[ 2 ] = _space_center[ 2 ] + new_half_size[ 2 ];
    } else if ( octant == 2 ) {
      new_center[ 0 ] = _space_center[ 0 ] - new_half_size[ 0 ];
      new_center[ 1 ] = _space_center[ 1 ] - new_half_size[ 1 ];
      new_center[ 2 ] = _space_center[ 2 ] + new_half_size[ 2 ];
    } else if ( octant == 3 ) {
      new_center[ 0 ] = _space_center[ 0 ] + new_half_size[ 0 ];
      new_center[ 1 ] = _space_center[ 1 ] - new_half_size[ 1 ];
      new_center[ 2 ] = _space_center[ 2 ] + new_half_size[ 2 ];
    } else if ( octant == 4 ) {
      new_center[ 0 ] = _space_center[ 0 ] + new_half_size[ 0 ];
      new_center[ 1 ] = _space_center[ 1 ] + new_half_size[ 1 ];
      new_center[ 2 ] = _space_center[ 2 ] - new_half_size[ 2 ];
    } else if ( octant == 5 ) {
      new_center[ 0 ] = _space_center[ 0 ] - new_half_size[ 0 ];
      new_center[ 1 ] = _space_center[ 1 ] + new_half_size[ 1 ];
      new_center[ 2 ] = _space_center[ 2 ] - new_half_size[ 2 ];
    } else if ( octant == 6 ) {
      new_center[ 0 ] = _space_center[ 0 ] - new_half_size[ 0 ];
      new_center[ 1 ] = _space_center[ 1 ] - new_half_size[ 1 ];
      new_center[ 2 ] = _space_center[ 2 ] - new_half_size[ 2 ];
    } else if ( octant == 7 ) {
      new_center[ 0 ] = _space_center[ 0 ] + new_half_size[ 0 ];
      new_center[ 1 ] = _space_center[ 1 ] - new_half_size[ 1 ];
      new_center[ 2 ] = _space_center[ 2 ] - new_half_size[ 2 ];
    }
  }

  /**
   * Computes the padding of a cluster which is necessary to ensure that all
   * elements are included in it. Clusters are padded uniformly in space (the
   * lower and upper bound is modified by the same amount along all dimensions)
   * and uniformly in time.
   * @param[out] space_padding  Padding in space.
   * @param[out] time_padding   Padding in time.
   * @note  Padding in time is not planned to be used and could probably be
   *        removed.
   */
  void compute_padding( sc & space_padding, sc & time_padding ) const {
    std::vector< linear_algebra::coordinates< 4 > > node_vector;
    node_vector.resize( 6 );

    // determine the appropriate mesh and start index (distinguish whether
    // cluster is local or in the nearfield)
    const spacetime_tensor_mesh * clusters_mesh;
    lo start_idx;
    if ( _mesh.get_rank( ) == _process_id ) {  // local cluster
      clusters_mesh = _mesh.get_local_mesh( );
      start_idx = _mesh.get_local_start_idx( );
    } else {
      clusters_mesh = _mesh.get_nearfield_mesh( );
      start_idx = _mesh.get_nearfield_start_idx( );
    }
    sc * curr_node;

    space_padding = 0.0;
    time_padding = 0.0;

    // loop over elements in cluster
    for ( lo i = 0; i < _n_elements; ++i ) {
      // clusters_mesh->get_nodes(
      //   _mesh.global_2_local( start_idx, _elements[ i ] ), node_vector );
      clusters_mesh->get_nodes(
        _mesh.global_2_local( start_idx, _elements[ i ] ), node_vector );
      // loop over element's nodes
      for ( lo j = 0; j < static_cast< lo >( node_vector.size( ) ); ++j ) {
        curr_node = node_vector.at( j ).data( );
        if ( ( ( _space_center[ 0 ] - _space_half_size[ 0 ] ) - curr_node[ 0 ]
               > space_padding ) ) {
          space_padding
            = _space_center[ 0 ] - _space_half_size[ 0 ] - curr_node[ 0 ];
        }
        if ( ( curr_node[ 0 ] - ( _space_center[ 0 ] + _space_half_size[ 0 ] )
               > space_padding ) ) {
          space_padding
            = curr_node[ 0 ] - ( _space_center[ 0 ] + _space_half_size[ 0 ] );
        }
        if ( ( ( _space_center[ 1 ] - _space_half_size[ 1 ] ) - curr_node[ 1 ]
               > space_padding ) ) {
          space_padding
            = _space_center[ 1 ] - _space_half_size[ 1 ] - curr_node[ 1 ];
        }
        if ( ( curr_node[ 1 ] - ( _space_center[ 1 ] + _space_half_size[ 1 ] )
               > space_padding ) ) {
          space_padding
            = curr_node[ 1 ] - ( _space_center[ 1 ] + _space_half_size[ 1 ] );
        }
        if ( ( ( _space_center[ 2 ] - _space_half_size[ 2 ] ) - curr_node[ 2 ]
               > space_padding ) ) {
          space_padding
            = _space_center[ 2 ] - _space_half_size[ 2 ] - curr_node[ 2 ];
        }
        if ( ( curr_node[ 2 ] - ( _space_center[ 2 ] + _space_half_size[ 2 ] )
               > space_padding ) ) {
          space_padding
            = curr_node[ 2 ] - ( _space_center[ 2 ] + _space_half_size[ 2 ] );
        }

        if ( ( ( _time_center - _time_half_size ) - curr_node[ 3 ]
               > time_padding ) ) {
          time_padding = _time_center - _time_half_size - curr_node[ 3 ];
        }
        if ( ( curr_node[ 3 ] - ( _time_center + _time_half_size )
               > time_padding ) ) {
          time_padding = curr_node[ 3 ] - ( _time_center + _time_half_size );
        }
      }
    }
  }

  /**
   * Returns level of the cluster in the cluster tree.
   */
  lo get_level( ) const {
    return _level;
  }

  /**
   * Returns coordinates of the box within boxes on given level.
   */
  const std::vector< slou > & get_box_coordinate( ) const {
    return _box_coordinate;
  }

  /**
   * Returns id of the process to which the cluster is assigned.
   */
  lo get_process_id( ) const {
    return _process_id;
  }

  /**
   * Allocates memory for elements.
   * @param[in] size Requested size.
   */
  void reserve_elements( lo size ) {
    _elements.reserve( size );
  }

  /**
   * Return numbers of spatial and temporal subdivisioning of the bounding box.
   * @param[out] n_space_div Number of spatial subdivisioning.
   * @param[out] n_time_div  Number of temporal subdivisioning.
   */
  void get_n_divs( lo & n_space_div, lo & n_time_div ) const {
    n_space_div = _n_space_div;
    n_time_div = _n_time_div;
  }

  /**
   * Sets new center of the cluster.
   */
  void set_time_center( sc center ) {
    _time_center = center;
  }

  /**
   * Sets new half size of the cluster.
   */
  void set_time_half_size( sc half_size ) {
    _time_half_size = half_size;
  }

  /**
   * Computes mapping from elements to local nodes and from local to global
   * nodes.
   */
  void compute_node_mapping( ) {
    // check first whether the mapping already exists
    if ( _local_2_global_nodes.size( ) == 0 ) {
      const spacetime_tensor_mesh * current_mesh;
      lo start_idx;
      if ( _elements_are_local ) {
        current_mesh = _mesh.get_local_mesh( );
        start_idx = _mesh.get_local_start_idx( );
      } else {
        current_mesh = _mesh.get_nearfield_mesh( );
        start_idx = _mesh.get_nearfield_start_idx( );
      }

      linear_algebra::indices< 6 > element;
      // _local_2_global_nodes.resize( 6 * _elements.size( ) );
      _elems_2_local_nodes.resize( 6 * _elements.size( ) );

      for ( lou i = 0; i < _elements.size( ); ++i ) {
        lo element_idx = _mesh.global_2_local( start_idx, _elements[ i ] );
        current_mesh->get_element( element_idx, element );
        _local_2_global_nodes.push_back( element[ 0 ] );
        _local_2_global_nodes.push_back( element[ 1 ] );
        _local_2_global_nodes.push_back( element[ 2 ] );
        _local_2_global_nodes.push_back( element[ 3 ] );
        _local_2_global_nodes.push_back( element[ 4 ] );
        _local_2_global_nodes.push_back( element[ 5 ] );
        // _elems_2_local_nodes[ 6 * i ] = 6 * i;
        // _elems_2_local_nodes[ 6 * i + 1 ] = 6 * i + 1;
        // _elems_2_local_nodes[ 6 * i + 2 ] = 6 * i + 2;
        // _elems_2_local_nodes[ 6 * i + 3 ] = 6 * i + 3;
        // _elems_2_local_nodes[ 6 * i + 4 ] = 6 * i + 4;
        // _elems_2_local_nodes[ 6 * i + 5 ] = 6 * i + 5;
      }
      std::sort( _local_2_global_nodes.begin( ), _local_2_global_nodes.end( ) );
      _local_2_global_nodes.erase( std::unique( _local_2_global_nodes.begin( ),
                                     _local_2_global_nodes.end( ) ),
        _local_2_global_nodes.end( ) );

      for ( lou i = 0; i < _elements.size( ); ++i ) {
        lo element_idx = _mesh.global_2_local( start_idx, _elements[ i ] );
        current_mesh->get_element( element_idx, element );

        auto idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 0 ] );
        _elems_2_local_nodes[ 6 * i ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );

        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 1 ] );
        _elems_2_local_nodes[ 6 * i + 1 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );

        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 2 ] );
        _elems_2_local_nodes[ 6 * i + 2 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );

        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 3 ] );
        _elems_2_local_nodes[ 6 * i + 3 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );

        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 4 ] );
        _elems_2_local_nodes[ 6 * i + 4 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );

        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 5 ] );
        _elems_2_local_nodes[ 6 * i + 5 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );
      }
    }
  }

  /**
   * Returns the local space node index corresponding to a given local spacetime
   * node index.
   * @param[in] local_spacetime_node_idx  Local spacetime node index whose
   *                                      corresponding local space node index
   *                                      is computed.
   */
  lo local_spacetime_node_idx_2_local_space_node_idx(
    lo local_spacetime_node_idx ) const {
    return local_spacetime_node_idx % _n_space_nodes;
  }

  /**
   * Returns mapping from local element indices to local cluster node indices.
   */
  const std::vector< lo > & get_elems_2_local_nodes( ) const {
    return _elems_2_local_nodes;
  }

  /**
   * Returns mapping from local cluster node indices to global node indices.
   */
  const std::vector< lo > & get_local_2_global_nodes( ) const {
    return _local_2_global_nodes;
  }

  /**
   * Sets the number of spatial nodes in the cluster.
   * @warning The vector @p _local_2_global_nodes.size( ) and the number of
   * time elements have to be computed before this routine is executed.
   */
  void set_n_space_nodes( ) {
    _n_space_nodes = _local_2_global_nodes.size( ) / ( _n_time_elements + 1 );
  }

  /**
   * Returns the number of spatial nodes contained in the cluster.
   */
  lo get_n_space_nodes( ) const {
    return _n_space_nodes;
  }

  /**
   * Sets the pointer to the moment.
   * @param[in] moment_address Address of the moment (in the array stored in the
   *                           associated scheduling_time_cluster)
   */
  void set_pointer_to_moment( sc * moment_address ) {
    _moment = moment_address;
  }

  /**
   * Returns a pointer to the moment of the current cluster.
   */
  sc * get_pointer_to_moment( ) {
    return _moment;
  }

  /**
   * Returns a pointer to the const moment of the current cluster.
   */
  const sc * get_pointer_to_moment( ) const {
    return _moment;
  }

  /**
   * Sets the pointer to the local_contribution.
   * @param[in] local_contribution_address  Address of the local contribution
   *                                        (in the array stored in the
   *                                        associated scheduling_time_cluster)
   */
  void set_pointer_to_local_contribution( sc * local_contribution_address ) {
    _local_contribution = local_contribution_address;
  }

  /**
   * Returns a pointer to the local contribution of the current cluster.
   */
  sc * get_pointer_to_local_contribution( ) {
    return _local_contribution;
  }

  /**
   * Returns a pointer to the const local contribution of the current cluster.
   */
  const sc * get_pointer_to_local_contribution( ) const {
    return _local_contribution;
  }

  /**
   * Sets the pointer to the spatial moments.
   * @param[in] spatial_moments_address Address of the spatial moments (in the
   * array stored in the associated scheduling_time_cluster)
   */
  void set_pointer_to_spatial_moments( sc * spatial_moments_address ) {
    _spatial_moments = spatial_moments_address;
  }

  /**
   * Returns a pointer to the spatial moments of the current cluster.
   */
  sc * get_pointer_to_spatial_moments( ) {
    return _spatial_moments;
  }

  /**
   * Returns a pointer to the const spatial moments of the current cluster.
   */
  const sc * get_pointer_to_spatial_moments( ) const {
    return _spatial_moments;
  }

  /**
   * Sets the pointer to the spatial local contributions
   * @param[in] spatial_lc_address Address of the spatial local contributions
   * (in the array stored in the associated scheduling_time_cluster)
   */
  void set_pointer_to_spatial_local_contributions( sc * spatial_lc_address ) {
    _spatial_local_contributions = spatial_lc_address;
  }

  /**
   * Returns a pointer to the spatial local contributions of the current
   * cluster.
   */
  sc * get_pointer_to_spatial_local_contributions( ) {
    return _spatial_local_contributions;
  }

  /**
   * Returns a pointer to the const spatial local contributions of the current
   * cluster.
   */
  const sc * get_pointer_to_spatial_local_contributions( ) const {
    return _spatial_local_contributions;
  }

  /**
   * Sets the pointer to the auxiliary spatial moments.
   * @param[in] spatial_moments_address Address of the spatial moments (in the
   * array stored in the associated scheduling_time_cluster)
   */
  void set_pointer_to_aux_spatial_moments( sc * spatial_moments_address ) {
    _aux_spatial_moments = spatial_moments_address;
  }

  /**
   * Returns a pointer to the auxiliary spatial moments of the current cluster.
   */
  sc * get_pointer_to_aux_spatial_moments( ) {
    return _aux_spatial_moments;
  }

  /**
   * Returns a pointer to the (const) auxiliary spatial moments of the current
   * cluster.
   */
  const sc * get_pointer_to_aux_spatial_moments( ) const {
    return _aux_spatial_moments;
  }

  /**
   * Sets @ref _n_aux_spatial_moments to a new value.
   * @param[in] new_value Value to be set.
   */
  void set_n_aux_spatial_moments( lo new_value ) {
    _n_aux_spatial_moments = new_value;
  }

  /**
   * Returns the value of @ref _n_aux_spatial_moments.
   */
  lo get_n_aux_spatial_moments( ) const {
    return _n_aux_spatial_moments;
  }

  /**
   * Returns the number of degrees of freedom in the cluster (depending on the
   * underlying space)
   */
  template< class space_type >
  lo get_n_dofs( ) const;

  /**
   * Returns the number of degrees of freedom in the cluster with respect to
   * space (depending on the underlying space)
   */
  template< class space_type >
  lo get_n_space_dofs( ) const;

  /**
   * Provides local indices for the degrees of freedom in space for a given,
   * possibly transformed, spatial element.
   * In case of p0 elements only the element index is returned
   * In case of p1 elements the local vertex indices are returned
   * @param[in] i_loc_elem  Local spatial element index.
   * @param[in] n_shared_vertices Number of shared vertices in current elements
   *                              (regularized quadrature).
   * @param[in] rotation  Virtual element rotation (regularized quadrature).
   * @param[in] swap  Virtual element inversion (regularized quadrature).
   * @param[out] indices  Local indices for the current (transformed) element.
   */
  template< class space_type >
  void local_elem_to_local_space_dofs( lo i_loc_elem, int n_shared_vertices,
    int rotation, bool swap, std::vector< lo > & indices ) const;

  /**
   * Computes a selected component of the surface curls of p1 basis functions
   * associated to the triangles of the space-time elements contained in the
   * cluster.
   * @param[out] surface_curls_along_dim  Vector in which the surface curls are
   *                                      stored.
   * @tparam dim Used to select the component of the surface curls (0,1 or 2).
   */
  template< slou dim >
  void compute_surface_curls_p1_along_dim(
    std::vector< sc > & surface_curls_along_dim ) const;

  void print_short( ) const {
    std::cout << _n_time_elements << " time elems, " << _n_space_elements
              << " space elems, time level " << _n_time_div << ", space level "
              << _n_space_div << ", coordinates " << _box_coordinate[ 1 ]
              << ", " << _box_coordinate[ 2 ] << ", " << _box_coordinate[ 3 ]
              << ", " << _box_coordinate[ 4 ];
  }

  /**
   * Prints info of the object.
   */
  void print( ) const {
    std::cout << "level: " << _level;
    std::cout << ", temporal center: " << _time_center;
    std::cout << ", temporal half size: " << _time_half_size;
    std::cout << ", n_elements: " << _n_elements;
    std::cout << ", n_time_elements: " << _n_time_elements;
    std::cout << ", n_space_nodes: " << _n_space_nodes;
    std::cout << ", box coordinates: (" << _box_coordinate[ 0 ] << ", "
              << _box_coordinate[ 1 ] << ", " << _box_coordinate[ 2 ] << ", "
              << _box_coordinate[ 3 ] << ", " << _box_coordinate[ 4 ] << ")";
    std::cout << ", space_center: (" << _space_center[ 0 ] << ", "
              << _space_center[ 1 ] << ", " << _space_center[ 2 ] << ")";
    std::cout << ", octant: " << _octant;
    std::cout << ", global_leaf_status: " << _global_leaf_status;
    std::cout << ", process id: " << _process_id;
    std::cout << ", n_space_div: " << _n_space_div;
    std::cout << ", is_auxiliary_ref_cluster: " << _is_auxiliary_ref_cluster;
    // std::cout << ", n_children: " << get_n_children( );
    // std::cout << ", _n_aux_spatial_moments: " << _n_aux_spatial_moments;
    // std::cout << ", elements: ";
    // for ( lou i = 0; i < _elements.size( ); ++i ) {
    //   std::cout << _elements[ i ] << " ";
    // }
    // if ( _nearfield_list != nullptr ) {
    //   std::cout << ", nearfield: n_clusters is " << _nearfield_list->size( );
    //   std::cout << ", list: ";
    //   for ( auto nf_cluster : *_nearfield_list ) {
    //     std::vector< slou > nf_box_coordinate
    //       = nf_cluster->get_box_coordinate( );
    //     lo dummy, nf_cluster_space_div;
    //     nf_cluster->get_n_divs( nf_cluster_space_div, dummy );
    //     std::cout << "(" << nf_box_coordinate[ 0 ] << ", "
    //               << nf_box_coordinate[ 1 ] << ", " << nf_box_coordinate[ 2 ]
    //               << ", " << nf_box_coordinate[ 3 ] << ", "
    //               << nf_box_coordinate[ 4 ] << ", " << nf_cluster_space_div
    //               << "), ";
    //   }
    // }
    // if ( _interaction_list != nullptr ) {
    //   std::cout << ", interaction list: ";
    //   for ( auto ff_cluster : *_interaction_list ) {
    //     std::vector< slou > ff_box_coordinate
    //       = ff_cluster->get_box_coordinate( );
    //     std::cout << "(" << ff_box_coordinate[ 0 ] << ", "
    //               << ff_box_coordinate[ 1 ] << ", " << ff_box_coordinate[ 2 ]
    //               << ", " << ff_box_coordinate[ 3 ] << ", "
    //               << ff_box_coordinate[ 4 ] << "), ";
    //   }
    // }
    // if ( _m2t_list != nullptr ) {
    //   std::cout << "m2t list (total = " << _m2t_list->size( )
    //             << ", n_hybrid = " << _n_hybrid_m2t_operations << "): ";
    //   for ( auto m2t_cluster : *_m2t_list ) {
    //     std::vector< slou > m2t_box_coordinate
    //       = m2t_cluster->get_box_coordinate( );
    //     lo dummy, m2t_cluster_space_div;
    //     m2t_cluster->get_n_divs( m2t_cluster_space_div, dummy );
    //     std::cout << "(" << m2t_box_coordinate[ 0 ] << ", "
    //               << m2t_box_coordinate[ 1 ] << ", " << m2t_box_coordinate[ 2
    //               ]
    //               << ", " << m2t_box_coordinate[ 3 ] << ", "
    //               << m2t_box_coordinate[ 4 ] << ", " << m2t_cluster_space_div
    //               << "), ";
    //   }
    // }

    // if ( _s2l_list != nullptr ) {
    //   std::cout << "s2l list (total = " << _s2l_list->size( )
    //             << ",n_hybrid = " << _n_hybrid_s2l_operations << "): ";
    //   for ( auto s2l_cluster : *_s2l_list ) {
    //     std::vector< slou > s2l_box_coordinate
    //       = s2l_cluster->get_box_coordinate( );
    //     std::cout << "(" << s2l_box_coordinate[ 0 ] << ", "
    //               << s2l_box_coordinate[ 1 ] << ", " << s2l_box_coordinate[ 2
    //               ]
    //               << ", " << s2l_box_coordinate[ 3 ] << ", "
    //               << s2l_box_coordinate[ 4 ] << "), ";
    //   }
    // }

    std::cout << std::endl;
    // if ( _children == nullptr ) {
    //   std::cout << "elements are: " << std::endl;
    //   for ( lou i = 0; i < _elements.size( ); ++i ) {
    //     // output global index of all elements
    //     std::cout << _elements[i] << ", ";
    //   }
    //   std::cout << std::endl;
    // }
  }

  /**
   * Checks whether the current cluster is separated from the other cluster.
   * @param[in] cluster Second cluster for the check.
   */
  bool check_for_separation_in_space(
    const general_spacetime_cluster * cluster ) const {
    lo cluster_space_level, dummy;
    cluster->get_n_divs( cluster_space_level, dummy );
    bool are_separated;
    if ( cluster_space_level == _n_space_div ) {
      are_separated = check_for_separation_in_space_same_space_level( cluster );
    } else if ( cluster_space_level < _n_space_div ) {
      are_separated
        = check_for_separation_in_space_coarser_space_level( cluster );
    } else {
      are_separated
        = cluster->check_for_separation_in_space_coarser_space_level( this );
    }
    return are_separated;
  }

  /**
   * Returns the pointer to the cluster from which this cluster was
   * created by an additional subdivision of the spatially large cluster.
   */
  general_spacetime_cluster * get_original_leaf_ascendant( ) {
    return compute_original_leaf_ascendant( this );
  }

 private:
  lo _n_elements;             //!< number of spacetime elements in the cluster.
  lo _n_time_elements;        //!< number of different time steps in the
                              //!< cluster (components of spacetime elements).
                              //!< This is only set for those clusters, to which
                              //!< elements are assigned.
  lo _n_space_elements;       //!< number of different spatial elements in the
                              //!< cluster (components of spacetime elements).
  sc _time_center;            //!< temporal center of the cluster
  vector_type _space_center;  //!< spatial center of the cluster
  sc _time_half_size;         //!< temporal half size of the cluster
  vector_type _space_half_size;  //!< half sizes of the cluster's faces (in [x,
                                 //!< y, z] directions)
  sc _max_element_space_diameter;  //!< maximal spatial diameter of the elements
                                   //!< contained in the cluster. Default value
                                   //!< is -1.0.
  std::vector< lo > _elements;     //!< indices of the cluster's elements within
                                   //!< global spacetime tensor mesh

  std::vector< lo > _elems_2_local_nodes;   //!< mapping from element nodes
                                            //!< vertices to local node list
  std::vector< lo > _local_2_global_nodes;  //!< mapping from local nodes
                                            //!< to the global ones
  lo _n_space_nodes;  //!< number of spatial nodes in the cluster.
  general_spacetime_cluster * _parent;  //!< parent of the cluster
  std::vector< general_spacetime_cluster * > *
    _children;               //!< children of the current cluster
  bool _global_leaf_status;  //!< indicates whether the cluster is a leaf in the
                             //!< global tree (true) or not (false). Note:
                             //!< Auxiliary spatial refinements are not taken
                             //!< into account, so an auxiliary refined cluster
                             //!< keeps global_leaf_status true, and its
                             //!< descendants' status is set to false.
  const distributed_spacetime_tensor_mesh &
    _mesh;  //!< distributed spacetime mesh associated with the cluster
  bool _is_mesh_available;  //!< Indicates if the mesh corresponding to the
                            //!< elements in this cluster is available for the
                            //!< current MPI rank. This is only set to true for
                            //!< reasonably fine clusters (= for clusters which
                            //!< are leaves at the end of the first phase of the
                            //!< space-time tree construction and all their
                            //!< descendants).
  bool _elements_are_local;  //!< Indicates if the elements contained in the
                             //!< cluster are in the local mesh of the
                             //!< distributed spacetime tensor mesh (true) or
                             //!< in the nearfield mesh (false).
  lo _level;                 //!< level within the cluster tree
  short _octant;             //!< octant of the parent cluster
  short _left_right;         //!< left (0), or right (1) child of the parent
  std::vector< slou >
    _box_coordinate;  //!< coordinates of the box within boxes on given level
  lo _global_time_index;  //!< Global index of the temporal component of the
                          //!< cluster. The children of a cluster with index k
                          //!< have the indices 2k+1 and 2k+2.
                          //!< @todo Include this index into
                          //!< @ref _box_coordinate instead of the 5th entry?
  std::vector< std::vector< lo > >
    _idx_2_coord;   //!< auxiliary mapping from octant indexing to coordinates
  lo _process_id;   //!< rank of an MPI process owning the cluster
  lo _n_space_div;  //!< number of splittings in space dimensions
  lo _n_time_div;   //!< number of splittings in temporal dimension

  std::vector< general_spacetime_cluster * > *
    _nearfield_list;  //!< nearfield list of the cluster; if aca recompression
                      //!< is used and the current cluster is an original leaf
                      //!< or an auxiliary cluster then this list contains only
                      //!< spatially inadmissible clusters.
  std::vector< general_spacetime_cluster * > *
    _spatially_admissible_nearfield_list;  //!< if aca recompression is used and
                                           //!< the current cluster is an
                                           //!< original leaf or an auxiliary
                                           //!< cluster, this list contains only
                                           //!< spatially admissible nearfield
                                           //!< clusters.
  std::vector< general_spacetime_cluster * > *
    _interaction_list;  //!< interaction list of the cluster
  std::vector< general_spacetime_cluster * > *
    _m2t_list;  //!< list of source clusters for which m2t operations have to be
                //!< executed.
  std::vector< general_spacetime_cluster * > *
    _s2l_list;  //!< list of source clusters for which s2l operations have to be
                //!< executed.
  lo _n_hybrid_m2t_operations;  //!< number of clusters in m2t list, for which
                                //!< hybrid m2t operations can be executed. They
                                //!< come first in _m2t_list.
  lo _n_hybrid_s2l_operations;  //!< number of clusters in s2l list, for which
                                //!< hybrid s2l operations can be executed. They
                                //!< come first in _s2l_list.
  sc * _moment;  //!< pointer to the moment of the cluster, which is stored in
                 //!< the associated scheduling_time_cluster
  sc * _local_contribution;  //!< pointer to the local contribution of the
                             //!< cluster, which is stored in the associated
                             //!< scheduling_time_cluster
  sc * _spatial_moments;  //!< pointer to all the spatial moments of the cluster
                          //!< (one per time-step), which are stored in the
                          //!< associated scheduling_time_cluster. The first
                          //!< _spat_contribution_size entries correspond to the
                          //!< spatial moment of the first time-step, ...
  sc * _spatial_local_contributions;  //!< pointer to the spatial local
                                      //!< contributions (one per time-step),
                                      //!< which are stored in the associated
                                      //!< scheduling_time_cluster. Same access
                                      //!< per time-step as for spatial moments.
  sc *
    _aux_spatial_moments;  //!< Pointer to all the auxiliary spatial moments
                           //!< of the cluster, which are stored in the
                           //!< associated scheduling_time_cluster. These are
                           //!< used to subdivide S2M/S2Ms operations for large
                           //!< clusters. If the current cluster is large, the
                           //!< pointer points to a contiguous array of all the
                           //!< auxiliary spatial moments of its descendants.
  lo _n_aux_spatial_moments;  //!< Number of all auxiliary spatial moments to
                              //!< which @ref _aux_spatial_moments points. This
                              //!< is relevant for leaf or auxiliary clusters
                              //!< containing many spatial elements, in which
                              //!< case it indicates the number of leaf
                              //!< descendants.
  bool _has_additional_spatial_children;  //!< determines whether the cluster
                                          //!< has been additionally subdivided
                                          //!< in space
  bool _is_auxiliary_ref_cluster;  //!< was the cluster created by additional
                                   //!< spatial refinement?

  /**
   * Computer the pointer to the cluster from which this cluster was
   * created by an additional subdivision of the spatially large cluster.
   */
  general_spacetime_cluster * compute_original_leaf_ascendant(
    general_spacetime_cluster * cluster ) {
    general_spacetime_cluster * ret;
    if ( !cluster->has_additional_spatial_children( ) ) {
      ret = compute_original_leaf_ascendant( cluster->_parent );
    } else {
      ret = cluster;
    }
    // std::cout << ret <<std::endl;
    return ret;
  }

  /**
   * Checks whether the current cluster is separated from the other cluster
   * which has the same spatial level.
   * @param[in] cluster Second cluster for the check.
   */
  bool check_for_separation_in_space_same_space_level(
    const general_spacetime_cluster * cluster ) const {
    const std::vector< slou > & cluster_coordinates
      = cluster->get_box_coordinate( );
    lo x_diff = _box_coordinate[ 1 ] - cluster_coordinates[ 1 ];
    lo y_diff = _box_coordinate[ 2 ] - cluster_coordinates[ 2 ];
    lo z_diff = _box_coordinate[ 3 ] - cluster_coordinates[ 3 ];
    bool are_separated = ( std::abs( x_diff ) > 1 || std::abs( y_diff ) > 1
      || std::abs( z_diff ) > 1 );
    return are_separated;
  }

  /**
   * Checks whether the current cluster is separated from the other cluster
   * which is coarser in space (i.e. it has a smaller spatial level)
   *
   * @param[in] cluster  Coarser cluster for the check.
   */
  bool check_for_separation_in_space_coarser_space_level(
    const general_spacetime_cluster * cluster ) const {
    // determine the difference in spatial levels
    lo its_space_level, dummy;
    cluster->get_n_divs( its_space_level, dummy );
    lo space_level_diff = _n_space_div - its_space_level;
    // determine the range of spatial box coordinates which the descendants at
    // the spatial level of this cluster would have
    const std::vector< slou > & cluster_coordinates
      = cluster->get_box_coordinate( );
    lo level_factor = 1 << ( space_level_diff );
    lo coord_0_low = level_factor * cluster_coordinates[ 1 ];
    lo coord_0_up = coord_0_low + level_factor - 1;
    lo coord_1_low = level_factor * cluster_coordinates[ 2 ];
    lo coord_1_up = coord_1_low + level_factor - 1;
    lo coord_2_low = level_factor * cluster_coordinates[ 3 ];
    lo coord_2_up = coord_2_low + level_factor - 1;
    // determine the "coordinate difference" for each spatial dimension
    lo x_diff = _box_coordinate[ 1 ] - coord_0_up;
    if ( x_diff < 0 ) {
      x_diff = coord_0_low - _box_coordinate[ 1 ];
    }
    lo y_diff = _box_coordinate[ 2 ] - coord_1_up;
    if ( y_diff < 0 ) {
      y_diff = coord_1_low - _box_coordinate[ 2 ];
    }
    lo z_diff = _box_coordinate[ 3 ] - coord_2_up;
    if ( z_diff < 0 ) {
      z_diff = coord_2_low - _box_coordinate[ 3 ];
    }
    bool are_separated = ( x_diff > 1 || y_diff > 1 || z_diff > 1 );
    return are_separated;
  }
};

/** specialization of @ref besthea::mesh::general_spacetime_cluster::get_n_dofs
 * for p0 basis functions */
template<>
inline lo besthea::mesh::general_spacetime_cluster::get_n_dofs< besthea::bem::
    distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >( )
  const {
  return _n_elements;
}

/** specialization of @ref besthea::mesh::general_spacetime_cluster::get_n_dofs
 * for p1 basis functions
 */
template<>
inline lo besthea::mesh::general_spacetime_cluster::get_n_dofs< besthea::bem::
    distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( )
  const {
  return _n_time_elements * _n_space_nodes;
}

/** specialization of
 * @ref besthea::mesh::general_spacetime_cluster::get_n_space_dofs for p0 basis
 * functions */
template<>
inline lo besthea::mesh::general_spacetime_cluster::get_n_space_dofs< besthea::
    bem::distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >( )
  const {
  return _n_space_elements;
}

/** specialization of
 * @ref besthea::mesh::general_spacetime_cluster::get_n_space_dofs for p1 basis
 * functions */
template<>
inline lo besthea::mesh::general_spacetime_cluster::get_n_space_dofs< besthea::
    bem::distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( )
  const {
  return _n_space_nodes;
}

/** specialization of
 * @ref besthea::mesh::general_spacetime_cluster::local_elem_to_local_space_dofs
 * for p0 basis functions */
template<>
inline void
besthea::mesh::general_spacetime_cluster::local_elem_to_local_space_dofs<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >( lo i_loc_elem,
  [[maybe_unused]] int n_shared_vertices, [[maybe_unused]] int rotation,
  [[maybe_unused]] bool swap, std::vector< lo > & indices ) const {
  indices[ 0 ] = i_loc_elem;
}

/** specialization of
 * @ref besthea::mesh::general_spacetime_cluster::local_elem_to_local_space_dofs
 * for p1 basis functions */
template<>
inline void
besthea::mesh::general_spacetime_cluster::local_elem_to_local_space_dofs<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >( lo i_loc_elem,
  [[maybe_unused]] int n_shared_vertices, [[maybe_unused]] int rotation,
  [[maybe_unused]] bool swap, std::vector< lo > & indices ) const {
  std::vector< lo > local_space_indices
    = { local_spacetime_node_idx_2_local_space_node_idx(
          _elems_2_local_nodes[ 6 * i_loc_elem ] ),
        local_spacetime_node_idx_2_local_space_node_idx(
          _elems_2_local_nodes[ 6 * i_loc_elem + 1 ] ),
        local_spacetime_node_idx_2_local_space_node_idx(
          _elems_2_local_nodes[ 6 * i_loc_elem + 2 ] ) };
  switch ( rotation ) {
    case 0:
      if ( n_shared_vertices == 2 && swap ) {
        indices[ 0 ] = local_space_indices[ 1 ];
        indices[ 1 ] = local_space_indices[ 0 ];
      } else {
        indices[ 0 ] = local_space_indices[ 0 ];
        indices[ 1 ] = local_space_indices[ 1 ];
      }
      indices[ 2 ] = local_space_indices[ 2 ];
      break;
    case 1:
      if ( n_shared_vertices == 2 && swap ) {
        indices[ 0 ] = local_space_indices[ 2 ];
        indices[ 1 ] = local_space_indices[ 1 ];
      } else {
        indices[ 0 ] = local_space_indices[ 1 ];
        indices[ 1 ] = local_space_indices[ 2 ];
      }
      indices[ 2 ] = local_space_indices[ 0 ];
      break;
    case 2:
      if ( n_shared_vertices == 2 && swap ) {
        indices[ 0 ] = local_space_indices[ 0 ];
        indices[ 1 ] = local_space_indices[ 2 ];
      } else {
        indices[ 0 ] = local_space_indices[ 2 ];
        indices[ 1 ] = local_space_indices[ 0 ];
      }
      indices[ 2 ] = local_space_indices[ 1 ];
      break;
  }
}
#endif /* INCLUDE_BESTHEA_GENERAL_SPACETIME_CLUSTER_H_ */
