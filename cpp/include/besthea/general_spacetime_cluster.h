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

// forward declaration of fast_spacetime_be_space and basis functions
namespace besthea {
  namespace bem {
    template< class basis_type >
    class fast_spacetime_be_space;
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
   * @param[in] left_right 0 - left temp. child, 1 - right temp. child
   * @param[in] coordinate Coordinates of the box within boxes on given level
   * @param[in] mesh Reference to the underlying distributed spacetime tensor
   * mesh.
   * @param[in] process_id Rank of an MPI process owning the cluster.
   * @param[in] reserve_elements Whether to allocate data for cluster's
   * elements.
   */
  general_spacetime_cluster( const vector_type & space_center, sc time_center,
    const vector_type & space_half_size, sc time_half_size, lo n_elements,
    general_spacetime_cluster * parent, lo level, short octant,
    std::vector< slou > & coordinate, short left_right, lo n_space_div,
    lo n_time_div, const distributed_spacetime_tensor_mesh & mesh,
    lo process_id, bool reserve_elements = false )
    : _n_elements( n_elements ),
      _time_center( time_center ),
      _space_center( space_center ),
      _time_half_size( time_half_size ),
      _space_half_size( space_half_size ),
      _parent( parent ),
      _children( nullptr ),
      _mesh( mesh ),
      _level( level ),
      _octant( octant ),
      _left_right( left_right ),
      _n_space_div( n_space_div ),
      _n_time_div( n_time_div ),
      _padding( 0.0 ),
      _box_coordinate( coordinate ),
      _process_id( process_id ) {
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
  lo get_n_children( ) {
    if ( _children != nullptr ) {
      return _children->size( );
    } else {
      return 0;
    }
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

  void compute_padding( sc & space_padding, sc & time_padding ) const {
    std::vector< linear_algebra::coordinates< 4 > > node_vector;
    node_vector.resize( 6 );

    sc * curr_node;

    space_padding = 0.0;
    time_padding = 0.0;

    // loop over elements in cluster
    for ( lo i = 0; i < _n_elements; ++i ) {
      _mesh.get_my_mesh( )->get_nodes(
        _mesh.global_2_local( _elements[ i ] ), node_vector );
      // loop over element's nodes
      for ( lo j = 0; j < node_vector.size( ); ++j ) {
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
  lo get_level( ) {
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
   * @param[in] n_space_div Number of spatial subdivisioning.
   * @param[in]
   */
  void get_n_divs( lo & n_space_div, lo & n_time_div ) {
    n_space_div = _n_space_div;
    n_time_div = _n_time_div;
  }

  void set_time_center( sc center ) {
    _time_center = center;
  }

  void set_time_half_size( sc half_size ) {
    _time_half_size = half_size;
  }

 private:
  lo _n_elements;             //!< number of spacetime elements in the cluster
  sc _time_center;            //!< temporal center of the cluster
  vector_type _space_center;  //!< spatial center of the cluster
  sc _time_half_size;         //!< temporal half size of the cluster
  vector_type _space_half_size;  //!< half sizes of the cluster's faces (in [x,
                                 //!< y, z] directions)
  std::vector< lo >
    _elements;  //!< indices of the cluster's elements within spacetime mesh
  general_spacetime_cluster * _parent;  //!< parent of the cluster
  std::vector< general_spacetime_cluster * > *
    _children;  //!< children of the current cluster
  const distributed_spacetime_tensor_mesh &
    _mesh;        //!< distributed spacetime mesh associated with the cluster
  lo _level;      //!< level within the cluster tree
  short _octant;  //!< octant of the parent cluster
  short _left_right;  //!< left (0), or right (0) child of the parent
  sc _padding;        //!< padding of the cluster
  std::vector< slou >
    _box_coordinate;  //!< coordinates of the box within boxes on given level
  std::vector< std::vector< lo > >
    _idx_2_coord;   //!< auxiliary mapping from octant indexing to coordinates
  lo _process_id;   //!< rank of an MPI process owning the cluster
  lo _n_space_div;  //!< number of splittings in space dimensions
  lo _n_time_div;   //!< number of splittings in temporal dimension
};

#endif /* INCLUDE_BESTHEA_GENERAL_SPACETIME_CLUSTER_H_ */
