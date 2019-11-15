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

/** @file space_cluster.h
 * @brief Cubic cluster of spatial elements.
 */

#ifndef INCLUDE_BESTHEA_SPACE_CLUSTER_H_
#define INCLUDE_BESTHEA_SPACE_CLUSTER_H_

#include "besthea/settings.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/vector.h"

#include <vector>

namespace besthea {
  namespace mesh {
    class space_cluster;
  }
}

/**
 * Class representing a cubic spatial cluster in 3 dimensions.
 */
class besthea::mesh::space_cluster {
 public:
  using vector_type = besthea::linear_algebra::vector; //!< Vector type.

  /**
   * Constructor.
   * @param[in] center Center of the cluster.
   * @param[in] half_size Half size of the cluster's face.
   * @param[in] n_elements Number of spatial elements in the cluster.
   * @param[in] parent Pointer to the cluster's parent cluster.
   * @param[in] level Level within the cluster tree.
   * @param[in] octant Index of the octant within the parent cluster.
   * @param[in] coordinate Coordinates of the box within boxes on given level.
   * @param[in] mesh Reference to the underlying spatial surface mesh.
   */
  space_cluster( const vector_type & center, const vector_type & half_size,
    lo n_elements, space_cluster * parent, lo level, short octant,
    std::vector< slou > & coordinate, const triangular_surface_mesh & mesh )
    : _n_elements( n_elements ),
      _center( center ),
      _half_size( half_size ),
      _parent( parent ),
      _children( nullptr ),
      _mesh( mesh ),
      _level( level ),
      _octant( octant ),
      _padding( 0.0 ),
      _box_coordinate( coordinate ) {
    _elements.reserve( _n_elements );
    _box_coordinate.shrink_to_fit( );
  }

  space_cluster( const space_cluster & that ) = delete;

  /**
   * Destructor.
   */
  virtual ~space_cluster( ) {
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
   * Adds surface element to the cluster.
   * @param[in] idx Index of the surface element in the underlying mesh.
   */
  void add_element( lo idx ) {
    _elements.push_back( idx );
  }

  /**
   * Adds cluster's child to the list
   * @param[in] child Child cluster.
   */
  void add_child( space_cluster * child ) {
    if ( _children == nullptr ) {
      _children = new std::vector< space_cluster * >( );
    }
    _children->push_back( child );
  }

  /**
   * Returns list of cluster's children
   */
  std::vector< space_cluster * > * get_children( ) {
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
   * @param[out] center Coordinates of the cluster's centroid.
   */
  void get_center( vector_type & center ) const {
    center[ 0 ] = _center[ 0 ];
    center[ 1 ] = _center[ 1 ];
    center[ 2 ] = _center[ 2 ];
  }

  /**
   * Returns half sizes of the cluster.
   * @param[out] half_size Half-sizes in individual directions.
   */
  void get_half_size( vector_type & half_size ) const {
    half_size[ 0 ] = _half_size[ 0 ];
    half_size[ 1 ] = _half_size[ 1 ];
    half_size[ 2 ] = _half_size[ 2 ];
  }

  /**
   * Returns number of elements in the cluster.
   */
  lo get_n_elements( ) const {
    return _n_elements;
  }

  /**
   * Returns element index in the mesh.
   * @param[in] idx Index of element in the cluster's internal storage.
   */
  lo get_element( lo idx ) const {
    return _elements[ idx ];
  }

  /**
   * Sets a number of children and allocates vector of pointers to children.
   * @param[in] n_children Number of cluster's children clusters.
   */
  void set_n_children( lo n_children ) {
    if ( n_children > 0 ) {
      _children = new std::vector< space_cluster * >( );
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
  short get_octant( ) const {
    return _octant;
  }

  /**
   * Computes center and half-sizes of the child in a given octant
   * @param[in] octant Suboctant of the cluster.
   * @param[out] new_center Center of the suboctant.
   * @param[out] new_half_size Half-size of the suboctant.
   */
  void compute_suboctant(
    lo octant, vector_type & new_center, vector_type & new_half_size ) {
    new_half_size[ 0 ] = _half_size[ 0 ] / 2;
    new_half_size[ 1 ] = _half_size[ 1 ] / 2;
    new_half_size[ 2 ] = _half_size[ 2 ] / 2;

    if ( octant == 0 ) {
      new_center[ 0 ] = _center[ 0 ] + new_half_size[ 0 ];
      new_center[ 1 ] = _center[ 1 ] + new_half_size[ 1 ];
      new_center[ 2 ] = _center[ 2 ] + new_half_size[ 2 ];
    } else if ( octant == 1 ) {
      new_center[ 0 ] = _center[ 0 ] - new_half_size[ 0 ];
      new_center[ 1 ] = _center[ 1 ] + new_half_size[ 1 ];
      new_center[ 2 ] = _center[ 2 ] + new_half_size[ 2 ];
    } else if ( octant == 2 ) {
      new_center[ 0 ] = _center[ 0 ] - new_half_size[ 0 ];
      new_center[ 1 ] = _center[ 1 ] - new_half_size[ 1 ];
      new_center[ 2 ] = _center[ 2 ] + new_half_size[ 2 ];
    } else if ( octant == 3 ) {
      new_center[ 0 ] = _center[ 0 ] + new_half_size[ 0 ];
      new_center[ 1 ] = _center[ 1 ] - new_half_size[ 1 ];
      new_center[ 2 ] = _center[ 2 ] + new_half_size[ 2 ];
    } else if ( octant == 4 ) {
      new_center[ 0 ] = _center[ 0 ] + new_half_size[ 0 ];
      new_center[ 1 ] = _center[ 1 ] + new_half_size[ 1 ];
      new_center[ 2 ] = _center[ 2 ] - new_half_size[ 2 ];
    } else if ( octant == 5 ) {
      new_center[ 0 ] = _center[ 0 ] - new_half_size[ 0 ];
      new_center[ 1 ] = _center[ 1 ] + new_half_size[ 1 ];
      new_center[ 2 ] = _center[ 2 ] - new_half_size[ 2 ];
    } else if ( octant == 6 ) {
      new_center[ 0 ] = _center[ 0 ] - new_half_size[ 0 ];
      new_center[ 1 ] = _center[ 1 ] - new_half_size[ 1 ];
      new_center[ 2 ] = _center[ 2 ] - new_half_size[ 2 ];
    } else if ( octant == 7 ) {
      new_center[ 0 ] = _center[ 0 ] + new_half_size[ 0 ];
      new_center[ 1 ] = _center[ 1 ] - new_half_size[ 1 ];
      new_center[ 2 ] = _center[ 2 ] - new_half_size[ 2 ];
    }
  }

  /**
   * Computes padding of the cluster (distance of the farthest point to the
   * cluster's boundary)
   *
   */
  sc compute_padding( ) const {
    linear_algebra::coordinates< 3 > node1;
    linear_algebra::coordinates< 3 > node2;
    linear_algebra::coordinates< 3 > node3;
    sc * nodes[ 3 ];

    sc * curr_node;

    sc padding = 0.0;

    // loop over elements in cluster
    for ( lo i = 0; i < _n_elements; ++i ) {
      _mesh.get_nodes( _elements[ i ], node1, node2, node3 );
      nodes[ 0 ] = node1.data( );
      nodes[ 1 ] = node2.data( );
      nodes[ 2 ] = node3.data( );
      // loop over triangle's nodes
      for ( lo j = 0; j < 3; ++j ) {
        curr_node = nodes[ j ];
        if ( ( ( _center[ 0 ] - _half_size[ 0 ] ) - curr_node[ 0 ]
               > padding ) ) {
          padding = _center[ 0 ] - _half_size[ 0 ] - curr_node[ 0 ];
        }
        if ( ( curr_node[ 0 ] - ( _center[ 0 ] + _half_size[ 0 ] )
               > padding ) ) {
          padding = curr_node[ 0 ] - ( _center[ 0 ] + _half_size[ 0 ] );
        }
        if ( ( ( _center[ 1 ] - _half_size[ 1 ] ) - curr_node[ 1 ]
               > padding ) ) {
          padding = _center[ 1 ] - _half_size[ 1 ] - curr_node[ 1 ];
        }
        if ( ( curr_node[ 1 ] - ( _center[ 1 ] + _half_size[ 1 ] )
               > padding ) ) {
          padding = curr_node[ 1 ] - ( _center[ 1 ] + _half_size[ 1 ] );
        }
        if ( ( ( _center[ 2 ] - _half_size[ 2 ] ) - curr_node[ 2 ]
               > padding ) ) {
          padding = _center[ 2 ] - _half_size[ 2 ] - curr_node[ 2 ];
        }
        if ( ( curr_node[ 2 ] - ( _center[ 2 ] + _half_size[ 2 ] )
               > padding ) ) {
          padding = curr_node[ 2 ] - ( _center[ 2 ] + _half_size[ 2 ] );
        }
      }
    }
    return padding;
  }

  /**
   * Sets padding of the cluster.
   * @param[in] padding Padding of the cluster.
   */
  void set_padding( sc padding ) {
    _padding = padding;
  }

  /**
   * Returns padding of the cluster.
   */
  sc get_padding( ) const {
    return _padding;
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

 private:
  lo _n_elements;          //!< number of elements in the cluster
  vector_type _center;     //!< center of the cluster
  vector_type _half_size;  //!< half sizes of the cluster's faces (in [x, y, z]
                           //!< directions)
  // TODO: this probably will have to be optimized to reduce memory consumption
  std::vector< lo >
    _elements;  //!< indices of the cluster's elements within the spatial mesh
  space_cluster * _parent;                     //!< parent of the cluster
  std::vector< space_cluster * > * _children;  //!< children of the cluster
  const triangular_surface_mesh &
    _mesh;        //!< spatial mesh associated with the cluster
  lo _level;      //!< level within the cluster tree
  short _octant;  //!< octant of the parent cluster
  sc _padding;    //!< padding of the cluster
  std::vector< slou >
    _box_coordinate;  //!< coordinates of the box within boxes on given level
};

#endif /* INCLUDE_BESTHEA_SPACE_CLUSTER_H_ */
