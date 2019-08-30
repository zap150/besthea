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
  using vector_type = besthea::linear_algebra::vector;

  /**
   * Constructor.
   * @param[in] center Center of the cluster.
   * @param[in] half_size Half size of the cluster's face.
   * @param[in] n_elements Number of spatial elements in the cluster.
   * @param[in] parent Pointer to the cluster's parent cluster.
   * @param[in] mesh Reference to the underlying spatial surface mesh.
   */
  space_cluster( const vector_type & center, const vector_type & half_size,
    lo n_elements, space_cluster * parent,
    const triangular_surface_mesh & mesh )
    : _n_elements( n_elements ),
      _center( center ),
      _half_size( half_size ),
      _elements( _n_elements ),
      _parent( parent ),
      _mesh( mesh ) {
  }

  space_cluster( const space_cluster & that ) = delete;

  /**
   * Destructor.
   */
  virtual ~space_cluster( ) {
    for ( auto it = _children.begin( ); it != _children.end( ); ++it ) {
      if ( *it != nullptr ) {
        delete *it;
      }
    }
  }

  /**
   * Adds surface element to the cluster.
   * param[in] idx Index of the surface element in the underlying mesh.
   */
  void add_element( lo idx ) {
    _elements.push_back( idx );
  }

  /**
   * Adds cluster's child to the list
   * param[in] child Child cluster.
   */
  void add_child( space_cluster * child ) {
    _children.push_back( child );
  }

  /**
   * Returns center of the cluster.
   * param[out] center Coordinates of the cluster's centroid.
   */
  void get_center( vector_type & center ) {
    center[ 0 ] = _center[ 0 ];
    center[ 1 ] = _center[ 1 ];
    center[ 2 ] = _center[ 2 ];
  }

  /**
   * Returns center of the cluster.
   * param[out] center Coordinates of the cluster's centroid.
   */
  void get_half_size( vector_type & half_size ) {
    half_size[ 0 ] = _half_size[ 0 ];
    half_size[ 1 ] = _half_size[ 1 ];
    half_size[ 2 ] = _half_size[ 2 ];
  }

  /**
   * Returns number of elements in the cluster.
   */
  lo get_n_elements( ) {
    return _n_elements;
  }

  /**
   * Returns element index in the mesh.
   * param[in] idx Index of element in the cluster's internal storage.
   */
  lo get_element( lo idx ) {
    return _elements[ idx ];
  }

  /**
   * Sets a number of children and allocates vector of pointers to children.
   * param[in] n_children Number of cluster's children clusters.
   */
  void set_n_children( lo n_children ) {
    _children.reserve( n_children );
  }

  /**
   * Computes center and half-sizes of the child in a given octant
   * param[in] octant Suboctant of the cluster.
   * param[out] center Center of the suboctant.
   * param[out] half_size Half-size of the suboctant.
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

 private:
  lo _n_elements;          //!< number of elements in the cluster
  vector_type _center;     //!< center of the cluster
  vector_type _half_size;  //!< half sizes of the cluster's faces (in [x, y, z]
                           //!< directions)
  std::vector< lo >
    _elements;  //!< indices of the cluster's elements within the spatial mesh
  space_cluster * _parent;                   //!< parent of the cluster
  std::vector< space_cluster * > _children;  //!< children of the cluster
  const triangular_surface_mesh &
    _mesh;  //!< spatial mesh associated with the cluster
};

#endif /* INCLUDE_BESTHEA_SPACE_CLUSTER_H_ */
