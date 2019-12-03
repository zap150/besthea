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

/** @file time_cluster.h
 * @brief Cluster of temporal elements.
 */

#ifndef INCLUDE_BESTHEA_TIME_CLUSTER_H_
#define INCLUDE_BESTHEA_TIME_CLUSTER_H_

#include "besthea/settings.h"
#include "besthea/temporal_mesh.h"
#include "besthea/vector.h"

#include <iostream>
#include <vector>

namespace besthea {
  namespace mesh {
    class time_cluster;
  }
}

/**
 * Class representing 1D temporal cluster.
 */
class besthea::mesh::time_cluster {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Block vector type.
  /**
   * Constructor.
   * @param[in] center Center of the cluster.
   * @param[in] half_size Half size of the cluster.
   * @param[in] n_elements Number of temporal elements in the cluster.
   * @param[in] parent Pointer to the cluster's parent.
   * @param[in] level Level within the cluster tree.
   * @param[in] mesh Reference to the underlying temporal mesh.
   */
  time_cluster( sc center, sc half_size, lo n_elements, time_cluster * parent,
    lo level, const temporal_mesh & mesh )
    : _n_elements( n_elements ),
      _center( center ),
      _half_size( half_size ),
      _parent( parent ),
      _children( nullptr ),
      _mesh( mesh ),
      _level( level ) {
    _elements.reserve( _n_elements );
    _lagrange_quad = new std::vector< vector_type >;
  }

  time_cluster( const time_cluster & that ) = delete;

  /**
   * Destructor.
   */
  virtual ~time_cluster( ) {
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
   * Adds temporal element to the cluster.
   * @param[in] idx Index of the temporal element in the underlying mesh.
   */
  void add_element( lo idx ) {
    _elements.push_back( idx );
  }

  /**
   * Adds cluster's child to the list
   * @param[in] child Child cluster.
   */
  void add_child( time_cluster * child ) {
    if ( _children == nullptr ) {
      _children = new std::vector< time_cluster * >( );
    }
    _children->push_back( child );
  }

  /**
   * Returns center of the cluster.
   */
  sc get_center( ) const {
    return _center;
  }

  /**
   * Returns half-size of the cluster..
   */
  sc get_half_size( ) const {
    return _half_size;
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
      _children = new std::vector< time_cluster * >( );
      _children->reserve( n_children );
    } else {
      _children = nullptr;
    }
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
   * Returns a pointer to the children.
   */
  std::vector< time_cluster * > * get_children( ) {
    return _children;
  }

  /**
   * Returns a pointer to the children.
   */
  const std::vector< time_cluster * > * get_children( ) const {
    return _children;
  }

  /**
   * Returns level of the cluster in the cluster tree.
   */
  lo get_level( ) const {
    return _level;
  }

  /**
   * Computes padding of the cluster (distance of the farthest point to the
   * cluster's boundary)
   */
  sc compute_padding( ) const {
    linear_algebra::coordinates< 1 > node1;
    linear_algebra::coordinates< 1 > node2;

    sc padding = 0.0;

    // loop over elements in cluster
    for ( lo i = 0; i < _n_elements; ++i ) {
      _mesh.get_nodes( _elements[ i ], node1, node2 );
      if ( ( ( _center - _half_size ) - node1( 0 ) > padding ) ) {
        padding = _center - _half_size - node1( 0 );
      }
      if ( ( node2( 0 ) - ( _center + _half_size ) > padding ) ) {
        padding = node2( 0 ) - ( _center + _half_size );
      }
    }
    return padding;
  }

  /**
   * Returns a pointer to left neighbour.
   */
  time_cluster * get_left_neighbour( ) {
    if ( _parent == nullptr ) {
      // for the root cluster
      return nullptr;
    }

    if ( this == _parent->_children->back( ) ) {
      return _parent->_children->front( );
    } else if ( ( _parent->get_left_neighbour( ) != nullptr )
      && ( _parent->get_left_neighbour( )->_children->size( ) == 2 ) ) {
      return _parent->get_left_neighbour( )->_children->back( );
    } else {
      return nullptr;
    }
  }

  /**
   * Returns vector storing quadrature of the Lagrange polynomials on a cluster.
   */
  std::vector< vector_type > * get_lagrange_quad( ) {
    return _lagrange_quad;
  }

  /**
   * Returns the associated mesh.
   */
  const temporal_mesh & get_mesh( ) {
    return _mesh;
  }

 private:
  lo _n_elements;  //!< number of elements in the cluster
  sc _center;      //!< center of the cluster
  sc _half_size;   //!< half size of the cluster
  std::vector< lo >
    _elements;  //!< indices of the cluster's elements within the temporal mesh
  time_cluster * _parent;                     //!< parent of the cluster
  std::vector< time_cluster * > * _children;  //!< children of the cluster
  const temporal_mesh & _mesh;  //!< temporal mesh associated with the cluster
  lo _level;                    //!< level within the cluster tree
  std::vector< vector_type > *
    _lagrange_quad;  //!< integrals of the Lagrange polynomials defined on
                     //!< temporal clusters over temporal elements; each
                     //!< std::vector entry of index i stores data associated
                     //!< with i-th order polynomial
};

#endif /* INCLUDE_BESTHEA_TIME_CLUSTER_H_ */
