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

#include "besthea/settings.h"
#include "besthea/vector.h"

#include <iostream>
#include <vector>

namespace besthea {
  namespace mesh {
    class scheduling_time_cluster;
  }
}

/**
 * Class representing 1D temporal cluster.
 */
class besthea::mesh::scheduling_time_cluster {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Block vector type.
  /**
   * Constructor.
   * @param[in] center Center of the cluster.
   * @param[in] parent Pointer to the cluster's parent.
   * @param[in] level Level within the cluster tree.
   */
  scheduling_time_cluster( sc center, sc half_size, 
    scheduling_time_cluster * parent, lo level )
    : _center( center ),
      _half_size ( half_size ),
      _parent( parent ),
      _children( nullptr ),
      _level( level ) {
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
  }

  /**
   * Adds cluster's child to the list
   * @param[in] child Child cluster.
   */
  void add_child( scheduling_time_cluster * child ) {
    if ( _children == nullptr ) {
      _children = new std::vector< scheduling_time_cluster * >( );
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
   * Sets a number of children and allocates vector of pointers to children.
   * @param[in] n_children Number of cluster's children clusters.
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
  std::vector< scheduling_time_cluster * > * get_children( ) {
    return _children;
  }
  
    /**
   * Returns a pointer to the parent.
   */
  scheduling_time_cluster * get_parent( ) {
    return _parent;
  }
    
  /**
   * Returns a pointer to the children.
   */
  const std::vector< scheduling_time_cluster * > * get_children( ) const {
    return _children;
  }

  /**
   * Returns level of the cluster in the cluster tree.
   */
  lo get_level( ) const {
    return _level;
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
   * \note If the current cluster is the root of the tree \p false is returned.
   */
  bool is_left_child( ) const {
    if ( _parent == nullptr )
      return false;
    else 
      return ( this == _parent->_children->front( ) );
  }

  /**
   * Prints info of the object.
   */
  void print( ) {
    std::cout << "level: " << _level << std::endl;
    std::cout << "center: " << _center << ", half size: " << _half_size
              << std::endl;
  }

 private:
  sc _center;      //!< center of the cluster
  sc _half_size;   //!< half size of the cluster
  scheduling_time_cluster * _parent;                     //!< parent of the cluster
  std::vector< scheduling_time_cluster * > * _children;  //!< children of the cluster
  lo _level;                    //!< level within the cluster tree
};

#endif /* INCLUDE_BESTHEA_SCHEDULING_TIME_CLUSTER_H_ */