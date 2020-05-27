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
   *                        cluster in an MPI parallelized FMM.
   */
  scheduling_time_cluster( sc center, sc half_size, 
    scheduling_time_cluster * parent, lo level, lo process_id = -1 )
    : _center( center ),
      _half_size ( half_size ),
      _parent( parent ),
      _children( nullptr ),
      _level( level ),
      _index ( -1 ),
      _process_id( process_id ),
      _nearfield( nullptr ),
      _interaction_list( nullptr ) {
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
    delete _nearfield;
    delete _interaction_list;
  }

  /**
   * Deletes the nearfield 
   */
  void delete_nearfield( ) {
    delete _nearfield;
    _nearfield = nullptr;
  }

  /**
   * Deletes the interaction list
   */
  void delete_interaction_list( ) {
    delete _interaction_list;
    _interaction_list = nullptr;
  }

  /**
   * Deletes the vector of children
   */
  void delete_children( ) {
    delete _children;
    _children = nullptr;
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
   * Returns level of the cluster in the cluster tree.
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
   * @note The clusters are enumerated levelwise from left to right. 
   */
  lo get_index( ) const {
    return _index;
  }

  /**
   * Sets the index of the cluster.
   * @param[in] index Index which is set.
   */
  void set_index( lo index ) {
    _index = index;
  }

  /**
   * Adds a pointer to a nearfield cluster to the vector _nearfield.
   * @param[in] src_cluster Pointer to a nearfield cluster.
   * @note If _nearfield is not allocated this is done here.
   */
  void add_to_nearfield( scheduling_time_cluster * src_cluster ) {
    if ( _nearfield == nullptr ) {
      _nearfield = new std::vector< scheduling_time_cluster * >( );
    }
    _nearfield->push_back( src_cluster );
  }

  /**
   * Returns a pointer to the const nearfield.
   */
  const std::vector< scheduling_time_cluster * > * get_nearfield( ) const {
    return _nearfield;
  }

  /**
   * Returns a pointer to the nearfield.
   */
  std::vector< scheduling_time_cluster * > * get_nearfield( ) {
    return _nearfield;
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
   * Adds a pointer to a farfield cluster to the vector _interaction_list.
   * @param[in] src_cluster Pointer to a farfield cluster.
   * @note If _interaction_list is not allocated this is done here.
   */
  void add_to_interaction_list( scheduling_time_cluster * src_cluster ) {
    if ( _interaction_list == nullptr ) {
      _interaction_list = new std::vector< scheduling_time_cluster * >( );
    }
    _interaction_list->push_back( src_cluster );
  }

  /**
   * Determines admissibility based on neighborhood criterion.
   * @param[in] src_cluster Source cluster whose admissibility is checked.
   * @warning This check of admissibility is only reasonable for clusters at the
   * same level of a tree.
   */
  bool determine_admissibility( scheduling_time_cluster * src_cluster ) {
    bool admissibility = false;
    if ( src_cluster != this && src_cluster->get_center( ) < _center 
                             && src_cluster != get_left_neighbour( ) ) {
      admissibility = true;
    } 
    return admissibility;
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
   * Prints info of the object.
   */
  void print( ) {
    std::cout << "level: " << _level 
              << ", center: " << _center << ", half size: " << _half_size
              << ", index: " << _index << ", proc_id: " << _process_id;
    // if ( _nearfield != nullptr ) {
    //   std::cout << ", nearfield: ";
    //   for ( lou i = 0; i < _nearfield->size( ); ++i ) {
    //     std::cout << "(" << ( *_nearfield )[ i ]->get_level( ) << ", "  
    //               << ( *_nearfield )[ i ]->get_index( ) << "), ";
    //   }
    // }
    // std::cout << std::endl;
    // if ( _interaction_list != nullptr ) {
    //   std::cout << ", interaction_list: ";
    //   for ( lou i = 0; i < _interaction_list->size( ); ++i ) {
    //     std::cout << "(" << ( *_interaction_list )[ i ]->get_level( ) << ", "  
    //               << ( *_interaction_list )[ i ]->get_index( ) << "), ";
    //   }
    // }
    std::cout << std::endl;
  }

 private:
  sc _center;       //!< center of the cluster
  sc _half_size;    //!< half size of the cluster
  scheduling_time_cluster * _parent;  //!< parent of the cluster
  std::vector< scheduling_time_cluster * > 
    * _children;    //!< children of the cluster
  lo _level;        //!< level within the cluster tree
  lo _index;        //!< index of the cluster at the current level according
                    //!< to an enumeration from left to right
  lo _process_id;   //!< id of the process to which the cluster is assigned
  std::vector< scheduling_time_cluster * >
    * _nearfield;   //!< nearfield of the cluster
  std::vector< scheduling_time_cluster * >
    * _interaction_list;  //!< interaction list of the cluster
};

#endif /* INCLUDE_BESTHEA_SCHEDULING_TIME_CLUSTER_H_ */