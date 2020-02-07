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

/** @file spacetime_cluster.h
 * @brief Combination of space and time clusters
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_CLUSTER_H_
#define INCLUDE_BESTHEA_SPACETIME_CLUSTER_H_

#include "besthea/full_matrix.h"
#include "besthea/settings.h"
#include "besthea/space_cluster.h"
#include "besthea/time_cluster.h"

#include <vector>

namespace besthea {
  namespace mesh {
    class spacetime_cluster;
  }
}

/**
 * Class representing a space-time cluster
 */
class besthea::mesh::spacetime_cluster {
 public:
   using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< Sparse matrix type.
  /**
   * Constructor.
   * @param[in] spatial_cluster Reference to a spatial cluster.
   * @param[in] temporal_cluster Reference to a temporal cluster.
   * @param[in] parent Pointer to the parent.
   * @param[in] level Assigns level to the cluster.
   */
  spacetime_cluster( space_cluster & spatial_cluster,
    time_cluster & temporal_cluster, spacetime_cluster * parent, lo level )
    : _spatial_cluster( spatial_cluster ),
      _temporal_cluster( temporal_cluster ),
      _parent( parent ),
      _children( nullptr ),
      _moment_contribution( nullptr ),
      _local_contribution( nullptr ),
      _interaction_list( nullptr ),
      _level( level ) {
  }

  spacetime_cluster( const space_cluster & that ) = delete;

  /**
   * Destructor.
   */
  virtual ~spacetime_cluster( ) {
    if ( _children != nullptr ) {
      for ( auto it = _children->begin( ); it != _children->end( ); ++it ) {
        if ( *it != nullptr ) {
          delete *it;
        }
      }
      delete _children;
    }
    if ( _moment_contribution != nullptr )
      delete _moment_contribution;
    if ( _local_contribution != nullptr )
      delete _local_contribution;
    if ( _interaction_list != nullptr )
      delete _interaction_list;
  }

  /**
   * Adds cluster's child to the list
   * @param[in] child Child cluster.
   */
  void add_child( spacetime_cluster * child ) {
    if ( _children == nullptr ) {
      _children = new std::vector< spacetime_cluster * >( );
    }
    _children->push_back( child );
  }

  /**
   * Returns list of cluster's children.
   */
  std::vector< spacetime_cluster * > * get_children( ) {
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
   * Adds cluster to the interaction list.
   * @param[in] cluster Cluster to be added.
   */
  void add_to_interaction_list( spacetime_cluster * cluster ) {
    if ( _interaction_list == nullptr ) {
      _interaction_list = new std::vector< spacetime_cluster * >( );
    }
    _interaction_list->push_back( cluster );
  }
  
  /**
   * Returns cluster's interaction list.
   */
  std::vector< spacetime_cluster * > * get_interaction_list( ) {
    return _interaction_list;
  }

  /**
   * Returns a reference to the underlying spatial cluster.
   */
  space_cluster & get_space_cluster( ) {
    return _spatial_cluster;
  }

  /**
   * Returns a reference to the underlying temporal cluster.
   */
  time_cluster & get_time_cluster( ) {
    return _temporal_cluster;
  }

  /**
   * Sets a number of children and allocates vector of pointers to children.
   * @param[in] n_children Number of cluster's children clusters.
   */
  void set_n_children( lo n_children ) {
    if ( n_children > 0 ) {
      _children = new std::vector< spacetime_cluster * >( );
      _children->reserve( n_children );
    } else {
      _children = nullptr;
    }
  }
  
  /**
   * Initialize local contribution as full matrix of zeros.
   * @param[in] n_rows_contribution Number of rows.
   * @param[in] n_columns_contribution Number of columns.
   * \note If the contributions are set already, then they are resized and
   * reset to zero.
   * \todo Is resizing of a full matrix allocated with new safe?
   */
  void set_local_contribution( lo n_rows_contribution, 
                               lo n_columns_contribution ) {
    if ( _local_contribution == nullptr ) {
      _local_contribution = new full_matrix_type( n_rows_contribution, 
                                                  n_columns_contribution );
    }
    else {
//       TODO: see comment above
      _local_contribution->resize( n_rows_contribution, 
                                   n_columns_contribution );
      _local_contribution->fill( 0.0 );
    }
  }
  
  /**
   * Initialize moment contribution as full matrix of zeros.
   * @param[in] n_rows_contribution Number of rows.
   * @param[in] n_columns_contribution Number of columns.
   * \note If the contributions are set already, then they are resized and
   * reset to zero.
   * \todo Is resizing of a full matrix allocated with new safe?
   */
  void set_moment_contribution( lo n_rows_contribution, 
                                lo n_columns_contribution ) {
    if ( _moment_contribution == nullptr ) {
      _moment_contribution = new full_matrix_type( n_rows_contribution, 
                                                   n_columns_contribution );
    }
    else {
//       TODO: see comment above
      _moment_contribution->resize( n_rows_contribution, 
                                    n_columns_contribution );
      _moment_contribution->fill( 0.0 );
    }
  }
  
  /**
   * Returns pointer to the local contribution.
   */
  full_matrix_type * get_local_contribution( ) {
    return _local_contribution;
  }
  
  /**
   * Returns pointer to the moment contribution.
   */
  full_matrix_type * get_moment_contribution( ) {
    return _moment_contribution;
  }

  /**
   * Returns level of the cluster in the cluster tree.
   */
  lo get_level( ) const {
    return _level;
  }

  /**
   * Prints info of the object.
   */
  void print( ) {
    std::cout << _level << ", space: " << _spatial_cluster.get_level( ) 
              << " time: " << _temporal_cluster.get_level( ) << std::endl;
    besthea::linear_algebra::vector spat_center( 3, false );
    _spatial_cluster.get_center( spat_center );
    sc temp_center = _temporal_cluster.get_center( );
    std::cout << "spatial center: (" << spat_center[ 0 ] << ", "
              << spat_center[ 1] << ", " << spat_center[ 2 ] << ")" 
              << std::endl;
    std::cout << "temporal center: " << temp_center << std::endl;
  }

 private:
  space_cluster & _spatial_cluster;  //!< underlying spatial cluster
  time_cluster & _temporal_cluster;  //!< underlying temporal cluster
  spacetime_cluster * _parent;       //!< parent of the space-time cluster
  std::vector< spacetime_cluster * > * _children;  //!< children of the cluster
  full_matrix_type *_moment_contribution; //!< matrix to store the intermediate 
                                          //!< products in the upward FMM step
  full_matrix_type *_local_contribution; //!< matrix to store the intermediate 
                                         //!< products in the downpward FMM step
  std::vector< spacetime_cluster * > * _interaction_list; //!< interaction list
                                                          //!< of the cluster
  lo _level;  //!< level within the cluster tree
};

#endif /* INCLUDE_BESTHEA_SPACETIME_CLUSTER_H_ */
