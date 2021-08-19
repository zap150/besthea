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

/** @file volume_space_cluster.h
 * @brief Cubic cluster of spatial elements.
 */

#ifndef INCLUDE_BESTHEA_VOLUME_SPACE_CLUSTER_H_
#define INCLUDE_BESTHEA_VOLUME_SPACE_CLUSTER_H_

// #include "besthea/fast_spacetime_be_space.h"
#include "besthea/full_matrix.h"
#include "besthea/settings.h"
#include "besthea/tetrahedral_volume_mesh.h"
#include "besthea/vector.h"

#include <algorithm>
#include <vector>

namespace besthea {
  namespace mesh {
    class volume_space_cluster;
  }
}

// forward declaration of fast_spacetime_be_space and basis functions
namespace besthea {
  namespace bem {
    template< class basis_type >
    class fe_space;

    class basis_tetra_p1;
  }
}

/**
 * Class representing a cubic spatial cluster containing elements in a
 * tetrahedral volume mesh in 3 dimensions.
 */
class besthea::mesh::volume_space_cluster {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< Vector type.

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
  volume_space_cluster( const vector_type & center,
    const vector_type & half_size, lo n_elements, volume_space_cluster * parent,
    lo level, short octant, std::vector< slou > & coordinate,
    const tetrahedral_volume_mesh & mesh )
    : _n_elements( n_elements ),
      _n_nodes( -1 ),
      _center( center ),
      _half_size( half_size ),
      _parent( parent ),
      _children( nullptr ),
      _mesh( mesh ),
      _level( level ),
      _octant( octant ),
      _box_coordinate( coordinate ) {
    _elements.reserve( _n_elements );
    _box_coordinate.shrink_to_fit( );
  }

  volume_space_cluster( const volume_space_cluster & that ) = delete;

  /**
   * Destructor.
   */
  virtual ~volume_space_cluster( ) {
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
   * Adds volume elements to the cluster.
   * @param[in] idx Index of the volume element in the underlying mesh.
   */
  void add_element( lo idx ) {
    _elements.push_back( idx );
  }

  /**
   * Adds a child to the current cluster's list of children.
   * @param[in] child Child cluster.
   */
  void add_child( volume_space_cluster * child ) {
    if ( _children == nullptr ) {
      _children = new std::vector< volume_space_cluster * >( );
    }
    _children->push_back( child );
  }

  /**
   * Returns the list of children
   */
  std::vector< volume_space_cluster * > * get_children( ) {
    return _children;
  }

  /**
   * Returns the number of children.
   */
  lo get_n_children( ) {
    if ( _children != nullptr ) {
      return _children->size( );
    } else {
      return 0;
    }
  }

  /**
   * Returns the cluster's center.
   * @param[out] center Coordinates of the cluster's centroid.
   */
  void get_center( vector_type & center ) const {
    center[ 0 ] = _center[ 0 ];
    center[ 1 ] = _center[ 1 ];
    center[ 2 ] = _center[ 2 ];
  }

  /**
   * Returns the half sizes of the cluster.
   * @param[out] half_size Vector containing half-sizes for each spatial
   * dimension.
   */
  void get_half_size( vector_type & half_size ) const {
    half_size[ 0 ] = _half_size[ 0 ];
    half_size[ 1 ] = _half_size[ 1 ];
    half_size[ 2 ] = _half_size[ 2 ];
  }

  /**
   * Returns the number of elements in the cluster.
   */
  lo get_n_elements( ) const {
    return _n_elements;
  }

  /**
   * Returns the global element index of a local element in the cluster.
   * @param[in] idx Index of element in the cluster's internal storage.
   */
  lo get_element( lo idx ) const {
    return _elements[ idx ];
  }

  /**
   * Returns a reference to the vector @ref _elements.
   */
  const std::vector< lo > & get_all_elements( ) const {
    return _elements;
  }

  /**
   * Returns the number of nodes in the cluster.
   */
  lo get_n_nodes( ) const {
    return _local_2_global_nodes.size( );
  }

  /**
   * Sets the cluster's number of children and allocates a vector to store the
   * pointers to the children.
   * @param[in] n_children Number of cluster's child clusters.
   */
  void set_n_children( lo n_children ) {
    if ( n_children > 0 ) {
      _children = new std::vector< volume_space_cluster * >( );
      _children->reserve( n_children );
    } else {
      _children = nullptr;
    }
  }

  /**
   * Shrinks the vector @ref _children.
   */
  void shrink_children( ) {
    if ( _children != nullptr ) {
      _children->shrink_to_fit( );
    }
  }

  /**
   * Returns the current cluster's octant with respect to its parent.
   *
   * For parent cluster with center at (0, 0, 0) the following octant ordering
   * holds:
   *
   * oct/coord 0 1 2 3 4 5 6 7
   * 	x		+ - - + + - - +
   * 	y		+ + - - + + - -
   * 	z		+ + + + - - - -
   */
  short get_octant( ) const {
    return _octant;
  }

  /**
   * Computes the center and half-sizes of a child in a given octant
   * @param[in] octant Suboctant of the child.
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
   * Computes the required padding of the cluster (distance of the farthest
   * point to the cluster's boundary)
   */
  sc compute_padding( ) const {
    linear_algebra::coordinates< 3 > node1;
    linear_algebra::coordinates< 3 > node2;
    linear_algebra::coordinates< 3 > node3;
    linear_algebra::coordinates< 3 > node4;
    sc * nodes[ 4 ];

    sc * curr_node;

    sc padding = 0.0;

    // loop over elements in cluster
    for ( lo i = 0; i < _n_elements; ++i ) {
      _mesh.get_nodes( _elements[ i ], node1, node2, node3, node4 );
      nodes[ 0 ] = node1.data( );
      nodes[ 1 ] = node2.data( );
      nodes[ 2 ] = node3.data( );
      nodes[ 3 ] = node4.data( );
      // loop over triangle's nodes
      for ( lo j = 0; j < 4; ++j ) {
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
   * Returns the level of the cluster in the cluster tree.
   */
  lo get_level( ) const {
    return _level;
  }

  /**
   * Returns a pointer to the cluster's parent.
   */
  volume_space_cluster * get_parent( ) const {
    return _parent;
  }

  /**
   * Returns @ref _box_coordinate.
   */
  const std::vector< slou > & get_box_coordinate( ) const {
    return _box_coordinate;
  }

  /**
   * Returns @ref _mesh.
   */
  const tetrahedral_volume_mesh & get_mesh( ) const {
    return _mesh;
  }

  /**
   * Returns @ref _elems_2_local_nodes.
   */
  const std::vector< lo > & get_elems_2_local_nodes( ) const {
    return _elems_2_local_nodes;
  }

  /**
   * Returns @ref _local_2_global_nodes.
   */
  const std::vector< lo > & get_local_2_global_nodes( ) const {
    return _local_2_global_nodes;
  }

  /**
   * Provides local indices for a given element.
   * In case of p1 elements the local vertex indices are returned.
   * @param[in] i_loc_elem Element index.
   * @param[out] indices Local indices for the current (transformed) element.
   * @note A permutation of indices (rotation, swap) is currently not supported.
   */
  template< class space_type >
  void local_elem_to_local_dofs(
    lo i_loc_elem, std::vector< lo > & indices ) const;

  /**
   * Returns the degrees of freedom depending on the space.
   * For p1 elements the number of vertices in the cluster is returned.
   */
  template< class space_type >
  lo get_n_dofs( ) const;

  /**
   * Computes mapping from elements to local nodes and from local to global
   * nodes.
   */
  void compute_node_mapping( ) {
    // check first whether the mapping already exists
    if ( _local_2_global_nodes.size( ) == 0 ) {
      linear_algebra::indices< 4 > element;
      for ( auto it = _elements.begin( ); it != _elements.end( ); ++it ) {
        _mesh.get_element( *it, element );
        _local_2_global_nodes.push_back( element[ 0 ] );
        _local_2_global_nodes.push_back( element[ 1 ] );
        _local_2_global_nodes.push_back( element[ 2 ] );
        _local_2_global_nodes.push_back( element[ 3 ] );
      }
      std::sort( _local_2_global_nodes.begin( ), _local_2_global_nodes.end( ) );
      _local_2_global_nodes.erase( std::unique( _local_2_global_nodes.begin( ),
                                     _local_2_global_nodes.end( ) ),
        _local_2_global_nodes.end( ) );

      _elems_2_local_nodes.resize( 4 * _elements.size( ) );

      lo counter = 0;
      for ( auto it = _elements.begin( ); it != _elements.end( ); ++it ) {
        _mesh.get_element( *it, element );
        auto idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 0 ] );
        _elems_2_local_nodes[ 4 * counter ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );
        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 1 ] );
        _elems_2_local_nodes[ 4 * counter + 1 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );
        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 2 ] );
        _elems_2_local_nodes[ 4 * counter + 2 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );
        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 3 ] );
        _elems_2_local_nodes[ 4 * counter + 3 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );
        ++counter;
      }

      _n_nodes = _local_2_global_nodes.size( );
    }
  }

  /**
   * Allocates @ref _moments as an array of scalar values of given size.
   */
  void resize_moments( const lou moment_size ) {
    _moments.resize( moment_size );
  }

  /**
   * Resets the vector @ref _moments to 0.
   */
  void clear_moments( ) {
    _moments.fill( 0.0 );
  }

  /**
   * Returns a reference to @ref _moments.
   */
  vector_type & get_moments( ) {
    return _moments;
  }

  /**
   * Returns a reference to @ref _moments (immutable).
   */
  const vector_type & get_moments( ) const {
    return _moments;
  }

  /**
   * Prints info of the object.
   */
  void print( ) {
    std::cout << "level: " << _level;
    std::cout << ", center: (" << _center[ 0 ] << ", " << _center[ 1 ] << ", "
              << _center[ 2 ] << "), half size: (" << _half_size[ 0 ] << ", "
              << _half_size[ 1 ] << ", " << _half_size[ 1 ]
              << "), elements: " << _n_elements;
    std::cout << ", coordinates: " << _box_coordinate[ 0 ] << ", "
              << _box_coordinate[ 1 ] << ", " << _box_coordinate[ 2 ] << ", "
              << _box_coordinate[ 3 ];
    std::cout << ", octant " << _octant;
    std::cout << ", nodes:";
    for ( auto node_idx : _local_2_global_nodes ) {
      std::cout << " " << node_idx;
    }
    std::cout << std::endl;
  }

 private:
  lo _n_elements;  //!< Number of elements in the cluster.
  lo _n_nodes;     //!< Number of nodes in the cluster. This is initialized in
                   //!< @ref compute_node_mapping.
  vector_type _center;     //!< Center of the cluster.
  vector_type _half_size;  //!< Half sizes of the cluster's faces (in [x, y, z]
                           //!< directions).
  // TODO: this probably will have to be optimized to reduce memory consumption
  std::vector< lo >
    _elements;  //!< Indices of the cluster's elements within the spatial mesh
  volume_space_cluster * _parent;  //!< Parent of the cluster
  std::vector< volume_space_cluster * > *
    _children;  //!< Children of the cluster
  const tetrahedral_volume_mesh &
    _mesh;        //!< Spatial volume mesh associated with the cluster.
  lo _level;      //!< Level within the spatial cluster tree.
  short _octant;  //!< Octant of the parent cluster. See @ref get_octant for
                  //!< more information.
  std::vector< slou >
    _box_coordinate;  //!< Coordinates of the box within a regular grid at the
                      //!< cluster's level.
  std::vector< lo >
    _elems_2_local_nodes;  //!< Mapping from the elements in the cluster to the
                           //!< local node list. At position [4*i+j] the index
                           //!< of the j-th node of the i-th element is stored.
  std::vector< lo > _local_2_global_nodes;  //!< Mapping from the local nodes
                                            //!< to the global ones in the mesh.
  vector_type _moments;  //!< Vector storing the moments of the volume cluster.
};

/** specialization of
 * @ref besthea::mesh::volume_space_cluster::get_n_dofs for p1 basis
 * functions */
template<>
inline lo besthea::mesh::volume_space_cluster::get_n_dofs<
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >( ) const {
  return _n_nodes;
}

/** specialization of
 * @ref besthea::mesh::volume_space_cluster::local_elem_to_local_dofs
 * for p0 basis functions */
template<>
inline void besthea::mesh::volume_space_cluster::local_elem_to_local_dofs<
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >(
  lo i_loc_elem, std::vector< lo > & indices ) const {
  indices[ 0 ] = _elems_2_local_nodes[ 4 * i_loc_elem ];
  indices[ 1 ] = _elems_2_local_nodes[ 4 * i_loc_elem + 1 ];
  indices[ 2 ] = _elems_2_local_nodes[ 4 * i_loc_elem + 2 ];
  indices[ 3 ] = _elems_2_local_nodes[ 4 * i_loc_elem + 3 ];
}

#endif /* INCLUDE_BESTHEA_VOLUME_SPACE_CLUSTER_H_ */
