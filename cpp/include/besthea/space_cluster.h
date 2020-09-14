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

// #include "besthea/fast_spacetime_be_space.h"
#include "besthea/full_matrix.h"
#include "besthea/settings.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/vector.h"

#include <algorithm>
#include <vector>

namespace besthea {
  namespace mesh {
    class space_cluster;
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
 * Class representing a cubic spatial cluster in 3 dimensions.
 */
class besthea::mesh::space_cluster {
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
      _box_coordinate( coordinate ),
      _cheb_T_p0( 1, 1 ),
      _cheb_T_p1( 1, 1 ),
      _cheb_normal_drv_T( 1, 1 ),
      _cheb_times_normal_dim0( 1, 1 ),
      _cheb_times_normal_dim1( 1, 1 ),
      _cheb_times_normal_dim2( 1, 1 ) {
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
   * Returns reference to vector of global element indices for elements in the
   * cluster
   */
  const std::vector< lo > & get_all_elements( ) const {
    return _elements;
  }

  /**
   * Returns number of nodes in the cluster.
   */
  lo get_n_nodes( ) const {
    return _local_2_global_nodes.size( );
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
   * Returns level of the cluster in the cluster tree.
   */
  lo get_level( ) const {
    return _level;
  }

  /**
   * Returns a pointer to the cluster's parent.
   */
  space_cluster * get_parent( ) const {
    return _parent;
  }

  /**
   * Returns coordinates of the box within boxes on given level.
   */
  const std::vector< slou > & get_box_coordinate( ) const {
    return _box_coordinate;
  }

  /**
   * Returns a reference to the matrix storing quadratures of Chebyshev
   * polynomials times p0 basis functions over the elements of the cluster.
   */
  full_matrix_type & get_chebyshev_quad_p0( ) {
    return _cheb_T_p0;
  }

  /**
   * Returns a reference to the matrix storing quadratures of Chebyshev
   * polynomials times p1 basis functions over the elements of the cluster.
   */
  full_matrix_type & get_chebyshev_quad_p1( ) {
    return _cheb_T_p1;
  }

  /**
   * Returns a reference to the matrix storing quadratures of the normal
   * derivatives of Chebyshev polynomials over the elements of the cluster.
   */
  full_matrix_type & get_normal_drv_chebyshev_quad( ) {
    return _cheb_normal_drv_T;
  }

  /**
   * Returns a reference to the matrix storing quadratures of the product
   * of Chebyshev polynomials, p1 basis functions and the selected component of
   * the normal vector over the elements of the cluster.
   * @param[in] dim   Indicates which component of the normal vector is used
   *                  (0,1,2).
   */
  full_matrix_type & get_cheb_times_normal_quad( lo dim ) {
    if ( dim == 0 )
      return _cheb_times_normal_dim0;
    else if ( dim == 1 )
      return _cheb_times_normal_dim1;
    else  // if dim == 2
      return _cheb_times_normal_dim2;
  }

  /**
   * Returns a reference to the vector storing the selected component of the
   * surface curls of p1 basis functions.
   * @param[in] dim   Indicates which component of the surface curls is returned
   *                  (0,1,2).
   */
  std::vector< sc > & get_surf_curls( lo dim ) {
    if ( dim == 0 )
      return _surf_curls_dim0;
    else if ( dim == 1 )
      return _surf_curls_dim1;
    else  // if dim == 2
      return _surf_curls_dim2;
  }

  /**
   * Returns the associated mesh.
   */
  const triangular_surface_mesh & get_mesh( ) {
    return _mesh;
  }

  /**
   * Returns mapping from elements to local local cluster nodes.
   */
  const std::vector< lo > & get_elems_2_local_nodes( ) {
    return _elems_2_local_nodes;
  }

  /**
   * Returns mapping from local cluster nodes to global nodes.
   */
  const std::vector< lo > & get_local_2_global_nodes( ) {
    return _local_2_global_nodes;
  }

  /**
   * Provides local indices for a given, possibly transformed element.
   * In case of p0 elements only the element index is returned
   * In case of p1 elements the local vertex indices are returned
   * @param[in] i_loc_elem Element index.
   * @param[in] n_shared_vertices Number of shared vertives in current elements
   * (regularized quadrature).
   * @param[in] rotation Virtual element rotation (regularized quadrature).
   * @param[in] swap Virtual element inversion (regularized quadrature).
   * @param[out] indices Local indices for the current (transformed) element.
   */
  template< class space_type >
  void local_elem_to_local_dofs( lo i_loc_elem, int n_shared_vertices,
    int rotation, bool swap, std::vector< lo > & indices ) const;

  /**
   * Returns the degrees of freedom depending on the space.
   * For p0 elements the number of elements in the cluster is returned.
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
      linear_algebra::indices< 3 > element;
      for ( auto it = _elements.begin( ); it != _elements.end( ); ++it ) {
        _mesh.get_element( *it, element );
        _local_2_global_nodes.push_back( element[ 0 ] );
        _local_2_global_nodes.push_back( element[ 1 ] );
        _local_2_global_nodes.push_back( element[ 2 ] );
      }
      std::sort( _local_2_global_nodes.begin( ), _local_2_global_nodes.end( ) );
      _local_2_global_nodes.erase( std::unique( _local_2_global_nodes.begin( ),
                                     _local_2_global_nodes.end( ) ),
        _local_2_global_nodes.end( ) );

      _elems_2_local_nodes.resize( 3 * _elements.size( ) );

      lo counter = 0;
      for ( auto it = _elements.begin( ); it != _elements.end( ); ++it ) {
        _mesh.get_element( *it, element );
        auto idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 0 ] );
        _elems_2_local_nodes[ 3 * counter ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );
        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 1 ] );
        _elems_2_local_nodes[ 3 * counter + 1 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );
        idx_it = std::find( _local_2_global_nodes.begin( ),
          _local_2_global_nodes.end( ), element[ 2 ] );
        _elems_2_local_nodes[ 3 * counter + 2 ]
          = std::distance( _local_2_global_nodes.begin( ), idx_it );
        ++counter;
      }
    }
  }

  /**
   * Prints info of the object.
   */
  void print( ) {
    std::cout << "level: " << _level;
    std::cout << ", center: (" << _center[ 0 ] << ", " << _center[ 1 ] << ", "
              << _center[ 2 ] << "), half size: (" << _half_size[ 0 ] << ", "
              << _half_size[ 1 ] << ", " << _half_size[ 1 ]
              << "), elements: " << _n_elements << std::endl;
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
  std::vector< slou >
    _box_coordinate;  //!< coordinates of the box within boxes on given level
  full_matrix_type
    _cheb_T_p0;  //!< matrix storing quadrature of the Chebyshev polynomials
                 //!< times p0 basis functions
                 //!< (rows element of the cluster,
                 //!< columns - order of the polynomial)
  full_matrix_type
    _cheb_T_p1;  //!< matrix storing quadrature of the Chebyshev polynomials
                 //!< times p1 basis functions
                 //!< (rows element of the cluster,
                 //!< columns - order of the polynomial)
  full_matrix_type
    _cheb_normal_drv_T;  //!< matrix storing quadrature of the normal
                         //!< derivatives of the Chebyshev polynomials (weighted
                         //!< with heat coefficient alpha!) times p1 basis
                         //!< function (rows - vertex of the cluster,
                         //!<  columns - order of the polynomial)
  full_matrix_type
    _cheb_times_normal_dim0;  //!< matrix storing quadrature of the Chebyshev
                              //!< polynomials times p1 basis function times
                              //!< 0th component of the normal vector
                              //!< (rows - vertex of the cluster,
                              //!<  columns - order of the polynomial)
  full_matrix_type
    _cheb_times_normal_dim1;  //!< matrix storing quadrature of the Chebyshev
                              //!< polynomials times p1 basis function times
                              //!< 1st component of the normal vector
                              //!< (rows - vertex of the cluster,
                              //!<  columns - order of the polynomial)
  full_matrix_type
    _cheb_times_normal_dim2;  //!< matrix storing quadrature of the Chebyshev
                              //!< polynomials times p1 basis function times
                              //!< 2nd component of the normal vector
                              //!< (rows - vertex of the cluster,
                              //!<  columns - order of the polynomial)
  std::vector< sc >
    _surf_curls_dim0;  //!< vector storing the 0th component of the surface
                       //!< curls of p1 basis functions elementwise (wrt. local
                       //!< indices); entry at pos 3*i + j (j=0,1,2) corresponds
                       //!< to the j_th local basis function for the i_th
                       //!< element
  std::vector< sc >
    _surf_curls_dim1;  //!< vector storing the 1st component of the surface
                       //!< curls of p1 basis functions elementwise (wrt. local
                       //!< indices); entry at pos 3*i + j (j=0,1,2) corresponds
                       //!< to the j_th local basis function for the i_th
                       //!< element
  std::vector< sc >
    _surf_curls_dim2;  //!< vector storing the 2nd component of the surface
                       //!< curls of p1 basis functions elementwise (wrt. local
                       //!< indices); entry at pos 3*i + j (j=0,1,2) corresponds
                       //!< to the j_th local basis function for the i_th
                       //!< element
  std::vector< lo > _elems_2_local_nodes;   //!< mapping from element nodes
                                            //!< vertices to local node list
  std::vector< lo > _local_2_global_nodes;  //!< mapping from local nodes
                                            //!< to the global ones
};

/** specialization for p0 basis functions */
template<>
inline void besthea::mesh::space_cluster::local_elem_to_local_dofs<
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >(
  lo i_loc_elem, int n_shared_vertices, int rotation, bool swap,
  std::vector< lo > & indices ) const {
  indices[ 0 ] = i_loc_elem;
}

/** specialization for p1 basis functions
 * @todo Is a more elegant implementation with map as in basis_tri_p1.cpp
 * possible without wasting too much storage.
 */
template<>
inline void besthea::mesh::space_cluster::local_elem_to_local_dofs<
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >(
  lo i_loc_elem, int n_shared_vertices, int rotation, bool swap,
  std::vector< lo > & indices ) const {
  std::vector< lo > local_indices = { _elems_2_local_nodes[ 3 * i_loc_elem ],
    _elems_2_local_nodes[ 3 * i_loc_elem + 1 ],
    _elems_2_local_nodes[ 3 * i_loc_elem + 2 ] };

  switch ( rotation ) {
    case 0:
      if ( n_shared_vertices == 2 && swap ) {
        indices[ 0 ] = local_indices[ 1 ];
        indices[ 1 ] = local_indices[ 0 ];
      } else {
        indices[ 0 ] = local_indices[ 0 ];
        indices[ 1 ] = local_indices[ 1 ];
      }
      indices[ 2 ] = local_indices[ 2 ];
      break;
    case 1:
      if ( n_shared_vertices == 2 && swap ) {
        indices[ 0 ] = local_indices[ 2 ];
        indices[ 1 ] = local_indices[ 1 ];
      } else {
        indices[ 0 ] = local_indices[ 1 ];
        indices[ 1 ] = local_indices[ 2 ];
      }
      indices[ 2 ] = local_indices[ 0 ];
      break;
    case 2:
      if ( n_shared_vertices == 2 && swap ) {
        indices[ 0 ] = local_indices[ 0 ];
        indices[ 1 ] = local_indices[ 2 ];
      } else {
        indices[ 0 ] = local_indices[ 2 ];
        indices[ 1 ] = local_indices[ 0 ];
      }
      indices[ 2 ] = local_indices[ 1 ];
      break;
  }
}

/** specialization for p0 basis functions */
template<>
inline lo besthea::mesh::space_cluster::get_n_dofs<
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >( )
  const {
  return _n_elements;
}

/** specialization for p1 basis functions */
template<>
inline lo besthea::mesh::space_cluster::get_n_dofs<
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( )
  const {
  return _local_2_global_nodes.size( );
}

#endif /* INCLUDE_BESTHEA_SPACE_CLUSTER_H_ */
