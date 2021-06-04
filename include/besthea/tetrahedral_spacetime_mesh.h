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

/** @file tetrahedral_spacetime_mesh.h
 * @brief Tetrahedral spacetime mesh for general decomposition of spacetime
 * domain
 */

#ifndef INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_MESH_H_
#define INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_MESH_H_

#include "besthea/coordinates.h"
#include "besthea/indices.h"
#include "besthea/mesh.h"
#include "besthea/settings.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/vector.h"

#include <optional>
#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class tetrahedral_spacetime_mesh;
  }
}

/**
 * Class representing tetrahedral spacetime mesh for general decomposition of
 * spacetime domain
 */
class besthea::mesh::tetrahedral_spacetime_mesh : public besthea::mesh::mesh {
 public:
  /**
   * Constructing mesh from a spacetime tensor mesh.
   * @param[in] stmesh Path to the file.
   */
  tetrahedral_spacetime_mesh(
    const besthea::mesh::spacetime_tensor_mesh & stmesh );

  /**
   * Constructing mesh from provided data.
   * @param[in] stmesh Path to the file.
   */
  tetrahedral_spacetime_mesh( const std::vector< sc > & nodes,
    const std::vector< lo > & elements, const std::vector< sc > & normals );

  /**
   * Copy constructor.
   */
  tetrahedral_spacetime_mesh( const tetrahedral_spacetime_mesh & that )
    = delete;

  /**
   * Destructor.
   */
  ~tetrahedral_spacetime_mesh( );

  /**
   * Prints info on the object.
   */
  void print_info( ) const;

  /**
   * Loads mesh from a file.
   * @param[in] file File name.
   */
  /*
  bool load( const std::string & file );
  */

  /**
   * Returns area (volume) of a single element
   */
  sc area( lo i_elem ) const {
    return _areas[ i_elem ];
  }

  /**
   * Returns number of elements.
   */
  lo get_n_elements( ) const {
    return _n_elements;
  }

  /**
   * Returns number of nodes.
   */
  lo get_n_nodes( ) const {
    return _n_nodes;
  }

  /**
   * Returns number of edges.
   */
  lo get_n_edges( ) const {
    return _n_edges;
  }

  /**
   * Returns node indices of an element.
   * @param[in] i_element Index of the element.
   * @param[out] element Element indices.
   */
  void get_element(
    lo i_element, linear_algebra::indices< 4 > & element ) const {
    element[ 0 ] = _elements[ 4 * i_element ];
    element[ 1 ] = _elements[ 4 * i_element + 1 ];
    element[ 2 ] = _elements[ 4 * i_element + 2 ];
    element[ 3 ] = _elements[ 4 * i_element + 3 ];
  }

  /**
   * Returns coordinates of a node.
   * @param[in] i_node Index of the node.
   * @param[out] node Element coordinates.
   */
  void get_node( lo i_node, linear_algebra::coordinates< 4 > & node ) const {
    node[ 0 ] = _nodes[ 4 * i_node ];
    node[ 1 ] = _nodes[ 4 * i_node + 1 ];
    node[ 2 ] = _nodes[ 4 * i_node + 2 ];
    node[ 3 ] = _nodes[ 4 * i_node + 3 ];
  }

  /**
   * Returns coordinates of all nodes of an element.
   * @param[in] i_element Index of the element.
   * @param[out] node1 Coordinates of the first node.
   * @param[out] node2 Coordinates of the second node.
   * @param[out] node3 Coordinates of the third node.
   * @param[out] node4 Coordinates of the fourth node.
   */
  void get_nodes( lo i_element, linear_algebra::coordinates< 4 > & node1,
    linear_algebra::coordinates< 4 > & node2,
    linear_algebra::coordinates< 4 > & node3,
    linear_algebra::coordinates< 4 > & node4 ) const {
    node1[ 0 ] = _nodes[ 4 * _elements[ 4 * i_element ] ];
    node1[ 1 ] = _nodes[ 4 * _elements[ 4 * i_element ] + 1 ];
    node1[ 2 ] = _nodes[ 4 * _elements[ 4 * i_element ] + 2 ];
    node1[ 3 ] = _nodes[ 4 * _elements[ 4 * i_element ] + 3 ];
    node2[ 0 ] = _nodes[ 4 * _elements[ 4 * i_element + 1 ] ];
    node2[ 1 ] = _nodes[ 4 * _elements[ 4 * i_element + 1 ] + 1 ];
    node2[ 2 ] = _nodes[ 4 * _elements[ 4 * i_element + 1 ] + 2 ];
    node2[ 3 ] = _nodes[ 4 * _elements[ 4 * i_element + 1 ] + 3 ];
    node3[ 0 ] = _nodes[ 4 * _elements[ 4 * i_element + 2 ] ];
    node3[ 1 ] = _nodes[ 4 * _elements[ 4 * i_element + 2 ] + 1 ];
    node3[ 2 ] = _nodes[ 4 * _elements[ 4 * i_element + 2 ] + 2 ];
    node3[ 3 ] = _nodes[ 4 * _elements[ 4 * i_element + 2 ] + 3 ];
    node4[ 0 ] = _nodes[ 4 * _elements[ 4 * i_element + 3 ] ];
    node4[ 1 ] = _nodes[ 4 * _elements[ 4 * i_element + 3 ] + 1 ];
    node4[ 2 ] = _nodes[ 4 * _elements[ 4 * i_element + 3 ] + 2 ];
    node4[ 3 ] = _nodes[ 4 * _elements[ 4 * i_element + 3 ] + 3 ];
  }

  /**
   * Returns reference to the node coordinates.
   */
  std::vector< sc > & get_nodes( ) {
    return _nodes;
  }

  /**
   * Returns reference to the node coordinates.
   */
  const std::vector< sc > & get_nodes( ) const {
    return _nodes;
  }

  /**
   * Sets coordinates of a node.
   * @param[in] i_node Index of the node.
   * @param[in] node Node coordinates.
   */
  void set_node( lo i_node, const linear_algebra::coordinates< 4 > & node ) {
    _nodes[ 4 * i_node ] = node[ 0 ];
    _nodes[ 4 * i_node + 1 ] = node[ 1 ];
    _nodes[ 4 * i_node + 2 ] = node[ 2 ];
    _nodes[ 4 * i_node + 3 ] = node[ 3 ];
  }

  /**
   * Returns node indices of an edge.
   * @param[in] i_edge Index of the edge.
   * @param[out] edge Element indices.
   */
  void get_edge( lo i_edge, linear_algebra::indices< 2 > & edge ) const {
    edge[ 0 ] = _edges[ 2 * i_edge ];
    edge[ 1 ] = _edges[ 2 * i_edge + 1 ];
  }

  /**
   * Returns edge indices of an element.
   * @param[in] i_element Index of an element.
   * @param[out] edges Edge indices.
   */
  void get_edges( lo i_element, linear_algebra::indices< 6 > & edges ) const {
    edges[ 0 ] = _element_to_edges[ 6 * i_element ];
    edges[ 1 ] = _element_to_edges[ 6 * i_element + 1 ];
    edges[ 2 ] = _element_to_edges[ 6 * i_element + 2 ];
    edges[ 3 ] = _element_to_edges[ 6 * i_element + 3 ];
    edges[ 4 ] = _element_to_edges[ 6 * i_element + 4 ];
    edges[ 5 ] = _element_to_edges[ 6 * i_element + 5 ];
  }

  /**
   * Returns the centroid of the element.
   * @param[in] i_elem element index.
   * @param[out] centroid Allocated array containing the element centroid on
   * return.
   */
  void get_centroid(
    lo i_elem, linear_algebra::coordinates< 4 > & centroid ) const {
    linear_algebra::coordinates< 4 > x1, x2, x3, x4;
    get_nodes( i_elem, x1, x2, x3, x4 );
    centroid[ 0 ] = ( x1[ 0 ] + x2[ 0 ] + x3[ 0 ] + x4[ 0 ] ) / 4.0;
    centroid[ 1 ] = ( x1[ 1 ] + x2[ 1 ] + x3[ 1 ] + x4[ 1 ] ) / 4.0;
    centroid[ 2 ] = ( x1[ 2 ] + x2[ 2 ] + x3[ 2 ] + x4[ 2 ] ) / 4.0;
    centroid[ 3 ] = ( x1[ 3 ] + x2[ 3 ] + x3[ 3 ] + x4[ 3 ] ) / 4.0;
  }

  /**
   * Returns element normal vector.
   * @param[in] i_element Index of the element.
   * @param[out] n Normal indices.
   */
  void get_spatial_normal(
    lo i_element, linear_algebra::coordinates< 3 > & n ) const {
    n[ 0 ] = _normals[ 3 * i_element ];
    n[ 1 ] = _normals[ 3 * i_element + 1 ];
    n[ 2 ] = _normals[ 3 * i_element + 2 ];
  }

  /**
   * Returns element normal vector.
   * @param[in] i_node Index of the node.
   * @param[out] n Normal indices.
   */
  void get_spatial_nodal_normal(
    lo i_node, linear_algebra::coordinates< 3 > & n ) const {
    n[ 0 ] = n[ 1 ] = n[ 2 ] = 0.0;
    lo size = _node_to_elements[ i_node ].size( );
    linear_algebra::coordinates< 3 > nn;
    lo i_elem;

    for ( lo i = 0; i < size; ++i ) {
      i_elem = _node_to_elements[ i_node ][ i ];
      get_spatial_normal( i_elem, nn );
      n[ 0 ] += _areas[ i_elem ] * nn[ 0 ];
      n[ 1 ] += _areas[ i_elem ] * nn[ 1 ];
      n[ 2 ] += _areas[ i_elem ] * nn[ 2 ];
    }

    sc norm = std::sqrt( n[ 0 ] * n[ 0 ] + n[ 1 ] * n[ 1 ] + n[ 2 ] * n[ 2 ] );
    n[ 0 ] /= norm;
    n[ 1 ] /= norm;
    n[ 2 ] /= norm;
  }

  /**
   * Scales the mesh around its spatial centroid (scales only the spatial
   * coordinates).
   * @param[in] factor Scaling multiplier.
   */
  void scale_in_space( sc factor );

  /**
   * Refines the mesh by quadrisection.
   * @param[in] level Number of refinements.
   */
  void refine( int level = 1 );

  /**
   * Returns the volume mesh.
   */
  virtual triangular_surface_mesh * get_spatial_surface_mesh( ) override {
    return nullptr;
  }

  /**
   * Returns the volume mesh.
   */
  virtual const triangular_surface_mesh * get_spatial_surface_mesh( )
    const override {
    return nullptr;
  }

  /**
   * Returns the volume mesh.
   */
  virtual tetrahedral_volume_mesh * get_spatial_volume_mesh( ) override {
    return nullptr;
  }

  /**
   * Returns the volume mesh.
   */
  virtual const tetrahedral_volume_mesh * get_spatial_volume_mesh( )
    const override {
    return nullptr;
  }

  /**
   * Returns the centroid of the mesh.
   * @param[out] centroid Allocated array containing the spatial centroid on
   * return.
   */
  void get_spatial_centroid( linear_algebra::coordinates< 3 > & centroid );

 protected:
  /**
   * Precomputes areas (volumes) of elements.
   */
  void init_areas( );

  /**
   * Initializes edges.
   */
  void init_edges( );

  /**
   * Initializes the mapping from nodes to elements.
   */
  void init_node_to_elements( );

  /**
   * Initializes faces.
   */
  /*
  void init_faces( );
  */

  lo _n_nodes;               //!< number of nodes
  std::vector< sc > _nodes;  //!< coordinates of nodes
  std::vector< std::vector< lo > >
    _node_to_elements;          //!< mapping from nodes to elements
  lo _n_elements;               //!< number of elements
  std::vector< lo > _elements;  //!< indices into #_nodes
  std::vector< sc > _areas;     //!< element areas

  lo _n_edges;                          //!< number of edges
  std::vector< lo > _edges;             //!< indices into #_nodes
  std::vector< lo > _element_to_edges;  //!< indices into #_edges

  /*
    lo _n_faces;                          //!< number of faces
    std::vector< lo > _faces;             //!< indices into #_nodes
    std::vector< lo > _element_to_faces;  //!< indices into #_faces
  */

  std::vector< sc > _normals;  //!< spatial normals (temporal part is zero)
};

#endif /* INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_MESH_H_ */
