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

/** @file triangular_surface_mesh.h
 * @brief Triangular mesh of a boundary of a 3D object.
 */

#ifndef INCLUDE_BESTHEA_TRIANGULAR_SURFACE_MESH_H_
#define INCLUDE_BESTHEA_TRIANGULAR_SURFACE_MESH_H_

#include "besthea/settings.h"

#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class triangular_surface_mesh;
  }
}

/**
 *  Class representing a triangular mesh of a 3D surface
 */
class besthea::mesh::triangular_surface_mesh {

 public:
  triangular_surface_mesh( );

  /**
   * Constructing mesh from a file.
   * @param[in] file Path to the file.
   */
  triangular_surface_mesh( const std::string & file );

  triangular_surface_mesh( const triangular_surface_mesh & that ) = delete;

  ~triangular_surface_mesh( );

  /**
   * Prints info on the object.
   */
  void print_info( ) const;

  /**
   * Prints the mesh into Paraview format.
   * @param[in] file File name.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] node_data Scalar nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] element_data Scalar elemental data.
   */
  bool print_vtu( const std::string & file,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< sc * > * node_data = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    const std::vector< sc * > * element_data = nullptr ) const;

  /**
   * Loads mesh from a file.
   * @param[in] file File name.
   */
  bool load( const std::string & file );

  /**
   * Returns area of a single element.
   * @param[in] i_elem Index of the element.
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
   * Returns node indices of an element.
   * @param[in] i_element Index of the element.
   * @param[out] element Element indices.
   */
  void get_element( lo i_element, lo * element ) const {
    element[ 0 ] = _elements[ 3 * i_element ];
    element[ 1 ] = _elements[ 3 * i_element + 1 ];
    element[ 2 ] = _elements[ 3 * i_element + 2 ];
  }

  /**
   * Returns coordinates of a node.
   * @param[in] i_node Index of the node.
   * @param[out] node Element coordinates.
   */
  void get_node( lo i_node, sc * node ) const {
    node[ 0 ] = _nodes[ 3 * i_node ];
    node[ 1 ] = _nodes[ 3 * i_node + 1 ];
    node[ 2 ] = _nodes[ 3 * i_node + 2 ];
  }

  /**
   * Sets coordinates of a node.
   * @param[in] i_node Index of the node.
   * @param[in] node Element coordinates.
   */
  void set_node( lo i_node, const sc * node ) {
    _nodes[ 3 * i_node ] = node[ 0 ];
    _nodes[ 3 * i_node + 1 ] = node[ 1 ];
    _nodes[ 3 * i_node + 2 ] = node[ 2 ];
  }

  /**
   * Returns node indices of an edge.
   * @param[in] i_edge Index of the edge.
   * @param[out] edge Element indices.
   */
  void get_edge( lo i_edge, lo * edge ) const {
    edge[ 0 ] = _edges[ 2 * i_edge ];
    edge[ 1 ] = _edges[ 2 * i_edge + 1 ];
  }

  /**
   * Returns edge indices of an element.
   * @param[in] i_edge Index of the edge.
   * @param[out] edge Element indices.
   */
  void get_edges( lo i_element, lo * edges ) const {
    edges[ 0 ] = _element_to_edges[ 3 * i_element ];
    edges[ 1 ] = _element_to_edges[ 3 * i_element + 1 ];
    edges[ 2 ] = _element_to_edges[ 3 * i_element + 2 ];
  }

  /**
   * Refines the mesh by quadrisection.
   * @param[in] level Number of refinements.
   */
  void refine( int level );

  /**
   * Maps the nodes to the unit sphere.
   */
  void map_to_unit_sphere( );

 protected:
  lo _n_nodes;               //!< number of nodes
  std::vector< sc > _nodes;  //!< coordinates of nodes

  lo _n_elements;               //!< number of elements
  std::vector< lo > _elements;  //!< indices into #_nodes
  std::pair< std::vector< lo >, std::vector< lo > >
    _orientation;              //!< orientation of n := (x2-x1)x(x3-x1) and -n
  std::vector< sc > _areas;    //!< element areas
  std::vector< sc > _normals;  //!< exterior normal vectors

  lo _n_edges;                          //!< number of edges
  std::vector< lo > _edges;             //!< indices into #_nodes
  std::vector< lo > _element_to_edges;  //!< indices into #_edges

  /**
   * Precomputes exterior normals and areas of elements.
   */
  void init_normals_and_areas( );

  /**
   * Initializes edges.
   */
  void init_edges( );
};

#endif /* INCLUDE_BESTHEA_TRIANGULAR_SURFACE_MESH_H_ */
