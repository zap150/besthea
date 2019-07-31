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

/** @file spacetime_tensor_mesh.h
 * @brief Tensor product spacetime mesh based on a triangular_surface_mesh.h and
 * temporal_mesh.h
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_TENSOR_MESH_H_
#define INCLUDE_BESTHEA_SPACETIME_TENSOR_MESH_H_

#include "besthea/mesh.h"
#include "besthea/temporal_mesh.h"
#include "besthea/triangular_surface_mesh.h"

namespace besthea {
  namespace mesh {
    class spacetime_tensor_mesh;
  }
}

/**
 * Class representing a tensor product spacetime mesh with general temporal
 * decomposition.
 */
class besthea::mesh::spacetime_tensor_mesh : public besthea::mesh::mesh {
 public:
  /**
   * Constructing tensor product mesh from spatial and temporal meshes.
   * @param[in] space_mesh Reference to a triangular_surface_mesh object.
   * @param[in] time_mesh Reference to a temporal_mesh object.
   */
  spacetime_tensor_mesh(
    triangular_surface_mesh & space_mesh, temporal_mesh & time_mesh );

  spacetime_tensor_mesh( const spacetime_tensor_mesh & that ) = delete;

  ~spacetime_tensor_mesh( );

  /**
   * Prints info on the object.
   */
  void print_info( ) const;

  /**
   * Refines the spatial mesh by quadrisection, temporal by bisection.
   * @param[in] level Number of spatial refinements.
   * @param[in] temporal_order Number of temporal refinements per single spatial
   * refinement.
   */
  void refine( int level, int temporal_order = 1 );

  /**
   * Maps the spatial nodes to the unit sphere.
   */
  void map_to_unit_sphere( );

  /**
   * Returns number of space-time elements.
   */
  lo get_n_elements( ) const {
    return _space_mesh->get_n_elements( ) * _time_mesh->get_n_elements( );
  }

  /**
   * Returns number of space-time nodes.
   */
  lo get_n_nodes( ) const {
    return _space_mesh->get_n_nodes( ) * _time_mesh->get_n_nodes( );
  }

  sc area( lo i_element ) const {
    lo space_elem_idx;
    lo time_elem_idx;
    map_index( i_element, space_elem_idx, time_elem_idx );
    return spatial_area( space_elem_idx ) * temporal_length( time_elem_idx );
  }

  /**
   * Returns node indices of a space-time element.
   * @param[in] i_element Index of the element.
   * @param[out] element Element indices (array of 6 indices).
   */
  void get_element( lo i_element, lo * element ) const {
    lo space_elem_idx;
    lo time_elem_idx;
    map_index( i_element, space_elem_idx, time_elem_idx );
    lo sp_element[ 3 ];
    lo t_element[ 2 ];

    _space_mesh->get_element( space_elem_idx, sp_element );
    _time_mesh->get_element( time_elem_idx, t_element );

    element[ 0 ] = sp_element[ 0 ] + t_element[ 0 ] * get_n_spatial_nodes( );
    element[ 1 ] = sp_element[ 1 ] + t_element[ 0 ] * get_n_spatial_nodes( );
    element[ 2 ] = sp_element[ 2 ] + t_element[ 0 ] * get_n_spatial_nodes( );
    element[ 3 ] = sp_element[ 0 ] + t_element[ 1 ] * get_n_spatial_nodes( );
    element[ 4 ] = sp_element[ 1 ] + t_element[ 1 ] * get_n_spatial_nodes( );
    element[ 5 ] = sp_element[ 2 ] + t_element[ 1 ] * get_n_spatial_nodes( );
  }

  void get_node( lo i_node, sc * node ) const {
    lo t_idx = i_node / get_n_spatial_nodes( );
    lo s_idx = i_node % get_n_spatial_nodes( );

    _space_mesh->get_node( s_idx, node );
    _time_mesh->get_node( t_idx, node + 3 );
  }

  /**
   * Returns number of spatial elements.
   */
  lo get_n_spatial_elements( ) const {
    return _space_mesh->get_n_elements( );
  }

  /**
   * Returns number of spatial nodes.
   */
  lo get_n_spatial_nodes( ) const {
    return _space_mesh->get_n_nodes( );
  }

  /**
   * Returns number of spatial edges.
   */
  lo get_n_spatial_edges( ) const {
    return _space_mesh->get_n_edges( );
  }

  /**
   * Returns node indices of a spatial element.
   * @param[in] i_element Index of the spatial element.
   * @param[out] element Spatial element indices.
   */
  void get_spatial_element( lo i_element, lo * element ) const {
    _space_mesh->get_element( i_element, element );
  }

  /**
   * Returns coordinates of a spatial node.
   * @param[in] i_node Index of the spatial node.
   * @param[out] node Spatial node coordinates.
   */
  void get_spatial_node( lo i_node, sc * node ) const {
    _space_mesh->get_node( i_node, node );
  }

  /**
   * Returns coordinates of all nodes of a spatial element.
   * @param[in] i_element Index of the element.
   * @param[out] node1 Coordinates of the first node.
   * @param[out] node2 Coordinates of the second node.
   * @param[out] node3 Coordinates of the third node.
   */
  void get_spatial_nodes(
    lo i_element, sc * node1, sc * node2, sc * node3 ) const {
    _space_mesh->get_nodes( i_element, node1, node2, node3 );
  }

  /**
   * Returns element normal vector.
   * @param[in] i_element Index of the element.
   * @param[out] n Normal indices.
   */
  void get_spatial_normal( lo i_element, sc * n ) const {
    _space_mesh->get_normal( i_element, n );
  }

  /**
   * Returns area of a single spatial element.
   * @param[in] i_elem Index of the element.
   */
  sc spatial_area( lo i_elem ) const {
    return _space_mesh->area( i_elem );
  }

  /**
   * Returns number of temporal elements.
   */
  lo get_n_temporal_elements( ) const {
    return _time_mesh->get_n_elements( );
  }

  /**
   * Returns number of temporal nodes.
   */
  lo get_n_temporal_nodes( ) const {
    return _time_mesh->get_n_nodes( );
  }

  /**
   * Returns node indices of a temporal element.
   * @param[in] i_element Index of the temporal element.
   * @param[out] element Temporal indices.
   */
  void get_temporal_element( lo i_element, lo * element ) const {
    _time_mesh->get_element( i_element, element );
  }

  /**
   * Returns a coordinate of a temporal node.
   * @param[in] i_node Index of the temporal node.
   */
  sc get_temporal_node( lo i_node ) const {
    return _time_mesh->get_node( i_node );
  }

  /**
   * Returns a coordinate of a temporal node.
   * @param[in] i_node Index of the temporal node.
   */
  void get_temporal_node( lo i_node, sc * node ) const {
    _time_mesh->get_node( i_node, node );
  }

  /**
   * Returns coordinates of all temporal nodes of an element.
   * @param[in] i_element Index of the temporal element.
   * @param[out] node1 Coordinate of the first node (beginning of the temporal
   * subinterval).
   * @param[out] node2 Coordinate of the second node (end of the temporal
   * subinterval).
   */
  void get_nodes( lo i_element, sc * node1, sc * node2 ) const {
    _time_mesh->get_nodes( i_element, node1, node2 );
  }

  sc temporal_length( lo i_element ) const {
    return _time_mesh->length( i_element );
  }

  /**
   * Prints the mesh into Paraview format.
   * @param[in] file File name.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] node_data Scalar nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] element_data Scalar elemental data.
   */
  bool print_vtu( const std::string & directory,
    const std::vector< std::string > * node_labels,
    const std::vector< linear_algebra::block_vector * > * node_data,
    const std::vector< std::string > * element_labels,
    const std::vector< linear_algebra::block_vector * > * element_data,
    lo time_stride ) const;

  virtual triangular_surface_mesh * get_spatial_mesh( ) {
    return _space_mesh;
  }

  virtual temporal_mesh * get_temporal_mesh( ) {
    return _time_mesh;
  }

 protected:
  /**
   * Refines the spatial mesh by quadrisection.
   * @param[in] level Number of spatial refinements.
   */
  void refine_space( int level );

  /**
   * Refines the temporal mesh by bisection.
   * @param[in] level Number of temporal refinements.
   */
  void refine_time( int level );

  /**
   * Maps global space-time index to separate space and time indices.
   * @param[in] i_element Space-time element index.
   * @param[out] i_space_element Spatial element index.
   * @param[out] i_time_element Temporal element index.
   */
  void map_index(
    lo i_element, lo & i_space_element, lo & i_time_element ) const {
    i_space_element = i_element % get_n_spatial_elements( );
    i_time_element = i_element / get_n_spatial_elements( );
  }

  triangular_surface_mesh *
    _space_mesh;               //!< pointer to a triangular_surface_mesh.h
  temporal_mesh * _time_mesh;  //!< pointer to a temporal_mesh.h
};

#endif /* INCLUDE_BESTHEA_SPACETIME_TENSOR_MESH_H_ */