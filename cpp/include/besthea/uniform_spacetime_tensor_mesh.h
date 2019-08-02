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

/** @file uniform_spacetime_tensor_mesh.h
 * @brief Tensor product spacetime mesh based on a triangular_surface_mesh.h and
 * uniform time stepping.
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_H_

#include "besthea/linear_algebra.h"
#include "besthea/mesh.h"
#include "besthea/triangular_surface_mesh.h"

namespace besthea {
  namespace mesh {
    class uniform_spacetime_tensor_mesh;
  }
}

/**
 *  Class representing a tensor product spacetime mesh with a uniform timestep.
 */
class besthea::mesh::uniform_spacetime_tensor_mesh
  : public besthea::mesh::mesh {
 public:
  /**
   * Constructing mesh from a file.
   * @param[in] space_mesh Reference to a triangular_surface_mesh.h.
   * @param[in] end_time Temporal interval set to (0,end_time).
   * @param[in] n_timesteps Number of timesteps.
   */
  uniform_spacetime_tensor_mesh(
    triangular_surface_mesh & space_mesh, sc end_time, lo n_timesteps );

  uniform_spacetime_tensor_mesh( const uniform_spacetime_tensor_mesh & that )
    = delete;

  ~uniform_spacetime_tensor_mesh( );

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
   * Returns number of spatial elements.
   */
  lo get_n_spatial_elements( ) const {
    return _space_mesh->get_n_elements( );
  }

  /**
   * Returns the uniform timestep.
   */
  sc get_timestep( ) const {
    return _timestep;
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
   * Returns coordinates of all nodes of aspatial element.
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
   * Returns nodal normal vector.
   * @param[in] i_node Index of the node.
   * @param[out] n Normal indices.
   */
  void get_spatial_nodal_normal( lo i_node, sc * n ) const {
    _space_mesh->get_nodal_normal( i_node, n );
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
    return _n_timesteps;
  }

  /**
   * Returns number of temporal nodes.
   */
  lo get_n_temporal_nodes( ) const {
    return _n_timesteps + 1;
  }

  /**
   * Returns node indices of a temporal element.
   * @param[in] i_element Index of the temporal element.
   * @param[out] element Spatial temporal indices.
   */
  void get_temporal_element( lo i_element, lo * element ) const {
    element[ 0 ] = i_element;
    element[ 1 ] = i_element + 1;
  }

  /**
   * Returns a coordinate of a temporal node.
   * @param[in] i_node Index of the temporal node.
   */
  sc get_temporal_node( lo i_node ) const {
    return i_node * _timestep;
  }

  /**
   * Returns the centroid of the mesh.
   * @param[in] i_elem element index.
   * @param[out] centroid Allocated array containing the element centroid on
   * return.
   */
  void get_spatial_centroid( lo i_elem, sc * centroid ) const {
    sc x1[ 3 ], x2[ 3 ], x3[ 3 ];
    _space_mesh->get_nodes( i_elem, x1, x2, x3 );
    centroid[ 0 ] = ( x1[ 0 ] + x2[ 0 ] + x3[ 0 ] ) / 3.0;
    centroid[ 1 ] = ( x1[ 1 ] + x2[ 1 ] + x3[ 1 ] ) / 3.0;
    centroid[ 2 ] = ( x1[ 2 ] + x2[ 2 ] + x3[ 2 ] ) / 3.0;
  }

  /**
   * Prints the mesh into Paraview format.
   * @param[in] directory Directory to which the vtu file is saved.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] node_data Scalar nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] element_data Scalar elemental data.
   * @param[in] time_stride Stride for printed time steps.
   */
  bool print_vtu( const std::string & directory,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< linear_algebra::block_vector * > * node_data = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    const std::vector< linear_algebra::block_vector * > * element_data
    = nullptr,
    lo time_stride = 1 ) const;

  /**
   * Prints the EnSight Gold case file.
   * @param[in] directory Directory to which the case file is saved.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] time_stride Stride for time steps
   */
  bool print_ensight_case( const std::string & directory,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    lo time_stride = 1 ) const {
    return _space_mesh->print_ensight_case( directory, node_labels,
      element_labels, ( _n_timesteps + time_stride - 1 ) / time_stride,
      _timestep * time_stride );
  }

  /**
   * Prints the EnSight Gold geometry file.
   * @param[in] directory Directory to which the geometry file is saved.
   */
  bool print_ensight_geometry( const std::string & directory ) const {
    return _space_mesh->print_ensight_geometry( directory );
  }

  /**
   * Prints the EnSight Variable files for per element and per node data.
   * @param[in] directory Directory that datafile are printed to.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] node_data Scalar nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] element_data Scalar elemental data.
   * @param[in] time_stride Stride for time steps.
   */
  bool print_ensight_datafiles( const std::string & directory,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< linear_algebra::block_vector * > * node_data = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    const std::vector< linear_algebra::block_vector * > * element_data
    = nullptr,
    lo time_stride = 1 ) const {
    lo n_nodal = node_data ? node_data->size( ) : 0;
    lo n_elem = element_data ? element_data->size( ) : 0;
    std::vector< linear_algebra::vector * > node_data_for_one_ts( n_nodal );
    std::vector< linear_algebra::vector * > elem_data_for_one_ts( n_elem );
    for ( lo ts = 0; ts < _n_timesteps; ts += time_stride ) {
      for ( lo i = 0; i < n_nodal; ++i ) {
        node_data_for_one_ts[ i ] = &( *node_data )[ i ]->get_block( ts );
      }

      for ( lo i = 0; i < n_elem; ++i ) {
        elem_data_for_one_ts[ i ] = &( *element_data )[ i ]->get_block( ts );
      }

      if ( !_space_mesh->print_ensight_datafiles( directory, node_labels,
             &node_data_for_one_ts, element_labels, &elem_data_for_one_ts,
             ts / time_stride ) ) {
        return false;
      }
    }

    return true;
  }

  virtual triangular_surface_mesh * get_spatial_mesh( ) {
    return _space_mesh;
  }

  virtual const triangular_surface_mesh * get_spatial_mesh( ) const {
    return _space_mesh;
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

  triangular_surface_mesh *
    _space_mesh;    //!< pointer to a triangular_surface_mesh.h
  sc _end_time;     //!< temporal interval set to (0,end_time)
  sc _timestep;     //!< size of the timestep
  lo _n_timesteps;  //!< number of timesteps
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_H_ */
