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

/** @file tetrahedral_volume_mesh.h
 * @brief Tetrahedral mesh of a volume of a 3D object.
 */

#ifndef INCLUDE_BESTHEA_TETRAHEDRAL_VOLUME_MESH_H_
#define INCLUDE_BESTHEA_TETRAHEDRAL_VOLUME_MESH_H_

#include "besthea/coordinates.h"
#include "besthea/indices.h"
#include "besthea/mesh.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

#include <optional>
#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class tetrahedral_volume_mesh;
  }
}

namespace besthea {
  namespace bem {
    template< class basis_type >
    class fe_space;

    class basis_tetra_p1;
  }
}

/**
 *  Class representing a tetrahedral mesh of a 3D volume
 */
class besthea::mesh::tetrahedral_volume_mesh : public besthea::mesh::mesh {
 public:
  /**
   * Constructor.
   */
  tetrahedral_volume_mesh( );

  /**
   * Constructing mesh from a file.
   * @param[in] file Path to the file.
   */
  tetrahedral_volume_mesh( const std::string & file );

  tetrahedral_volume_mesh( const tetrahedral_volume_mesh & that ) = delete;

  ~tetrahedral_volume_mesh( );

  /**
   * Prints info on the object.
   */
  void print_info( ) const;

  /**
   * Prints the mesh file.
   * @param[in] file File name.
   */
  void print( const std::string & file );

  /**
   * Prints the EnSight Gold case file.
   * @param[in] directory Directory to which the case file is saved.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] n_timesteps Number of time steps.
   * @param[in] timestep_size Size of the timestep.
   *
   * @note Timesteps are printed with reduced precision to output files.
   */
  bool print_ensight_case( const std::string & directory,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    lo n_timesteps = 0, sc timestep_size = 0 ) const;

  /**
   * Prints the EnSight Gold geometry file.
   * @param[in] directory Directory to which the geometry file is saved.
   */
  bool print_ensight_geometry( const std::string & directory ) const;

  /**
   * Prints the EnSight Variable files for per element and per node data.
   * @param[in] directory Directory that datafile are printed to.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] node_data Scalar nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] element_data Scalar elemental data.
   * @param[in] timestep Time step index
   */
  bool print_ensight_datafiles( const std::string & directory,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< linear_algebra::vector * > * node_data = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    const std::vector< linear_algebra::vector * > * element_data = nullptr,
    std::optional< lo > timestep = std::nullopt ) const;

  /**
   * Prints the mesh into Ensight gold format.
   * @param[in] directory Directory name
   * @param[in] node_labels Labels for nodal data.
   * @param[in] node_data Scalar nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] element_data Scalar elemental data.
   *
   * @note Floating point values are printed with single precision in the output
   * files.
   */
  bool print_ensight( const std::string & directory,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< linear_algebra::vector * > * node_data = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    const std::vector< linear_algebra::vector * > * element_data
    = nullptr ) const;

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
   * Returns number of edges.
   */
  lo get_n_edges( ) const {
    return _n_edges;
  }

  /**
   * Returns number of surface elements.
   */
  lo get_n_surface_elements( ) const {
    return _n_surface_elements;
  }

  /**
   * Returns number of surface nodes.
   */
  lo get_n_surface_nodes( ) const {
    return _n_surface_nodes;
  }

  /**
   * Returns number of surface edges.
   */
  lo get_n_surface_edges( ) const {
    return _n_surface_edges;
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
   * Returns node indices of a surface element.
   * @param[in] i_element Index of the element.
   * @param[out] element Element indices.
   */
  void get_surface_element(
    lo i_element, linear_algebra::indices< 3 > & element ) const {
    element[ 0 ] = _surface_elements[ 3 * i_element ];
    element[ 1 ] = _surface_elements[ 3 * i_element + 1 ];
    element[ 2 ] = _surface_elements[ 3 * i_element + 2 ];
  }

  /**
   * Returns coordinates of a node.
   * @param[in] i_node Index of the node.
   * @param[out] node Element coordinates.
   */
  void get_node( lo i_node, linear_algebra::coordinates< 3 > & node ) const {
    node[ 0 ] = _nodes[ 3 * i_node ];
    node[ 1 ] = _nodes[ 3 * i_node + 1 ];
    node[ 2 ] = _nodes[ 3 * i_node + 2 ];
  }

  /**
   * Returns coordinates of all nodes of an element.
   * @param[in] i_element Index of the element.
   * @param[out] node1 Coordinates of the first node.
   * @param[out] node2 Coordinates of the second node.
   * @param[out] node3 Coordinates of the third node.
   * @param[out] node4 Coordinates of the fourth node.
   */
  void get_nodes( lo i_element, linear_algebra::coordinates< 3 > & node1,
    linear_algebra::coordinates< 3 > & node2,
    linear_algebra::coordinates< 3 > & node3,
    linear_algebra::coordinates< 3 > & node4 ) const {
    node1[ 0 ] = _nodes[ 3 * _elements[ 4 * i_element ] ];
    node1[ 1 ] = _nodes[ 3 * _elements[ 4 * i_element ] + 1 ];
    node1[ 2 ] = _nodes[ 3 * _elements[ 4 * i_element ] + 2 ];
    node2[ 0 ] = _nodes[ 3 * _elements[ 4 * i_element + 1 ] ];
    node2[ 1 ] = _nodes[ 3 * _elements[ 4 * i_element + 1 ] + 1 ];
    node2[ 2 ] = _nodes[ 3 * _elements[ 4 * i_element + 1 ] + 2 ];
    node3[ 0 ] = _nodes[ 3 * _elements[ 4 * i_element + 2 ] ];
    node3[ 1 ] = _nodes[ 3 * _elements[ 4 * i_element + 2 ] + 1 ];
    node3[ 2 ] = _nodes[ 3 * _elements[ 4 * i_element + 2 ] + 2 ];
    node4[ 0 ] = _nodes[ 3 * _elements[ 4 * i_element + 3 ] ];
    node4[ 1 ] = _nodes[ 3 * _elements[ 4 * i_element + 3 ] + 1 ];
    node4[ 2 ] = _nodes[ 3 * _elements[ 4 * i_element + 3 ] + 2 ];
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
  void set_node( lo i_node, const linear_algebra::coordinates< 3 > & node ) {
    _nodes[ 3 * i_node ] = node[ 0 ];
    _nodes[ 3 * i_node + 1 ] = node[ 1 ];
    _nodes[ 3 * i_node + 2 ] = node[ 2 ];
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
   * Returns edge indices of a surface element.
   * @param[in] i_element Index of an element.
   * @param[out] edges Edge indices.
   */
  void get_surface_edges(
    lo i_element, linear_algebra::indices< 3 > & edges ) const {
    edges[ 0 ] = _surface_element_to_edges[ 3 * i_element ];
    edges[ 1 ] = _surface_element_to_edges[ 3 * i_element + 1 ];
    edges[ 2 ] = _surface_element_to_edges[ 3 * i_element + 2 ];
  }

  /**
   * Returns the centroid of the element.
   * @param[in] i_elem element index.
   * @param[out] centroid Allocated array containing the element centroid on
   * return.
   */
  void get_centroid(
    lo i_elem, linear_algebra::coordinates< 3 > & centroid ) const {
    linear_algebra::coordinates< 3 > x1, x2, x3, x4;
    get_nodes( i_elem, x1, x2, x3, x4 );
    centroid[ 0 ] = ( x1[ 0 ] + x2[ 0 ] + x3[ 0 ] + x4[ 0 ] ) / 4.0;
    centroid[ 1 ] = ( x1[ 1 ] + x2[ 1 ] + x3[ 1 ] + x4[ 1 ] ) / 4.0;
    centroid[ 2 ] = ( x1[ 2 ] + x2[ 2 ] + x3[ 2 ] + x4[ 2 ] ) / 4.0;
  }

  /**
   * Scales the mesh around its centroid.
   * @param[in] factor Scaling multiplier.
   */
  void scale( sc factor );

  /**
   * Refines the mesh by quadrisection.
   * @param[in] level Number of refinements.
   */
  void refine( int level = 1 );

  /**
   * Returns true for a surface node.
   */
  bool is_surface_node( lo i_node ) const {
    return _is_surface_node[ i_node ];
  }

  /**
   * Returns the orientation of surface elements.
   */
  const std::pair< std::vector< lo >, std::vector< lo > > &
  get_surface_orientation( ) const {
    return _surface_orientation;
  }

  /**
   * Returns the global number of dofs in the mesh, depending on the considered
   * discrete space.
   * @tparam space_type  @ref besthea::bem::fe_space representing p1 (or other)
   * basis functions. It determines the DOFs.
   */
  template< class space_type >
  lo get_n_dofs( ) const;

 protected:
  /**
   * Precomputes exterior normals and areas of elements.
   */
  void init_areas( );

  /**
   * Initializes edges.
   */
  void init_edges( );

  /**
   * Initializes number of surface nodes.
   */
  void init_surface_nodes( );

  /**
   * Initializes faces.
   */
  void init_faces( );

  /**
   * Returns the centroid of the mesh.
   * @param[out] centroid Allocated array containing the centroid on return.
   */
  void get_centroid( linear_algebra::coordinates< 3 > & centroid );

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
    return this;
  }

  /**
   * Returns the volume mesh.
   */
  virtual const tetrahedral_volume_mesh * get_spatial_volume_mesh( )
    const override {
    return this;
  }

  lo _n_nodes;               //!< number of nodes
  std::vector< sc > _nodes;  //!< coordinates of nodes
  std::vector< std::vector< lo > >
    _node_to_elements;                   //!< mapping from nodes to elements
  lo _n_surface_nodes;                   //!< number of surface nodes
  std::vector< bool > _is_surface_node;  //!< True if surface node

  lo _n_elements;               //!< number of elements
  std::vector< lo > _elements;  //!< indices into #_nodes
  std::vector< sc > _areas;     //!< element areas

  lo _n_surface_elements;               //!< number of elements
  std::vector< lo > _surface_elements;  //!< indices into #_nodes
  std::pair< std::vector< lo >, std::vector< lo > >
    _surface_orientation;  //!< orientation of n := (x2-x1)x(x3-x1) and -n

  lo _n_edges;                          //!< number of edges
  std::vector< lo > _edges;             //!< indices into #_nodes
  std::vector< lo > _element_to_edges;  //!< indices into #_edges

  lo _n_surface_edges;                          //!< number of surface edges
  std::vector< lo > _surface_element_to_edges;  //!< indices into #_edges

  lo _n_faces;                          //!< number of faces
  std::vector< lo > _faces;             //!< indices into #_nodes
  std::vector< lo > _element_to_faces;  //!< indices into #_faces
};

template<>
inline lo besthea::mesh::tetrahedral_volume_mesh::get_n_dofs<
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >( ) const {
  return _n_nodes;
}

#endif /* INCLUDE_BESTHEA_TETRAHEDRAL_VOLUME_MESH_H_ */
