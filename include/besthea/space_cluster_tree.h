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

/** @file space_cluster_tree.h
 * @brief Octree of spatial clusters
 */

#ifndef INCLUDE_BESTHEA_SPACE_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_SPACE_CLUSTER_TREE_H_

#include "besthea/settings.h"
#include "besthea/space_cluster.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/vector.h"

#include <map>
#include <optional>

namespace besthea {
  namespace mesh {
    class space_cluster_tree;
  }
}

/**
 * Class representing an octree of spatial clusters.
 */
class besthea::mesh::space_cluster_tree {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Constructor.
   * @param[in] mesh Reference to the underlying mesh.
   * @param[in] levels Maximum number of levels in the tree.
   * @param[in] n_min_elems Minimum number of elements so that cluster can be
   *                        split into octants
   */
  space_cluster_tree(
    const triangular_surface_mesh & mesh, lo levels, lo n_min_elems );

  /**
   * Destructor.
   */
  virtual ~space_cluster_tree( ) {
    delete _root;
  }

  /**
   * Returns neighbors of a given cluster based on the limit number of clusters.
   *
   * @param[in] cluster Reference to the space cluster whos neighbors should be
   * found.
   * @param[in] limit Number of clusters which should be considered neighbors in
   * each direction.
   * @param[in,out] neighbors Reference to the std::vector in which the pointers
   * to the neighbors should be included.
   */
  void find_neighbors( space_cluster & cluster, slou limit,
    std::vector< space_cluster * > & neighbors ) const;

  /**
   * Prints cluster centers and half_sizes to visualizable datafile tree.vtu.
   *
   * To visualize the tree in ParaView:
   * 0. To ensure ParaView will work in 3D mode, load the underlying mesh file.
   * 1. Load the tree.vtu file.
   * 2. Change representation to Point Gaussians to see the centers of clusters
   * and possibly change the Gaussian radius.
   * 3. Apply the Glyph filter.
   * 4. Change Glyph type to Box, Orientation array to No orientation array,
   * Scale array to Half_sizes, Vector Scale Mode to Scale by Components, Glyph
   * Mode to All points, Scale Factor to 1, Representation to either Wireframe
   * or Surface with Edges (and play with its opacity).
   *
   * @param[in] directory Output directory.
   * @param[in] include_padding Adds padding to cluster's half-sizes.
   * @param[in] level If set, prints only a given level of the tree. For
   * -1, prints all levels.
   * @param[in] suffix Suffix for the filename.
   */
  bool print_tree( const std::string & directory, bool include_padding = false,
    lo level = -1, std::optional< lo > suffix = std::nullopt ) const;

  /**
   * Prints cluster centers and half_sizes on all levels to visualizable
   * datafiles tree.vtu.n.
   *
   * To visualize the tree in ParaView:
   * 0. To ensure ParaView will work in 3D mode, load the underlying mesh file.
   * 1. Load all files into ParaView.
   * 2. Change representation to Point Gaussians to see the centers of clusters
   * and possibly change the Gaussian radius.
   * 3. Apply the Glyph filter.
   * 4. Change Glyph type to Box, Orientation array to No orientation array,
   * Scale array to Half_sizes, Vector Scale Mode to Scale by Components, Glyph
   * Mode to All points, Scale Factor to 1, Representation to either Wireframe
   * or Surface with Edges (and play with its opacity).
   *
   * @param[in] directory Output directory.
   * @param[in] include_padding Adds padding to cluster's half-sizes.
   */
  bool print_tree_separately(
    const std::string & directory, bool include_padding = false ) const {
    bool ret = true;
    for ( lo i = 0; i < _levels; ++i ) {
      ret = print_tree( directory, include_padding, i, i );
    }
    return ret;
  }

  /**
   * Returns the root of the tree.
   */
  space_cluster * get_root( ) {
    return _root;
  }

  /**
   * Returns the number of levels in the tree.
   */
  lo get_levels( ) const {
    return _levels;
  }

  /**
   * Returns the maximal number of elements in a leaf cluster
   */
  lo get_n_max_elems_leaf( ) const {
    return _n_max_elems_leaf;
  }

  /**
   * Returns the vector of levelwise paddings.
   */
  const std::vector< sc > & get_paddings( ) const {
    return _paddings;
  }

  /**
   * Returns size of the underlying mesh bounding box.
   */
  const std::vector< sc > & get_bounding_box( ) const {
    return _bounding_box_size;
  }

  /**
   * Returns clusters without descendants.
   */
  std::vector< space_cluster * > & get_leaves( ) {
    return _leaves;
  }

  /**
   * Prints levels of the tree.
   */
  void print( ) {
    // print cluster information recursively
    print_internal( _root );
    // print general tree information
    std::cout << "number of levels: " << _levels << std::endl;
    // print vector of paddings
    std::cout << "padding: " << std::endl;
    for ( lou i = 0; i < _paddings.size( ); ++i ) {
      std::cout << _paddings[ i ] << " ";
    }
    std::cout << std::endl;
  }

 private:
  space_cluster * _root;                  //!< root cluster of the tree
  const triangular_surface_mesh & _mesh;  //!< underlying mesh
  lo _levels;                             //!< number of levels in the tree
  lo _real_max_levels;  //!< auxiliary value to determine number of real tree
                        //!< levels (depending on _n_min_elems)
  lo _n_min_elems;  //!< minimum number of elements so that cluster can be split
                    //!< into octants
  lo _n_max_elems_leaf;  //!< maximal number of elements in a leaf cluster after
                         //!< construction.
  std::vector< std::vector< space_cluster * > >
    _non_empty_nodes;           //!< vectors of nonempty tree
                                //!< nodes in each level
  std::vector< sc > _paddings;  //!< vector of paddings on each level
  lo _n_nonempty_nodes;         //!< number of nonempty clusters in the tree
  std::vector< std::vector< lo > >
    _idx_2_coord;  //!< auxiliary mapping from octant indexing to coordinates
  std::map< std::vector< slou >, space_cluster * >
    _coord_2_cluster;  //!< map from cluster coordinates to its location in
                       //!< memory
  std::vector< sc > _bounding_box_size;  //!< size of the mesh bounding box;
  std::vector< space_cluster * >
    _leaves;  //!< vector of all clusters without descendants

  /**
   * Computes the bounding box of the underlying mesh.
   * @param[in,out] xmin Minimum x coordinate of element's centroids.
   * @param[in,out] xmax Maximum x coordinate of element's centroids.
   * @param[in,out] ymin Minimum y coordinate of element's centroids.
   * @param[in,out] ymax Maximum y coordinate of element's centroids.
   * @param[in,out] zmin Minimum z coordinate of element's centroids.
   * @param[in,out] zmax Maximum z coordinate of element's centroids.
   */
  void compute_bounding_box(
    sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax );

  /**
   * Builds tree recursively
   * @param[in] root Node to stem from.
   * @param[in] level Current level.
   */
  void build_tree( space_cluster & root, lo level );

  /**
   * Recursively computes padding of clusters in the tree.
   * @param[in] root Node to stem from.
   */
  sc compute_padding( space_cluster & root );

  /**
   * Collects all clusters without descendants and stores them in the internal
   * _leaves vector.
   * @param[in] root Root cluster of the tree.
   */
  void collect_leaves( space_cluster & root );

  /**
   * Aux for printing
   */
  void print_internal( space_cluster * root ) {
    if ( root != nullptr ) {
      root->print( );
      std::vector< space_cluster * > * children = root->get_children( );
      if ( children != nullptr )
        for ( auto it = children->begin( ); it != children->end( ); ++it ) {
          for ( lo i = 0; i < ( *it )->get_level( ); ++i ) std::cout << " ";
          print_internal( *it );
        }
    }
  }
};

#endif /* INCLUDE_BESTHEA_SPACE_CLUSTER_TREE_H_ */
