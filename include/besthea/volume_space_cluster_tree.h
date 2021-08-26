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

/** @file volume_space_cluster_tree.h
 * @brief Octree of spatial clusters
 */

#ifndef INCLUDE_BESTHEA_VOLUME_SPACE_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_VOLUME_SPACE_CLUSTER_TREE_H_

#include "besthea/settings.h"
#include "besthea/tetrahedral_volume_mesh.h"
#include "besthea/vector.h"
#include "besthea/volume_space_cluster.h"

#include <map>
#include <optional>

namespace besthea {
  namespace mesh {
    class volume_space_cluster_tree;
  }
}

/**
 * Class representing an octree of spatial clusters.
 */
class besthea::mesh::volume_space_cluster_tree {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Constructor.
   * @param[in] mesh Reference to the underlying mesh.
   * @param[in] levels Maximum number of levels in the tree.
   * @param[in] n_min_elems Minimum number of elements so that cluster can be
   * split into octants
   * @param[in] print_warnings  If true, potential warnings are printed during
   * the tree construction
   */
  volume_space_cluster_tree( const tetrahedral_volume_mesh & mesh, lo levels,
    lo n_min_elems, bool print_warnings );

  /**
   * Destructor.
   */
  virtual ~volume_space_cluster_tree( ) {
    delete _root;
  }

  /**
   * Returns neighbors of a given cluster based on the limit number of clusters.
   *
   * @param[in] level Level on which neighbors are searched.
   * @param[in] target_grid_coordinates Coordinates determining the cluster
   * whose neighbors are searched. (only 3 coordinates: x,y,z)
   * @param[in] limit Number of clusters which should be considered neighbors in
   * each direction.
   * @param[in,out] neighbors Reference to the std::vector in which the pointers
   * to the neighbors should be included.
   */
  void find_neighbors( const lo level,
    const std::vector< slou > & target_grid_coordinates, slou limit,
    std::vector< volume_space_cluster * > & neighbors ) const;

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
   *
   * @note Floating point values are printed with single precision in the output
   * files.
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
    for ( lo i = 0; i < _real_n_levels; ++i ) {
      ret = print_tree( directory, include_padding, i, i );
    }
    return ret;
  }

  /**
   * Returns the associated distributed spacetime tensor mesh.
   */
  const tetrahedral_volume_mesh & get_mesh( ) const {
    return _mesh;
  }

  /**
   * Returns @ref _root.
   */
  volume_space_cluster * get_root( ) {
    return _root;
  }

  /**
   * Returns @ref _max_n_levels.
   */
  lo get_max_n_levels( ) const {
    return _max_n_levels;
  }

  /**
   * Returns @ref _n_max_elems_leaf.
   */
  lo get_n_max_elems_leaf( ) const {
    return _n_max_elems_leaf;
  }

  /**
   * Returns @ref _paddings.
   */
  const std::vector< sc > & get_paddings( ) const {
    return _paddings;
  }

  /**
   * Returns @ref _bounding_box_size.
   */
  const std::vector< sc > & get_bounding_box( ) const {
    return _bounding_box_size;
  }

  /**
   * Returns @ref _leaves.
   */
  std::vector< volume_space_cluster * > & get_leaves( ) {
    return _leaves;
  }

  /**
   * Traverses the cluster tree recursively and allocates and initializes the
   * moments for all clusters.
   * @param[in] current_cluster  Current cluster in the tree traversal.
   * @param[in] contribution_size Size of the moment of a cluster.
   */
  void initialize_moment_contributions(
    volume_space_cluster & current_cluster, lou contribution_size );

  /**
   * Traverses the cluster tree recursively and resets the moments of all
   * clusters to 0.
   * @param[in] current_cluster Current cluster in the tree traversal.
   */
  void clear_moment_contributions(
    volume_space_cluster & current_cluster ) const;

  /**
   * Prints the tree levelwise together with some additional information.
   */
  void print( ) {
    // print cluster information recursively
    print_internal( _root );
    // print general tree information
    std::cout << "maximal allowed number of levels: " << _max_n_levels
              << std::endl;
    std::cout << "real number of levels: " << _real_n_levels << std::endl;
    // print vector of paddings
    std::cout << "padding: " << std::endl;
    for ( lou i = 0; i < _paddings.size( ); ++i ) {
      std::cout << _paddings[ i ] << " ";
    }
    std::cout << std::endl;
  }

 private:
  /**
   * Computes a cubic bounding box of the underlying mesh.
   *
   * First the minimal axis parallel rectangular box in which the mesh is
   * contained is computed. This box is extended to a cube by expanding
   * the shorter sides of the box by increasing the upper bound.
   * @param[in,out] xmin Minimum x coordinate of element's centroids.
   * @param[in,out] xmax Maximum x coordinate of element's centroids.
   * @param[in,out] ymin Minimum y coordinate of element's centroids.
   * @param[in,out] ymax Maximum y coordinate of element's centroids.
   * @param[in,out] zmin Minimum z coordinate of element's centroids.
   * @param[in,out] zmax Maximum z coordinate of element's centroids.
   * @todo Instead of computing the bounding box for the surface and then again
   * for the volume one could load it the second time.
   */
  void compute_cubic_bounding_box(
    sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax );

  /**
   * Builds the tree recursively.
   * @param[in] root Node to stem from.
   */
  void build_tree( volume_space_cluster & root );

  /**
   * Computes the padding of clusters in the tree recursively.
   * @param[in] root Node to stem from.
   */
  sc compute_padding( volume_space_cluster & root );

  /**
   * Collects all clusters without descendants and stores them in the internal
   * vector @ref _leaves.
   * @param[in] root Root cluster of the tree.
   */
  void collect_leaves( volume_space_cluster & root );

  /**
   * Initializes @ref _levelwise_cluster_grids.
   *
   * It resizes @ref _levelwise_cluster_grids appropriately and fills it using
   * the routine @ref fill_levelwise_cluster_grids_recursively.
   */
  void initialize_levelwise_cluster_grids( );

  /**
   * Routine used to initialize @ref _levelwise_cluster_grids.
   *
   * The routine is based on a recursive tree traversal.
   * @param[in] current_cluster Current cluster in the tree traversal.
   */
  void fill_levelwise_cluster_grids_recursively(
    volume_space_cluster & current_cluster );

  /**
   * Aux for printing
   */
  void print_internal( volume_space_cluster * root ) {
    if ( root != nullptr ) {
      root->print( );
      const std::vector< volume_space_cluster * > * children
        = root->get_children( );
      if ( children != nullptr )
        for ( auto it = children->begin( ); it != children->end( ); ++it ) {
          for ( lo i = 0; i < ( *it )->get_level( ); ++i ) std::cout << " ";
          print_internal( *it );
        }
    }
  }

  volume_space_cluster * _root;           //!< root cluster of the tree
  const tetrahedral_volume_mesh & _mesh;  //!< underlying mesh
  lo _max_n_levels;   //!< maximal number of levels allowed in the tree
  lo _real_n_levels;  //!< real number of levels in the tree
                      //!< levels (depending on _n_min_elems)
  lo _n_min_elems;  //!< minimum number of elements so that cluster can be split
                    //!< into octants
  lo _n_max_elems_leaf;  //!< maximal number of elements in a leaf cluster after
                         //!< construction.
  std::vector< std::vector< volume_space_cluster * > >
    _non_empty_clusters_per_level;  //!< vectors of nonempty tree
                                    //!< clusters in each level
  std::vector< sc > _paddings;      //!< vector of paddings on each level
  lo _n_nonempty_clusters;          //!< number of nonempty clusters in the tree
  std::vector< std::vector< lo > >
    _idx_2_coord;  //!< auxiliary mapping from octant indexing to coordinates
  std::map< std::vector< slou >, volume_space_cluster * >
    _coord_2_cluster;  //!< map from cluster coordinates to its location in
                       //!< memory
  std::vector< sc > _bounding_box_size;  //!< size of the mesh bounding box;
  std::vector< volume_space_cluster * >
    _leaves;  //!< vector of all clusters without descendants

  std::vector< std::vector< volume_space_cluster * > >
    _levelwise_cluster_grids;  //!< For each level a cluster grid vector is
                               //!< stored. In this vector the clusters from a
                               //!< certain level l of the tree are arranged
                               //!< along the appropriate regular grid. A
                               //!< pointer to a cluster with grid coordinates
                               //!< (x,y,z) is stored at position
                               //!< x*4^l + y*2^l + z in the grid vector at
                               //!< level l. If there is an early leaf cluster X
                               //!< at some level in the tree all the entries on
                               //!< finer levels which would correspond to
                               //!< descendants of X point to X itself.
                               //!< Nullpointers mark non-existing clusters.
};

#endif /* INCLUDE_BESTHEA_VOLUME_SPACE_CLUSTER_TREE_H_ */
