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

/** @file space_cluster_tree.h
 * @brief Octree of spatial clusters
 */

#ifndef INCLUDE_BESTHEA_SPACE_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_SPACE_CLUSTER_TREE_H_

#include "besthea/settings.h"
#include "besthea/space_cluster.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/vector.h"

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
  using vector_type = besthea::linear_algebra::vector;

  /**
   * Constructor.
   * @param[in] triangular_surface_mesh Reference to the underlying mesh.
   * @param[in] levels Maximum number of levels in the tree.
   *
   *
   */
  space_cluster_tree(
    const triangular_surface_mesh & mesh, lo levels, lo n_min_elems );

  /**
   * Destructor.
   */
  virtual ~space_cluster_tree( ) {
    delete _root;
  }

 private:
  space_cluster * _root;                  //!< root cluster of the tree
  const triangular_surface_mesh & _mesh;  //!< underlying mesh
  lo _levels;                             //!< number of levels in the tree
  lo _n_min_elems;  //!< minimum number of elements so that cluster can be split
                    //!< into octants
  std::vector< std::vector< space_cluster * > >
    _non_empty_nodes;           //!< vectors of nonempty tree
                                //!< nodes in each level
  std::vector< sc > _paddings;  //!< vector of paddings on each level

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
};

#endif /* INCLUDE_BESTHEA_SPACE_CLUSTER_TREE_H_ */
