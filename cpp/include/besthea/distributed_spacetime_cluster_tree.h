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

/** @file distributed_spacetime_cluster_tree.h
 * @brief Spacetime cluster tree distributed among MPI processes
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_CLUSTER_TREE_H_

#include "besthea/block_vector.h"
#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/space_cluster_tree.h"
#include "besthea/spacetime_cluster.h"
#include "besthea/time_cluster_tree.h"
#include "besthea/vector.h"

#include <map>
#include <mpi.h>
#include <vector>

namespace besthea {
  namespace mesh {
    class distributed_spacetime_cluster_tree;
  }
}

/**
 * Class representing spacetime cluster tree distributed among MPI processes
 */
class besthea::mesh::distributed_spacetime_cluster_tree {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  /**
   * Constructor
   */
  distributed_spacetime_cluster_tree(
    const distributed_spacetime_tensor_mesh & spacetime_mesh, lo time_levels,
    lo n_min_elems, sc st_coeff, lo spatial_nearfield_limit, MPI_Comm * comm );

 private:
  /**
   *
   */
  void build_tree( general_spacetime_cluster * root );
  lo _levels;  //!< number of levels in the tree
  const distributed_spacetime_tensor_mesh &
    _spacetime_mesh;                  //!< underlying distributed spacetime mesh
  general_spacetime_cluster * _root;  //!< root of the cluster tree
  lo _start_spatial_level;   //!< auxiliary variable determining the appropriate
                             //!< starting level in the space cluster tree
  lo _start_temporal_level;  //!< auxiliary variable to determine in which level
                             //!< the spatial refinement starts
                             //!< (meaningful only if _start_spatial_level = 0)
  sc _s_t_coeff;    //!< coefficient to determine the coupling of the spatial
                    //!< and temporal levels
  lo _n_min_elems;  //!< minimum number of elements so that cluster can be split
  lo _spatial_nearfield_limit;  //!< number of the clusters in the vicinity to
                                //!< be considered as nearfield
  const std::vector< std::vector< lo > > _idx_2_coord = { { 1, 1, 1 },
    { 0, 1, 1 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 0 }, { 0, 1, 0 },
    { 0, 0, 0 },
    { 1, 0, 0 } };  //!< auxiliary mapping from octant indexing to coordinates
  std::vector< sc > _bounding_box_size;  //!< size of the mesh bounding box;
  time_cluster *
    _time_root;            //!< temporal part of the spacetime tree root cluster
  const MPI_Comm * _comm;  //!< MPI communicator associated with the tree
  int _my_rank;            //!< MPI rank of current processor
  int _n_processes;  //!< total number of MPI processes in the communicator

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
   * Collectively computes number of elements in subdivisioning of 1Dx3D
   * bounding box
   * @param[in] space_order How many times is space divided.
   * @param[in] time_order How many times is time divided.
   * @param[inout] n_el_per_part Vector for storing number of spacetime
   * elements.
   */
  void get_n_elements_in_subdivisioning( general_spacetime_cluster * root,
    bool split_space, std::vector< lo > & n_elems_per_subd );
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_CLUSTER_TREE_H_ */
