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

/** @file distributed_spacetime_tensor_mesh.h
 * @brief Container holding sequence of space-time meshes distributed among MPI
 * processes.
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_TENSOR_MESH_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_TENSOR_MESH_H_

#include "besthea/io_routines.h"
#include "besthea/scheduling_time_cluster.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/temporal_mesh.h"
#include "besthea/tree_structure.h"
#include "besthea/triangular_surface_mesh.h"

#include <mpi.h>
#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class distributed_spacetime_tensor_mesh;
  }
}

/**
 * Class serving as a container holding spacetime tensor product meshes
 * distributed among MPI processes.
 */
class besthea::mesh::distributed_spacetime_tensor_mesh {
 public:
  /**
   * Constructor taking decomposition file and MPI communicator. Based on the
   * communicator, the class automatically evenly distributes meshes among MPI
   * processes.
   * @param[in] decomposition_file Path to the file describing the input mesh.
   * @param[in] tree_file Path to the file describing the initial temporal
   * cluster tree
   * @param[in] comm Pointer to the MPI communicator associated with
   * decomposition.
   */
  distributed_spacetime_tensor_mesh( const std::string & decomposition_file,
    const std::string & tree_file, const std::string & distribution_file,
    MPI_Comm * comm );

  /**
   * Destructor.
   */
  ~distributed_spacetime_tensor_mesh( );

  /**
   * Mapping from local temporal element index to global temporal element
   * index.
   * @param[in] local_idx Local time index.
   */
  lo local_2_global_time( lo local_idx ) const {
    return _my_start_idx + local_idx;
  }

  /**
   * Mapping from local spacetime element index to global spacetime element
   * index.
   * @param[in] local_idx Local spacetime index.
   */
  lo local_2_global( lo local_idx ) const {
    return _my_start_idx * _space_mesh->get_n_elements( ) + local_idx;
  }

  /**
   * Mapping from global temporal element index to local temporal element index.
   * @param[in] global_idx Global temporal element index.
   */
  lo global_2_local_time( lo global_idx ) const {
    return global_idx - _my_start_idx;
  }

  /**
   * Mapping from global spacetime element index to local spacetime element
   * index.
   * @param[in] global_idx Global spacetime element index.
   */
  lo global_2_local( lo global_idx ) const {
    return global_idx - _my_start_idx * _space_mesh->get_n_elements( );
  }

  /**
   * Returns the mesh associated with current MPI process.
   */
  spacetime_tensor_mesh * const get_my_mesh( ) const {
    return _my_mesh;
  }

  /**
   * Returns the tree composed of scheduling_time_clusters.
   */
  tree_structure * const get_distribution_tree( ) const {
    return _dist_tree;
  }

  /**
   * Returns start of the global time interval
   */
  sc get_start( ) const {
    return _t_start;
  }

  /**
   * Returns end of the global time interval
   */
  sc get_end( ) const {
    return _t_end;
  }

  /**
   * Returns total number of elements in the distributed mesh
   */
  lo get_n_elements( ) const {
    return _n_global_elements;
  }

  /**
   * Returns vector defining time slices
   */
  const std::vector< sc > & get_slices( ) const {
    return _slices;
  }

 protected:
  /**
   * Loads submeshes assigned to the current rank and merges them into one mesh.
   * @param[in] decomposition_file Path to the file describing the input mesh.
   * @param[in] tree_file Path to the file describing the initial temporal
   * cluster tree
   * @param[in] distribution_file Path to the file describing the time-slices
   * distribution among MPI processes.
   */
  bool load( const std::string & decomposition_file,
    const std::string & tree_file, const std::string & distribution_file );

  /**
   * Goes through the temporal tree and finds leaf clusters with the same owner
   * @param[in] root Root cluster of the tree.
   * @param[in] center Center of the current cluster.
   * @param[in] half_size Half size of the current cluster.
   * @param[in,out] slices_indices Indices of the slices owned by the process.
   * @param[in] start Start index of curren cluster.
   * @param[in] end End index of current clusters.
   */
  void find_my_slices( scheduling_time_cluster * root, sc center, sc half_size,
    std::vector< lo > & slices_indices, lo start, lo end );

  MPI_Comm * _comm;  //!< MPI communicator associated with the distributed mesh
  int _my_rank;      //!< MPI rank of current processor
  int _n_processes;  //!< total number of MPI processes in the communicator
  lo _n_meshes;      //!< total number of input meshes
  lo _n_meshes_per_rank;             //!< number of meshes owned by this process
  std::vector< sc > _slices;         //!< vector defining the global mesh slices
  spacetime_tensor_mesh * _my_mesh;  //!< part of the global mesh owned by the
                                     //!< rank (including the nearfield)
  spacetime_tensor_mesh * _nearfield_mesh;  //!< temporal nearfield mesh
  triangular_surface_mesh *
    _space_mesh;  //!< surface mesh the spacetime mesh is composed of
  temporal_mesh *
    _time_mesh;  //!< temporal mesh the spacetime mesh is composed of
  temporal_mesh *
    _nearfield_time_mesh;  //!< nearfiel temporal mesh the nearfield spacetime
                           //!< mesh is composed of
  lo _my_start_idx;  //!< initial timestep on this MPI rank (used for loc/glob
                     //!< mapping)
  tree_structure * _dist_tree;  //!< temporal tree with distribution of clusters
                                //!< among MPI processes (reduced to essential)
  sc _t_start;                  //!< start of the global time interval
  sc _t_end;                    //!< end of the global time interval
  lo _n_global_elements;  //!< total number of elements in the distributed mesh
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_TENSOR_MESH_H_ */
