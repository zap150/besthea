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
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/temporal_mesh.h"
#include "besthea/triangular_surface_mesh.h"

#include <mpi.h>
#include <set>
#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class distributed_spacetime_tensor_mesh;
    class scheduling_time_cluster;
    class tree_structure;
  }
}

/**
 * Class serving as a container holding spacetime tensor product meshes
 * distributed among MPI processes.
 */
class besthea::mesh::distributed_spacetime_tensor_mesh : 
      public besthea::mesh::mesh{
 public:
  /**
   * Constructor taking decomposition file and MPI communicator. Based on the
   * communicator, the class automatically evenly distributes meshes among MPI
   * processes.
   * @param[in] decomposition_file Path to the file describing the input mesh.
   * @param[in] tree_file Path to the file describing the initial temporal
   * cluster tree.
   * @param[in] cluster_bounds_file Path to the file containing the cluster 
   * bounds of the clusters in the initial temporal cluster tree.
   * @param[in] comm Pointer to the MPI communicator associated with
   * decomposition.
   */
  distributed_spacetime_tensor_mesh( const std::string & decomposition_file,
    const std::string & tree_file, const std::string & cluster_bounds_file,
    const std::string & distribution_file, MPI_Comm * comm );

  /**
   * Destructor.
   */
  ~distributed_spacetime_tensor_mesh( );

  /**
   * Mapping from local temporal element index to global temporal element
   * index using a given start index.
   * @param[in] start_idx Start index of local time mesh.
   * @param[in] local_idx Local time index.
   * @note Use @ref _my_nearfield_start_idx as @p start_idx to map from temporal
   * nearfield to global indices.
   * @note Use @ref _local_start_idx as @p start_idx to map from local to global 
   * indices.
   */
  lo local_2_global_time( lo start_idx, lo local_idx ) const {
    return start_idx + local_idx;  
  }

  /**
   * Mapping from local spacetime element index to global spacetime element
   * index using a given start index.
   * @param[in] start_idx Start index of local time mesh. (NOT spacetime)
   * @param[in] local_idx Local spacetime index.
   * @note Use @ref _my_nearfield_start_idx as @p start_idx to map from temporal
   * nearfield to global indices.
   * @note Use @ref _local_start_idx as @p start_idx to map from local to global 
   * indices.
   */
  lo local_2_global( lo start_idx, lo local_idx ) const {
    return start_idx * _space_mesh->get_n_elements( ) + local_idx;
  }

  /**
   * Mapping from global temporal element index to local temporal element index
   * using a given start index.
   * @param[in] start_idx Start index of local time mesh.
   * @param[in] global_idx Global temporal element index.
   * @note Use @ref _my_nearfield_start_idx as @p start_idx to map from global 
   * to temporal nearfield indices.
   * @note Use @ref _local_start_idx as @p start_idx to map from global to local
   * indices.
   */
  lo global_2_local_time( lo start_idx, lo global_idx ) const {
    return global_idx - start_idx;
  }

  /**
   * Mapping from global spacetime element index to local spacetime element
   * index using a given start index.
   * @param[in] start_idx Start index of local time mesh. (NOT spacetime)
   * @param[in] global_idx Global spacetime element index.
   * @note Use @ref _my_nearfield_start_idx as @p start_idx to map from global 
   * to temporal nearfield indices.
   * @note Use @ref _local_start_idx as @p start_idx to map from global to local
   * indices.
   */
  lo global_2_local( lo start_idx, lo global_idx ) const {
    return global_idx - start_idx * _space_mesh->get_n_elements( );
  }

  /**
   * Returns a pointer to the internally stored spatial mesh.
   * @todo adapt this.
   */
  virtual triangular_surface_mesh * get_spatial_surface_mesh( ) override {
    // return _my_mesh->get_spatial_surface_mesh( );
    return nullptr;
  }

  /**
   * Returns a pointer to the internally stored spatial mesh.
   * @todo adapt this.
   */
  virtual const triangular_surface_mesh * get_spatial_surface_mesh( )
    const override {
    // return _my_mesh;
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
   * Returns the mesh associated with current MPI process.
   */
  spacetime_tensor_mesh const * get_local_mesh( ) const {
    return _my_mesh;
  }

  /**
   * Returns the nearfield mesh of the current MPI process.
   */
  spacetime_tensor_mesh const * get_nearfield_mesh( ) const {
    return _nearfield_mesh;
  }

  /**
   * Returns the tree composed of scheduling_time_clusters (unmodifiable).
   */
  tree_structure const * get_distribution_tree( ) const;

  /**
   * Returns the tree composed of scheduling_time_clusters (modifiable).
   */
  tree_structure* get_distribution_tree( );

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
   * @todo global number of temporal elements is currently not available 
   * (or is it?)
   */
  // lo get_global_n_temporal_elements( ) const {
  //   return;
  // }

  /**
   * Returns the start index of the local mesh.
   */
  lo get_local_start_idx( ) const {
    return _local_start_idx;
  }

  /**
   * Returns the start index of the nearfield mesh.
   */
  lo get_nearfield_start_idx( ) const {
    return _my_nearfield_start_idx;
  }

  /**
   * Returns the rank of the current cluster.
   */
  int get_rank( ) const {
    return _my_rank;
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
   * @param[in] cluster_bounds_file Path to the file containing the cluster 
   * bounds of the clusters in the initial temporal cluster tree
   * @param[in] distribution_file Path to the file describing the time-slices
   * distribution among MPI processes.
   */
  bool load( const std::string & decomposition_file,
    const std::string & tree_file, 
    const std::string & cluster_bounds_file,
    const std::string & distribution_file );

  /**
   * Goes through the temporal tree and finds leaf clusters with the same owner
   * @param[in] root Root cluster of the tree.
   * @param[in,out] slices_indices Indices of the slices owned by the process.
   * @param[in] start Start index of curren cluster.
   * @param[in] end End index of current clusters.
   * @todo replace by a loop over all leaves (now the cluster bounds are 
   * initialized properly)
   */
  void find_my_slices( scheduling_time_cluster * root,
    std::vector< lo > & slices_indices, lo start, lo end );

  /**
   * Goes through the leaves of the temporal tree and adds time slice indices
   * whose meshes have to be loaded to the appropriate sets.
   * @param[in, out] nearfield_slice_indices  Set of all slice indices 
   *                                          corresponding to slices contained
   *                                          in the nearfield of local 
   *                                          clusters.
   * @param[in, out] local_slice_indices  Set of all slice indices corresponding
   *                                      to slices contained in local leaf 
   *                                      clusters.
   * @note (Sorted) sets are used instead of vectors to avoid repeated entries
   * (in particular relevant for nearfield_slices in case of general
   * admissibility criteria).
   */
  void find_slices_to_load( std::set< lo > & nearfield_slice_indices,
    std::set< lo > & local_slice_indices ) const ;

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
    _nearfield_time_mesh;  //!< nearfield temporal mesh the nearfield spacetime
                           //!< mesh is composed of
  lo _local_start_idx;  //!< initial timestep on this MPI rank (used for loc/glob
                     //!< mapping)
  lo _my_nearfield_start_idx; //!< initial timestep of the nearfield meshes on
                              //!< this MPI rank (used for nearfield/glob 
                              //!< mapping)
  tree_structure * _dist_tree;  //!< temporal tree with distribution of clusters
                                //!< among MPI processes (reduced to essential)
  sc _t_start;                  //!< start of the global time interval
  sc _t_end;                    //!< end of the global time interval
  lo _n_global_elements;  //!< total number of elements in the distributed mesh
  const MPI_Comm * _comm; //!< MPI communicator associated with the distributed 
                          //!< mesh
  int _my_rank;           //!< MPI rank of the current process
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_TENSOR_MESH_H_ */
