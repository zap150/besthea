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
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/space_cluster_tree.h"
#include "besthea/spacetime_cluster.h"
#include "besthea/time_cluster_tree.h"
#include "besthea/timer.h"
#include "besthea/vector.h"

#include <map>
#include <mpi.h>
#include <omp.h>
#include <vector>

namespace besthea {
  namespace mesh {
    class distributed_spacetime_cluster_tree;
    class distributed_spacetime_tensor_mesh;
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
   * @param[in] spacetime_mesh Distributed spacetime mesh.
   * @param[in] max_levels Bound on the number of levels in the tree.
   * @param[in] n_min_elems Minimum number of spacetime elements in cluster.
   * @param[in] st_coeff Coefficient to determine the coupling of the spatial
   * and temporal levels.
   * @param[in] spatial_nearfield_limit Number of clusters in the vicinity of a
   * given clusters to be considered as nearfield
   * @param[in] comm MPI communicator associated with the tree.
   */
  distributed_spacetime_cluster_tree(
    distributed_spacetime_tensor_mesh & spacetime_mesh, lo max_levels,
    lo n_min_elems, sc st_coeff, slou spatial_nearfield_limit,
    MPI_Comm * comm );

  /**
   * Destructor.
   */
  ~distributed_spacetime_cluster_tree( ) {
    if ( _root )
      delete _root;
  }

  /**
   * Returns the root of the tree.
   */
  general_spacetime_cluster * get_root( ) {
    return _root;
  }

  /**
   * Returns the associated distributed spacetime tensor mesh.
   */
  const distributed_spacetime_tensor_mesh & get_mesh( ) {
    return _spacetime_mesh;
  }

  /**
   * Returns the distribution tree corresponding to the spacetime cluster tree
   */
  tree_structure * get_distribution_tree( ) {
    return _spacetime_mesh.get_distribution_tree( );
  }

  /**
   * Returns the vector of local leaves.
   */
  const std::vector< general_spacetime_cluster * > & get_local_leaves( ) {
    return _local_leaves;
  }

  /**
   * Returns the vector of levelwise spatial paddings.
   */
  const std::vector< sc > & get_spatial_paddings( ) const {
    return _spatial_paddings;
  }

  /**
   * Returns the bound for the maximal number of refinements in space of the
   * clusters in the local part of the distributed tree.
   */
  lo get_local_max_space_level( ) const {
    return _local_max_space_level;
  }

  /**
   * Returns the level where space is first refined.
   * @note: If @p _initial_space_refinement != 0 this is always 2.
   */
  lo get_start_space_refinement( ) const {
    return _start_space_refinement;
  }

  /**
   * Returns the number of refinements in space applied for the clusters at
   * level 0.
   */
  lo get_initial_space_refinement( ) const {
    return _initial_space_refinement;
  }

  /**
   * Prints levels of the tree.
   */
  void print( ) {
    // print general tree information
    std::cout << "number of levels of spacetime tree " << _real_max_levels
              << std::endl;
    std::cout << "spatial padding: " << std::endl;
    for ( lou i = 0; i < _spatial_paddings.size( ); ++i ) {
      std::cout << "level " << i << ": " << _spatial_paddings[ i ] << std::endl;
    }
    // print cluster information recursively
    print_internal( *_root );
  }

 private:
  /**
   *
   */
  void build_tree( general_spacetime_cluster * root );

  /**
   * Expands the distribution tree included in @p _spacetime_mesh by adding
   * relevant time clusters which appear as components of spacetime clusters in
   * the current tree but are not in the distribution tree.
   * @note The clusters which are refined are determined using the routine
   *       @ref tree_structure::determine_clusters_to_refine and the refinement
   *       is executed by @ref expand_distribution_tree_recursively.
   * @note The nearfields, interaction lists and send lists of the distribution
   *       tree are cleared using the routine
   *       @ref tree_structure::clear_cluster_lists and filled anew.
   * @note After execution the distribution tree is possibly not locally
   *       essential anymore. Call the routine
   *       @ref tree_structure::reduce_2_essential on the distribution tree to
   *       make it locally essential again.
   */
  void expand_distribution_tree_locally( );

  /**
   * Expands the distribution tree included in @p _spacetime_mesh by
   * exchanging information with other processes (such that the distribution
   * tree includes the locally essential subtree of the global distribution
   * tree after refinement).
   * @param[in] cluster_send_list Contains pairs of process ids and clusters.
   *                              For a pair (p, idx) the subtree with root idx
   *                              is send to process p.
   * @param[in] cluster_receive_list  Contains pairs of process ids and
   *                                  clusters. For a pair (p, idx) the subtree
   *                                  with root idx is received from process p.
   * @note The input sets can be determined using
   * @ref tree_structure::determine_refinement_communication_lists.
   */
  void expand_distribution_tree_communicatively(
    const std::set< std::pair< lo, scheduling_time_cluster * > > &
      cluster_send_list,
    const std::set< std::pair< lo, scheduling_time_cluster * > > &
      cluster_receive_list );

  /**
   * Expands the temporal tree structure by recursively traversing the current
   * distributed spacetime cluster tree and the distribution tree given in
   * @p _spacetime_mesh. It uses @p refine_map and the spacetime cluster tree
   * to determine if clusters should be added to the temporal tree structure.
   * @param[in] distribution_tree Distribution tree which is modified and in
   *                              which @p root has to lie.
   * @param[in] spacetime_root Current cluster in the spacetime cluster tree.
   * @param[in] root  Current cluster in the tree structure.
   * @param[in,out] refine_map  Map which indicates if the tree should be
   *                            expanded at a leaf cluster or not. This is
   *                            updated if new clusters are added.
   * @note  Call this routine for the spacetime roots at level 0, not the
   *        spacetime root at level -1.
   */
  void expand_tree_structure_recursively( tree_structure * distribution_tree,
    general_spacetime_cluster * spacetime_root,
    scheduling_time_cluster * time_root,
    std::unordered_map< lo, bool > & refine_map );

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

  //  /**
  //   * Collectively computes number of elements in subdivisioning of 1Dx3D
  //   * bounding box
  //   * @param[in] space_order How many times is space divided.
  //   * @param[in] time_order How many times is time divided.
  //   * @param[inout] n_el_per_part Vector for storing number of spacetime
  //   * elements.
  //   */
  //  void get_n_elements_in_subdivisioning( general_spacetime_cluster * root,
  //    bool split_space, std::vector< lo > & n_elems_per_subd );

  /**
   * Collectively computes number of elements in subdivisioning (given by
   * numbers of space and time divisioning) of the bounding box.
   * @param[in] root Root of the tree.
   * @param[in] n_space_div Number of the spatial octasections.
   * @param[in] n_time_div Number of temporal bisections.
   * @param[inout] elems_in_clusters Vector consisting of numbers of elements in
   * individual subclusters of the bounding box (ordered as pos_t *
   n_space_clusters * n_space_clusters * n_space_clusters
      + pos_x * n_space_clusters * n_space_clusters + pos_y * n_space_clusters
      + pos_z).
   * @todo Since exact bounds of scheduling time clusters are now available they
   * could be used to make this routine simpler.
   */
  void get_n_elements_in_subdivisioning( general_spacetime_cluster & root,
    lo n_space_div, lo n_time_div, std::vector< lo > & elems_in_clusters );

  /**
   * Recursively splits an interval into subintervals.
   * @param[in] center Center of the interval to be split.
   * @param[in] half_size Radius of the interval to be split.
   * @param[in] left_bound Left boundary of the interval to be split.
   * @param[in] n_ref Number of recursive refinement.
   * @param[in] curr_level Current level of refinement.
   */
  void decompose_line( sc center, sc half_size, sc left_bound, lo n_ref,
    lo curr_level, std::vector< sc > & steps );

  /**
   * Creates the spacetime root clusters at level 0 which are the clusters
   * formed by the initial time cluster and spatial clusters resulting after
   * @p _initial_space_refinement refinements of the initial space cluster.
   * @param[in] elems_in_clusters Vector consisting of numbers of elements in
   *                              individual subclusters of the bounding box
   *                              (ordered as pos_t *  n_space_clusters^3
   *                               + pos_x * n_space_clusters^2
   *                               + pos_y * n_space_clusters + pos_z).
   * @param[out] spacetime_root_pairs The resulting spacetime root clusters are
   *                                  added to this structure as pairs with the
   *                                  initial scheduling time cluster. This
   *                                  structure is used for the further tree
   *                                  construction.
   */
  void create_spacetime_roots( std::vector< lo > & elems_in_clusters,
    std::vector< std::pair< general_spacetime_cluster *,
      scheduling_time_cluster * > > & spacetime_root_pairs );

  /**
   * Splits all suitably large clusters at an implicitly given level into
   * subclusters
   * @param[in] split_space Whether to split space as well.
   * @param[in] n_space_div Number of previous space subdivisions.
   * @param[in] n_time_div Number of previous time subdivisions.
   * @param[in] elems_in_clusters Vector of numbers of elements in the
   *                              subdivisioned bounding box (generated by the
   *                              method @ref get_n_elements_in_subdivisioning).
   * @param[in, out] cluster_pairs  Vector of pairs of spacetime clusters and
   *                                scheduling time clusters on the current
   *                                level. A spacetime cluster in the vector is
   *                                refined, if it contains enough elements and
   *                                if the corresponding time cluster is an
   *                                essential non-leaf cluster. The vector is
   *                                overwritten by the vector of pairs of
   *                                children.
   * @todo What to do with early space-time leaf clusters?
   */
  void split_clusters_levelwise( bool split_space, lo n_space_div,
    lo n_time_div, std::vector< lo > & elems_in_clusters,
    std::vector< std::pair< general_spacetime_cluster *,
      scheduling_time_cluster * > > & cluster_pairs );

  /**
   * Collect the leaves of the cluster tree which are owned by the current MPI
   * process, i.e. local. The routine is based on a tree traversal.
   * @param[in] root Current cluster in the tree traversal.
   */
  void collect_local_leaves( general_spacetime_cluster & root );

  /**
   * Collects space-time leaf clusters in the local part of the distributed tree
   * which are leaves in the global tree (i.e. real leaves). The leaves include
   * non-local leaves, which are in the nearfield of local leaf clusters.
   *
   * The routine is based on a traversal of the local part of the spacetime
   * cluster tree and the temporal tree structure.
   * @param[in] st_root Current cluster in the local spacetime tree.
   * @param[in] t_root  Current cluster in the temporal tree structure.
   * @param[in,out] leaves  Vector containing all the leaves.
   * @note  Call this routine for the spacetime roots at level 0, not the
   *        spacetime root at level -1.
   */
  void collect_real_leaves( general_spacetime_cluster & st_root,
    scheduling_time_cluster & t_root,
    std::vector< general_spacetime_cluster * > & leaves );

  /**
   * Adds elements to the cluster.
   * @param[in] cluster Cluster to be filled with elements.
   */
  void fill_elements( general_spacetime_cluster & cluster );

  // void fill_elements2( std::vector< general_spacetime_cluster * > & leaves,
  //  spacetime_tensor_mesh const * current_mesh );

  // void insert_local_element( lo elem_idx, general_spacetime_cluster & root,
  //  vector_type & space_center, vector_type & half_size,
  //  linear_algebra::coordinates< 4 > & centroid, std::vector< sc > & boundary
  //  );

  /**
   * Builds subtree starting from a given root cluster.
   * @param[in] root Root to the subtree.
   * @param[in] split_space Indicates whether to split space when
   *                        constructing the children of root.
   * @todo check correctness of determination of temporal cluster data when more
   * general meshes are used.
   */
  void build_subtree( general_spacetime_cluster & root, bool split_space );

  /**
   * Finds the associated spacetime clusters for each scheduling time cluster in
   * the distribution tree.
   * @note The routines
   * @ref associate_scheduling_clusters_and_space_time_leaves and
   * @ref associate_scheduling_clusters_and_space_time_non_leaves are executed
   * to find the leaves.
   * @warning: Space-time leaves and space-time non-leaves are distinguished.
   * This distinction is reasonable only for local clusters in the
   * distribution tree, i.e. clusters with process id @p _my_rank. Only for such
   * clusters this distinction is required in the FMM.
   */
  void associate_scheduling_clusters_and_space_time_clusters( );

  /**
   * Recursively finds the associated spacetime leaf clusters for each
   * scheduling time cluster in the distribution tree. For this purpose the
   * distribution tree and the distributed spacetime cluster tree are traversed
   * simultaneously in a recursvie manner.
   * @param[in] t_root  Current cluster in the distribution tree.
   * @param[in] st_root  Current cluster in the distributed spacetime cluster
   *                     tree.
   * @warning If the global leaf status of a cluster in the distribution tree
   * is not correct anymore (after a local extension of the distribution tree
   * this is possible), "false" leaves are detected by this method. However,
   * for such scheduling time clusters a distinction of leaves and non-leaves
   * is not necessary.
   * @note  Call this routine for the spacetime roots at level 0, not the
   *        spacetime root at level -1.
   */
  void associate_scheduling_clusters_and_space_time_leaves(
    scheduling_time_cluster * t_root, general_spacetime_cluster * st_root );

  /**
   * Recursively finds the associated spacetime non-leaf clusters for each
   * scheduling time cluster in the distribution tree. For this purpose the
   * distribution tree and the distributed spacetime cluster tree are traversed
   * simultaneously in a recursvie manner.
   * @param[in] t_root  Current cluster in the distribution tree.
   * @param[in] st_root  Current cluster in the distributed spacetime cluster
   *                     tree.
   * @warning Some non-leaves might be missing, see
   * @ref associate_scheduling_clusters_and_space_time_leaves for more details.
   * @note  Call this routine for the spacetime roots at level 0, not the
   *        spacetime root at level -1.
   */
  void associate_scheduling_clusters_and_space_time_non_leaves(
    scheduling_time_cluster * t_root, general_spacetime_cluster * st_root );

  /**
   * Computes and sets the nearfield list and interaction list for every
   * cluster in the distributed spacetime cluster tree by recursively traversing
   * the tree.
   * @param[in] root  Current cluster in the tree traversal.
   * @warning The construction is based only on the local part of the cluster
   * tree. Only for local clusters the nearfield and interaction lists are the
   * same as in the global tree.
   */
  void fill_nearfield_and_interaction_lists( general_spacetime_cluster & root );

  /**
   * Used for the construction of nearfields of leaf clusters by
   * @ref fill_nearfield_and_interaction_lists. It recursively traverses the
   * tree starting from the initial @p current_cluster, and adds all descendant
   * leaves to the nearfield of the leaf @p target_cluster.
   * @param[in] current_cluster Current cluster in the tree traversal.
   * @param[in] target_cluster  Cluster to whose nearfield the leaves are added.
   */
  void add_leaves_to_nearfield_list(
    general_spacetime_cluster & current_cluster,
    general_spacetime_cluster & target_cluster );

  /**
   * Recursively computes padding of clusters in the tree
   * @param[in] root Node to stem from
   */
  sc compute_local_spatial_padding( general_spacetime_cluster & root );

  /**
   * Computes the subtree structure and cluster bounds of all clusters in a
   * given vector and sends them to the process which needs them for its locally
   * essential distribution tree.
   * @param[in] send_cluster_vector Vector containing the clusters whose
   *                                subtrees are considered.
   * @param[in] global_tree_levels  Number of levels in the global distribution
   *                                tree.
   * @param[in] communication_offset  The data is send to the process whose id
   *                                  is @p _my_rank minus this offset.
   * @note  This routine is solely called by
   *        @ref expand_distribution_tree_communicatively.
   */
  void send_subtree_data_of_distribution_tree(
    const std::vector< scheduling_time_cluster * > & send_cluster_vector,
    const lo global_tree_levels, const lo communication_offset ) const;

  /**
   * Receives the subtree structure and cluster bounds of all clusters in a
   * given vector and appends the distribution tree according to this data.
   * @param[in] send_cluster_vector Vector containing the clusters whose
   *                                subtrees are received
   * @param[in] global_tree_levels  Number of levels in the global distribution
   *                                tree.
   * @param[in] communication_offset  The data is received from the process
   *                                  whose id is @p _my_rank plus this offset.
   * @note  This routine is solely called by
   *        @ref expand_distribution_tree_communicatively.
   */
  void receive_subtree_data_of_distribution_tree(
    const std::vector< scheduling_time_cluster * > & receive_clusters_vector,
    const lo global_tree_levels, const lo communication_offset );

  /**
   * Aux for printing
   */
  void print_internal( general_spacetime_cluster & root ) {
    root.print( );
    std::vector< general_spacetime_cluster * > * children
      = root.get_children( );
    // std::cout << children->size( ) << std::endl;
    if ( children != nullptr )
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        for ( lo i = 0; i < ( *it )->get_level( ); ++i ) std::cout << " ";
        print_internal( **it );
      }
  }

  lo _max_levels;       //!< number of levels in the tree
  lo _real_max_levels;  //!< auxiliary value to determine number of real tree
                        //!< levels (depending on _n_min_elems)
  distributed_spacetime_tensor_mesh &
    _spacetime_mesh;                  //!< underlying distributed spacetime mesh
                                      //!< @todo discuss: why reference and not
                                      //!< pointer? why was it const originally?
                                      //!< problem: distribution tree is
                                      //!< modified!
  general_spacetime_cluster * _root;  //!< root of the cluster tree

  std::vector< general_spacetime_cluster * >
    _local_leaves;  //!< Vector containing the leaves of the distributed
                    //!< space-time cluster tree which are local, i.e.
                    //!< assigned to the process with rank @p _my_rank.

  lo _initial_space_refinement;  //!< auxiliary variable determining the
                                 //!< number of spatial refinements executed to
                                 //!< get clusters at level 0
  lo _start_space_refinement;    //!< auxiliary variable to determine in which
                                 //!< level the spatial refinement starts
  //!< (relevant if _initial_space_refinement == 0)
  lo _local_max_space_level;  //!< bound for the maximal number of spatial
                              //!< refinements in the local part of the
                              //!< distributed tree.

  sc _s_t_coeff;    //!< coefficient to determine the coupling of the spatial
                    //!< and temporal levels
  lo _n_min_elems;  //!< minimum number of elements so that cluster can be split
  std::vector< sc >
    _spatial_paddings;            //!< vector of spatial paddings on each level
                                  //!< @note: for each space-time level a value
                                  //!< is stored for simpler access (not only
                                  //!< each spatial level, which is harder to
                                  //!< access)
  slou _spatial_nearfield_limit;  //!< number of the clusters in the vicinity to
                                  //!< be considered as nearfield
  const std::vector< std::vector< lo > > _idx_2_coord = { { 1, 1, 1 },
    { 0, 1, 1 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 0 }, { 0, 1, 0 },
    { 0, 0, 0 },
    { 1, 0, 0 } };  //!< auxiliary mapping from octant indexing to coordinates
  std::vector< sc > _bounding_box_size;  //!< size of the mesh bounding box;
  const MPI_Comm * _comm;  //!< MPI communicator associated with the tree
  int _my_rank;            //!< MPI rank of current processor
  int _n_processes;  //!< total number of MPI processes in the communicator
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_CLUSTER_TREE_H_ */
