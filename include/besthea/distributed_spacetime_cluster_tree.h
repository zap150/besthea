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

/** @file distributed_spacetime_cluster_tree.h
 * @brief Spacetime cluster tree distributed among MPI processes
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_CLUSTER_TREE_H_

#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/scheduling_time_cluster.h"
#include "besthea/timer.h"
#include "besthea/vector.h"

#include <map>
#include <mpi.h>
#include <omp.h>
#include <unordered_map>
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
   * @param[in] alpha Heat capacity constant. It influences the shape of the
   * space-time clusters.
   * @param[in] spatial_nearfield_limit Number of clusters in the vicinity of a
   * given clusters to be considered as nearfield
   * @param[in] comm MPI communicator associated with the tree.
   * @param[in,out] status  Indicates if the tree construction was successfull
   *                        (status 0) or not (status 1)
   */
  distributed_spacetime_cluster_tree(
    distributed_spacetime_tensor_mesh & spacetime_mesh, lo max_levels,
    lo n_min_elems, sc st_coeff, sc alpha, slou spatial_nearfield_limit,
    MPI_Comm * comm, lo & status );

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
   * Computes and returns the spatial padding for each spatial level from
   * @ref _spatial_paddings
   */
  std::vector< sc > get_spatial_paddings_per_spatial_level( ) const;

  /**
   * Returns the effective number of levels in the cluster tree.
   */
  lo get_max_levels( ) const {
    return _max_levels;
  }

  /**
   * Returns the bound for the maximal number of refinements in space of the
   * clusters in the local part of the distributed tree.
   */
  lo get_local_max_space_level( ) const {
    return _local_max_space_level;
  }

  /**
   * Returns @ref _global_max_space_level.
   */
  lo get_global_max_space_level( ) const {
    return _global_max_space_level;
  }

  /**
   * Returns the level where space is first refined.
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
   * Returns the value of @ref _spatial_nearfield_limit.
   */
  lo get_spatial_nearfield_limit( ) const {
    return _spatial_nearfield_limit;
  }

  /**
   * Returns the associated MPI communicator.
   */
  const MPI_Comm * get_MPI_comm( ) const {
    return _comm;
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

  /**
   * Prints various information about the cluster tree
   * @param[in] root_process  Process responsible for printing the information.
   */
  void print_information( const int root_process );

  /**
   * Prints a grid of the spatial clusters in the tree for each space-time
   * level.
   * @warning This makes sense only for space-time meshes with a uniform
   * structure in time.
   */
  void print_spatial_grids( const lo root_proc_id ) const;

  /**
   * Collects the leaves of the cluster tree which are owned by the current MPI
   * process, i.e. local. The routine is based on a tree traversal. It can also
   * be used to find local leaves of a subtree.
   * @param[in] current_cluster Current cluster in the tree traversal.
   * @param[in,out] leaf_vector Vector to which the detected local leaves are
   * added.
   */
  void collect_local_leaves( general_spacetime_cluster & current_cluster,
    std::vector< general_spacetime_cluster * > & leaf_vector ) const;

 private:
  /**
   * Builds the spacetime cluster tree in a communicative way in the upper part
   * (where clusters can contain elements located in meshes of two or more
   * processes) and in a non-communicative way in the lower part.
   * @param[in] status  Indicates if the tree has been built correctly
   *                    (status 0). Status 1 means that for some cluster the
   *                    number of assigned elements did not match with the
   *                    predetermined value.
   */
  void build_tree( lo & status );

  /**
   * Builds the spacetime cluster tree in a communicative way in the upper part
   * (where clusters can contain elements located in meshes of two or more
   * processes) and in a non-communicative way in the lower part.
   * @param[in] status  Indicates if the tree has been built correctly
   *                    (status 0). Status 1 means that for some cluster the
   *                    number of assigned elements did not match with the
   *                    predetermined value.
   */
  void build_tree_new( lo & status );

  /**
   * Expands the distribution tree included in @p _spacetime_mesh by adding
   * relevant time clusters which appear as components of spacetime clusters in
   * the current tree but are not in the distribution tree.
   * @note The clusters which are refined are determined using the routine
   *       @ref tree_structure::determine_clusters_to_refine and the refinement
   *       is executed by @ref expand_tree_structure_recursively.
   * @note The nearfields, interaction lists and send lists of the distribution
   *       tree are cleared using the routine
   *       @ref tree_structure::clear_nearfield_send_and_interaction_lists and
   *       filled anew.
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
   * @ref tree_structure::determine_cluster_communication_lists.
   */
  void expand_distribution_tree_communicatively(
    const std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      cluster_send_list,
    const std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      cluster_receive_list );

  /**
   * Exchanges necessary leaf information of non-local clusters with other
   * processes.
   * @param[in] leaf_info_send_list Contains pairs of process ids and clusters.
   *                                For a pair (p, idx) the leaf information of
   *                                clusters associated with cluster idx is sent
   *                                to process p.
   * @param[in] leaf_info_receive_list  Contains pairs of process ids and
   *                                    clusters. For a pair (p, idx) the leaf
   *                                    information of clusters associated with
   *                                    cluster idx is received from process p.
   * @note The input sets can be determined using
   * @ref tree_structure::determine_cluster_communication_lists.
   * @todo Discuss: We could get rid of this communication step: If we get rid
   * of the ordering of associated spacetime clusters (currently the first few
   * are leaves, the remaining non-leaves) we do not need to know the leaf
   * information of non-local spacetime clusters. However, the execution of S2M
   * and L2T operations has to be changed in this case. (basic idea: store the
   * number of associated leaves. search for these leaves in the vector of
   * associated spacetime clusters)
   */
  void communicate_necessary_leaf_information(
    const std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      leaf_info_send_list,
    const std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      leaf_info_receive_list );

  /**
   * Expands the temporal tree structure by recursively traversing the current
   * distributed spacetime cluster tree and the distribution tree given in
   * @p _spacetime_mesh. It uses @p refine_map and the spacetime cluster tree
   * to determine if clusters should be added to the temporal tree structure.
   * @param[in] distribution_tree Distribution tree which is modified and in
   *                              which @p time_root has to lie.
   * @param[in] spacetime_root Current cluster in the spacetime cluster tree.
   * @param[in] time_root  Current cluster in the tree structure.
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
   */
  void compute_cubic_bounding_box(
    sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax );

  /**
   * Collectively computes number of elements in subdivisioning (given by
   * numbers of space divisioning and vector of time clusters) of the bounding
   * box.
   * @param[in] n_space_div Number of the spatial octasections.
   * @param[in] level_time The level in time determining the subdivision. On
   *                       each new level the clusters are split into halves
   *                       with respect to time.
   * @param[in] time_clusters_on_level  Contains the time clusters into which
   *                                    the temporal domain is subdivided (
   *                                    the clusters do not have to span the
   *                                    complete time interval).
   * @param[in] fine_box_bounds Auxiliary structure which contains the bounds of
   *                            the spatial boxes at the finest required spatial
   *                            level for all three spatial dimensions. It is
   *                            used to determine the bounds of the boxes for
   *                            each subdivision consistently.
   * @param[inout] elems_in_clusters Vector consisting of numbers of elements in
   * individual subclusters of the bounding box (ordered as pos_t *
   n_space_clusters * n_space_clusters * n_space_clusters
      + pos_x * n_space_clusters * n_space_clusters + pos_y * n_space_clusters
      + pos_z).
   */
  void get_n_elements_in_subdivisioning( lo n_space_div, lo level_time,
    const std::vector< scheduling_time_cluster * > & time_clusters_on_level,
    const std::vector< std::vector< sc > > & fine_box_bounds,
    std::vector< lo > & elems_in_clusters );

  /**
   * Collectively computes the number of elements in a subdivisioning of the
   * space-time bounding box corresponding to the leaves in the distribution
   * tree.
   * @param[in] n_space_div Number of spatial refinements used for the
   *                        subdivisioning.
   * @param[out] elems_in_clusters  Vector containing the number of elements in
   * the individual subclusters of the subdivisioning (ordered as
   * pos_t * n_space_clusters * n_space_clusters * n_space_clusters
   * + pos_x * n_space_clusters * n_space_clusters + pos_y * n_space_clusters
   * + pos_z).
   * @param[out] boxes_of_local_elements For each element in the local mesh this
   *                                     vector contains the index of the
   *                                     cluster in which it lies.
   * @note If the distribution tree contains early leaf clusters they are
   *       refined artificially for the counting process. Thus, the clusters in
   *       the subdivisioning correspond to a regular refinement.
   */
  void get_n_elements_in_fine_subdivisioning( lo n_space_div,
    std::vector< lo > & elems_in_clusters,
    std::vector< lo > & boxes_of_local_elements );

  /**
   * Initializes some data which is needed for counting elements in (and
   * assigning elements to) clusters in some subdivisioning of the space-time
   * bounding box.
   * @param[in] temporal_leaf_clusters  Temporal leaf clusters corresponding to
   *                                    the mesh of the elements elements
   *                                    (nearfield mesh or local mesh)
   * @param[in] n_space_clusters_per_dim  Number of space clusters along each
   *                                      spatial dimension in the
   *                                      subdivisioning.
   * @param[out] leaves_start_index Temporal index of the first cluster in
   *                                @p temporal_leaf_clusters. If this is an
   *                                early cluster, the temporal index of its
   *                                leftmost (artificially refined) successor is
   *                                returned.
   * @param[out] starts_time_intervals  Start time of the provided time
   *                                    clusters.
   * @param[out] endings_time_intervals End time of the provided time clusters.
   * @param[out] spatial_step_sizes Spatial step sizes for each spatial
                                    dimension.
   * @param[out] space_bounds_x Bounds of the spatial clusters of the
   *                            subdivisioning along x axis.
   * @param[out] space_bounds_y Bounds of the spatial clusters of the
   *                            subdivisioning along y axis.
   * @param[out] space_bounds_z Bounds of the spatial clusters of the
   *                            subdivisioning along z axis.

   */
  void prepare_data_for_element_counting(
    const std::vector< scheduling_time_cluster * > & temporal_leaf_clusters,
    const lo n_space_clusters_per_dim, lo & leaves_start_index,
    std::vector< sc > & starts_time_intervals,
    std::vector< sc > & endings_time_intervals,
    std::vector< sc > & spatial_step_sizes, std::vector< sc > & space_bounds_x,
    std::vector< sc > & space_bounds_y, std::vector< sc > & space_bounds_z );

  /**
   * Assigns the elements contained in the nearfield mesh of
   * @ref _spacetime_mesh to the appropriate clusters in the tree.
   * @param[in] n_space_clusters_per_dim Number of space clusters along each
   *                                      spatial dimension in the fine
   *                                      subdivisioning from *
   *                                  @ref get_n_elements_in_fine_subdivisioning
   * @param[in] boxes_of_nearfield_elements For each element in the nearfield
   *                                        mesh this vector contains the index
   *                                        of the cluster in which it lies.
   */
  void assign_nearfield_elements_to_boxes( const lo n_space_clusters_per_dim,
    std::vector< lo > & boxes_of_nearfield_elements );

  /**
   * Computes the number of elements in all clusters of a certain level of the
   * tree by summing up the numbers of their children.
   * @param[in] time_level  Level of the clusters at the parent level.
   * @param[in] space_level Number of spatial refinements of the clusters at the
   *                        parent level.
   * @param[in] elems_per_subdivisioning_child_level  Number of elements in the
   *                                                  clusters at child level.
   * @param[in] space_refined Indicates if the child clusters are refined with
   *                          respect to space and time (true) or only with
   *                          respect to time.
   * @param[out] elems_per_subdivisioning_this_level  Number of elements in the
   *                                                  clusters at parent level.
   *
   * @note This routine is an auxiliary routine for the construction of the
   * upper part of the space-time cluster tree.
   */
  void sum_up_elements_in_boxes( const lo time_level, const lo space_level,
    const std::vector< lo > & elems_per_subdivisioning_child_level,
    const bool space_refined,
    std::vector< lo > & elems_per_subdivisioning_this_level ) const;

  /**
   * Recursively splits an interval into subintervals.
   * @param[in] center Center of the interval to be split.
   * @param[in] half_size Radius of the interval to be split.
   * @param[in] left_bound Left boundary of the interval to be split.
   * @param[in] n_ref Number of recursive refinement.
   * @param[in] curr_level Current level of refinement.
   * @param[out] steps  Vector in which the bounds of the subintervals are
   *                    stored.
   * @warning The right bound of the largest subinterval is not added to
   *          @p steps.
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
   * @param[out] status If clusters containing few elements (<_n_min_elems) are
   *                    refined by the routine  @p status is set to 1, otherwise
   *                    it is set to 0.
   * @todo What to do with early space-time leaf clusters?
   * @todo Check if documentation is ok.
   */
  void split_clusters_levelwise( bool split_space, lo n_space_div,
    lo n_time_div, std::vector< lo > & elems_in_clusters,
    std::vector< std::pair< general_spacetime_cluster *,
      scheduling_time_cluster * > > & cluster_pairs,
    lo & status );

  /**
   * Collects the leaves of the cluster tree which correspond to elements in the
   * nearfield mesh of the current process. The routine is based on a tree
   * traversal.
   * @param[in] root Current cluster in the tree traversal.
   * @param[out]  non_local_leaves  Non local leaves are added to this vector in
   *                                the order in which they are detected.
   */
  void collect_non_local_leaves( general_spacetime_cluster & root,
    std::vector< general_spacetime_cluster * > & non_local_leaves ) const;

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
   * @warning Clusters whose meshes are not available are not collected.
   */
  void collect_real_leaves( general_spacetime_cluster & st_root,
    scheduling_time_cluster & t_root,
    std::vector< general_spacetime_cluster * > & leaves );

  /**
   * Adds elements to the cluster.
   * @param[in] cluster Cluster to be filled with elements.
   * @param[in] fine_box_bounds Auxiliary structure which contains the bounds of
   *                            the spatial boxes at the finest required spatial
   *                            level for all three spatial dimensions. It is
   *                            used to determine the bounds of the cluster for
   *                            which elements are filled in consistently.
   * @param[out]  status  Indicates if the cluster has been filled correctly
   *                      (status 0). If not the exact amount of elements as
   *                      expected is assigned to the cluster, status is 1.
   */
  void fill_elements( general_spacetime_cluster & cluster,
    const std::vector< std::vector< sc > > & fine_box_bounds, lo & status );

  /**
   * Assigns elements to the leaf clusters in the upper part of the tree in the
   * first part of the tree assembly.
   * @param[in] leaves  Vector containing all the space-time leaf clusters in
   *                    the upper part of the tree.
   * @param[in] global_dist_tree_depth  Depth of the global distribution tree.
   * @param[in] space_levels  For each space-time level this vector contains the
   *                          associated space level.
   * @param[in] boxes_of_local_elements For each element in the nearfield mesh
   *                                    this vector contains the index of the
   *                                    cluster in which the element is located.
   * @param[in,out] status If a leaf at levels <= 1 is found the status is set
   *                       to 3, otherwise it is not changed.
   */
  void fill_elements_new(
    const std::vector< general_spacetime_cluster * > & leaves,
    const lo global_dist_tree_depth, const std::vector< lo > & space_levels,
    const std::vector< lo > & boxes_of_local_elements, lo & status );

  // void fill_elements2( std::vector< general_spacetime_cluster * > & leaves,
  //  spacetime_tensor_mesh const * current_mesh );

  // void insert_local_element( lo elem_idx, general_spacetime_cluster & root,
  //  vector_type & space_center, vector_type & half_size,
  //  linear_algebra::coordinates< 4 > & centroid, std::vector< sc > &
  //  boundary
  //  );

  /**
   * Builds subtree starting from a given root cluster.
   * @param[in] root Root to the subtree.
   * @param[in] split_space Indicates whether to split space when
   *                        constructing the children of root.
   * @todo check correctness of determination of temporal cluster data when
   * more general meshes are used.
   * @note The member variables @p _real_max_levels and
   * @p _local_max_space_level are updated in this routine.
   *
   */
  void build_subtree(
    general_spacetime_cluster & root, const bool split_space );

  /**
   * Finds the associated spacetime clusters for each scheduling time cluster in
   * the distribution tree.
   * @note The routine
   * @ref associate_scheduling_clusters_and_space_time_clusters_recursively
   * is executed to find the associated clusters.
   * @warning: Space-time leaves and space-time non-leaves are not distinguished
   * here. The routine @ref sort_associated_space_time_clusters_recursively has
   * to be called in addition to distinguish them and resort the associated
   * clusters such that the first clusters are leaves and the rest non-leaves.
   */
  void associate_scheduling_clusters_and_space_time_clusters( );

  /**
   * Recursively finds the associated spacetime clusters for each
   * scheduling time cluster in the distribution tree. For this purpose the
   * distribution tree and the distributed spacetime cluster tree are traversed
   * simultaneously in a recursvie manner.
   * @param[in] t_root  Current cluster in the distribution tree.
   * @param[in] st_root  Current cluster in the distributed spacetime cluster
   *                     tree.
   * @note  Call this routine for the spacetime roots at level 0, not the
   *        spacetime root at level -1.
   */
  void associate_scheduling_clusters_and_space_time_clusters_recursively(
    scheduling_time_cluster * t_root, general_spacetime_cluster * st_root );

  /**
   * Traverses the distribution tree recursively. For all time clusters it
   * determines the number n of associated spacetime leaf clusters and sorts the
   * associated spacetime clusters such that the first n clusters are the
   * leaves.
   * @param[in] t_root  Current cluster in the tree traversal.
   * @param[in] leaf_buffer Vector in which leaves are temporarily stored.
   * @param[in] non_leaf_buffer Vector in which non-leaves are temporarily
   *                            stored.
   * @note  At input the buffer sizes can be 0. The buffers are resized
   *        in the routine as necessary. After the execution of the routine,
   *        the buffers should be cleared manually, if storage is limited.
   */
  void sort_associated_space_time_clusters_recursively(
    scheduling_time_cluster * t_root,
    std::vector< general_spacetime_cluster * > & leaf_buffer,
    std::vector< general_spacetime_cluster * > & non_leaf_buffer );

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
   * @param[in] receive_clusters_vector Vector containing the clusters whose
   *                                    subtrees are received.
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
   * Sends leaf information for all clusters associated with the time clusters
   * in the given vector to the process whose index is determined by
   * @p communication_offset.
   * @param[in] send_cluster_vector List of all time clusters, for which the
   *                                leaf information of associated spacetime
   *                                clusters is sent to another process.
   * @param[in] communication_offset  The data is sent to the process whose id
   *                                  is @p _my_rank plus this offset.
   * @note  This routine is solely called by
   *        @ref communicate_necessary_leaf_information.
   */
  void send_leaf_info(
    const std::vector< scheduling_time_cluster * > & send_cluster_vector,
    const lo communication_offset ) const;

  /**
   * Receives leaf information of all clusters associated with the time clusters
   * in the given vector from the process whose index is determined by
   * @p communication_offset and updates it.
   * @param[in] receive_cluster_vector  List of all time clusters, for which the
   *                                    leaf information of associated spacetime
   *                                    clusters is received from another
   *                                    process.
   * @param[in] communication_offset  The data is receive from the process whose
   *                                  id is @p _my_rank minus this offset.
   * @note  This routine is solely called by
   *        @ref communicate_necessary_leaf_information.
   */
  void receive_leaf_info(
    const std::vector< scheduling_time_cluster * > & receive_cluster_vector,
    const lo communication_offset ) const;

  /**
   * Auxiliary routine for @ref print_spatial_grids. Based on a recursive tree
   * traversal.
   * @param[in] current_cluster Current cluster in the tree traversal.
   * @param[in] print_level Space-time level for which the grid image vector is
   * determined.
   * @param[in] n_space_clusters_per_dim Number of spatial clusters per
   * dimension at level @p print_level.
   * @param[out] image_vector For each cluster in the spatial grid it contains
   * an entry (0 if there is no cluster associated with the spatial cluster at
   * this position, 1 if all clusters are non-leaves, 2 if there is at least one
   * leaf). The boxes are numerated by
   * i_x + i_y * n_space_cluster_per_dim
   * + i_z * n_space_cluster_per_dim * n_space_cluster_per_dim
   */
  void determine_grid_image_vector( general_spacetime_cluster & current_cluster,
    const lo print_level, const lo n_space_clusters_per_dim,
    std::vector< lo > & image_vector ) const;

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
                                 //!< (relevant if
                                 //!< _initial_space_refinement == 0)
  lo _local_max_space_level;     //!< bound for the maximal number of spatial
                                 //!< refinements in the local part of the
                                 //!< distributed tree.
  lo _global_max_space_level;    //!< maximal spatial level in the global
                                 //!< distributed tree

  // sc _s_t_coeff;  //!< coefficient to determine the coupling of the spatial
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
