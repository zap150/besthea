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

/** @file tree_structure.h
 * @brief General tree structure.
 */

#ifndef INCLUDE_BESTHEA_TREE_STRUCTURE_H_
#define INCLUDE_BESTHEA_TREE_STRUCTURE_H_

#include "besthea/io_routines.h"
#include "besthea/scheduling_time_cluster.h"
#include "besthea/settings.h"

#include <iostream>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace besthea {
  namespace mesh {
    class tree_structure;
  }
}

namespace besthea {
  namespace mesh {
    class distributed_spacetime_cluster_tree;
  }
}

/**
 * Class representing the structure of a tree of temporal clusters.
 * It is meant to be used for the scheduling of jobs in a parallel FMM.
 * @todo Discuss: Should we change the constructor such that it incorporates the
 * time slices and the process assignments only. (Currently we load the
 * structure from file, but still rebuild the other information
 * (slices, cluster bounds). If the time slices are already assigned to clusters
 * we could also use them to build the tree instead of the tree structure
 * vector)
 * @warning: if the tree structure is computed on each process separately it is
 * not guaranteed that it is the same for all processes!
 */
class besthea::mesh::tree_structure {
 public:
  /**
   * Constructs a tree structure by reading it from a file.
   * @param[in] filename Name of the input file containing the tree structure.
   * @param[in] start_time  Start time of the time mesh.
   * @param[in] end_time  End time of the time mesh.
   * @param[in] process_id  Id of the process which calls the method. Default
   *                        value is -1.
   * @param[in] enable_m2t_and_s2l  If true, structures for the
   * realization of m2t and s2l operations are initialized.
   * @note The start and end point of the mesh are used to generate the
   * geometrical data of the clusters.
   */
  tree_structure( const std::string & filename, const sc start_time,
    const sc end_time, lo process_id = -1, bool enable_m2t_and_s2l = false );

  /**
   * Constructs a tree structure by reading it from files
   * @param[in] structure_file Name of the binary input file containing the tree
   *                          structure.
   * @param[in] cluster_bounds_file   Name of the binary input file containing
   *                                  the bounds of all the clusters in the
   *                                  tree structure.
   * @param[in] process_id  Id of the process which calls the method. Default
   *                        value is -1.
   * @param[in] enable_m2t_and_s2l  If true, structures for the
   * realization of m2t and s2l operations are initialized.
   */
  tree_structure( const std::string & structure_file,
    const std::string & cluster_bounds_file, lo process_id = -1,
    bool enable_m2t_and_s2l = false );

  /**
   * Destructor.
   */
  virtual ~tree_structure( ) {
    if ( _root != nullptr ) {
      delete _root;
    }
  }

  /**
   * Returns number of levels in the tree.
   */
  lo get_levels( ) const {
    return _levels;
  }

  bool supports_m2t_and_s2l_operations( ) const {
    return _supports_m2t_and_s2l;
  }

  /**
   * Returns the root of the tree.
   */
  scheduling_time_cluster * get_root( ) {
    return _root;
  }

  /**
   * Returns the root of the tree.
   */
  const scheduling_time_cluster * get_root( ) const {
    return _root;
  }

  /**
   * Returns clusters without descendants.
   */
  const std::vector< scheduling_time_cluster * > & get_leaves( ) const {
    return _leaves;
  }

  /**
   * Loads process assignments from a given file and assigns to each cluster in
   * the tree structure its respective process.
   * @param[in] filename File containing a vector of process assignments.
   */
  void load_process_assignments( const std::string & filename );

  /**
   * Assigns slices to the leaf clusters in the tree structure.
   * @param[in] slice_nodes Nodes of the slices in ascending order. The index
   *                        of the slice consisiting of the nodes i and i+1 is
   *                        assumed to have the index i.
   * @warning This routine should only be called for the time slices for which
   * the tree structure was originally constructed.
   */
  void assign_slices_to_clusters( const std::vector< sc > & slice_nodes );

  /**
   * Reduces the tree structure by deleting all clusters which are not contained
   * in the locally essential tree of the current process (which has the id
   * @p _my_process_id ).
   * @note @p _leaves and @p _levels are reset.
   * @note The original global_indices of the clusters are not modified. This
   * allows to identify clusters between different processes.
   * @note A slightly larger part of the tree is kept then the usual locally
   * essential tree: If a non-local leaf cluster has a local cluster in its
   * nearfield it is also kept.
   */
  void reduce_2_essential( );

  /**
   * Determines the value of
   * @ref scheduling_time_cluster::_status_initial_op_downward_path for all
   * scheduling time clusters in the subtree of current_cluster by a recursive
   * tree traversal.
   * @param[in] current_cluster Current cluster in the tree traversal.
   * @note This routine has to be called after the complete construction of a
   * distributed space-time cluster tree associated with the tree structure for
   * which it is called. In particular, the assignment of space-time clusters to
   * scheduling time clusters has to be completed.
   */
  char determine_downward_path_initial_op_status_recursively(
    scheduling_time_cluster & current_cluster );

  /**
   * Updates the m2t and s2l lists of all local non-leaf scheduling time
   * clusters, whose associated space-time clusters have non-empty m2t or s2l
   * lists.
   * @note This routine is basically just a wrapper for
   * @ref update_m2t_and_s2l_lists_recursively
   */
  void update_m2t_and_s2l_lists( );

  /**
   * Initializes the m2t lists of clusters by traversing the temporal tree
   * structure recursively.
   *
   * This routine has to be called also when no single sided expansions are used
   * in the later FMM algorithm. In fact, m2t lists are used to decide on local
   * subtree communication after local extensions of the tree.
   *
   * m2t lists are constructed for local and non-local clusters
   *
   * @param[in] current_cluster Current cluster in the tree traversal.
   */
  void set_m2t_lists_for_subtree_communication(
    scheduling_time_cluster & current_cluster );

  /**
   * Traverses the tree recursively and adds all relevant clusters assigned to
   * the process @p _my_process_id to the 4 lists for scheduling operations in
   * the FMM.
   * @param[in] root Current cluster in the tree traversal.
   * @param[in,out] m_list  List for scheduling upward path operations.
   * @param[in,out] m2l_list  List for scheduling interactions and downward
   * pass operations.
   * @param[in,out] m2t_task_list  List for scheduling m2t operations.
   * @param[in,out] s2l_task_list  List for scheduling s2l operations.
   * @param[in,out] l_list  List for scheduling downward path operations.
   * @param[in,out] n_list  List for scheduling nearfield operations.
   * @note The routine is solely called by
   * @ref linear_algebra::distributed_pFMM_matrix::prepare_fmm.
   * @note Please distinguish the task lists (e.g. m2t_task_list) and the
   * respective lists of the individual clusters (e.g. m2t_list of a given
   * cluster)
   */
  void init_fmm_lists( scheduling_time_cluster & root,
    std::list< scheduling_time_cluster * > & m_list,
    std::list< scheduling_time_cluster * > & m2l_list,
    std::list< scheduling_time_cluster * > & m2t_task_list,
    std::list< scheduling_time_cluster * > & s2l_task_list,
    std::list< scheduling_time_cluster * > & l_list,
    std::list< scheduling_time_cluster * > & n_list ) const;

  /**
   * Traverses the tree structure recursively and allocates the moments for all
   * clusters which are active in the upward path of the FMM.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in] contribution_size Size of the contribution of a single
   * spacetime cluster.
   */
  void allocate_moments_in_tree(
    scheduling_time_cluster & root, lou contribution_size );

  /**
   * Fills the associated (spatial and space-time) moments of all clusters in
   * the tree with zeros. The tree is traversed recursively by the routine.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void clear_moment_contributions( scheduling_time_cluster & root );

  /**
   * Traverses the tree structure recursively and allocates the local
   * contributions for all clusters which are active in the downward path of the
   * FMM.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in] contribution_size Size of the contribution of a single
   * spacetime cluster.
   */
  void allocate_local_contributions_in_tree(
    scheduling_time_cluster & root, lou contribution_size );

  /**
   * Fills the associated (spatial and space-time) local contributions of all
   * clusters in the tree with zeros. The tree is traversed recursively by the
   * routine.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void clear_local_contributions( scheduling_time_cluster & root );

  /**
   * Traverses the tree structure recursively and allocates the spatial moments
   * for all clusters which are active in the upward path of the FMM.
   * @param[in] current_cluster Current time cluster in the tree traversal.
   * @param[in] spatial_contribution_size Size of the spatial contribution of a
   * single spacetime cluster.
   * @param[in] first_space_refinement_level  First space-time level in the
   * underlying space-time tree where clusters are refined in space. This is
   * used to determine the clusters for which spatial moments have to be
   * allocated.
   */
  void allocate_spatial_moments_in_tree(
    scheduling_time_cluster & current_cluster, lou spatial_contribution_size,
    lo first_space_refinement_level );

  /**
   * Traverses the tree structure recursively and allocates the spatial local
   * contributions for all clusters which are active in the downward path of the
   * FMM.
   * @param[in] current_cluster Current time cluster in the tree traversal.
   * @param[in] spatial_contribution_size Size of the spatial contribution of a
   * single spacetime cluster.
   * @param[in] first_space_refinement_level  First space-time level in the
   * underlying space-time tree where clusters are refined in space. This is
   * used to determine the clusters for which spatial local contributions have
   * to be allocated.
   */
  void allocate_spatial_local_contributions_in_tree(
    scheduling_time_cluster & current_cluster, lou spatial_contribution_size,
    lo first_space_refinement_level );

  /**
   * Traverses the tree structure recursively and allocates and initializes the
   * local contributions for all clusters which are active in the downward path
   * of an initial pFMM operator.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in] contribution_size Size of the contribution of a single
   * spacetime cluster.
   */
  void allocate_local_contributions_in_tree_initial_op(
    scheduling_time_cluster & root, lou contribution_size );

  /**
   * Returns the structure of the tree represented as a vector of chars.
   *
   * The chars are sorted according to the traversal of the tree (recursive,
   * left subtree first). For every non-leaf cluster two chars are in the list
   * indicating the status of its children
   * (0: not existent, 1: non-leaf, 2: leaf)
   */
  std::vector< char > compute_tree_structure( ) const;

  /**
   * Comparison operator for two clusters.
   * @param[in] first Pointer to the first cluster.
   * @param[in] second Pointer to the second cluster.
   * @return True if first's level is greater than second's level, or, in case
   *         of equality of the levels, if first's global index is greater than
   *         second's global index
   */
  static bool compare_clusters_bottom_up_right_2_left(
    scheduling_time_cluster * first, scheduling_time_cluster * second );

  /**
   * Comparison operator for two clusters.
   * @param[in] first Pointer to the first cluster.
   * @param[in] second Pointer to the second cluster.
   * @return True if first's level is less than second's level, or, in case
   *         of equality of the levels, if first's global index is greater than
   *         second's global index
   */
  static bool compare_clusters_top_down_right_2_left(
    scheduling_time_cluster * first, scheduling_time_cluster * second );

  /**
   * Counts the number of contributions of the associated spacetime clusters of
   * all clusters in the tree structure.
   *
   * This is an auxiliary routine which helps to estimate the memory
   * requirements of the farfield part of distributed pFMM matrices in
   * @ref distributed_pFMM_matrix.h. It is based on a recursive tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] n_moments Counter for the number of moments in the tree.
   * @param[in,out] n_moments_receive Counter for the number of separately
   * received moments in the tree (such moments are only used to store received
   * moments for a parent cluster in the upward path of the FMM)
   * @param[in,out] n_local_contributions Counter for the number of local
   * contributions in the tree.
   * @param[in,out] n_spat_moments  Counter for the number of spatial moments in
   * the tree.
   * @param[in,out] n_spat_local_contribution Counter for the number of spatial
   * local contributions in the tree.
   */
  void count_number_of_contributions( scheduling_time_cluster * root,
    lo & n_moments, lo & n_moments_receive, lo & n_local_contributions,
    lo & n_spat_moments, lo & n_spat_local_contributions ) const;

  /**
   * Computes the tree structure and prints it to a binary file
   * @param[in] filename Name of the output file
   */
  void print_tree_structure( const std::string & filename ) const;

  /**
   * Print the tree in a recursive manner to the console.
   */
  void print( ) {
    std::cout << "number of levels: " << _levels << std::endl;
    std::cout << "leaves are: " << std::endl;
    for ( auto it : _leaves ) {
      std::cout << it->get_global_index( ) << " ";
    }
    std::cout << std::endl;
    // print cluster information recursively
    print_internal( _root );
  }

  /**
   * Prints the process ids or the global indices assigned to the clusters in a
   * real binary tree format to console.
   * @param[in] digits  Number of digits which are used to print the process
   *                    information. Its value should be larger than the number
   *                    of digits the highest process id or global index has.
   * @param[in] print_process_ids Bool to decide whether process ids of the
   *                              clusters or their global indices are printed.
   * @note For deep trees the ouput lines will get quite long, and automatic
           line breaks destroy the readability.
   */
  void print_tree_human_readable(
    const lo digits, const bool print_process_ids ) const;

 private:
  scheduling_time_cluster * _root;  //!< root cluster of the tree structure
  lo _levels;                       //!< number of levels in the tree
  bool _supports_m2t_and_s2l;       //!< Indicates whether the tree structure is
                               //!< built such that m2t and s2l operations are
                               //!< supported in the later FMM algorithm.
  std::vector< scheduling_time_cluster * >
    _leaves;          //!< vector of all clusters without descendants
  lo _my_process_id;  //!< id of the process executing the operations
                      //!< @todo Rename consistently?

  friend class besthea::mesh::distributed_spacetime_cluster_tree;

  /**
   * Set number of levels in the tree to a new value
   * @param[in] levels New number of levels in the tree.
   */
  void set_levels( lo levels ) {
    _levels = levels;
  }

  /**
   * Recursively constructs the tree structure from a given array.
   * @param[in] tree_array Contains the data needed for tree construction.
   * @param[in,out] root  Current cluster, to which the next clusters are
   * added.
   * @param[in,out] position  Auxiliary variable to keep track of the current
   *                          position in the tree_vector.
   * @note  The child of a cluster inherits the process id from its parent in
   *        this routine.
   * @note This method is supposed to be called by the constructor.
   */
  void array_2_tree(
    const char * tree_array, scheduling_time_cluster & root, lou & position );

  /**
   * Recursively constructs the tree structure from a given structure array
   * and a given array of cluster bounds.
   * @param[in] tree_array Contains the tree structure data.
   * @param[in] cluster_bounds_array  Contains the data of the clusters' bounds.
   * @param[in] n_time_elements_array Contains the information about the number
   * of time steps of each cluster. @p nullptr can be inserted if this is
   * unknown.
   * @param[in,out] root  Current cluster, to which the next clusters are
   * added.
   * @param[in,out] position  Auxiliary variable to keep track of the current
   *                          position in the vectors.
   * @note  This method is supposed to be called by the constructor.
   */
  void create_tree_from_arrays( const char * tree_array,
    const sc * cluster_bounds_array, const lo * n_time_elements_array,
    scheduling_time_cluster & root, lou & position );

  /** Assigns to each cluster in the tree structure its respective process
   * given in the vector @p process_assignments by traversing the tree.
   * @param[in] process_assignments Vector of the process assignments given in
   *                                the tree structure format.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] position  Current position in process_assignments.

   */
  void set_process_assignments( const std::vector< lo > process_assignments,
    scheduling_time_cluster & root, lou & position );

  /**
   * Collects all clusters without descendants and stores them in the internal
   * @p _leaves vector. The routine is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void collect_leaves( scheduling_time_cluster & root );

  /**
   * Sets the global indices of all clusters in the tree structure by
   * traversing it recursively. The parent cluster with id k sets the indices
   * of its children: the left gets the index 2k+1, the right the index 2k+2.
   * @param[in] root Current cluster in the tree traversal.
   * @note The index of the root s assumed to be 0 before the routine is
   * called.
   */
  void set_indices( scheduling_time_cluster & root );

  /**
   * Computes and sets the operation lists (nearfield, interaction, m2t, s2l,
   * send, diagonal send) for every cluster in the tree structure by recursively
   * traversing the tree.
   *
   * @note if @ref _supports_m2t_and_s2l is set to false, m2t, s2l and
   * diagonal send lists are not set.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void set_cluster_operation_lists( scheduling_time_cluster & root );

  /**
   * Used for the construction of nearfields and m2t lists of leaf clusters by
   * @ref set_cluster_operation_lists.
   *
   * It recursively traverses the tree starting from the initial
   * @p current_cluster, and adds all appropriate descendants to the
   * corresponding list of the leaf @p target_cluster.
   *
   * If a cluster I is added to the m2t list of a cluster J, J is added to I's
   * diagonal send list.
   * @param[in] current_cluster Current cluster in the tree traversal.
   * @param[in] target_cluster  Cluster to whose nearfield the leaves are
   * added.
   */
  void determine_operation_lists_in_source_subtree(
    scheduling_time_cluster & current_cluster,
    scheduling_time_cluster & target_cluster );

  /**
   * Auxiliary routine used to traverse a subtree starting at a given source
   * cluster to update the m2t list of a given target cluster. The source
   * subtree is traversed recursively.
   *
   * @note Diagonal send lists are not set by this routine.
   * @param[in] current_source  Current source cluster in the tree traversal.
   * @param[in] target_cluster  Target cluster whose m2t list is updated.
   */
  void determine_m2t_list_in_subtree( scheduling_time_cluster & current_source,
    scheduling_time_cluster & target_cluster );

  /**
   * Updates the m2t and s2l lists of all local non-leaf scheduling time
   * clusters, whose associated space-time clusters have non-empty m2t or s2l
   * lists, by a recursive tree traversal.
   * @param[in] current_cluster Current cluster in the tree traversal. Its m2t
   * and s2l list are initialized if necessary.
   * @param[in] global_index_to_cluster Auxiliary structure to access
   * scheduling time clusters by knowing only their global index in the tree
   * structure.
   * @note The unordered map @p global_index_to_cluster can be constructed
   * using
   * @ref initialize_map_global_index_to_cluster.
   */
  void update_m2t_and_s2l_lists_recursively(
    scheduling_time_cluster & current_cluster,
    const std::unordered_map< lo, scheduling_time_cluster * > &
      global_index_to_cluster );

  /**
   * Auxiliary routine that constructs a mapping from global indices to time
   * clusters in the tree structure.
   *
   * The routine is based on a recursive tree traversal.
   * @param[in] current_cluster Current cluster in the tree traversal.
   * @param[in,out] global_index_to_cluster Map which is constructed by the
   * routine.
   */
  void initialize_map_global_index_to_cluster(
    scheduling_time_cluster & current_cluster,
    std::unordered_map< lo, scheduling_time_cluster * > &
      global_index_to_cluster ) const;

  /**
   * Determines all clusters which should be refined during a local expansion of
   * the tree structure. Therefore, an entry is added to @p refine_map for every
   * leaf cluster in the tree structure, with the global cluster index as key
   * and a bool indicating if it should be refined or not.
   * A leaf cluster should be refined if:
   * - it is handled by process @p _my_process_id
   * - it is in the nearfield of a cluster which is handled by process
   *   @p _my_process_id
   * - it is in the s2l-list of a cluster which is handled by process
   *   @p _my_process_id
   * The routine is based on a recursive tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in] refine_map  Stores which clusters should be refined. The keys
   *                        of the entries are the global indices of the
   *                        clusters.
   */
  void determine_clusters_to_refine_locally( scheduling_time_cluster * root,
    std::unordered_map< lo, bool > & refine_map ) const;

  /**
   * Checks if the subtree of @p root contains a cluster handled by process
   * @p _my_process_id by a recursive tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   */
  bool subtree_contains_local_cluster(
    const scheduling_time_cluster * root ) const;

  /**
   * Determines those clusters in the tree structure for which data has to be
   * exchanged.
   *
   * The subtree send and receive lists are relevant, when the tree structure
   * is refined based on a locally essential space-time cluster tree, see
   * @ref distributed_spacetime_cluster_tree::expand_distribution_tree_locally.
   *
   * The leaf info send and receive list are required to determine leaf
   * statuses of non-local space-time clusters, whose moments have to be
   * communicated in the FMM algorithm.
   *
   * The output vectors are filled with pairs (p, idx) of process ids and
   * pointers to clusters. They are used as follows:
   * - In @p subtree_send_list : the subtree starting at cluster idx has to be
   *   sent to process p after refinement.
   * - In @p subtree_receive_list : the subtree starting at cluster idx has to
   *   be received from process p after refinement.
   * - In @p leaf_info_send_list : the leaf information of the spacetime
   *   clusters associated with cluster idx has to be sent to process p.
   * - In @p leaf_info_receive_list : the leaf information of the spacetime
   *   clusters associated with cluster idx has to be received from process p.
   *
   * The routine is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] subtree_send_list Set storing process ids and pointers of
   *                                  clusters for which subtree data has to be
   *                                  sent.
   * @param[in,out] subtree_receive_list  Set storing process ids and pointers
   *                                      of clusters for which subtree data
   * has to be received.
   * @param[in,out] leaf_info_send_list Set storing process ids and pointers of
   *                                    clusters for which leaf information has
   *                                    to be sent.
   * @param[in,out] leaf_info_receive_list  Set storing process ids and
   * pointers of clusters for which leaf information has to be received.
   * @note We call this routine after executing
   * @ref remove_clusters_with_no_association. This routine gets rid of
   * unnecessary scheduling time clusters, and updates the global leaf status
   * of clusters which turn into leaves in this process. Such clusters are also
   * added to the send and receive list, even though this would not be
   * necessary.
   */
  void determine_cluster_communication_lists( scheduling_time_cluster * root,
    std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      subtree_send_list,
    std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      subtree_receive_list,
    std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      leaf_info_send_list,
    std::set< std::pair< lo, scheduling_time_cluster * >,
      compare_pairs_of_process_ids_and_scheduling_time_clusters > &
      leaf_info_receive_list ) const;

  /**
   * Clears the operation lists (nearfield, interaction, m2t, s2l, send,
   * diagonal send) of all clusters in the tree.
   *
   * The method relies on a tree traversal.
   *
   * @param[in] root  Current cluster in the tree traversal.
   * @note m2t, s2l and diagonal send lists are not only cleared, but deleted.
   */
  void clear_cluster_operation_lists( scheduling_time_cluster * root );

  /**
   * Clears the lists of associated spacetime clusters of all clusters in the
   * tree structure.
   *
   * The method relies on a recursive tree traversal.
   * @param[in] current_cluster Current cluster in the tree traversal.
   */
  void clear_lists_of_associated_clusters(
    scheduling_time_cluster & current_cluster );

  /**
   * Determines if clusters are active in the upward or downward path (needed
   * for FMM).
   * @param[in] root  Current cluster in the tree traversal.
   */
  void determine_cluster_activity( scheduling_time_cluster & root );

  /**
   * Prepares the reduction of the tree structure to the locally essential
   * part, by updating nearfields, interaction lists and send lists.
   *
   * In addition, @p _levels is reset. The method is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @note This method is solely used by @ref reduce_2_essential.
   */
  void prepare_essential_reduction( scheduling_time_cluster & root );

  /**
   * Deletes clusters form the tree structure which are not locally essential
   * (in the time cluster tree). The method is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void execute_essential_reduction( scheduling_time_cluster & root );

  /**
   * Determines the clusters which are essential for the current process by
   * traversing the tree recursively twice.
   *
   * A cluster I is essential in the temporal tree structure if it meets one
   * of the following requirements:
   * -#  I is assigned to the process (i.e. I is local).
   * -#  I is in the interaction list of a cluster which is assigned to the
   *     process.
   * -#  I is in the nearfield of a cluster J which is assigned to the
   *     process and either I or J are leaves. (cases where I is a leaf are
   *     included to enable S2L operations)
   * -#  I is in the S2L-list or M2T-list of a cluster which is assigned to the
   *     process (if such lists are constructed)
   * -#  I contains a cluster in its interaction list which is assigned to the
   *     process.
   * -#  I is a child of a cluster which is assigned to the process.
   * -#  (*) I is a leaf cluster and one of the clusters in its physical
   *     nearfield is assigned to the process. (Such a cluster might be needed
   *     to extend the tree communicatively or for M2T operations)
   * -#  (*) I is a cluster on a path between the root of a tree and another
   *     essential cluster.
   * -#  (*) I is in the interaction list of a cluster K, where K is on a path
   *     between the root of the tree and a local cluster.
   * -#  (*) I contains a cluster K in its interaction list, and K is on a path
   *     between the root of the tree and a local cluster.
   *
   * If a cluster satisfies one of the first three conditions the associated
   * space-time clusters in a space-time cluster tree are also locally
   * essential.
   *
   * The criteria are checked for all clusters by traversing the tree structure
   * twice. Criteria marked by (*) are checked during the second tree
   * traversal, the other ones during the first.
   *
   * The member @p essential_status of the clusters is set by this
   * function. (see @ref scheduling_time_cluster::_essential_status for a list
   * of possible status )
   *
   * @note This method is solely used by @ref reduce_2_essential. It executes
   * the routines @ref determine_essential_clusters_first_traversal and
   * @ref determine_essential_clusters_second_traversal.
   */
  void determine_essential_clusters( ) const;

  /**
   * Auxiliary routine that is used to determine essential clusters in the tree
   * structure by a first recursive tree traversal.
   *
   * See @ref determine_essential_clusters for details.
   */
  void determine_essential_clusters_first_traversal(
    scheduling_time_cluster & current_cluster ) const;

  /**
   * Auxiliary routine that is used to determine essential clusters in the tree
   * structure by a second recursive tree traversal.
   *
   * See @ref determine_essential_clusters for details.
   */
  void determine_essential_clusters_second_traversal(
    scheduling_time_cluster & current_cluster ) const;

  /**
   * Removes scheduling cluster from the tree structure which are not
   * associated with any space-time clusters.
   *
   * The clusters are not only removed but deleted. The routine is based on a
   * recursive tree traversal.
   * @param[in] current_cluster Current cluster in the tree traversal.
   */
  void remove_clusters_with_no_association(
    scheduling_time_cluster & current_cluster );

  /**
   * Considers all clusters in a given vector and removes those which have
   * essential_status 0.
   * @param[in,out] vector_of_clusters  Pointer to vector which is considered.
   * @note This routine can be very inefficient for long vectors, because it
   * iterates through the vector and calls erase for all relevant entries. In
   * practice it is only executed for short vectors, though.
   */
  void remove_non_essential_clusters_from_vector(
    std::vector< scheduling_time_cluster * > * vector_of_clusters ) const;

  /**
   * Aux for printing
   */
  void print_internal( scheduling_time_cluster * root ) {
    if ( root != nullptr ) {
      root->print( _my_process_id );
      std::vector< scheduling_time_cluster * > * children
        = root->get_children( );
      if ( children != nullptr )
        for ( auto it = children->begin( ); it != children->end( ); ++it ) {
          for ( lo i = 0; i < ( *it )->get_level( ); ++i ) std::cout << " ";
          print_internal( *it );
        }
    }
  }

  /**
   * Auxiliary method to determine the strings for the output by
   * @ref print_tree_human_readable. The method is based on a tree
   * traversal.
   * @param[in] digits  Number of digits used for the output of a process id.
   * @param[in] print_process_ids Bool to decide whether process ids of the
   *                              clusters or their global indices are printed.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] levelwise_output_strings  For each level this vector
   *                                          contains a string to which the
   *                                          output strings for each cluster
   *                                          are added.
   */
  void determine_levelwise_output_string( const lo digits,
    bool print_process_ids, scheduling_time_cluster * root,
    std::vector< std::string > & levelwise_output_strings ) const;
};

#endif /* INCLUDE_BESTHEA_TREE_STRUCTURE_H_ */
