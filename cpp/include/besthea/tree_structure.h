/*
 * Copyright 2020, VSB - Technical University of Ostrava and Graz University of
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

/** @file tree_structure.h
 * @brief General tree structure.
 */

#ifndef INCLUDE_BESTHEA_TREE_STRUCTURE_H_
#define INCLUDE_BESTHEA_TREE_STRUCTURE_H_

#include "besthea/io_routines.h"
#include "besthea/scheduling_time_cluster.h"
#include "besthea/settings.h"
#include "besthea/spacetime_cluster.h"
#include "besthea/spacetime_cluster_tree.h"

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
   * @note The start and end point of the mesh are used to generate the 
   * geometrical data of the clusters.
   */
  tree_structure( const std::string & filename, const sc start_time, 
    const sc end_time, lo process_id = -1  );

  /**
   * Constructs a tree structure by reading it from files
   * @param[in] structure_file Name of the binary input file containing the tree 
   *                          structure.
   * @param[in] cluster_bounds_file   Name of the binary input file containing
   *                                  the bounds of all the clusters in the 
   *                                  tree structure.
   * @param[in] process_id  Id of the process which calls the method. Default 
   *                        value is -1.
   */
  tree_structure( const std::string & structure_file, 
    const std::string & cluster_bounds_file,
    lo process_id = -1 );

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
  std::vector< scheduling_time_cluster * > & get_leaves( ) {
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
   * Traverses the tree recursively and adds all relevant clusters assigned to 
   * the process @p _my_process_id to the 4 lists for scheduling operations in 
   * the FMM.
   * @param[in] root Current cluster in the tree traversal.
   * @param[in,out] m_list  List for scheduling upward path operations.
   * @param[in,out] m2l_list  List for scheduling interactions and downward pass
   *                          operations.
   * @param[in,out] l_list  List for scheduling downward path operations.
   * @param[in,out] n_list  List for scheduling nearfield operations.
   * @note The routine is solely called by @ref prepare_fmm.
   */
  void init_fmm_lists_and_dependency_data( scheduling_time_cluster & root,
    std::list< scheduling_time_cluster* > & m_list,
    std::list< scheduling_time_cluster* > & m2l_list,
    std::list< scheduling_time_cluster* > & l_list,
    std::list< scheduling_time_cluster* > & n_list ) const;
    
  /**
   * Fills the 4 lists used for scheduling the FMM operations by adding pointers
   * to clusters assigned to the process with id @p _my_process_id . In addition
   * it determines all pairs of clusters and process ids from which data is 
   * received, and initializes the data in the scheduling time clusters which is
   * used to check the dependencies.
   * @param[in,out] m_list  List for scheduling upward path operations.
   * @param[in,out] m2l_list  List for scheduling interaction and downward pass
   *                          operations.
   * @param[in,out] l_list  List for scheduling downward path operations.
   * @param[in,out] n_list  List for scheduling nearfield operations.
   * @param[in,out] receive_vector  Is filled with the pairs of clusters and 
   *                                process ids from which data is received.
   * @param[out] n_moments_upward  Used to store the number of entries in the 
   *                               receive vector, which corresponds to receive 
   *                               operations in the upward path. These entries 
   *                               come first in the vector.
   * @param[out] n_moments_m2l  Used to store the number of entries in the  
   *                            receive vector, which corresponds to receive 
   *                            operations for M2L. These entries come second in  
   *                            the vector, and are followed by the entries for 
   *                            the receive operations in the downward path.
   * @note All lists are constructed anew, existing values are overwritten.
   * @note The clusters in the m_list are sorted using the comparison operator
   *       @ref compare_clusters_bottom_up_right_2_left, the others using
   *       @ref compare_clusters_top_down_right_2_left .
   * @todo Determine n_list differently, when coupling with space-time cluster
   *       tree is done.
   * @todo Delete this later. The routine was transferred to 
   * @ref besthea::linear_algebra::distributed_pFMM_matrix.
   */
  void prepare_fmm( std::list< scheduling_time_cluster* > & m_list,
    std::list< scheduling_time_cluster* > & m2l_list,
    std::list< scheduling_time_cluster* > & l_list,
    std::list< scheduling_time_cluster* > & n_list,
    std::vector< std::pair< scheduling_time_cluster*, lo > > & receive_vector,
    lou & n_moments_upward, lou & n_moments_m2l ) const;

  /**
   * Traverses the tree structure recursively and allocates and initializes the 
   * moments for all clusters which are active in the upward path of the FMM.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in] contribution_size Size of the contribution of a single spacetime
   *                              cluster.
   */
  void initialize_moment_contributions( 
    scheduling_time_cluster& root, lou contribution_size );

  /**
   * Fills the associated moments of all clusters in the tree with zeros.
   * The tree is traversed recursively by the routine.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void clear_moment_contributions( scheduling_time_cluster& root );

  /**
   * Traverses the tree structure recursively and allocates and initializes the 
   * local contributions for all clusters which are active in the downward path
   * of the FMM.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in] contribution_size Size of the contribution of a single spacetime
   *                              cluster.
   */
  void initialize_local_contributions( 
    scheduling_time_cluster& root, lou contribution_size );

  /**
   * Fills the associated local contributions of all clusters in the tree with 
   * zeros. The tree is traversed recursively by the routine.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void clear_local_contributions( scheduling_time_cluster& root );

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
    scheduling_time_cluster* first, scheduling_time_cluster* second );

  /**
   * Comparison operator for two clusters.
   * @param[in] first Pointer to the first cluster.
   * @param[in] second Pointer to the second cluster.
   * @return True if first's level is less than second's level, or, in case 
   *         of equality of the levels, if first's global index is greater than
   *         second's global index
   */
  static bool compare_clusters_top_down_right_2_left( 
    scheduling_time_cluster* first, scheduling_time_cluster* second );

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
  std::vector< scheduling_time_cluster * >
    _leaves;  //!< vector of all clusters without descendants 
  lo _my_process_id;  //!< id of the process executing the operations
                      //!< @todo Exchange by an MPI query later?

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
   * @param[in,out] root  Current cluster, to which the next clusters are added.
   * @param[in,out] position  Auxiliary variable to keep track of the current
   *                          position in the tree_vector.
   * @note This method is supposed to be called by the constructor.
   */
  void array_2_tree( const char * tree_array, scheduling_time_cluster & root, 
    lou & position );

  /**
   * Recursively constructs the tree structure from a given structure vector
   * and a given vector of cluster bounds.
   * @param[in] tree_vector Contains the tree structure data.
   * @param[in] cluster_bounds  Contains the data of the clusters' bounds.
   * @param[in,out] root  Current cluster, to which the next clusters are added.
   * @param[in,out] position  Auxiliary variable to keep track of the current
   *                          position in the vectors. 
   * @note  This method is supposed to be called by the constructor.
   */
  void create_tree_from_vectors( const std::vector< char > & tree_vector, 
    const std::vector< sc > & cluster_bounds, scheduling_time_cluster & root, 
    lou & position );

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
   * Assigns consecutive indices to all leaf clusters. The routine is based on a
   * tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] next_id Next id which is set.
   * @todo Delete this later, if not needed (used originally for dummy test)
   */
  void set_leaf_ids( scheduling_time_cluster & root, lo & next_id );

  /**
   * Sets the global indices of all clusters in the tree structure by traversing
   * it recursively. The parent cluster with id k sets the indices of its 
   * children: the left gets the index 2k+1, the right the index 2k+2.
   * @param[in] root Current cluster in the tree traversal.
   * @note The index of the root s assumed to be 0 before the routine is called.
   */
  void set_indices( scheduling_time_cluster & root );

  /**
   * Computes and sets the nearfield, interaction list and send list for every 
   * cluster in the tree structure by recursively traversing the tree.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void set_nearfield_interaction_and_send_list( 
    scheduling_time_cluster & root );

  /**
   * Used for the construction of nearfields of leaf clusters by 
   * @ref set_nearfield_interaction_and_send_list. It recursively traverses the
   * tree starting from the initial @p current_cluster, and adds all descendant
   * leaves to the nearfield of the leaf @p target_cluster.
   * @param[in] current_cluster Current cluster in the tree traversal.
   * @param[in] target_cluster  Cluster to whose nearfield the leaves are added.
   */
  void add_leaves_to_nearfield_list( scheduling_time_cluster & current_cluster, 
    scheduling_time_cluster & target_cluster );

    /**
   * Determines all clusters which should be refined during an expansion of the
   * tree structure. Therefore, an entry is added to @p refine_map for every 
   * leaf cluster in the tree structure, with the global cluster index as key
   * and a bool indicating if it should be refined or not.
   * A leaf cluster should be refined if: 
   * - it is handled by process @p _my_process_id
   * - it is in the nearfield of a cluster which is handled by process 
   *    @p _my_process_id
   * - there is a cluster handled by process @p _my_process_id in its direct
   *   nearfield or one of the descendants of such a cluster is handled by it.
   * The routine is based on a recursive tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in] refine_map  Stores which clusters should be refined. The keys
   *                        of the entries are the global indices of the 
   *                        clusters.
   */
  void determine_clusters_to_refine( scheduling_time_cluster* root, 
    std::unordered_map< lo, bool > & refine_map ) const;

  /**
   * Checks if the subtree of @p root contains a cluster handled by process
   * @p _my_process_id by a recursive tree traversal.
   * @param[in] root  Current cluster in the tree traversal. 
   */
  bool subtree_contains_local_cluster( 
    const scheduling_time_cluster* root ) const;

  /**
   * Expands the temporal tree structure by recursively traversing the current 
   * tree structure and the given spacetime cluster tree. It uses @p refine_map
   * and the spacetime cluster tree to determine if clusters should be added to
   * the temporal tree structure.
   * @param[in] spacetime_root Current cluster in the spacetime cluster tree.
   * @param[in] root  Current cluster in the tree structure.
   * @param[in,out] refine_map  Map which indicates if the tree should be 
   *                            expanded at a leaf cluster or not. This is
   *                            updated if new clusters are added.
   * @todo Delete?! (obsolete, since a similar method exists in 
   * @ref distributed_spacetime_cluster_tree )
   */
  void expand_tree_structure_recursively(
    spacetime_cluster* spacetime_root, scheduling_time_cluster* root,
    std::unordered_map< lo, bool > & refine_map );


  /**
   * Determines those clusters in the tree structure for which data has to be 
   * exchanged when the tree structure is refined based on a locally essential
   * space-time cluster tree (see 
   * @ref distributed_spacetime_cluster_tree::expand_distribution_tree_locally ).
   * 
   * The output vectors @p cluster_send_list and @p cluster_receive_list
   * contain pairs of process ids and global cluster indices:
   * - For a pair (p, idx) in @p cluster_send_list the subtree starting at 
   *   cluster idx has to be sent to process p after refinement. 
   * - For a pair (p, idx) in @p cluster_receive_list the subtree starting at 
   *   cluster idx has to be received from process p after refinement.
   * 
   * The routine is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] cluster_send_list Vector storing the clusters and process
   *                                  ids for which data has to be sent.
   * @param[in,out] cluster_receive_list Vector storing the clusters and process
   *                                     ids for which data has to be received.
   */
  void determine_refinement_communication_lists(
    scheduling_time_cluster* root,
    std::set< std::pair< lo, scheduling_time_cluster* > > & cluster_send_list,
    std::set< std::pair< lo, scheduling_time_cluster* > > 
      & cluster_receive_list ) const;

  /**
   * Clears the nearfield, interaction and send list of all clusters in the 
   * tree. The method relies on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void clear_cluster_lists( scheduling_time_cluster* root );
  /**
   * Determines if clusters are active in the upward or downward path (needed 
   * for FMM).
   * @param[in] root  Current cluster in the tree traversal.
   */
  void determine_cluster_activity( scheduling_time_cluster & root );

  /**
   * Prepares the reduction of the tree structure to the locally essential part,
   * by updating nearfields, interaction lists and send lists and detecting
   * the remaining essential clusters (those which lie on a path between the
   * root and another essential cluster). In addition, @p _levels is reset. The 
   * method is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @note This method is solely used by @ref reduce_2_essential .
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
   * traversing the tree recursively.
   * A cluster is essential in the temporal tree structure if it meets one of 
   * the following requirements:
   * -#  It is assigned to the process.
   * -#  It is in the interaction list of a cluster which is assigned to the 
   *     process.
   * -#  It is in the nearfield of a leaf cluster which is assigned to the 
   *     process.
   * -#  It contains a cluster in its interaction list which is assigned to the
   *     process.
   * -#  It is a child of a cluster which is assigned to the process.
   * -#  It is a leaf cluster and one of the clusters in its nearfield is 
   *     assigned to the process. (Such a cluster is strictly speaking not
   *     essential, but we keep it to make expansions simpler)
   * 
   * If a cluster satisfies one of the first three conditions the associated
   * space-time clusters in a space-time cluster tree are also locally 
   * essential. 
   * 
   * The member @p essential_status of the clusters is set by this 
   * function. (see @ref scheduling_time_cluster::_essential_status for a list
   * of possible status )
   * @param[in] my_process_id Id of the current process.
   * @param[in] root  Current cluster in the tree traversal.
   * @note The locally essential tree should also contain clusters which are 
   * contained in a path from the root of the tree structure to a cluster which
   * meets one of the above requirements. Such clusters are not detected here, 
   * but in the routine @ref prepare_essential_reduction .
   * @note This method is solely used by @ref reduce_2_essential .
   */
  void determine_essential_clusters( const lo my_process_id, 
    scheduling_time_cluster & root ) const;

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
