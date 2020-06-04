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
#include "besthea/time_cluster.h"

#include <iostream>
#include <list>
#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class tree_structure;
  }
}

/**
 * Class representing the structure of a tree of temporal clusters.
 * It is meant to be used for the scheduling of jobs in a parallel FMM.
 */
class besthea::mesh::tree_structure {
 public:
  /**
   * Constructs a tree structure by reading it from a file.
   * @param[in] filename Name of the input file containing the tree structure.
   * @param[in] start_time  Start time of the time mesh.
   * @param[in] end_time  End time of the time mesh.
   * @note The start and end point of the mesh are used to generate the 
   * geometrical data of the clusters.
   */
  tree_structure( const std::string & filename, const sc start_time, 
    const sc end_time );

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
   * Reduces the tree structure by deleting all clusters which are not contained
   * in the locally essential tree of the current process.
   * @param[in] my_process_id Id of the current process.
   * @note @p _leaves and @p _levels are reset.
   * @note The original global_indices of the clusters are not modified. This 
   * allows to identify clusters between different processes.
   */
  void reduce_2_essential( const lo my_process_id );

  /**
   * Fills the 4 lists used for scheduling the FMM operations by adding pointers
   * to clusters assigned to the process with id @p _my_process_id . In addition
   * it determines all pairs of clusters and process ids from which data is 
   * received, and initializes the data in the scheduling time clusters which is
   * used to check the dependencies.
   * @param[in,out] m_list  List for scheduling upward path operations.
   * @param[in,out] m2l_list  List for scheduling interactions and downward pass
   *                          operations.
   * @param[in,out] l_list  List for scheduling downward path operations.
   * @param[in,out] n_list  List for scheduling nearfield operations.
   * @param[in,out] receive_vector  Is filled with the pairs of clusters and 
   *                                process ids from which data is received.
   * @param[out]  n_moments_to_receive  The first @p n_moments_to_receive pairs
   *                                    in the receive vector correspond to 
   *                                    operations where moments are received,
   *                                    not local contributions.
   * @note All lists are constructed anew, existing values are overwritten.
   * @note The clusters in the m_list are sorted using the comparison operator
   *       @ref compare_clusters_bottom_up_right_2_left, the others using
   *       @ref compare_clusters_top_down_right_2_left .
   * @todo Determine n_list differently, when coupling with space-time cluster
   *       tree is ready.
   */
  void prepare_fmm( std::list< scheduling_time_cluster* > & m_list,
    std::list< scheduling_time_cluster* > & m2l_list,
    std::list< scheduling_time_cluster* > & l_list,
    std::list< scheduling_time_cluster* > & n_list,
    std::vector< std::pair< scheduling_time_cluster*, lo > > & receive_vector,
    lou & n_moments_to_receive ) const;

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
   * Computes the tree structure and prints it to a binary file
   * @param[in] filename Name of the output file
   */
  void print_tree_structure( const std::string & filename ) const;

  /**
   * Print the tree in a recursive manner to the console.
   */
  void print( ) {
    std::cout << "number of levels: " << _levels << std::endl;
    // print cluster information recursively
    print_internal( _root );
  }

  /**
   * Prints the process ids assigned to the clusters in a real binary tree
   * format to console.
   * @param[in] digits  Number of digits which are used to print the process 
   *                    ids. Its value should be larger than the number of
   *                    digits the highest process id has.
   */
  void print_tree_human_readable( 
    const lo digits, const bool print_process_ids ) const; 

 private:
  scheduling_time_cluster * _root;  //!< root cluster of the tree structure
  lo _levels;                       //!< number of levels in the tree
  std::vector< scheduling_time_cluster * >
    _leaves;  //!< vector of all clusters without descendants
  lo _my_process_id;  //!< id of the process executing the operations
                      //!< @todo This can later be replaced by an MPI query. 

  /**
   * Recursively constructs the structural vector of a tree structure.
   * @param[in] root Current cluster, whose children are considered to determine
   *                 the next characters in the structural vector.
   * @param[in,out] tree_vector Vector to store the tree structure.
   * @note This method is supposed to be called by @ref compute_tree_structure
   * @warning currently this works only for time clusters
   */
  void tree_2_vector( const scheduling_time_cluster & root,
    std::vector< char > & tree_vector ) const;

  /**
   * Recursively constructs the tree structure from a given vector.
   * @param[in] tree_vector Contains the data needed for tree construction.
   * @param[in,out] root  Current cluster, to which the next clusters are added.
   * @param[in,out] position  Auxiliary variable to keep track of the current
   *                          position in the tree_vector.
   * @note  This method is supposed to be called by the corresponding 
   *        constructor.
   * @todo adapt this for the individual cluster types
   */
  void vector_2_tree( const std::vector<char> & tree_vector, 
    scheduling_time_cluster & root, lou & position );

  /** Assigns to each cluster in the tree structure its respective process 
   * given in process_assignments by traversing the tree structure.
   * @param[in] process_assignments Vector of the process assignments given in 
   *                                the tree structure format
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
   */
  void set_leaf_ids( scheduling_time_cluster & root, lo & next_id );

  /**
   * Sets the global indices of all clusters in the tree structure by traversing
   * it recursively. The indices correspond to a consecutive enumeration of the 
   * clusters according to the tree traversal. The index of a parent is set
   * after its children and therefore greater.
   * @param[in] root Current cluster in the tree traversal.
   * @param[in,out] next_index Next index which is assigned to a cluster.
   */
  void set_indices( scheduling_time_cluster & root, lo & next_index );

  /**
   * Computes and sets the nearfield and interaction list for every cluster in 
   * the tree structure by recursively traversing the tree.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void set_nearfield_interaction_and_send_list( 
    scheduling_time_cluster & root );

  void add_leaves_to_nearfield( scheduling_time_cluster & current_cluster, 
    scheduling_time_cluster & target_cluster );

  /**
   * Determines if clusters are active in the upward or downward path (needed 
   * for FMM).
   * @param[in] root  Current cluster in the tree traversal.
   */
  void determine_cluster_activity( scheduling_time_cluster & root );

  /**
   * Traverses the tree recursively and adds all clusters assigned to the 
   * process @p _my_process_id to the given list.
   * @param[in] root Current cluster in the tree traversal.
   * @param[in,out] cluster_list  List which is filled with the local clusters.
   * @note The routine is called solely by @ref prepare_fmm_lists .
   */
  void init_fmm_lists_and_dependency_data( scheduling_time_cluster & root,
    std::list< scheduling_time_cluster* > & m_list,
    std::list< scheduling_time_cluster* > & m2l_list,
    std::list< scheduling_time_cluster* > & l_list,
    std::list< scheduling_time_cluster* > & n_list ) const;

  /**
   * Prepares the reduction of the tree structure to the locally essential part,
   * by updating nearfields, interaction lists and send lists and detecting
   * the remaining essential clusters (those which lie on a path between the
   * root and another essential cluster). In addition, @p _levels is reset. The 
   * method is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] status_vector  Status of the clusters in the tree, 
   *                               indicating if a cluster is essential or not. 
   *                               This is updated. 
   * @note This method is solely used by @ref reduce_2_essential .
   * @note Clusters in the tree structure which are not contained in the
   * locally essential part of the tree are deleted. 
   */
  void prepare_essential_reduction( scheduling_time_cluster & root, 
    std::vector< char > & status_vector );

  /**
   * Deletes clusters form the tree structure which are not locally essential.
   * The method is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in] status_vector Status of the clusters in the tree, ordered by the
   *                          global indices of the clusters.
   */
  void execute_essential_reduction( scheduling_time_cluster & root, 
    const std::vector< char > & status_vector ); 

  /**
   * Determines the clusters which are essential for the current process by
   * traversing the tree recursively.
   * A cluster is essential if it meets one of the following requirements:
   * -  It is assigned to the process.
   * -  It is in the interaction list of a cluster which is assigned to the 
   *    process.
   * -  It contains a cluster in its interaction list which is assigned to the
   *    process.
   * -  It is a child of a cluster which is assigned to the process.
   * -  It is in the nearfield of a leaf cluster which is assigned to the 
   *    process.
   * @param[in] my_process_id Id of the current process.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] status_vector  Contains the status of the clusters in the 
   *                               order of their indices. Tree status are used: 
   *                               - 0 (not essential), 
   *                               - 1 (assigned to the process), 
   *                               - 2 (other essential).
   * @note The locally essential tree should also contain clusters which are 
   * contained in a path from the root of the tree structure to a cluster which
   * meets one of the above requirements. Such clusters are not detected here, 
   * but in the routine @ref execute_essential_reduction .
   * @note This method is solely used by @ref reduce_2_essential .
   */
  void determine_essential_clusters( const lo my_process_id, 
    const scheduling_time_cluster & root, 
    std::vector< char > & status_vector ) const;

  /**
   * Aux for printing
   */
  void print_internal( scheduling_time_cluster * root ) {
    if ( root != nullptr ) {
      root->print( );
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
   * Auxiliary method to determine the strings for the output of process ids by
   * @ref print_processes_human_readable. The method is based on a tree 
   * traversal.
   * @param[in] digits  Number of digits used for the output of a process id.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] levelwise_output_strings  For each level this vector 
   *                                          contains a string to which the 
   *                                          output strings for each cluster
   *                                          are added.
   */
  void determine_levelwise_output_string( const lo digits, 
  const bool print_process_ids, scheduling_time_cluster * root, 
  std::vector< std::string > & levelwise_output_strings ) const;

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
};

#endif /* INCLUDE_BESTHEA_TREE_STRUCTURE_H_ */
