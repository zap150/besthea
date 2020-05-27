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
#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    template< class cluster_type >
    class tree_structure;
  }
}

/**
 * Class representing (not necessarily binary) tree of temporal clusters.
 */
template< class cluster_type >
class besthea::mesh::tree_structure {
 public:
  /**
   * Constructs a tree structure by reading it from a file.
   * @param[in] filename Name of the input file containing the tree structure.
   * @param[in] start_time  Start time of the time mesh.
   * @param[in] end_time  End time of the time mesh.
   * @note The start and end point of the mesh are used to generate the 
   * geometrical data of the clusters.
   * @warning Only the structure of the tree is reconstructed. The elements of
   * the mesh are not added to the clusters.
   * @todo use a different constructor if the tree structure is used for more
   * general trees, not only temporal
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
  cluster_type * get_root( ) {
    return _root;
  }

  /**
   * Returns clusters without descendants.
   */
  std::vector< cluster_type * > & get_leaves( ) {
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
   * @param[in] my_id Id of the current process.
   * @note @p _leaves and @p _levels are reset, as well as the indices of the
   * time clusters.
   */
  void reduce_2_essential( const lo my_id );

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
  void print_processes_human_readable( lo digits ) {
    std::vector< std::string > print_strings;
    print_strings.resize( _levels );
    determine_processes_string( digits, _root, print_strings );
    for ( lou i = 0; i < print_strings.size( ); ++i ) {
      std::cout << print_strings[ i ] << std::endl;
    }
  } 

 private:
  cluster_type * _root;         //!< root cluster of the tree structure
  lo _levels;                   //!< number of levels in the tree
  std::vector< cluster_type * >
    _leaves;  //!< vector of all clusters without descendants

  /**
   * Recursively constructs the structural vector of a tree structure.
   * @param[in] root Current cluster, whose children are considered to determine
   *                 the next characters in the structural vector.
   * @param[in,out] tree_vector Vector to store the tree structure.
   * @note This method is supposed to be called by @ref compute_tree_structure
   * @warning currently this works only for time clusters
   */
  void tree_2_vector( const cluster_type & root,
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
    cluster_type & root, lou & position );

  /** Assigns to each cluster in the tree structure its respective process 
   * given in process_assignments by traversing the tree structure.
   * @param[in] process_assignments Vector of the process assignments given in 
   *                                the tree structure format
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] position  Current position in process_assignments.

   */
  void set_process_assignments( const std::vector< lo > process_assignments, 
    cluster_type & root, lou & position );

  /**
   * Collects all clusters without descendants and stores them in the internal
   * _leaves vector.
   */
  void collect_leaves( cluster_type & root );

  /**
   * Sets the levelwise indices of all clusters in the tree structure by 
   * traversing it recursively. The indices correspond to a levelwise 
   * consecutive enumeration of the clusters from left to right.
   * @param[in] root Current cluster in the tree traversal.
   * @param[in,out] index_counters  A counter for each level to keep track of 
   *                                the indices.
   */
  void set_indices( cluster_type & root, std::vector< lo > & index_counters );

  /**
   * Computes and sets the nearfield and interaction list for every cluster in 
   * the tree structure by recursively traversing the tree.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void set_nearfield_and_interaction_list( cluster_type & root );

  /**
   * Executes the reduction of the tree structure to the locally essential part.
   * The method is based on a tree traversal.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] levelwise_status  Status of the clusters in the tree,
   *                                  indicating if a cluster is essential or 
   *                                  not. This is updated by setting clusters
   *                                  to essential which are on a path from the
   *                                  root of the tree structure to an existing
   *                                  essential cluster. 
   * @note This method is solely used by @ref reduce_2_essential .
   * @note Clusters in the tree structure which are not contained in the
   * locally essential part of the tree are deleted. 
   * @note @p _leaves and @p _levels are reset, as well as the indices of the
   * time clusters.
   */
  void execute_essential_reduction( cluster_type & root, 
    std::vector< std::vector< char > > & levelwise_status ); 

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
   * @param[in] my_id Id of the current process.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in,out] levelwise_status  Consists of a vector for each level in 
   *                                  which the status of the clusters is given
   *                                  in the order of their indices. Tree status
   *                                  are used: 0 (not essential), 1 (assigned
   *                                  to the process), 2 (other essential)
   * @note The locally essential tree should also contain clusters which are 
   * contained in a path from the root of the tree structure to a cluster which
   * meets one of the above requirements. Such clusters are not detected.
   * @note This method is solely used by @ref reduce_2_essential .
   */
  void determine_essential_clusters( const lo my_id, const cluster_type & root, 
    std::vector< std::vector< char > > & levelwise_status );

  /**
   * Aux for printing
   */
  void print_internal( cluster_type * root ) {
    if ( root != nullptr ) {
      root->print( );
      std::vector< cluster_type * > * children = root->get_children( );
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
  void determine_processes_string( const lo digits, cluster_type * root,
    std::vector< std::string > & levelwise_output_strings ) {
    if ( root != nullptr ) {
      lo current_level = root->get_level( );
      lo process_id = root->get_process_id( );
      // compute the number of digits the process_id needs
      lo id_digits = 1;
      if ( process_id > 0 ) {
        id_digits = ( lo ) ceil( log10( process_id + 1 ) );
      }
      // construct the string for the cluster and append it to the output string
      // at the appropriate level
      lo n_digits_level = ( 1 << ( _levels - 1 - current_level ) ) * digits;
      lo n_trailing_whitespace = n_digits_level - id_digits;
      std::string next_string = std::to_string( process_id )
        + std::string( n_trailing_whitespace, ' ');
      levelwise_output_strings[ current_level ] += next_string;
      // proceed for the children
      if ( root->get_n_children( ) == 2 ) {
        // call the routine for each child
        auto children = root->get_children( );
        for ( auto child : *children ) {
          determine_processes_string( digits, child, levelwise_output_strings );
        }
      }
      else if ( root->get_n_children( ) == 1 ) {
        // call the routine for the existing child and add / and whitespaces for
        // the non-existing child
        auto child = ( * root->get_children( ) )[ 0 ];
        sc parent_center = root->get_center( );
        sc child_center = child->get_center( );
        std::vector< bool > child_exists( 2, false );
        if ( child_center < parent_center ) {
          child_exists[ 0 ] = true;
        } else {
          child_exists[ 1 ] = true;
        }
        for ( lou i = 0; i < 2; ++i ) {
          if ( child_exists[ i ] == true ) {
            determine_processes_string( digits, child, 
              levelwise_output_strings );
          } else {
            // add / and whitespaces for non-existing clusters starting from
            // the non-existing leaf to the bottom of the tree
            lo n_children = 1;
            lo n_digits_level_mod = n_digits_level;
            for ( lo level = current_level + 1; level < _levels; ++level ) {
              n_digits_level_mod /= 2;
              std::string child_string 
                = '/' + std::string(n_digits_level_mod - 1, ' ');
              for ( lo child = 0; child < n_children; ++child ) {
                levelwise_output_strings[ level ] += child_string; 
              }
              n_children *= 2;
            }
          }
        }
      } else {
        // add / and whitespaces for non-existing clusters starting from the
        // non-existing leaves to the bottom of the tree
        lo n_children = 1;
        for ( lo level = current_level + 1; level < _levels; ++level ) {
          n_children *= 2;
          n_digits_level /= 2;
          std::string child_string = '/' + std::string(n_digits_level - 1, ' ');
          for ( lo child = 0; child < n_children; ++child ) {
            levelwise_output_strings[ level ] += child_string; 
          }
        }
      }
    }
  }
};

#endif /* INCLUDE_BESTHEA_TREE_STRUCTURE_H_ */
