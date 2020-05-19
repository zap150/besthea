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
   * @note The start and end point of the mesh are used to generate the 
   * geometrical data of the clusters.
   * @warning Only the structure of the tree is reconstructed. The elements of
   * the mesh are not added to the clusters.
   * \todo update this
   */
  tree_structure( const std::string filename );

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
  void print_tree_structure( const std::string filename ) const;

  /**
   * Reads a vector corresponding to a tree structure from a binary file
   * @param[in] filename Name of the input file
   */
  std::vector< char > load_tree_structure( const std::string filename ) const;

  /**
   * Prints levels of the tree.
   */
  void print( ) {
    std::cout << "number of levels: " << _levels << std::endl;
    // print cluster information recursively
    print_internal( _root );
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
   * \note This method is supposed to be called by @ref compute_tree_structure
   * \warning currently this works only for time clusters
   */
  void tree_2_vector( const cluster_type & root,
    std::vector<char> & tree_vector ) const;

  /**
   * Recursively constructs the tree structure from a given vector.
   * @param[in] tree_vector Contains the data needed for tree construction.
   * @param[in,out] root  Current cluster, to which the next clusters are added.
   * @param[in,out] position  Auxiliary variable to keep track of the current
   *                          position in the tree_vector.
   * \note  This method is supposed to be called by the corresponding 
   *        constructor.
   * \todo adapt this for the individual cluster types
   */
  void vector_2_tree( const std::vector<char> & tree_vector, 
    cluster_type & root, lou & position );

  /**
   * Collects all clusters without descendants and stores them in the internal
   * _leaves vector.
   */
  void collect_leaves( cluster_type & root );

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
};

#endif /* INCLUDE_BESTHEA_TREE_STRUCTURE_H_ */
