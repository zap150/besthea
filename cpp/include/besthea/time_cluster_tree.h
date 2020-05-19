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

/** @file time_cluster_tree.h
 * @brief Tree of temporal cluster.
 */

#ifndef INCLUDE_BESTHEA_TIME_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_TIME_CLUSTER_TREE_H_

#include "io_routines.h"
#include "besthea/settings.h"
#include "besthea/temporal_mesh.h"
#include "besthea/time_cluster.h"

#include <iostream>
#include <string>

namespace besthea {
  namespace mesh {
    class time_cluster_tree;
  }
}

/**
 * Class representing (not necessarily binary) tree of temporal clusters.
 */
class besthea::mesh::time_cluster_tree {
 public:
  /**
   * Constructor
   * @param[in] mesh Reference to the underlying mesh.
   * @param[in] levels Maximum number of levels in the tree.
   * @param[in] n_min_elems Minimum number of elements so that a cluster can be 
   *                        split in halves in the construction of the tree.
   */
  time_cluster_tree( const temporal_mesh & mesh, lo levels, lo n_min_elems );

  /**
   * Destructor.
   */
  virtual ~time_cluster_tree( ) {
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
   * Returns the maximal number of elements in a leaf cluster
   */
  lo get_n_max_elems_leaf( ) const {
    return _n_max_elems_leaf;
  }

  /**
   * Returns the root of the tree.
   */
  time_cluster * get_root( ) {
    return _root;
  }

  /**
   * Returns the vector of levelwise paddings.
   */
  const std::vector< sc > & get_paddings( ) const {
    return _paddings;
  }

  /**
   * Computes padding of temporal clusters
   */
  sc compute_padding( time_cluster & root );

  /**
   * Returns the underlying temporal mesh.
   */
  const temporal_mesh & get_mesh( ) {
    return _mesh;
  }

  /**
   * Returns clusters without descendants.
   */
  std::vector< time_cluster * > & get_leaves( ) {
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
   * Prints levels of the tree.
   */
  void print( ) {
    // print cluster information recursively
    print_internal( _root );
    // print general tree information
    std::cout << "number of levels: " << _levels << std::endl;
    // print vector of paddings
    std::cout << "padding: " << std::endl;
    for ( lou i = 0; i < _paddings.size( ); ++ i ) {
      std::cout << _paddings[ i ] << " ";
    }
    std::cout << std::endl;
  }


 private:
  time_cluster * _root;         //!< root cluster of the tree
  const temporal_mesh & _mesh;  //!< underlying mesh
  lo _levels;                   //!< number of levels in the tree
  lo _real_max_levels;  //!< auxiliary value to determine number of real tree
                        //!< levels (depending on _n_min_elems)
  lo _n_min_elems;  //!< minimum number of elements so that cluster can be split
                    //!< in halves
  lo _n_max_elems_leaf; //!< maximal number of elements in a leaf cluster after 
                        //!< construction.
  std::vector< sc > _paddings;  //!< vector of paddings on each level
  std::vector< time_cluster * >
    _leaves;  //!< vector of all clusters without descendants

  /**
   * Builds tree recursively
   * @param[in] root Node to stem from.
   * @param[in] level Current level.
   */
  void build_tree( time_cluster & root, lo level );

  /**
   * Recursively constructs the structural vector of a tree.
   * @param[in] root Current cluster, whose children are considered to determine
   *                 the next characters in the structural vector.
   * @param[in,out] tree_vector Vector to store the tree structure.
   * \note This method is supposed to be called by @ref compute_tree_structure
   */
  void tree_2_vector( const time_cluster & root,
    std::vector<char> & tree_vector ) const;

  /**
   * Collects all clusters without descendants and stores them in the internal
   * _leaves vector.
   */
  void collect_leaves( time_cluster & root );

  /**
   * Aux for printing
   */
  void print_internal( time_cluster * root ) {
    if ( root != nullptr ) {
      root->print( );
      std::vector< time_cluster * > * children = root->get_children( );
      if ( children != nullptr )
        for ( auto it = children->begin( ); it != children->end( ); ++it ) {
          for ( lo i = 0; i < ( *it )->get_level( ); ++i ) std::cout << " ";
          print_internal( *it );
        }
    }
  }
};

#endif /* INCLUDE_BESTHEA_TIME_CLUSTER_TREE_H_ */
