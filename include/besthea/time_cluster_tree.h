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

/** @file time_cluster_tree.h
 * @brief Tree of temporal cluster.
 */

#ifndef INCLUDE_BESTHEA_TIME_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_TIME_CLUSTER_TREE_H_

#include "besthea/settings.h"
#include "besthea/temporal_mesh.h"
#include "besthea/time_cluster.h"
#include "io_routines.h"

#include <iostream>
#include <string>
#include <unordered_map>

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
   * Computes the bounds of all the clusters in the tree, i.e. for each cluster
   * the interval [t0,t1] which is the union of all elements in it.
   *
   * The bounds are stored in an unordered map with the pointer to the cluster
   * as key. They are created by traversing the tree recursively.
   * @param[in] root  Current cluster in the tree traversal.
   * @param[in, out] cluster_bounds   Map in which the bounds are stored.
   */
  void compute_cluster_bounds( time_cluster & root,
    std::unordered_map< time_cluster *, std::pair< sc, sc > > & cluster_bounds )
    const;

  /**
   * Creates a vector of cluster bounds sorted according to the tree format and
   * writes it to file. For each cluster two successive floating point numbers
   * are contained in the vector.
   * @param[in] filename  Name of the (binary!) file in which the bounds are
   *                      stored.
   */
  void print_cluster_bounds( const std::string filename ) const;

  /**
   * Computes the tree structure and prints it to a binary file
   * @param[in] filename Name of the output file
   */
  void print_tree_structure( const std::string filename ) const {
    write_vector_to_bin_file( compute_tree_structure( ), filename );
  }

  /**
   * Assigns clusters of the tree to a given number of processes.
   * Only those clusters whose level is less or equal than the level of the
   * earliest leaf and greater than 1 are assigned to processes. The clusters
   * are distributed levelwise:
   *  - For levels where the number of clusters is larger than the number of
   *    processes the clusters are distributed as uniformly as possible (each
   *    process gets the same amount of clusters +/- 1) with the constraint that
   *    if a cluster is assigned to a certain process at least one of its
   *    children has to be assigned to the same process.
   *  - Clusters at the first level where the number of processes is larger than
   *    the number of clusters are assigned to the same process as their left
   *    child.
   *  - Clusters above this level are distributed according to one of three
   *    strategies:
   *    * Strategy 0: Assign clusters to those processes which handle the fewest
   *      clusters at higher levels.
   *    * Strategy 1: Similar to 0, but split the processes into groups for each
   *      level and pick a process from each group (underlying idea: pick the
   *      process with the fewest clusters in the subtree).
   *    * Strategy 2: Always assign a cluster to the same process which handles
   *      its left child.
   *
   * @param[in] n_processes Number of processes.
   * @param[in] strategy  Indicates which strategy is used (default 2).
   * @param[in,out] status  Indicates if the process assignment was successful
   *                        (status 0). If the temporal tree has less than 2
   *                        levels, status is set to 1 (too coarse!). If there
   *                        are too many processes for the assignment, status is
   *                        set to 2.
   * \return  A vector representing the distribution of the clusters. It
   *          contains the process ids according to the tree format.
   */
  std::vector< lo > compute_process_assignments(
    const lo n_processes, const lo strategy, lo & status ) const;

  /**
   * Computes the process assignments and prints them to a binary file
   * @param[in] n_processes Number of processes used for the assignment.
   * @param[in] strategy  Value between 0 and 2 indicating one of three
   *                      strategies to assign the processes. See documentation
   *                      of @ref compute_process_assignments for details.
   * @param[in] filename Name of the output file.
   * @param[out] status  Indicates if the process assignment was successful
   *                     (status 0). If the temporal tree has less than 2
   * levels, status is set to 1 (too coarse!). If there are too many processes
   * for the assignment, status is set to 2.
   */
  void print_process_assignments( const lo n_processes, const lo strategy,
    const std::string filename, lo & status ) const {
    write_vector_to_bin_file(
      compute_process_assignments( n_processes, strategy, status ), filename );
  }

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
    for ( lou i = 0; i < _paddings.size( ); ++i ) {
      std::cout << _paddings[ i ] << " ";
    }
    std::cout << std::endl;
  }

 private:
  time_cluster * _root;         //!< root cluster of the tree
  const temporal_mesh & _mesh;  //!< underlying mesh
  lo _levels;                   //!< number of levels in the tree
  lo _real_n_levels;  //!< auxiliary value to determine number of real tree
                      //!< levels (depending on _n_min_elems)
  lo _n_min_elems;  //!< minimum number of elements so that cluster can be split
                    //!< in halves
                    //!< @todo: can we rename this? the name is somehow
                    //!< counterintuitive
  lo _n_max_elems_leaf;  //!< maximal number of elements in a leaf cluster after
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
   * @note This method is supposed to be called by @ref compute_tree_structure.
   */
  void tree_2_vector(
    const time_cluster & root, std::vector< char > & tree_vector ) const;

  /**
   * Takes a process assignment vector created in the routine
   * @ref compute_process_assignments and converts it into a vector of process
   * assignments in the format, which is used to represent the tree structure.
   * This is done by traversing the tree recursively.
   * @param[in] root  Current cluster in the recursion.
   * @param[in] levelwise_assignment  Process assignment in the original format
   * @param[in] thresh_level  First level in the tree which is larger or equal
   *                          then the number of processes.
   * @param[in] trunc_level Level of the earliest leaf cluster in the tree.
   * @param[in] n_processes Number of processes for the assignment.
   * @param[in] my_id Process id of the cluster which calls the routine. This is
   *                  used to assign the same process id to the children of a
   *                  cluster at levels below @p trunc_level.
   * @param[in,out] assigned_clusters Counters for each level to keep track of
   *                                  the number of clusters which have been
   *                                  assigned.
   * @param[in,out] process_pointers  Auxiliary vector for the assignment of
   *                                  processes to clusters starting from
   *                                  trunc_level. Contains an index for each
   *                                  such level which indicates the process
   *                                  which was last assigned.
   * @param[in,out] process_assignment  Vector of process assignments in the
   *                                    tree structure format.
   * @note This method is solely used by @ref compute_process_assignments.
   */
  void convert_assignment_vector_2_tree_format( const time_cluster & root,
    const std::vector< lo > & levelwise_assignment, const lo thresh_level,
    const lo trunc_level, const lo n_processes, const lo my_id,
    std::vector< lo > & assigned_clusters, std::vector< lo > & process_pointers,
    std::vector< lo > & process_assignment ) const;

  /**
   * Traverses the cluster tree recursively and adds cluster bounds from the
   * unordered map to a vector sorted according to the tree format.
   * @param[in] root  Current non-leaf cluster in the tree traversal.
   * @param[in] bounds_map  Unordered map containing the bounds of all clusters.
   *                        The bounds of a cluster are given as pair of
   *                        floating point values and are accessed by the
   *                        pointer to the cluster.
   * @param[in,out] bounds_vector Vector to which the bounds are written
   *                              according to the order of the tree format.
   * @note This function is only called by @ref print_cluster_bounds.
   * @warning It is assumed, that the cluster tree is a full binary tree, i.e.
   * no cluster has just one child. This is currently guaranteed by the
   * construction of the tree.
   */
  void convert_cluster_bounds_map_2_tree_format( const time_cluster & root,
    const std::unordered_map< time_cluster *, std::pair< sc, sc > > &
      bounds_map,
    std::vector< sc > & bounds_vector ) const;

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
