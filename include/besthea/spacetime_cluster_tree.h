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

/** @file spacetime_cluster_tree.h
 * @brief Product of spatial and temporal clusters.
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_SPACETIME_CLUSTER_TREE_H_

#include "besthea/block_vector.h"
#include "besthea/full_matrix.h"
#include "besthea/space_cluster_tree.h"
#include "besthea/spacetime_cluster.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/time_cluster_tree.h"
#include "besthea/vector.h"

#include <map>
#include <vector>

namespace besthea {
  namespace mesh {
    class spacetime_cluster_tree;
  }
}

/**
 * Class representing a product of spatial and temporal trees.
 */
class besthea::mesh::spacetime_cluster_tree {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Constructor.
   * @param[in] spacetime_mesh Spacetime mesh.
   * @param[in] time_levels Number of bisections in time.
   * @param[in] n_min_time_elems Minimum number of temporal elements in leafs.
   * @param[in] n_min_space_elems Minimum number of spatial elements in leafs.
   * @param[in] st_coeff Coefficient to determine the coupling of the spatial
   * and temporal levels.
   * @param[in] spatial_nearfield_limit Number of clusters in the vicinity of a
   * given clusters to be considered as nearfield
   */
  spacetime_cluster_tree( const spacetime_tensor_mesh & spacetime_mesh,
    lo time_levels, lo n_min_time_elems, lo n_min_space_elems, sc st_coeff,
    slou spatial_nearfield_limit = 3 );

  /**
   * Destructor.
   */
  ~spacetime_cluster_tree( ) {
    if ( _space_tree )
      delete _space_tree;
    if ( _time_tree )
      delete _time_tree;
    if ( _root ) {
      delete _root;
    }
  }

  /**
   * Returns the underlying time cluster.
   */
  time_cluster_tree * get_time_cluster_tree( ) {
    return _time_tree;
  }

  /**
   * Returns the underlying space cluster.
   */
  space_cluster_tree * get_space_cluster_tree( ) {
    return _space_tree;
  }

  /**
   * Prints levels of the tree.
   */
  void print( ) {
    // print cluster information recursively
    print_internal( _root );
    // print general tree information
    std::cout << "number of levels of spacetime tree " << _levels << std::endl;
    std::cout << "number of levels of space tree " << _space_tree->get_levels( )
              << std::endl;
    std::cout << "number of levels of time tree " << _time_tree->get_levels( )
              << std::endl;
    std::cout << "_start_spatial_level is " << _start_spatial_level
              << std::endl;
    std::cout << "_start_temporal_level is " << _start_temporal_level
              << std::endl;
    // print vector of paddings in time and space
    const std::vector< sc > time_paddings = _time_tree->get_paddings( );
    std::cout << "padding in time (level-wise): " << std::endl;
    for ( lou i = 0; i < time_paddings.size( ); ++i ) {
      std::cout << time_paddings[ i ] << " ";
    }
    std::cout << std::endl;
    const std::vector< sc > space_paddings = _space_tree->get_paddings( );
    std::cout << "padding in space (level-wise): " << std::endl;
    for ( lou i = 0; i < space_paddings.size( ); ++i ) {
      std::cout << space_paddings[ i ] << " ";
    }
    std::cout << std::endl;
  }

  /**
   * Returns a pointer to the root.
   */
  spacetime_cluster * get_root( ) {
    return _root;
  }

  /**
   * Returns a pointer to the root.
   */
  const spacetime_cluster * get_root( ) const {
    return _root;
  }

  /**
   * Returns the spacetime mesh.
   */
  const spacetime_tensor_mesh & get_mesh( ) const {
    return _spacetime_mesh;
  }

  /**
   * Returns the spatial mesh.
   */
  const triangular_surface_mesh & get_spatial_mesh( ) const {
    return _space_mesh;
  }

  /**
   * Returns the temporal mesh.
   */
  const temporal_mesh & get_temporal_mesh( ) const {
    return _time_mesh;
  }

  /**
   * Returns a pointer to the temporal tree.
   */
  time_cluster_tree * get_time_tree( ) {
    return _time_tree;
  }

  /**
   * Returns a pointer to the temporal tree.
   */
  const time_cluster_tree * get_time_tree( ) const {
    return _time_tree;
  }

  /**
   * Returns a pointer to the spatial tree.
   */
  space_cluster_tree * get_space_tree( ) {
    return _space_tree;
  }

  /**
   * Returns a pointer to the spatial tree.
   */
  const space_cluster_tree * get_space_tree( ) const {
    return _space_tree;
  }

  /**
   * Returns clusters without descendants.
   */
  std::vector< spacetime_cluster * > & get_leaves( ) {
    return _leaves;
  }

  /**
   * Returns the number of levels in the tree.
   */
  lo get_levels( ) const {
    return _levels;
  }

  /**
   * Initializes data storage for moments contribution.
   */
  void initialize_moment_contributions( spacetime_cluster * root,
    const lo & n_rows_contribution, const lo & n_columns_contribution );

  /**
   * Initializes data storage for local contribution.
   */
  void initialize_local_contributions( spacetime_cluster * root,
    const lo & n_rows_contribution, const lo & n_columns_contribution );

  /**
   * Fills the local contributions with zeros.
   */
  void clean_local_contributions( spacetime_cluster * root );

  /**
   * Fills the moment contributions with zeros.
   */
  void clean_moment_contributions( spacetime_cluster * root );

  /**
   * Recursively fills the interaction lists of clusters starting at root.
   * @param[in] root  Starting cluster for the recursion.
   */
  void determine_interactions( spacetime_cluster & root );

  /**
   * Fills the nearfield lists of all leaf clusters.
   */
  void determine_nearfield( );

 private:
  using full_matrix_type = besthea::linear_algebra::
    full_matrix;  //!< shortcut for the
                  //!< besthea::linear_algebra::full_matrix_type

  /**
   * Auxiliary method to get all spatial clusters on a given level
   */
  void get_space_clusters_on_level(
    space_cluster * root, lo level, std::vector< space_cluster * > & clusters );

  /**
   * Recursively builds the tree of spacetime clusters
   * param[in] root Current root cluster
   * param[in] level Current level
   * param[in] split_space Indicating whether to split space or only time
   */
  void build_tree( spacetime_cluster * root, lo level, bool split_space );

  /**
   * Collects all clusters without descendants and stores them in the internal
   * _leaves vector.
   * @param[in] root Root cluster of the tree.
   */
  void collect_leaves( spacetime_cluster & root );

  /**
   * Aux for printing
   */
  void print_internal( spacetime_cluster * root ) {
    //     if ( root->get_level( ) == -1 ) {
    //       // print temporal m2m matrices
    //       lo n_levels = _time_tree->get_levels( );
    //       std::cout << "printing exemplarily temporal m2m matrices" <<
    //       std::endl; for ( lo curr_lev = 2; curr_lev < n_levels; ++curr_lev )
    //       {
    //         std::cout << "printing m2m matrix on level " << curr_lev <<
    //         std::endl; for ( lo j = 0; j <= _temp_order; ++j ) {
    //           for ( lo k = 0; k <= _temp_order; ++k )
    //             printf( "%.4f ", _m2m_matrices_t_right[ curr_lev ]( j, k ) );
    //           std::cout << std::endl;
    //         }
    //       }
    //       // print spatial m2m coefficients
    //       n_levels = _space_tree->get_levels( );
    //       std::cout << "printing exemplarily spatial m2m coefficients" <<
    //       std::endl; for ( lo curr_lev = 0; curr_lev < n_levels; ++curr_lev )
    //       {
    //         std::cout << "printing m2m coeffs on level " << curr_lev <<
    //         std::endl; std::cout << "curr_lev = " << curr_lev << std::endl;
    //         for ( lo j = 0; j <= _spat_order; ++j ) {
    //           for ( lo k = 0; k <= _spat_order; ++k )
    //             printf( "(%ld, %ld): %.4f ", (long) j, (long) k,
    //               _m2m_coeffs_s_dim_0_left[ curr_lev ]
    //                                       [ j * ( _spat_order + 1 ) + k ] );
    //           std::cout << std::endl;
    //         }
    //       }
    //     }

    std::cout << "root levels: ";
    root->print( );
    std::vector< spacetime_cluster * > * children = root->get_children( );
    //     std::vector< spacetime_cluster * > * interaction_list = root->
    //     get_interaction_list( );

    //     if ( interaction_list != nullptr ) {
    //       std::cout << "nr interacting clusters: " << interaction_list->size(
    //       ) <<
    //         std::endl;
    //       std::cout << "##############################" << std::endl;
    //     }
    // std::cout << children->size( ) << std::endl;
    if ( children != nullptr )
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        for ( lo i = 0; i < ( *it )->get_level( ); ++i ) std::cout << " ";
        print_internal( *it );
      }
  }

  lo _levels;  //!< number of levels in the tree
  const spacetime_tensor_mesh & _spacetime_mesh;  //!< underlying spacetime mesh
  const triangular_surface_mesh & _space_mesh;    //!< underlying spatial mesh
  const temporal_mesh & _time_mesh;               //!< underlying temporal mesh
  time_cluster_tree * _time_tree;                 //!< temporal tree
  space_cluster_tree * _space_tree;               //!< spatial tree
  spacetime_cluster * _root;                      //!< root of the cluster tree
  lo _start_spatial_level;   //!< auxiliary variable determining the appropriate
                             //!< starting level in the space cluster tree
  lo _start_temporal_level;  //!< auxiliary variable to determine in which level
                             //!< the spatial refinement starts
                             //!< (meaningful only if _start_spatial_level = 0)
  // sc _s_t_coeff;  //!< coefficient to determine the coupling of the spatial
  //!< and temporal levels
  slou _spatial_nearfield_limit;  //!< number of the clusters in the vicinity to
  //!< be considered as nearfield

  std::vector< spacetime_cluster * >
    _leaves;  //!< vector of all clusters without descendants
  std::map< std::pair< space_cluster *, time_cluster * >, spacetime_cluster * >
    _map_to_spacetime_clusters;  //!< map from pairs of space and time clusters
                                 //!< to spacetime clusters
};

#endif /* INCLUDE_BESTHEA_SPACETIME_CLUSTER_TREE_H_ */
