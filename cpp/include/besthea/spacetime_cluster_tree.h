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

/** @file spacetime_cluster_tree.h
 * @brief Product of spatial and temporal clusters.
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_SPACETIME_CLUSTER_TREE_H_

#include "besthea/chebyshev_evaluator.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/space_cluster_tree.h"
#include "besthea/spacetime_cluster.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/time_cluster_tree.h"
#include "besthea/vector.h"

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
  using vector_type = besthea::linear_algebra::vector;
  spacetime_cluster_tree( const spacetime_tensor_mesh & spacetime_mesh,
    lo time_levels, lo n_min_time_elems, lo n_min_space_elems, sc st_coeff );

  ~spacetime_cluster_tree( ) {
    if ( _space_tree )
      delete _space_tree;
    if ( _time_tree )
      delete _time_tree;
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
    print_internal( _root );
  }

  spacetime_cluster * get_root( ) {
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

 private:
  using full_matrix_type = besthea::linear_algebra::full_matrix;
  std::vector< full_matrix_type >
    _m2m_matrices_t_left;  //! left temporal
                           //! m2m matrices stored levelwise
  std::vector< full_matrix_type >
    _m2m_matrices_t_right;  //! right temporal
                            //! m2m matrices stored levelwise
  lo _temp_order = 5;  //! degree of interpolation polynomials in time for pFMM
  std::vector< vector_type >
    _m2m_coeffs_s_dim_0_left;  //! left spatial
                               //! m2m matrices along dimension 0 stored
                               //! levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_0_right;  //! right spatial
                                //! m2m matrices along dimension 0 stored
                                //! levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_1_left;  //! left spatial
                               //! m2m matrices along dimension 1 stored
                               //! levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_1_right;  //! right spatial
                                //! m2m matrices along dimension 1 stored
                                //! levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_2_left;  //! left spatial
                               //! m2m matrices along dimension 2 stored
                               //! levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_2_right;  //! right spatial
                                //! m2m matrices along dimension 2 stored
                                //! levelwise
  lo _spat_order = 5;  //! degree of Chebyshev polynomials for expansion in
                       //! space in pFMM

  // TODO relocate the above members to pFMM matrix
  // TODO substitute hardcoded _temp_order and _spatial_order with settable ones

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
  sc _s_t_coeff;  //!< coefficient to determine the coupling of the spatial
                  //!< and temporal levels

  /**
   * Auxiliary method to get all spatial clusters on a given level
   */
  void get_space_clusters_on_level(
    space_cluster * root, lo level, std::vector< space_cluster * > & clusters );

  /**
   * Recursively builds the tree of spacetime clusters
   */
  void build_tree( spacetime_cluster * root, lo level, bool split_space );

  /*
   * Aux for printing
   */
  void print_internal( spacetime_cluster * root ) {
    if ( root->get_level( ) == -1 ) {
      // print temporal m2m matrices
      lo n_levels = _time_tree->get_levels( );
      std::cout << "printing exemplarily temporal m2m matrices" << std::endl;
      for ( lo curr_lev = 2; curr_lev < n_levels; ++curr_lev ) {
        std::cout << "printing m2m matrix on level " << curr_lev << std::endl;
        for ( lo j = 0; j <= _temp_order; ++j ) {
          for ( lo k = 0; k <= _temp_order; ++k )
            printf( "%.4f ", _m2m_matrices_t_right[ curr_lev ]( j, k ) );
          std::cout << std::endl;
        }
      }
      // print spatial m2m coefficients
      n_levels = _space_tree->get_levels( );
      std::cout << "printing exemplarily spatial m2m coefficients" << std::endl;
      for ( lo curr_lev = 0; curr_lev < n_levels; ++curr_lev ) {
        std::cout << "printing m2m coeffs on level " << curr_lev << std::endl;
        std::cout << "curr_lev = " << curr_lev << std::endl;
        for ( lo j = 0; j <= _spat_order; ++j ) {
          for ( lo k = 0; k <= _spat_order; ++k )
            printf( "(%ld, %ld): %.4f ", (long) j, (long) k,
              _m2m_coeffs_s_dim_0_left[ curr_lev ]
                                      [ j * ( _spat_order + 1 ) + k ] );
          std::cout << std::endl;
        }
      }
    }

    root->print( );
    std::vector< spacetime_cluster * > * children = root->get_children( );
    // std::cout << children->size( ) << std::endl;
    if ( children != nullptr )
      for ( auto it = children->begin( ); it != children->end( ); ++it ) {
        for ( lo i = 0; i < ( *it )->get_level( ); ++i ) std::cout << " ";
        print_internal( *it );
      }
  }

  /*
   * Compute the temporal m2m matrices for all levels.
   */
  void set_temporal_m2m_matrices( );

  /*
   * Compute the spatial m2m coefficients for all levels.
   */
  void set_spatial_m2m_coeffs( );

  /*
   * Apply the temporal m2m operation for given parent and child moments
   */
  void apply_temporal_m2m( full_matrix_type const & child_moment,
    const lo level, const bool is_left_child,
    full_matrix_type & parent_moment );

  /*
   * Apply the spatial m2m operation for given parent and child moments
   */
  void apply_spatial_m2m( full_matrix_type const & child_moment, const lo level,
    const slou octant, full_matrix_type & parent_moment );
};

#endif /* INCLUDE_BESTHEA_SPACETIME_CLUSTER_TREE_H_ */
