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

/** @file pFMM_matrix.h
 * @brief Represents matrix approximated by the pFMM method
 */

#ifndef INCLUDE_BESTHEA_PFMM_MATRIX_H_
#define INCLUDE_BESTHEA_PFMM_MATRIX_H_

#include "besthea/block_linear_operator.h"
#include "besthea/matrix.h"
#include "besthea/settings.h"
#include "besthea/spacetime_cluster_tree.h"
#include "besthea/sparse_matrix.h"
#include "besthea/time_cluster.h"
#include "besthea/vector.h"

#include <utility>

namespace besthea {
  namespace linear_algebra {
    class pFMM_matrix;
  }
}

/**
 * Class representing a matrix approximated by the pFMM method.
 */
class besthea::linear_algebra::pFMM_matrix
  : public besthea::linear_algebra::block_linear_operator {
 public:
  using sparse_matrix_type
    = besthea::linear_algebra::sparse_matrix;  //!< Sparse matrix type.
  using spacetime_tree_type
    = besthea::mesh::spacetime_cluster_tree;  //!< Spacetime tree type.
  using time_cluster_type
    = besthea::mesh::time_cluster;  //!< Time cluster type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Block vector type.

  /**
   * Default constructor.
   */
  pFMM_matrix( )
    : _spacetime_tree( nullptr ),
      _uniform( false ),
      _temp_order( 5 ),
      _spat_order( 5 ) {
  }

  pFMM_matrix( const pFMM_matrix & that ) = delete;

  /**
   * Destructor
   */
  virtual ~pFMM_matrix( ) {
    lo matrix_idx = 0;
    for ( auto it = _nearfield_matrices.begin( );
          it != _nearfield_matrices.end( ); ++it ) {
      const std::pair< lo, lo > & indices
        = _nearfield_block_map.at( matrix_idx );
      if ( *it != nullptr && indices.second == 0 ) {
        delete *it;
      }
      matrix_idx++;
    }

    for ( auto it = _farfield_matrices.begin( );
          it != _farfield_matrices.end( ); ++it ) {
      if ( *it != nullptr ) {
        delete *it;
      }
    }
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x using block vectors.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const block_vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * Sets the underlying spacetime tree.
   * @param[in] spacetime_tree The tree.
   */
  void set_tree( spacetime_tree_type * spacetime_tree ) {
    _spacetime_tree = spacetime_tree;
  }

  /*!
   * Sets the dimension of the matrix.
   * @param[in] n_rows Number of rows.
   * @param[in] n_cols Number of columns.
   */
  void resize( lo block_dim, lo dim_domain, lo dim_range ) {
    this->_block_dim = block_dim;
    this->_dim_domain = dim_domain;
    this->_dim_range = dim_range;
  }

  /**
   * Creates a nearfield matrix.
   * @param[in] test_idx Index of test interval.
   * @param[in] trial_idx Index of trial interval.
   * @param[in] n_duplications Number of matrix duplications for uniform-time
   * assembly.
   */
  sparse_matrix_type * create_nearfield_matrix(
    lo test_idx, lo trial_idx, lo n_duplications = 1 );

  /*!
   * Allocates sparse matrix of given farfield nonapproximated block and
   * returns a pointer.
   * @param[in] test_idx Index of the testing function.
   * @param[in] trial_idx Index of the trial function
   */
  sparse_matrix_type * create_farfield_matrix( lo test_idx, lo trial_idx );

 private:
  spacetime_tree_type * _spacetime_tree;  //!< tree hierarchically decomposing
                                          //!< spatial and temporal domains
  std::vector< sparse_matrix_type * >
    _nearfield_matrices;  //!< temporal nearfield blocks
  std::vector< std::pair< lo, lo > >
    _nearfield_block_map;  //!< mapping from block index to pair of matching
                           //!< temporal clusters
  std::vector< sparse_matrix_type * >
    _farfield_matrices;  //!< nonapproximated temporal farfield blocks
  std::vector< std::pair< lo, lo > >
    _farfield_block_map;  //!< mapping from block index to pair of matching
                          //!< temporal clusters
  bool _uniform;          //!< specifies whether time-discretization is uniform
                          //!< (duplicates blocks)
  using full_matrix_type = besthea::linear_algebra::full_matrix;
  std::vector< full_matrix_type >
    _m2m_matrices_t_left;  //! left temporal
                           //! m2m matrices stored levelwise
  std::vector< full_matrix_type >
    _m2m_matrices_t_right;  //! right temporal
                            //! m2m matrices stored levelwise

  lo _temp_order;  //! degree of interpolation polynomials in time for pFMM
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
  lo _spat_order;  //! degree of Chebyshev polynomials for expansion in
                   //! space in pFMM
};

#endif /* INCLUDE_BESTHEA_PFMM_MATRIX_H_ */
