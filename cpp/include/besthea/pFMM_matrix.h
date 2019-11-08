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

//#include "besthea/block_matrix.h"
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
  : public besthea::linear_algebra::matrix {
 public:
  using sparse_matrix_type = besthea::linear_algebra::sparse_matrix;
  using spacetime_tree_type = besthea::mesh::spacetime_cluster_tree;
  using time_cluster_type = besthea::mesh::time_cluster;

  /**
   * Default constructor.
   */
  pFMM_matrix( )
    : _spacetime_tree( nullptr ), _temp_order( 5 ), _space_order( 5 ) {
  }

  pFMM_matrix( const pFMM_matrix & that ) = delete;

  /**
   * Destructor
   */
  virtual ~pFMM_matrix( ) {
    for ( auto it = _nearfield_matrices.begin( );
          it != _nearfield_matrices.end( ); ++it ) {
      if ( *it != nullptr ) {
        delete *it;
      }
    }
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const vector_type & x, vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const {
  }

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
  void resize( lo n_rows, lo n_cols ) {
    _n_rows = n_rows;
    _n_columns = n_cols;
  }

  sparse_matrix_type * create_nearfield_matrix( lo test_idx, lo trial_idx );

 private:
  spacetime_tree_type * _spacetime_tree;  //!< tree hierarchically decomposing
                                          //!< spatial and temporal domains
  std::vector< sparse_matrix_type * >
    _nearfield_matrices;  //!< temporal nearfield blocks
  std::vector< std::pair< lo, lo > >
    _nearfield_block_map;  //!< mapping from block index to pair of matching
                           //!< temporal clusters
  lo _temp_order;   //!< degree of interpolation polynomials in time for pFMM
  lo _space_order;  //!< degree of truncated Chebyshev expansion
};

#endif /* INCLUDE_BESTHEA_PFMM_MATRIX_H_ */
