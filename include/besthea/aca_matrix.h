/*
Copyright (c) 2022, VSB - Technical University of Ostrava and Graz University of
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

/** @file aca_matrix.h
 * @brief Contains class of matrices compressed by ACA and corresponding
 * routines.
 */

#ifndef INCLUDE_BESTHEA_ACA_MATRIX_H_
#define INCLUDE_BESTHEA_ACA_MATRIX_H_

#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/low_rank_kernel.h"
#include "besthea/settings.h"

namespace besthea {
  namespace linear_algebra {
    class aca_matrix;
  }
}

/**
 * Class for evaluation adaptive cross approximation method
 * for compression of full matrices. Collection of matrix blocks (approximated
 * or not)
 */
class besthea::linear_algebra::aca_matrix {
  using full_matrix_type = besthea::linear_algebra::full_matrix;
  using vector_type = besthea::linear_algebra::vector;

 public:
  /**
   * Construct a new aca matrix object
   *
   * @param[in] eps Relative accuracy of the ACA method.
   * @param[in] max_rank Maximum rank of the produced ACA blocks.
   *
   */
  aca_matrix( sc eps, lo max_rank );

  /**
   * Destructor.
   */
  ~aca_matrix( ) {
    for ( lou i = 0; i < _u.size( ); ++i ) {
      delete _u[ i ];
      delete _v[ i ];
      delete _y_tmp_1[ i ];
      delete _y_tmp_2[ i ];
    }
    for ( lou i = 0; i < _full.size( ); ++i ) {
      delete _full[ i ];
    }
  }

  /**
   * Adds approximated matrix to the list of matrices.
   *
   * @param[in] matrix Full matrix block to be approximated
   * @param[in] source Pointer to the corresponding source cluster.
   * @param[in] target Pointer to the corresponding target cluster.
   */
  void add_aca_matrix( full_matrix_type & matrix,
    mesh::general_spacetime_cluster * source,
    mesh::general_spacetime_cluster * target,
    sc svd_recompression_reference_value = -1.0 );

  /**
   * Adds full matrix to the list of matrices.
   *
   * @param[in] matrix Full matrix block to be approximated
   * @param[in] source Pointer to the corresponding source cluster.
   * @param[in] target Pointer to the corresponding target cluster.
   */
  void add_full_matrix( full_matrix_type & matrix,
    mesh::general_spacetime_cluster * source,
    mesh::general_spacetime_cluster * target );

  /**
   * @brief Applies the stored ACA block with a given index
   * y = beta * y + alpha * (this)[block]^trans * x.
   *
   * @param block Index of the block to be applied
   * @param x
   * @param y
   * @param trans
   * @param alpha
   * @param beta
   */
  void apply_aca_block( lou block, const vector_type & x, vector_type & y,
    bool trans, sc alpha, sc beta ) const;

  /**
   * @brief Applies the stored full block with a given index
   * y = beta * y + alpha * (this)[block]^trans * x.
   *
   * @param block Index of the block to be applied
   * @param x
   * @param y
   * @param trans
   * @param alpha
   * @param beta
   */
  void apply_full_block( lou block, const vector_type & x, vector_type & y,
    bool trans, sc alpha, sc beta ) const;

  /**
   * Returns the number of approximated blocks stored in the matrix.
   */
  lou get_n_approximated_clusters( ) const {
    return _u.size( );
  }

  /**
   * Returns the number of full blocks stored in the matrix.
   */
  lou get_n_full_clusters( ) const {
    return _full.size( );
  }

  /**
   * Returns the source and target clusters associated with a certain
   * approximated matrix block whose index is given.
   * @param[in] block_index Index of the approximated block.
   * @param[in,out] source Pointer to the source cluster
   * @param[in,out] target Pointer to the target cluster
   */
  void get_approximated_clusters( lou block_index,
    mesh::general_spacetime_cluster ** source,
    mesh::general_spacetime_cluster ** target ) {
    if ( block_index < _sources_aca.size( ) ) {
      *source = _sources_aca[ block_index ];
      *target = _targets_aca[ block_index ];
    }
  }

  /**
   * Returns full clusters associated with a given index.
   *
   * @param[in] block_index Index of the full block.
   * @param[in,out] source Pointer to the source cluster
   * @param[in,out] target Pointer to the target cluster
   */
  void get_full_clusters( lou block_index,
    mesh::general_spacetime_cluster ** source,
    mesh::general_spacetime_cluster ** target ) {
    if ( block_index < _sources_full.size( ) ) {
      *source = _sources_full[ block_index ];
      *target = _targets_full[ block_index ];
    }
  }

  /**
   * Returns the compression ratio of the matrix
   */

  sc get_compression( ) const {
    return (sc) _compressed_size / (sc) _full_size;
  }

  /**
   * Returns the total size of the data associated with the compressed
   * blocks (including those that could not be approximated).
   */
  lo get_compressed_size( ) const {
    return _compressed_size;
  }

  /**
   * Returns the theoretical total size of the matrix when all blocks are
   * stored as full matrices.
   */
  lo get_full_size( ) const {
    return _full_size;
  }

 private:
  // todo: move aca routines to assembler (or add them to a separate friend
  // class?)

  /**
   * TODO: comment me
   */
  template< class T >
  bool compute_low_rank_block( lo row_dim, lo col_dim, full_matrix & u,
    full_matrix & v, besthea::bem::low_rank_kernel< T > & kernel, lo & rank,
    bool enable_svd_recompression = false,
    sc svd_recompression_reference_value = -1.0 );

  /**
   * TODO: comment me
   */
  template< class T >
  int compute_aca_block( lo row_dim, lo col_dim, full_matrix & u,
    full_matrix & v, besthea::bem::low_rank_kernel< T > & kernel,
    bool enable_svd_recompression, sc svd_recompression_reference_value,
    sc & eps, lo & rank );

  /**
   * Tries to recompress a given low rank matrix via an SVD and truncation.
   * @param[in] u Matrix U of the low rank matrix U*V^T.
   * @param[in] v Matrix V of the low rank matrix U*V^T.
   * @param[in] rows  Number of rows of the low rank matrix.
   * @param[in] cols  Number of columns of the low rank matrix.
   * @param[in] svd_recompression_reference_value Reference value used to decide
   * which singular values are truncated. If a value < 0 is provided, the
   * largest singular value of the low rank matrix is used instead.
   * @param[out] rank New rank after recompression.
   */
  void recompression( full_matrix & u, full_matrix & v, const lo rows,
    const lo cols, sc svd_recompression_reference_value, lo & rank );

  std::vector< mesh::general_spacetime_cluster * >
    _sources_aca;  //!< source clusters associated with the appropriate aca
                   //!< matrices
  std::vector< mesh::general_spacetime_cluster * >
    _targets_aca;  //!< target cluster associated with the appropriate aca
                   //!< matrices
  std::vector< mesh::general_spacetime_cluster * >
    _sources_full;  //!< source clusters associated with the appropriate full
                    //!< matrices
  std::vector< mesh::general_spacetime_cluster * >
    _targets_full;  //!< target cluster associated with the appropriate full
                    //!< matrices
  std::vector< full_matrix_type * > _u;  //!< ACA matrices U
  std::vector< full_matrix_type * > _v;  //!< ACA matrices V
  std::vector< full_matrix_type * >
    _full;       //!< Full matrices stored if no compression is possible.
  sc _eps;       //!< ACA accuracy
  lo _max_rank;  //!< maximum allowed rank
  mutable std::vector< vector_type * > _y_tmp_1;  //!< temporary vector
  mutable std::vector< vector_type * > _y_tmp_2;  //!< temporary vector
  lo _full_size;        //!< possible size of nonapproximated matrix
  lo _compressed_size;  //!< size after compression
};

#endif  // INCLUDE_BESTHEA_ACA_MATRIX_H_
