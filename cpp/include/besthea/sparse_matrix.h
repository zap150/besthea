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

/** @file sparse_matrix.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_SPARSE_MATRIX_H_
#define INCLUDE_BESTHEA_SPARSE_MATRIX_H_

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "besthea/matrix.h"
#include "besthea/settings.h"

#include <vector>

namespace besthea {
  namespace linear_algebra {
    class sparse_matrix;
  }
}

/**
 *  Class representing a sparse matrix.
 */
class besthea::linear_algebra::sparse_matrix
  : public besthea::linear_algebra::matrix {
  using vector = besthea::linear_algebra::vector;

 public:
  /**
   * Default constructor.
   */
  sparse_matrix( );

  /**
   * Copy constructor.
   * @param[in] that Matrix to be deep copied.
   */
  sparse_matrix( const sparse_matrix & that );

  /**
   * Constructs sparse matrix from triplets.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   * @param[in] row_indices Indices of rows.
   * @param[in] column_indices Indices of columns.
   * @param[in] values Values to be stored at positions specified by
   * `row_indices` and `column_indices`.
   */
  sparse_matrix( los n_rows, los n_columns, std::vector< los > & row_indices,
    std::vector< los > & column_indices, std::vector< sc > & values );

  virtual ~sparse_matrix( );

  /**
   * Sets the sparse matrix from triplets.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   * @param[in] row_indices Indices of rows.
   * @param[in] column_indices Indices of columns.
   * @param[in] values Values to be stored at positions specified by
   * `row_indices` and `column_indices`.
   */
  void set_from_triplets( los n_rows, los n_columns,
    std::vector< los > & row_indices, std::vector< los > & column_indices,
    std::vector< sc > & values );

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( vector const & x, vector & y, bool trans = false,
    sc alpha = 1.0, sc beta = 0.0 ) const;

  /**
   * Prints the triplets.
   * @param[in] stream Stream to print to.
   */
  void print( std::ostream & stream = std::cout ) const {
    using iterator_type =
      typename Eigen::SparseMatrix< sc, Eigen::ColMajor, los >::InnerIterator;

    for ( Eigen::Index k = 0; k < _data.outerSize( ); ++k )
      for ( iterator_type it( _data, k ); it; ++it ) {
        std::cout << it.row( ) << " ";  // row index
        std::cout << it.col( ) << " ";  // col index (here it is equal to k)
        std::cout << it.value( ) << std::endl;
      }
  }

 protected:
  Eigen::SparseMatrix< sc, Eigen::ColMajor, los > _data;  //!< Eigen data.
};

#endif /* INCLUDE_BESTHEA_SPARSE_MATRIX_H_ */
