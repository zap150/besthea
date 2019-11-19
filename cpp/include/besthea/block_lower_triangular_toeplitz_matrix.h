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

/** @file block_lower_triangular_toeplitz_matrix.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_BLOCK_LOWER_TRIANGULAR_TOEPLITZ_MATRIX_H_
#define INCLUDE_BESTHEA_BLOCK_LOWER_TRIANGULAR_TOEPLITZ_MATRIX_H_

#include "besthea/block_matrix.h"
#include "besthea/full_matrix.h"

namespace besthea {
  namespace linear_algebra {
    class block_lower_triangular_toeplitz_matrix;
  }
}

/**
 *  Class representing a block lower triangular Toeplitz matrix.
 */
class besthea::linear_algebra::block_lower_triangular_toeplitz_matrix
  : public besthea::linear_algebra::block_matrix {
 public:
  using matrix_type = besthea::linear_algebra::full_matrix;  //!< Matrix type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Default constructor.
   */
  block_lower_triangular_toeplitz_matrix( );

  block_lower_triangular_toeplitz_matrix(
    const block_lower_triangular_toeplitz_matrix & that )
    = delete;

  /**
   * Constructor with an initializer list.
   * @param[in] block_dim Number of blocks per row (column).
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   * @param[in] list Initializer list for full_matrix.
   */
  block_lower_triangular_toeplitz_matrix(
    lo block_dim, lo n_rows, lo n_columns, std::initializer_list< sc > list );

  /**
   * Constructing a matrix of the given dimension.
   * @param[in] block_dim Number of blocks per row (column).
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   * @param[in] zero Initialize to 0 if true.
   */
  block_lower_triangular_toeplitz_matrix(
    lo block_dim, lo n_rows, lo n_columns, bool zero = true );

  /**
   * Destructor.
   */
  virtual ~block_lower_triangular_toeplitz_matrix( );

  /**
   * Returns a reference to a single block.
   * @param[in] d Index of the block.
   */
  matrix_type & get_block( lo d ) {
    return _data[ d ];
  }

  /**
   * Returns a reference to a single block.
   * @param[in] d Index of the block.
   */
  const matrix_type & get_block( lo d ) const {
    return _data[ d ];
  }

  /*!
   * @brief Returns the (i,j)-th element of the matrix.
   * @param[in] d Index of the block.
   * @param[in] i Row index.
   * @param[in] j Column index.
   */
  sc get( lo d, lo i, lo j ) const {
    return _data[ d ].get( i, j );
  }

  /*!
   * @brief Sets the (i,j)-th element of the matrix.
   * @param[in] d Index of the block.
   * @param[in] i Row index.
   * @param[in] j Column index.
   * @param[in] value Value to be set.
   */
  void set( lo d, lo i, lo j, sc value ) {
    _data[ d ].set( i, j, value );
  }

  /*!
   * @brief Adds value to the (i,j)-th element of the matrix.
   * @param[in] d Index of the block.
   * @param[in] i Row index.
   * @param[in] j Column index.
   * @param[in] value Value to be set.
   */
  void add( lo d, lo i, lo j, sc value ) {
    _data[ d ].add( i, j, value );
  }

  /*!
   * @brief Atomically adds value to the (i,j)-th element of the matrix.
   * @param[in] d Index of the block.
   * @param[in] i Row index.
   * @param[in] j Column index.
   * @param[in] value Value to be set.
   */
  void add_atomic( lo d, lo i, lo j, sc value ) {
    _data[ d ].add_atomic( i, j, value );
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
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
   * @brief In-place Cholesky decomposition of the first block and solution.
   * @param[in,out] rhs Right-hand side overwritten by the result.
   */
  void cholesky_decompose_solve( block_vector_type & rhs );

  /*!
   * @brief In-place Cholesky decomposition of the first block.
   */
  void cholesky_decompose( );

  /*!
   * @brief Cholesky solution
   * @param[in,out] rhs Right-hand side overwritten by the result.
   */
  void cholesky_solve( block_vector_type & rhs );

  /*!
   * @brief Prints the matrix.
   * @param[in] stream
   */
  void print( std::ostream & stream = std::cout ) const;

  /**
   * Prints info on the object.
   */
  void print_info( ) const {
    std::cout
      << "besthea::linear_algebra::block_lower_triangular_toeplitz_matrix"
      << std::endl;
    std::cout << "  number of blocks: " << _data.size( ) << std::endl;
    std::cout << "  dimension of each block: " << _data[ 0 ].get_n_rows( )
              << " x " << _data[ 0 ].get_n_columns( ) << std::endl;
  }

  /*!
   * Resizes all matrix blocks.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   */
  void resize_blocks( lo n_rows, lo n_columns );

  /*!
   * Resizes the block dimension.
   * @param[in] block_dim Number of blocks per row (column).
   */
  void resize( lo block_dim );

 protected:
  std::vector< matrix_type > _data;  //!< Raw data.
};

#endif /* INCLUDE_BESTHEA_BLOCK_LOWER_TRIANGULAR_TOEPLITZ_MATRIX_H_ */
