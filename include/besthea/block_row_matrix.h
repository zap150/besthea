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

/** @file block_row_matrix.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_BLOCK_ROW_MATRIX_H_
#define INCLUDE_BESTHEA_BLOCK_ROW_MATRIX_H_

#include "besthea/block_row_linear_operator.h"
#include "besthea/full_matrix.h"
#include "besthea/settings.h"

namespace besthea {
  namespace linear_algebra {
    class block_row_matrix;
  }
}

/**
 *  Class representing a full matrix.
 */
class besthea::linear_algebra::block_row_matrix
  : public besthea::linear_algebra::block_row_linear_operator {
 public:
  using matrix_type = besthea::linear_algebra::full_matrix;  //!< Matrix type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Default constructor.
   */
  block_row_matrix( );

  /**
   * Constructor.
   * @param[in] block_dim Block dimension.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   * @param[in] zero Sets all to zero.
   */
  block_row_matrix( lo block_dim, lo n_rows, lo n_columns, bool zero = true );

  /**
   * Constructor.
   * @param[in] block_dim Block dimension.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   * @param[in] list Initializer list for full_matrix.
   */
  block_row_matrix(
    lo block_dim, lo n_rows, lo n_columns, std::initializer_list< sc > list );

  /**
   * Destructor.
   */
  virtual ~block_row_matrix( ) {
  }

  /**
   * Returns number of blocks in a row (column).
   */
  lo get_block_dim( ) const {
    return _block_dim;
  }

  /**
   * Returns number of rows.
   */
  lo get_n_rows( ) const {
    return _n_rows;
  }

  /**
   * Returns number of columns.
   */
  lo get_n_columns( ) const {
    return _n_columns;
  }

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
   * @brief Prints the matrix.
   * @param[in] stream
   */
  void print( std::ostream & stream = std::cout ) const;

  /**
   * Prints info on the object.
   */
  void print_info( ) const {
    std::cout << "besthea::linear_algebra::block_row_matrix" << std::endl;
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

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const override;

 protected:
  lo & _n_rows{
    block_row_linear_operator::_dim_range
  };  //!< number of rows (range dimension)

  lo & _n_columns{
    block_row_linear_operator::_dim_domain
  };  //!< number of columns (domain dimension)

  std::vector< matrix_type > _data;  //!< Raw data.
};

#endif /* INCLUDE_BESTHEA_BLOCK_ROW_MATRIX_H_ */
