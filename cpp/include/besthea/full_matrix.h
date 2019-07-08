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

/** @file full_matrix.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_FULL_MATRIX_H_
#define INCLUDE_BESTHEA_FULL_MATRIX_H_

#include "besthea/blas_lapack_wrapper.h"
#include "besthea/matrix.h"
#include "besthea/settings.h"

#include <iostream>
#include <vector>

namespace besthea {
  namespace linear_algebra {
    class full_matrix;
  }
}

/**
 *  Class representing a full matrix.
 */
class besthea::linear_algebra::full_matrix
  : public besthea::linear_algebra::matrix {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  /**
   * Default constructor.
   */
  full_matrix( );

  /**
   * Copy constructor.
   * @param[in] that Matrix to be deep copied.
   */
  full_matrix( const full_matrix & that );

  /**
   * Constructor with an initializer list.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   * @param[in] list Initializer list for std::vector.
   */
  full_matrix( lo n_rows, lo n_columns, std::initializer_list< sc > list );

  /**
   * Constructing a matrix of the given dimension.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   * @param[in] zero Initialize to 0 if true.
   */
  full_matrix( lo n_rows, lo n_columns, bool zero = true );

  /**
   * Destructor.
   */
  virtual ~full_matrix( );

  /*!
   * @brief Prints the matrix.
   * @param[in] stream
   */
  void print( std::ostream & stream = std::cout ) const;

  /*!
   * @brief Fills the matrix with the given value.
   * @param[in] value
   */
  void fill( sc value ) {
    std::fill( _data.begin( ), _data.end( ), value );
  }

  /*!
   * @brief Fills the diagonal of the matrix with the given value.
   * @param[in] value
   */
  void fill_diag( sc value );

  /*!
   * @brief Fills the matrix with random numbers (uniform distribution).
   * @param[in] lower Lower bound.
   * @param[in] upper Upper bound.
   */
  void random_fill( sc lower, sc upper );

  /*!
   * @brief Fills the matrix diagonal with random numbers (uniform
   * distribution).
   * @param[in] lower Lower bound.
   * @param[in] upper Upper bound.
   */
  void random_fill_diag( sc lower, sc upper );

  /*!
   * @brief Returns the (i,j)-th element of the matrix.
   * @param[in] i Row index.
   * @param[in] j Column index.
   */
  sc get( lo i, lo j ) const {
    return _data[ i + j * _n_rows ];
  }

  /*!
   * @brief Sets the (i,j)-th element of the vector.
   * @param[in] i Row index.
   * @param[in] j Column index.
   * @param[in] value Value to be set.
   */
  void set( lo i, lo j, sc value ) {
    _data[ i + j * _n_rows ] = value;
  }

  /*!
   * @brief Overloads the () operator.
   * @param[in] i Row index.
   * @param[in] j Column index.
   */
  sc & operator( )( lo i, lo j ) {
    return _data[ i + j * _n_rows ];
  }

  /*!
   * @brief Overloads the () operator.
   * @param[in] i Row index.
   * @param[in] j Column index.
   */
  sc operator( )( lo i, lo j ) const {
    return _data[ i + j * _n_rows ];
  }

  /*!
   * @brief Returns the raw data.
   */
  sc * data( ) {
    return _data.data( );
  }

  /*!
   * @brief Returns the raw data.
   */
  const sc * data( ) const {
    return _data.data( );
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose.
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( vector_type const & x, vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * @brief y = beta * y + alpha * this * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] alpha
   * @param[in] beta
   */
  void apply_symmetric(
    vector const & x, vector_type & y, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * @brief In-place LU decomposition and solution.
   * @param[in,out] rhs Right-hand side overwritten by the result.
   * @param[in] n_rhs Number of right-hand sides.
   * @param[in] trans Flag for transpose.
   */
  void lu_decompose_solve(
    vector_type & rhs, lo n_rhs = 1, bool trans = false );

  /*!
   * @brief In-place Choleski decomposition and solution.
   * @param[in,out] rhs Right-hand side overwritten by the result.
   * @param[in] n_rhs Number of right-hand sides.
   */
  void choleski_decompose_solve( vector_type & rhs, lo n_rhs = 1 );

  /*!
   * @brief In-place Choleski decomposition.
   */
  void choleski_decompose( );

  /*!
   * @brief Choleski solution
   * @param[in,out] rhs Right-hand side overwritten by the result.
   * @param[in] n_rhs Number of right-hand sides.
   */
  void choleski_solve( vector_type & rhs, lo n_rhs = 1 );

  /*!
   * Resizes the matrix.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   */
  void resize( lo n_rows, lo n_columns ) {
    _data.resize( n_rows * n_columns );
    _n_rows = n_rows;
    _n_columns = n_columns;
  }

 protected:
  std::vector< sc > _data;  //!< Raw data.
};

#endif /* INCLUDE_BESTHEA_FULL_MATRIX_H_ */
