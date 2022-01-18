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

/** @file low_rank_matrix.h
 * @brief Contains class of matrices compressed by ACA and corresponding
 * routines.
 */

#ifndef INCLUDE_BESTHEA_LOW_RANK_MATRIX_H_
#define INCLUDE_BESTHEA_LOW_RANK_MATRIX_H_

#include "besthea/full_matrix.h"
#include "besthea/matrix.h"
#include "besthea/settings.h"

namespace besthea {
  namespace linear_algebra {
    class low_rank_matrix;
  }
}

/**
 * Class representing a matrix in low rank format
 */
class besthea::linear_algebra::low_rank_matrix
  : public besthea::linear_algebra::matrix {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Default constructor.
   */
  low_rank_matrix( );

  /**
   * Copy constructor
   */
  low_rank_matrix( const low_rank_matrix & that );

  /**
   * Constructor for a new low rank matrix U*V^T using move semantics.
   * @param[in] u Matrix U in the low rank representation.
   * @param[in] v Matrix V in the low rank representation.
   * @warning @p u and @p v need to have the same number of columns. This is not
   * checked.
   */
  low_rank_matrix( full_matrix && u, full_matrix && v );

  /**
   * Constructor for a new low rank matrix U*V^T
   * @param[in] u Matrix U in the low rank representation.
   * @param[in] v Matrix V in the low rank representation.
   * @note In this version the matrices U and V are copied explicitly.
   *  @p u and @p v need to have the same number of columns. This is not
   * checked.
   */
  low_rank_matrix( const full_matrix & u, const full_matrix & v );

  /**
   * Destructor.
   */
  virtual ~low_rank_matrix( );

  /**
   * Move assignment operator for low rank matrices
   * @param[in] other Matrix to be assigned (by moving its data)
   */
  low_rank_matrix & operator=( low_rank_matrix && other );

  /**
   * Returns the rank of the low rank matrix.
   */
  virtual lo get_rank( ) const {
    return _u.get_n_columns( );
  }

  /**
   * Returns the number of stored entries of the matrix.
   * @note This is the sum of the number of entries of the low rank components
   * @ref _u and @ref _v.
   */
  virtual lo get_n_stored_entries( ) const {
    return _u.get_n_columns( ) * ( _u.get_n_rows( ) * _v.get_n_rows( ) );
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose.
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const vector_type & x, vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

 private:
  full_matrix _u;  //!< Matrix U in the low rank representation U*V^T
  full_matrix _v;  //!< Matrix V in the low rank representation U*V^T
};

#endif  // INCLUDE_BESTHEA_LOW_RANK_MATRIX_H_
