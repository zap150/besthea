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

/** @file block_matrix.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_BLOCK_MATRIX_H_
#define INCLUDE_BESTHEA_BLOCK_MATRIX_H_

#include "besthea/block_linear_operator.h"
#include "besthea/settings.h"

namespace besthea {
  namespace linear_algebra {
    class block_matrix;
  }
}

/**
 *  Class representing a full matrix.
 */
class besthea::linear_algebra::block_matrix
  : public besthea::linear_algebra::block_linear_operator {
 public:
  /**
   * Default constructor.
   */
  block_matrix( ) {
    _block_dim = 0;
    _n_rows = 0;
    _n_columns = 0;
  }

  /**
   * Constructor.
   * @param[in] block_dim Block dimension.
   * @param[in] n_rows Number of rows.
   * @param[in] n_columns Number of columns.
   */
  block_matrix( lo block_dim, lo n_rows, lo n_columns )
    : block_linear_operator( block_dim, n_columns, n_rows ) {
  }

  /**
   * Destructor.
   */
  virtual ~block_matrix( ) {
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

 protected:
  lo & _n_rows{
    block_linear_operator::_dim_range
  };  //!< number of rows (range dimension)
  lo & _n_columns{
    block_linear_operator::_dim_domain
  };  //!< number of columns (domain dimension)
};

#endif /* INCLUDE_BESTHEA_BLOCK_MATRIX_H_ */
