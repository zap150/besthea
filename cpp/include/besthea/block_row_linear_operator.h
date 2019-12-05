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

/** @file block_row_linear_operator.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_BLOCK_ROW_LINEAR_OPERATOR_H_
#define INCLUDE_BESTHEA_BLOCK_ROW_LINEAR_OPERATOR_H_

#include "besthea/block_vector.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

namespace besthea {
  namespace linear_algebra {
    class block_row_linear_operator;
  }
}

/**
 *  Class representing a linear operator.
 */
class besthea::linear_algebra::block_row_linear_operator {
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.

 public:
  block_row_linear_operator( )
    : _block_dim( 0 ), _dim_domain( 0 ), _dim_range( 0 ) {
  }

  /*!
   * @brief Constructor.
   * @param[in] block_dim Block dimension.
   * @param[in] dim_domain Dimension of domain per block.
   * @param[in] dim_range Dimension of range per block.
   */
  block_row_linear_operator( lo block_dim, lo dim_domain, lo dim_range )
    : _block_dim( block_dim ),
      _dim_domain( dim_domain ),
      _dim_range( dim_range ) {
  }

  /**
   * Destructor.
   */
  virtual ~block_row_linear_operator( ) {
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const = 0;

  /**
   * Returns the domain dimension.
   */
  lo get_dim_domain( ) const {
    return _dim_domain;
  }

  /**
   * Returns the range dimension.
   */
  lo get_dim_range( ) const {
    return _dim_range;
  }

  /**
   * Returns the block dimension.
   */
  lo get_block_dim( ) const {
    return _block_dim;
  }

  /**
   * Sets the domain dimension.
   * @param[in] dim_domain Domain dimension.
   */
  void set_dim_domain( lo dim_domain ) {
    _dim_domain = dim_domain;
  }

  /**
   * Sets the range dimension.
   * @param[in] dim_range Range dimension.
   */
  void set_dim_range( lo dim_range ) {
    _dim_range = dim_range;
  }

  /**
   * Sets the block dimension.
   * @param[in] block_dim Block dimension.
   */
  void set_block_dim( lo block_dim ) {
    _block_dim = block_dim;
  }

 protected:
  lo _block_dim;   //!< Number of blocks in a row (column).
  lo _dim_domain;  //!< domain dimension
  lo _dim_range;   //!< range dimension
};

#endif /* INCLUDE_BESTHEA_BLOCK_ROW_LINEAR_OPERATOR_H_ */
