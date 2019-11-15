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

/** @file block_linear_operator.h
 * @brief Parent class for block_linear operators.
 */

#ifndef INCLUDE_BESTHEA_BLOCK_LINEAR_OPERATOR_H_
#define INCLUDE_BESTHEA_BLOCK_LINEAR_OPERATOR_H_

#include "besthea/block_vector.h"
#include "besthea/settings.h"

namespace besthea {
  namespace linear_algebra {
    class block_linear_operator;
  }
}

/**
 *  Class representing a linear operator.
 */
class besthea::linear_algebra::block_linear_operator {
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.

 public:
  block_linear_operator( )
    : _block_dim( 0 ), _dim_domain( 0 ), _dim_range( 0 ) {
  }

  /*!
   * @brief Constructor.
   * @param[in] block_dim Block dimension.
   * @param[in] dim_domain Dimension of domain per block.
   * @param[in] dim_range Dimension of range per block.
   */
  block_linear_operator( lo block_dim, lo dim_domain, lo dim_range )
    : _block_dim( block_dim ),
      _dim_domain( dim_domain ),
      _dim_range( dim_range ) {
  }

  /**
   * Destructor.
   */
  virtual ~block_linear_operator( ) {
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const block_vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const = 0;

  /**
   * CG as implemented in MKL.
   * @param[in] rhs Right-hand side vector.
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|, actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations, actual value on
   * exit.
   */
  bool mkl_cg_solve( const block_vector_type & rhs,
    block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations ) const;

  /**
   * FGMRES as implemented in MKL.
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|, actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations, actual value on
   * exit.
   * @param[in] n_iterations_until_restart Maximal number of iterations before
   * restart.
   */
  bool mkl_fgmres_solve( block_vector_type & rhs, block_vector_type & solution,
    sc & relative_residual_error, lo & n_iterations,
    lo n_iterations_until_restart ) const;

 protected:
  lo _block_dim;   //!< Number of blocks in a row (column).
  lo _dim_domain;  //!< domain dimension
  lo _dim_range;   //!< range dimension
};

#endif /* INCLUDE_BESTHEA_BLOCK_LINEAR_OPERATOR_H_ */
