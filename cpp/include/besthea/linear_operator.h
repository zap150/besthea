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

/** @file linear_operator.h
 * @brief Parent class for linear operators.
 */

#ifndef INCLUDE_BESTHEA_LINEAR_OPERATOR_H_
#define INCLUDE_BESTHEA_LINEAR_OPERATOR_H_

#include "besthea/settings.h"
#include "besthea/vector.h"

namespace besthea {
  namespace linear_algebra {
    class linear_operator;
  }
}

/**
 *  Class representing a linear operator.
 */
class besthea::linear_algebra::linear_operator {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Destructor.
   */
  virtual ~linear_operator( ) {
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
  bool mkl_cg_solve( const vector_type & rhs, vector_type & solution,
    sc & relative_residual_error, lo & n_iterations ) const;

  /**
   * Preconditioned CG as implemented in MKL.
   * @param[in] preconditioner Linear operator as a preconditioner.
   * @param[in] rhs Right-hand side vector.
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|, actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations, actual value on
   * exit.
   */
  bool mkl_cg_solve( const linear_operator & preconditioner,
    const vector_type & rhs, vector_type & solution,
    sc & relative_residual_error, lo & n_iterations ) const;

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
   * @param[in] trans Use transpose of this.
   */
  bool mkl_fgmres_solve( const vector_type & rhs, vector_type & solution,
    sc & relative_residual_error, lo & n_iterations,
    lo n_iterations_until_restart = 0, bool trans = false ) const;

  /**
   * Preconditioned FGMRES as implemented in MKL.
   * @param[in] preconditioner Linear operator as a preconditioner.
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|, actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations, actual value on
   * exit.
   * @param[in] n_iterations_until_restart Maximal number of iterations before
   * restart.
   * @param[in] trans Use transpose of this.
   * @param[in] trans_preconditioner Use transpose of preconditioner.
   */
  bool mkl_fgmres_solve( const linear_operator & preconditioner,
    const vector_type & rhs, vector_type & solution,
    sc & relative_residual_error, lo & n_iterations,
    lo n_iterations_until_restart = 0, bool trans = false,
    bool trans_preconditioner = false ) const;

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

 protected:
  lo _dim_domain;  //!< domain dimension
  lo _dim_range;   //!< range dimension
};

#endif /* INCLUDE_BESTHEA_LINEAR_OPERATOR_H_ */
