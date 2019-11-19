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

/** @file block_iterative_inverse.h
 * @brief Parent class for iterative inverses of block linear operators.
 */

#ifndef INCLUDE_BESTHEA_BLOCK_ITERATIVE_INVERSE_H_
#define INCLUDE_BESTHEA_BLOCK_ITERATIVE_INVERSE_H_

#include "besthea/block_linear_operator.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

namespace besthea {
  namespace linear_algebra {
    class block_iterative_inverse;
  }
}

/**
 *  Class representing an iterative inverses of a linear operator.
 */
class besthea::linear_algebra::block_iterative_inverse
  : public besthea::linear_algebra::block_linear_operator {
 public:
  using block_vector_type
    = besthea::linear_algebra::vector;  //!< Block vector type.

  /**
   * Destructor.
   */
  virtual ~block_iterative_inverse( ) {
  }

 protected:
  /**
   * Constructor.
   * @param[in] op Linear operator to be inverted.
   * @param[in] relative_residual_error Relative residual error.
   * @param[in] n_iterations Maximal number of iterations.
   */
  block_iterative_inverse(
    block_linear_operator & op, sc relative_residual_error, lo n_iterations )
    : _operator( &op ),
      _preconditioner( nullptr ),
      _relative_residual_error( relative_residual_error ),
      _n_iterations( n_iterations ) {
  }

  /**
   * Constructor.
   * @param[in] op Linear operator to be inverted.
   * @param[in] precond Linear operator as a preconditioner.
   * @param[in] relative_residual_error Relative residual error.
   * @param[in] n_iterations Maximal number of iterations.
   */
  block_iterative_inverse( block_linear_operator & op,
    block_linear_operator & precond, sc relative_residual_error,
    lo n_iterations )
    : _operator( &op ),
      _preconditioner( &precond ),
      _relative_residual_error( relative_residual_error ),
      _n_iterations( n_iterations ) {
  }

  block_linear_operator * _operator;  //!< block linear operator to be inverted
  block_linear_operator *
    _preconditioner;            //!< block linear operator as a preconditioner
  sc _relative_residual_error;  //!< relative residual error
  lo _n_iterations;             //!< maximal number of iterations
};

#endif /* INCLUDE_BESTHEA_BLOCK_ITERATIVE_INVERSE_H_ */
