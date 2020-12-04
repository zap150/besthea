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

/** @file block_mkl_fgmres_inverse.h
 * @brief MKL FGMRES inverse of block linear operators.
 */

#ifndef INCLUDE_BESTHEA_BLOCK_MKL_FGMRES_INVERSE_H_
#define INCLUDE_BESTHEA_BLOCK_MKL_FGMRES_INVERSE_H_

#include "besthea/block_iterative_inverse.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

namespace besthea {
  namespace linear_algebra {
    class block_mkl_fgmres_inverse;
  }
}

/**
 *  Class representing a MKL CG inverse of a linear operator.
 */
class besthea::linear_algebra::block_mkl_fgmres_inverse
  : public besthea::linear_algebra::block_iterative_inverse {
 public:
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.

  /**
   * Constructor.
   * @param[in] op Linear operator to be inverted.
   * @param[in] relative_residual_error Relative residual error.
   * @param[in] n_iterations Maximal number of iterations.
   * @param[in] n_iterations_until_restart Number of iterations before restart.
   */
  block_mkl_fgmres_inverse( block_linear_operator & op,
    sc relative_residual_error, lo n_iterations,
    lo n_iterations_until_restart = 0 );

  /**
   * Constructor.
   * @param[in] op Linear operator to be inverted.
   * @param[in] precond Linear operator as a preconditioner.
   * @param[in] relative_residual_error Relative residual error.
   * @param[in] n_iterations Maximal number of iterations.
   * @param[in] n_iterations_until_restart Number of iterations before restart.
   */
  block_mkl_fgmres_inverse( block_linear_operator & op,
    block_linear_operator & precond, sc relative_residual_error,
    lo n_iterations, lo n_iterations_until_restart = 0 );

  /**
   * Destructor.
   */
  virtual ~block_mkl_fgmres_inverse( ) {
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
    [[maybe_unused]] bool trans = false, [[maybe_unused]] sc alpha = 1.0,
    [[maybe_unused]] sc beta = 0.0 ) const override;

 protected:
  lo _n_iterations_until_restart;  //!< maximal number of iterations before
                                   //!< restart
};

#endif /* INCLUDE_BESTHEA_BLOCK_MKL_FGMRES_INVERSE_H_ */
