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

/** @file iterative_inverse.h
 * @brief Parent class for iterative inverses of linear operators.
 */

#ifndef INCLUDE_BESTHEA_ITERATIVE_INVERSE_H_
#define INCLUDE_BESTHEA_ITERATIVE_INVERSE_H_

#include "besthea/linear_operator.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

namespace besthea {
  namespace linear_algebra {
    class iterative_inverse;
  }
}

/**
 *  Class representing an iterative inverses of a linear operator.
 */
class besthea::linear_algebra::iterative_inverse
  : public besthea::linear_algebra::linear_operator {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Destructor.
   */
  virtual ~iterative_inverse( ) {
  }

 protected:
  /**
   * Constructor.
   * @param[in] op Linear operator to be inverted.
   * @param[in] relative_residual_error Relative residual error.
   * @param[in] n_iterations Maximal number of iterations.
   * @param[in] trans Use transpose.
   */
  iterative_inverse( linear_operator & op, sc relative_residual_error,
    lo n_iterations, bool trans = false )
    : _operator( &op ),
      _preconditioner( nullptr ),
      _relative_residual_error( relative_residual_error ),
      _n_iterations( n_iterations ),
      _trans( trans ) {
    if ( !trans ) {
      set_dim_domain( op.get_dim_domain( ) );
      set_dim_range( op.get_dim_range( ) );
    } else {
      set_dim_domain( op.get_dim_range( ) );
      set_dim_range( op.get_dim_domain( ) );
    }
  }

  /**
   * Constructor.
   * @param[in] op Linear operator to be inverted.
   * @param[in] precond Linear operator as a preconditioner.
   * @param[in] relative_residual_error Relative residual error.
   * @param[in] n_iterations Maximal number of iterations.
   * @param[in] trans Use transpose.
   */
  iterative_inverse( linear_operator & op, linear_operator & precond,
    sc relative_residual_error, lo n_iterations, bool trans = false )
    : _operator( &op ),
      _preconditioner( &precond ),
      _relative_residual_error( relative_residual_error ),
      _n_iterations( n_iterations ),
      _trans( trans ) {
    if ( !trans ) {
      set_dim_domain( op.get_dim_domain( ) );
      set_dim_range( op.get_dim_range( ) );
    } else {
      set_dim_domain( op.get_dim_range( ) );
      set_dim_range( op.get_dim_domain( ) );
    }
  }

  linear_operator * _operator;        //!< linear operator to be inverted
  linear_operator * _preconditioner;  //!< linear operator as a preconditioner
  sc _relative_residual_error;        //!< relative residual error
  lo _n_iterations;                   //!< maximal number of iterations
  bool _trans;                        //!< Use transpose.
};

#endif /* INCLUDE_BESTHEA_ITERATIVE_INVERSE_H_ */
