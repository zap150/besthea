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

/** @file block_linear_operator.h
 * @brief Parent class for block_linear operators.
 * @note updated
 */

#ifndef INCLUDE_BESTHEA_BLOCK_LINEAR_OPERATOR_H_
#define INCLUDE_BESTHEA_BLOCK_LINEAR_OPERATOR_H_

#include "besthea/block_vector.h"
#include "besthea/distributed_block_vector.h"
#include "besthea/settings.h"

namespace besthea {
  namespace linear_algebra {
    class block_linear_operator;
  }
}

/**
 *  Class representing a linear operator with block structure.
 */
class besthea::linear_algebra::block_linear_operator {
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.
  using distributed_block_vector_type
    = besthea::linear_algebra::distributed_block_vector;  //!< Block vector
                                                          //!< type.

 public:
  block_linear_operator( )
    : _block_dim( 0 ), _dim_domain( 0 ), _dim_range( 0 ) {
  }

  /*!
   * @brief Constructor.
   * @param[in] block_dim Block dimension.
   * @param[in] dim_domain Dimension of domain per block.
   * @param[in] dim_range Dimension of range per block.
   * @example In case of block matrices, \p block_dim is the number of blocks in
   * each row and column, \p dim_domain the number of rows per block and
   * \p dim_range the number of columns per block.
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
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const block_vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const = 0;
  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( [[maybe_unused]] const distributed_block_vector_type & x,
    [[maybe_unused]] distributed_block_vector_type & y,
    [[maybe_unused]] bool trans = false, [[maybe_unused]] sc alpha = 1.0,
    [[maybe_unused]] sc beta = 0.0 ) const { };
  /**
   * CG as implemented in MKL to solve (this) * x = y.
   * @param[in] rhs Right-hand side vector y.
   * @param[out] solution Solution vector x.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   */
  bool mkl_cg_solve( const block_vector_type & rhs,
    block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations ) const;

  /**
   * CG as implemented in MKL to solve (this) * x = y for distributed block
   * vectors.
   * @param[in] rhs Right-hand side vector y.
   * @param[out] solution Solution vector x.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @warning This is not a proper parallel version of CG. The distributed
   * vectors are serialized!
   */
  bool mkl_cg_solve( const distributed_block_vector_type & rhs,
    distributed_block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations ) const;

  /**
   * Preconditioned CG as implemented in MKL to solve (this) * x = y.
   * @param[in] preconditioner Block linear operator used as preconditioner.
   * @param[in] rhs Right-hand side vector.
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   */
  bool mkl_cg_solve( const block_linear_operator & preconditioner,
    const block_vector_type & rhs, block_vector_type & solution,
    sc & relative_residual_error, lo & n_iterations ) const;

  /**
   * Preconditioned CG as implemented in MKL to solve (this) * x = y for
   * distributed block vectors.
   * @param[in] preconditioner Block linear operator used as preconditioner.
   * @param[in] rhs Right-hand side vector.
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @warning This is not a proper parallel version of preconditioned CG. The
   * distributed vectors are serialized!
   */
  bool mkl_cg_solve( const block_linear_operator & preconditioner,
    const distributed_block_vector_type & rhs,
    distributed_block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations ) const;

  /**
   * FGMRES as implemented in MKL to solve (this)^trans * x = y.
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @param[in] n_iterations_until_restart Maximal number of iterations before
   * restart.
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   */
  bool mkl_fgmres_solve( const block_vector_type & rhs,
    block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations, lo n_iterations_until_restart = 0,
    bool trans = false ) const;

  /**
   * Preconditioned FGMRES as implemented in MKL to solve (this)^trans * x = y.
   * @param[in] preconditioner Block linear operator used as preconditioner.
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @param[in] n_iterations_until_restart Maximal number of iterations before
   * restart.
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   * @param[in] trans_preconditioner Indicates if the block linear operator used
   * for preconditioning is transposed or not.
   */
  bool mkl_fgmres_solve( const block_linear_operator & preconditioner,
    const block_vector_type & rhs, block_vector_type & solution,
    sc & relative_residual_error, lo & n_iterations,
    lo n_iterations_until_restart = 0, bool trans = false,
    bool trans_preconditioner = false ) const;

  /**
   * FGMRES as implemented in MKL to solve (this)^trans * x = y for distributed
   * block vectors
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @param[in] n_iterations_until_restart Maximal number of iterations before
   * restart.
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   * @warning This is not a proper parallel version of preconditioned CG. The
   * distributed vectors are serialized!
   */
  bool mkl_fgmres_solve( const distributed_block_vector_type & rhs,
    distributed_block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations, lo n_iterations_until_restart = 0,
    bool trans = false ) const;

  /**
   * Preconditioned FGMRES as implemented in MKL to solve (this)^trans * x = y
   * for distributed block vectors.
   * @param[in] preconditioner Block linear operator used as preconditioner.
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @param[in] n_iterations_until_restart Maximal number of iterations before
   * restart.
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   * @param[in] trans_preconditioner Indicates if the block linear operator used
   * for preconditioning is transposed or not.
   * @warning This is not a proper parallel version of preconditioned CG. The
   * distributed vectors are serialized!
   */
  bool mkl_fgmres_solve( const block_linear_operator & preconditioner,
    const distributed_block_vector_type & rhs,
    distributed_block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations, lo n_iterations_until_restart = 0, bool trans = false,
    bool trans_preconditioner = false ) const;

  /**
   * Preconditioned GMRES solver for the solution of (this)^trans * x = y.
   * @param[in] rhs Right-hand side vector.
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @param[in] prec Block linear operator used as preconditioner.
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   * @todo Discuss new output for @p relative_residual_error.
   * @todo Discuss: What is relative error in case that solution is not 0 at
   * function call? Should we fill solution with 0 in the routine?
   */
  bool gmres_solve( const block_vector_type & rhs, block_vector_type & solution,
    sc & relative_residual_error, lo & n_iterations,
    const block_linear_operator & prec, bool trans = false ) const;

  /**
   * GMRES solver for the solution of (this)^trans * x = y.
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   * @todo Discuss new output for @p relative_residual_error
   * @todo Discuss: What is relative error in case that solution is not 0 at
   * function call? Should we fill solution with 0 in the routine?
   */
  bool gmres_solve( const block_vector_type & rhs, block_vector_type & solution,
    sc & relative_residual_error, lo & n_iterations, bool trans = false ) const;

  /**
   * Preconditioned GMRES solver for the solution of (this)^trans * x = y for
   * distributed block vectors.
   * @param[in] rhs Right-hand side vector.
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @param[in] prec Block linear operator used as preconditioner.
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   * @todo Currently all ranks execute the GMRES algorithm, but computations
   * like matrix-vector products and scalar products are parallelized.
   * @todo Discuss new output for @p relative_residual_error
   * @todo Discuss: What is relative error in case that solution is not 0 at
   * function call? Should we fill solution with 0 in the routine?
   */
  bool gmres_solve( const distributed_block_vector_type & rhs,
    distributed_block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations, const block_linear_operator & prec,
    bool trans = false ) const;

  /**
   * GMRES solver for the solution of (this)^trans * x = y for distributed block
   * vectors.
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|. Overwritten with the actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations. Overwritten with
   * the actual number of iterations on exit.
   * @param[in] trans Indicates if the block linear operator is transposed or
   * not.
   * @todo Currently all ranks execute the GMRES algorithm, but
   * computations like matrix-vector products and scalar products are
   * parallelized.
   * @todo Discuss new output for @p relative_residual_error
   * @todo Discuss: What is relative error in case that solution is not 0 at
   * function call? Should we fill solution with 0 in the routine?
   */
  bool gmres_solve( const distributed_block_vector_type & rhs,
    distributed_block_vector_type & solution, sc & relative_residual_error,
    lo & n_iterations, bool trans = false ) const;

  /**
   * * Returns the dimension of the domain of each block.
   */
  lo get_dim_domain( ) const {
    return _dim_domain;
  }

  /**
   * Returns the dimension of the range of each block.
   */
  lo get_dim_range( ) const {
    return _dim_range;
  }

  /**
   * Returns the block dimension, i.e. the number of blocks per row and column.
   */
  lo get_block_dim( ) const {
    return _block_dim;
  }

  /**
   * Sets the dimension of the domain of each block.
   * @param[in] dim_domain Domain dimension.
   */
  void set_dim_domain( lo dim_domain ) {
    _dim_domain = dim_domain;
  }

  /**
   * Sets the dimension of the range of each block.
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
  lo _dim_domain;  //!< domain dimension (number of rows in a block)
  lo _dim_range;   //!< range dimension (number of columns in a block)
};

#endif /* INCLUDE_BESTHEA_BLOCK_LINEAR_OPERATOR_H_ */
