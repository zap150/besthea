/*
Copyright (c) 2021, VSB - Technical University of Ostrava and Graz University of
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

/** @file distributed_diagonal_matrix.h
 * @brief Represents a diagonal matrix distributed among a set of processes.
 */

#include "besthea/block_linear_operator.h"

#include <mpi.h>

namespace besthea {
  namespace linear_algebra {
    class distributed_diagonal_matrix;
    class distributed_block_vector;
  }
}

/**
 *  Class representing a diagonal matrix distributed among a set of MPI
 * processes.
 */
class besthea::linear_algebra::distributed_diagonal_matrix
  : public besthea::linear_algebra::block_linear_operator {
 public:
  /**
   * Constructor.
   * @param[in] diagonal  Diagonal of the matrix, distributed among a set of MPI
   * processes.
   */
  distributed_diagonal_matrix( const distributed_block_vector & diagonal )
    : _diagonal( diagonal ) {
  }

  distributed_diagonal_matrix( const distributed_diagonal_matrix & that )
    = delete;

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x using block vectors.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   * @note This routine is just a dummy here. Please use the corresponding
   * version with distributed block vectors.
   */
  virtual void apply( const block_vector & x, block_vector & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x using block vectors.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const distributed_block_vector & x,
    distributed_block_vector & y, bool trans = false, sc alpha = 1.0,
    sc beta = 0.0 ) const;

 private:
  distributed_block_vector
    _diagonal;  //!< Diagonal of the matrix, distributed among MPI processes.
};
