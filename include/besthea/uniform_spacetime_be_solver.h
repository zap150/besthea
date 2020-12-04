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

/** @file uniform_spacetime_be_solver.h
 * @brief Solver for spacetime BEM.
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SOLVER_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SOLVER_H_

#include "besthea/block_lower_triangular_toeplitz_matrix.h"
#include "besthea/block_vector.h"
#include "besthea/sparse_matrix.h"

namespace besthea {
  namespace bem {
    class uniform_spacetime_be_solver;
  }
}

/**
 *  Class representing a boundary element solver.
 */
class besthea::bem::uniform_spacetime_be_solver {
 public:
  using block_matrix_type = besthea::linear_algebra::
    block_lower_triangular_toeplitz_matrix;  //!< Block matrix type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.
  using sparse_matrix_type
    = besthea::linear_algebra::sparse_matrix;  //!< Sparse matrix type.

  /**
   * Time marching solver for the Dirichlet problem Vt=(1/2M+K)u (V11 is
   * decomposed in-place!)
   * @param[in] V Single-layer matrix.
   * @param[in] K Double-layer matrix.
   * @param[in] M Identity matrix for a single timestep.
   * @param[in] dirichlet Dirichlet data.
   * @param[out] neumann Neumann data.
   */
  static void time_marching_dirichlet( block_matrix_type & V,
    const block_matrix_type & K, const sparse_matrix_type & M,
    const block_vector_type & dirichlet, block_vector_type & neumann );

  /**
   * Time marching solver for the Neumann problem Du=(1/2M-K')t (D11 is
   * decomposed in-place!)
   * @param[in] D hypersingular matrix.
   * @param[in] K Double-layer matrix.
   * @param[in] M Identity matrix for a single timestep.
   * @param[in] neumann Neumann data.
   * @param[out] dirichlet Dirichlet data.
   */
  static void time_marching_neumann( block_matrix_type & D,
    const block_matrix_type & K, const sparse_matrix_type & M,
    const block_vector_type & neumann, block_vector_type & dirichlet );
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SOLVER_H_ */
