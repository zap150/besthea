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

#include "besthea/block_mkl_fgmres_inverse.h"

besthea::linear_algebra::block_mkl_fgmres_inverse::block_mkl_fgmres_inverse(
  block_linear_operator & op, sc relative_residual_error, lo n_iterations,
  lo n_iterations_until_restart )
  : block_iterative_inverse( op, relative_residual_error, n_iterations ) {
  if ( n_iterations_until_restart == 0 ) {
    _n_iterations_until_restart = n_iterations;
  } else {
    _n_iterations_until_restart = n_iterations_until_restart;
  }
}

besthea::linear_algebra::block_mkl_fgmres_inverse::block_mkl_fgmres_inverse(
  block_linear_operator & op, block_linear_operator & precond,
  sc relative_residual_error, lo n_iterations, lo n_iterations_until_restart )
  : block_iterative_inverse(
    op, precond, relative_residual_error, n_iterations ) {
  if ( n_iterations_until_restart == 0 ) {
    _n_iterations_until_restart = n_iterations;
  } else {
    _n_iterations_until_restart = n_iterations_until_restart;
  }
}

void besthea::linear_algebra::block_mkl_fgmres_inverse::apply(
  const block_vector_type & x, block_vector_type & y,
  [[maybe_unused]] bool trans, [[maybe_unused]] sc alpha,
  [[maybe_unused]] sc beta ) const {
  sc relative_residual_error = _relative_residual_error;
  lo n_iterations = _n_iterations;
  _operator->mkl_fgmres_solve( x, y, relative_residual_error, n_iterations,
    _n_iterations_until_restart, _trans );
}

void besthea::linear_algebra::block_mkl_fgmres_inverse::apply(
  const distributed_block_vector_type & x, distributed_block_vector_type & y,
  [[maybe_unused]] bool trans, [[maybe_unused]] sc alpha,
  [[maybe_unused]] sc beta ) const {
  sc relative_residual_error = _relative_residual_error;
  lo n_iterations = _n_iterations;
  _operator->mkl_fgmres_solve( x, y, relative_residual_error, n_iterations,
    _n_iterations_until_restart, _trans );
}
