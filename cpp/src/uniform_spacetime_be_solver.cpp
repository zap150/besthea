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

#include "besthea/uniform_spacetime_be_solver.h"

void besthea::bem::uniform_spacetime_be_solver::time_marching_dirichlet(
  block_matrix_type & V, const block_matrix_type & K,
  const sparse_matrix_type & M, const block_vector_type & dirichlet,
  block_vector_type & neumann ) {
  lo n_timesteps = V.get_block_dim( );
  lo n_dofs = V.get_n_rows( );
  neumann.resize( n_timesteps );
  neumann.resize_blocks( n_dofs, true );

  V.get_block( 0 ).cholesky_decompose( );

  for ( lo d = 0; d < n_timesteps; ++d ) {
    auto & rhs = neumann.get_block( d );
    M.apply( dirichlet.get_block( d ), rhs, false, 0.5, 0.0 );
    for ( lo j = 0; j <= d; ++j ) {
      K.get_block( j ).apply(
        dirichlet.get_block( d - j ), rhs, false, 1.0, 1.0 );
    }
    for ( lo j = 1; j <= d; ++j ) {
      V.get_block( j ).apply(
        neumann.get_block( d - j ), rhs, false, -1.0, 1.0 );
    }
    V.get_block( 0 ).cholesky_solve( rhs );
  }
}

void besthea::bem::uniform_spacetime_be_solver::time_marching_neumann(
  block_matrix_type & D, const block_matrix_type & K,
  const sparse_matrix_type & M, const block_vector_type & neumann,
  block_vector_type & dirichlet ) {
  lo n_timesteps = D.get_block_dim( );
  lo n_dofs = D.get_n_rows( );
  dirichlet.resize( n_timesteps );
  dirichlet.resize_blocks( n_dofs, true );

  D.get_block( 0 ).cholesky_decompose( );

  for ( lo d = 0; d < n_timesteps; ++d ) {
    auto & rhs = dirichlet.get_block( d );
    M.apply( neumann.get_block( d ), rhs, true, 0.5, 0.0 );
    for ( lo j = 0; j <= d; ++j ) {
      K.get_block( j ).apply(
        neumann.get_block( d - j ), rhs, true, -1.0, 1.0 );
    }
    for ( lo j = 1; j <= d; ++j ) {
      D.get_block( j ).apply(
        dirichlet.get_block( d - j ), rhs, false, -1.0, 1.0 );
    }
    D.get_block( 0 ).cholesky_solve( rhs );
  }
}
