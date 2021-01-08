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

#include "besthea/block_lower_triangular_toeplitz_matrix.h"

besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::
  block_lower_triangular_toeplitz_matrix( )
  : block_matrix( 0, 0, 0 ), _data( ) {
}

besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::
  block_lower_triangular_toeplitz_matrix(
    lo block_dim, lo n_rows, lo n_columns, std::initializer_list< sc > list )
  : block_matrix( block_dim, n_columns, n_rows ),
    _data( block_dim, matrix_type( n_rows, n_columns, list ) ) {
}

besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::
  block_lower_triangular_toeplitz_matrix(
    lo block_dim, lo n_rows, lo n_columns, bool zero )
  : block_matrix( block_dim, n_columns, n_rows ),
    _data( block_dim, matrix_type( n_rows, n_columns, zero ) ) {
}

besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::
  ~block_lower_triangular_toeplitz_matrix( ) {
}

void besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::apply(
  const block_vector_type & x, block_vector_type & y, bool trans, sc alpha,
  sc beta ) const {
  const matrix_type * m;
  const vector_type * subx;
  vector_type * suby;

  sc block_beta = beta;
  for ( lo diag = 0; diag < _block_dim; ++diag ) {
    m = &( _data[ diag ] );
    for ( lo block = 0; block < _block_dim - diag; ++block ) {
      subx = &( x.get_block( block ) );
      suby = &( y.get_block( block + diag ) );
      m->apply( *subx, *suby, trans, alpha, block_beta );
    }
    block_beta = 1.0;
  }
}

void besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::print(
  std::ostream & stream ) const {
  for ( const matrix_type & m : _data ) {
    m.print( stream );
  }
}

void besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::
  resize_blocks( lo n_rows, lo n_columns ) {
  for ( matrix_type & m : _data ) {
    m.resize( n_rows, n_columns );
  }
  _n_rows = n_rows;
  _n_columns = n_columns;
}

void besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::resize(
  lo block_dim ) {
  _data.resize( block_dim );
  _data.shrink_to_fit( );
  _block_dim = block_dim;
}

void besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::
  cholesky_decompose_solve( block_vector_type & rhs ) {
  _data[ 0 ].cholesky_decompose( );

  for ( lo d = 0; d < _block_dim; ++d ) {
    for ( lo j = 1; j <= d; ++j ) {
      _data[ j ].apply(
        rhs.get_block( d - j ), rhs.get_block( d ), false, -1.0, 1.0 );
    }
    _data[ 0 ].cholesky_solve( rhs.get_block( d ) );
  }
}

void besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::
  cholesky_decompose( ) {
  _data[ 0 ].cholesky_decompose( );
}

void besthea::linear_algebra::block_lower_triangular_toeplitz_matrix::
  cholesky_solve( block_vector_type & rhs ) {
  for ( lo d = 0; d < _block_dim; ++d ) {
    for ( lo j = 1; j <= d; ++j ) {
      _data[ j ].apply(
        rhs.get_block( d - j ), rhs.get_block( d ), false, -1.0, 1.0 );
    }
    _data[ 0 ].cholesky_solve( rhs.get_block( d ) );
  }
}
