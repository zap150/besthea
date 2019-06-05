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

#include "besthea/full_matrix.h"

#include "besthea/blas_lapack_wrapper.h"

#include <algorithm>
#include <random>

besthea::linear_algebra::full_matrix::full_matrix( ) : _data( 1, 0.0 ) {
  this->_n_rows = 1;
  this->_n_columns = 1;
}

besthea::linear_algebra::full_matrix::full_matrix( const full_matrix & that )
  : _data( that._data ) {
  this->_n_rows = that._n_rows;
  this->_n_columns = that._n_columns;
}

besthea::linear_algebra::full_matrix::full_matrix(
  lo n_rows, lo n_columns, bool zero )
  : _data( n_rows * n_columns ) {
  this->_n_rows = n_rows;
  this->_n_columns = n_columns;
  if ( zero ) {
    fill( 0.0 );
  }
}

besthea::linear_algebra::full_matrix::~full_matrix( ) {
}

void besthea::linear_algebra::full_matrix::print(
  std::ostream & stream ) const {
  for ( lo i_row = 0; i_row < _n_rows; ++i_row ) {
    for ( lo i_col = 0; i_col < _n_columns; ++i_col ) {
      stream << _data[ i_row + i_col * _n_rows ] << " ";
    }
    stream << std::endl;
  }
  stream << std::endl;
}

void besthea::linear_algebra::full_matrix::random_fill( sc lower, sc upper ) {
  std::random_device rd;
  std::mt19937 gen( rd( ) );
  std::uniform_real_distribution< sc > dis( lower, upper );
  std::generate(
    _data.begin( ), _data.end( ), [&gen, &dis]( ) { return dis( gen ); } );
}

void besthea::linear_algebra::full_matrix::random_fill_diag(
  sc lower, sc upper ) {
  std::random_device rd;
  std::mt19937 gen( rd( ) );
  std::uniform_real_distribution< sc > dis( lower, upper );
  lo n = _n_rows < _n_columns ? _n_rows : _n_columns;
  for ( lo i = 0; i < n; ++i ) {
    _data[ i + i * _n_rows ] = dis( gen );
  }
}

void besthea::linear_algebra::full_matrix::fill_diag( sc value ) {
  lo n = _n_rows < _n_columns ? _n_rows : _n_columns;
  for ( lo i = 0; i < n; ++i ) {
    _data[ i + i * _n_rows ] = value;
  }
}

void besthea::linear_algebra::full_matrix::apply(
  const vector & x, vector & y, bool trans, sc alpha, sc beta ) const {
  CBLAS_TRANSPOSE cblas_trans = trans ? CblasTrans : CblasNoTrans;

  cblas_dgemv( CblasColMajor, cblas_trans, _n_rows, _n_columns, alpha,
    _data.data( ), _n_rows, x.data( ), 1, beta, y.data( ), 1 );
}

void besthea::linear_algebra::full_matrix::apply_symmetric(
  const vector & x, vector & y, sc alpha, sc beta ) const {
  CBLAS_UPLO cblas_uplo = CblasUpper;

  cblas_dsymv( CblasColMajor, cblas_uplo, _n_rows, alpha,
    _data.data( ), _n_rows, x.data( ), 1, beta, y.data( ), 1 );
}

void besthea::linear_algebra::full_matrix::lu_decompose_solve(
  vector & rhs, lo n_rhs, bool trans ) {
  char lapacke_trans = trans ? 'T' : 'N';
  lo ipiv_size = _n_rows < _n_columns ? _n_rows : _n_columns;

  lo * ipiv = new lo[ ipiv_size ];
  LAPACKE_dgetrf(
    LAPACK_COL_MAJOR, _n_rows, _n_columns, _data.data( ), _n_rows, ipiv );
  LAPACKE_dgetrs( LAPACK_COL_MAJOR, lapacke_trans, _n_rows, n_rhs,
    _data.data( ), _n_rows, ipiv, rhs.data( ), _n_rows );
  delete[] ipiv;
}

void besthea::linear_algebra::full_matrix::choleski_decompose_solve(
  vector & rhs, lo n_rhs ) {
  char uplo = 'U';
  LAPACKE_dpotrf( LAPACK_COL_MAJOR, uplo, _n_rows, _data.data( ), _n_rows );
  LAPACKE_dpotrs( LAPACK_COL_MAJOR, uplo, _n_rows, n_rhs, _data.data( ),
    _n_rows, rhs.data( ), _n_rows );
}

void besthea::linear_algebra::full_matrix::choleski_decompose( ) {
  char uplo = 'U';
  LAPACKE_dpotrf( LAPACK_COL_MAJOR, uplo, _n_rows, _data.data( ), _n_rows );
}

void besthea::linear_algebra::full_matrix::choleski_solve(
  vector & rhs, lo n_rhs ) {
  char uplo = 'U';
  LAPACKE_dpotrs( LAPACK_COL_MAJOR, uplo, _n_rows, n_rhs, _data.data( ),
    _n_rows, rhs.data( ), _n_rows );
}
