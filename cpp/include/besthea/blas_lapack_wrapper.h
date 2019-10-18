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

/** @file blas_lapack_wrapper.h
 * @brief Wrapper for BLAS and LAPACK methods.
 */

#ifndef INCLUDE_BESTHEA_BLAS_LAPACK_WRAPPER_H_
#define INCLUDE_BESTHEA_BLAS_LAPACK_WRAPPER_H_

#include "besthea/settings.h"

enum CBLAS_ORDER : int { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE : int {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
};
enum CBLAS_UPLO : int { CblasUpper = 121, CblasLower = 122 };
// enum CBLAS_DIAG : int { CblasNonUnit = 131, CblasUnit = 132 };
// enum CBLAS_SIDE : int { CblasLeft = 141, CblasRight = 142 };

//#ifndef LAPACK_ROW_MAJOR
//#define LAPACK_ROW_MAJOR 101
//#endif
#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif

extern "C" {

// LEVEL 1 BLAS

void cblas_daxpy( lo N, sc alpha, const sc * X, lo incX, sc * Y, lo incY );

sc cblas_ddot( lo N, const sc * X, lo incX, const sc * Y, lo incY );

sc cblas_dnrm2( lo N, const sc * X, lo incX );

// LEVEL 2 BLAS

void cblas_dgemv( CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, lo M, lo N,
  sc alpha, const sc * A, lo lda, const sc * X, lo incX, sc beta, sc * Y,
  lo incY );

void cblas_dsymv( CBLAS_ORDER order, CBLAS_UPLO Uplo, lo M, sc alpha,
  const sc * A, lo lda, const sc * X, lo incX, sc beta, sc * Y, lo incY );

// LEVEL 3 BLAS

void cblas_dgemm( CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
  CBLAS_TRANSPOSE TransB, lo m, lo n, lo k, sc alpha, const sc * A, lo lda,
  const sc * B, lo ldb, sc beta, sc * C, lo ldc );

// LAPACK

lo LAPACKE_dgetrf( int order, lo m, lo n, sc * a, lo lda, lo * ipiv );

lo LAPACKE_dgetrf2( int order, lo m, lo n, sc * a, lo lda, lo * ipiv );

lo LAPACKE_dgetrs( int order, char trans, lo n, lo nrhs, const sc * a, lo lda,
  const lo * ipiv, sc * b, lo ldb );

lo LAPACKE_dpotrf( int order, char uplo, lo n, sc * a, lo lda );

lo LAPACKE_dpotrf2( int order, char uplo, lo n, sc * a, lo lda );

lo LAPACKE_dpotrs(
  int order, char uplo, lo n, lo nrhs, const sc * a, lo lda, sc * b, lo ldb );

}  // extern "C"

#endif /* INCLUDE_BESTHEA_BLAS_LAPACK_WRAPPER_H_ */
