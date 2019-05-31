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

// LEVEL 1 BLAS

// scale a vector (matrix) by a scalar value
extern "C" {
void dscal_( lo * N, sc * ALPHA, sc * X, lo * INCX );
}

// vector (matrix) addition y = y + alpha*x
extern "C" {
void daxpy_( lo * N, sc * A, sc * X, lo * INCX, sc * Y, lo * INCY );
}

// vector 2-norm
extern "C" {
sc dnrm2_( lo const * N, sc * X, lo * INCX );
}

// dot product
extern "C" {
sc ddot_( lo const * N, sc * X, lo * INCX, sc * Y, lo * INCY );
}

// LEVEL 2 BLAS

// BLAS matrix-vector multiplication
extern "C" {
void dgemv_( char * TRANS, lo * M, lo * N, sc * ALPHA, sc * A, lo * LDA, sc * X,
  lo * INCX, sc * BETA, sc * Y, lo * INCY );
}

// LEVEL 3 BLAS

// matrix-matrix multiplication
extern "C" {
void dgemm_( char * TRANSA, char * TRANSB, lo * M, lo * N, lo * K, sc * ALPHA,
  sc * A, lo * LDA, sc * B, lo * LDB, sc * BETA, sc * C, lo * LDC );
}

// LAPACK

// matrix norm
extern "C" {
sc dlange_( char * norm, lo const * M, lo const * N, sc const * A,
  lo const * LDA, sc * WORK );
}

// LAPACK LU factorization of a general matrix
extern "C" {
void dgetrf_( lo * M, lo * N, sc * A, lo * LDA, lo * IPIV, lo * INFO );
}

extern "C" {  // LAPACK LU solver
void dgetrs_( char * TRANS, lo * N, lo * NRHS, sc * A, lo * LDA, lo * IPIV,
  sc * B, lo * LDB, lo * INFO );
}

// LAPACK Choleski factorization of a general matrix
extern "C" {
void dpotrf_( char * UPLO, lo * M, sc * A, lo * LDA, lo * INFO );
}

// LAPACK Choleski solver
extern "C" {
void dpotrs_( char * UPLO, lo * N, lo * NRHS, sc * A, lo * LDA, sc * B,
  lo * LDB, lo * INFO );
}

// reduces real sc symmetric matrix to tridiagonal form
extern "C" {
void dsytrd_( char * UPLO, lo * N, sc * A, lo * LDA, sc * D, sc * E, sc * TAU,
  sc * WORK, lo * LWORK, lo * INFO );
}

// computes selected eigenvalues and, optionally, eigenvectors of a real
// symmetric matrix A
extern "C" {
void dsyevx_( char * JOBZ, char * RANGE, char * UPLO, lo * N, sc * A, lo * LDA,
  sc * VL, sc * VU, lo * IL, lo * IU, sc * ABSTOL, lo * M, sc * W, sc * Z,
  lo * LDZ, sc * WORK, lo * LWORK, lo * IWORK, lo * IFAIL, lo * INFO );
}

// computes Schur decomposition of a Hessenberg matrix H
extern "C" {
void dhseqr_( char * job, char * compz, lo * n, lo * ilo, lo * ihi, sc * h,
  lo * ldh, sc * wr, sc * wi, sc * z, lo * ldz, sc * work, lo * lwork,
  lo * info );
}

#endif /* INCLUDE_BESTHEA_BLAS_LAPACK_WRAPPER_H_ */
