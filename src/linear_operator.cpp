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

#include "besthea/linear_operator.h"

#include <math.h>
#include <mkl_rci.h>
#include <vector>

bool besthea::linear_algebra::linear_operator::mkl_cg_solve(
  const vector_type & rhs, vector_type & solution, sc & relative_residual_error,
  lo & n_iterations ) const {
  if ( _dim_domain != _dim_range || _dim_domain != rhs.size( )
    || _dim_domain != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  lo size = _dim_domain;

  lo rci;
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp( size * 3 );
  sc * tmp_data = tmp.data( );

  vector_type tmp_1( size );
  vector_type tmp_2( size );

  dcg_init( &size, solution.data( ), rhs.data( ), &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "Failed to initialize MKL CG." << std::endl;
    return false;
  }

  ipar[ 0 ] = size;          // size of the problem
  ipar[ 4 ] = n_iterations;  // maximum number of iterations
  ipar[ 7 ] = 1;             // perform the iteration stopping test
  ipar[ 8 ] = 1;             // do the residual stopping test
  ipar[ 9 ] = 0;             // do not request user stopping test
  ipar[ 10 ] = 0;            // non-preconditioned

  dpar[ 0 ] = relative_residual_error;  // relative tolerance

  dcg_check( &size, solution.data( ), rhs.data( ), &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "MKL parameters incorrect." << std::endl;
    return false;
  }

  while ( true ) {
    dcg( &size, solution.data( ), rhs.data( ), &rci, ipar, dpar, tmp_data );

    if ( rci == 1 ) {  // apply operator
      tmp_1.copy_from_raw( _dim_domain, tmp_data );
      apply( tmp_1, tmp_2, false, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + _dim_domain );
      continue;
    } else if ( rci == 0 ) {  // success
      dcg_get( &size, solution.data( ), rhs.data( ), &rci, ipar, dpar, tmp_data,
        &iter );
      n_iterations = iter;
      relative_residual_error = dpar[ 4 ] / dpar[ 2 ];
      break;
    } else {
      std::cout << "Only RCI codes 0,1 supported." << std::endl;
      return false;
    }
  }

  return true;
}

bool besthea::linear_algebra::linear_operator::mkl_cg_solve(
  const linear_operator & preconditioner, const vector_type & rhs,
  vector_type & solution, sc & relative_residual_error,
  lo & n_iterations ) const {
  if ( _dim_domain != _dim_range || _dim_domain != rhs.size( )
    || _dim_domain != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  lo size = _dim_domain;

  lo rci;
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp( size * 4 );
  sc * tmp_data = tmp.data( );

  vector_type tmp_1( size );
  vector_type tmp_2( size );

  dcg_init( &size, solution.data( ), rhs.data( ), &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "Failed to initialize MKL CG." << std::endl;
    return false;
  }

  ipar[ 0 ] = size;          // size of the problem
  ipar[ 4 ] = n_iterations;  // maximum number of iterations
  ipar[ 7 ] = 1;             // perform the iteration stopping test
  ipar[ 8 ] = 1;             // do the residual stopping test
  ipar[ 9 ] = 0;             // do not request user stopping test
  ipar[ 10 ] = 1;            // preconditioned

  dpar[ 0 ] = relative_residual_error;  // relative tolerance

  dcg_check( &size, solution.data( ), rhs.data( ), &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "MKL parameters incorrect." << std::endl;
    return false;
  }

  while ( true ) {
    dcg( &size, solution.data( ), rhs.data( ), &rci, ipar, dpar, tmp_data );

    if ( rci == 1 ) {  // apply operator
      tmp_1.copy_from_raw( _dim_domain, tmp_data );
      apply( tmp_1, tmp_2, false, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + _dim_domain );
      continue;
    } else if ( rci == 3 ) {  // apply preconditioner
      tmp_1.copy_from_raw( _dim_domain, tmp_data );
      preconditioner.apply( tmp_1, tmp_2, false, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + _dim_domain );
      continue;
    } else if ( rci == 0 ) {  // success
      dcg_get( &size, solution.data( ), rhs.data( ), &rci, ipar, dpar, tmp_data,
        &iter );
      n_iterations = iter;
      relative_residual_error = dpar[ 4 ] / dpar[ 2 ];
      break;
    } else {
      std::cout << "Only RCI codes 0,1,3 supported." << std::endl;
      return false;
    }
  }

  return true;
}

bool besthea::linear_algebra::linear_operator::mkl_fgmres_solve(
  const vector_type & rhs, vector_type & solution, sc & relative_residual_error,
  lo & n_iterations, lo n_iterations_until_restart, bool trans ) const {
  if ( _dim_domain != _dim_range || _dim_domain != rhs.size( )
    || _dim_domain != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  if ( n_iterations_until_restart == 0 ) {
    n_iterations_until_restart = n_iterations;
  }

  lo size = _dim_domain;

  lo rci;
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp( ( 2 * n_iterations_until_restart + 1 ) * size
    + n_iterations_until_restart * ( n_iterations_until_restart + 9 ) / 2 + 1 );
  sc * tmp_data = tmp.data( );

  vector_type tmp_1( size );
  vector_type tmp_2( size );
  vector_type rhs_copy( rhs );

  dfgmres_init(
    &size, solution.data( ), rhs_copy.data( ), &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "Failed to initialize MKL CG." << std::endl;
    return false;
  }

  /*
   * New behaviour of init and check
   * https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/
   * RCI-ISS-solver-FGMRES-working-well-in-IPS-XE-2020-Update-4-does/m-p/1271247
   *
   * https://software.intel.com/content/www/us/en/develop/documentation/
   * onemkl-developer-reference-fortran/top/sparse-solver-routines/
   * iterative-sparse-solvers-based-on-reverse-communication-interface-rci-iss/
   * rci-iss-routines/dfgmres-check.html
   */

  bool silent = ( n_iterations_until_restart != ipar[ 4 ] );

  ipar[ 4 ] = n_iterations;  // maximum number of iterations
  ipar[ 7 ] = 1;             // perform the iteration stopping test
  ipar[ 8 ] = 1;             // do the residual stopping test
  ipar[ 9 ] = 0;             // do not request user stopping test
  ipar[ 10 ] = 0;            // non-preconditioned
  ipar[ 11 ] = 1;  // perform test for zero norm of generated direction
  ipar[ 14 ]
    = n_iterations_until_restart;  // number of iterations before restart
  if ( silent ) {
    ipar[ 6 ] = 0;  // disable the verbosity in case of non-default iterations
                    // until restart
  }

  dpar[ 0 ] = relative_residual_error;  // relative tolerance

  dfgmres_check(
    &size, solution.data( ), rhs_copy.data( ), &rci, ipar, dpar, tmp_data );
  if ( rci == -1100 ) {
    std::cout << "MKL parameters incorrect." << std::endl;
    return false;
  }

  while ( true ) {
    dfgmres(
      &size, solution.data( ), rhs_copy.data( ), &rci, ipar, dpar, tmp_data );

    if ( rci == 1 ) {  // apply operator
      tmp_1.copy_from_raw( _dim_domain, tmp_data + ipar[ 21 ] - 1 );
      apply( tmp_1, tmp_2, trans, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + ipar[ 22 ] - 1 );
      continue;
    } else if ( rci == 0 ) {  // success
      dfgmres_get( &size, solution.data( ), rhs_copy.data( ), &rci, ipar, dpar,
        tmp_data, &iter );
      n_iterations = iter;
      relative_residual_error = dpar[ 4 ] / dpar[ 2 ];
      break;
    } else {
      std::cout << "Only RCI codes 0,1 supported." << std::endl;
      return false;
    }
  }

  return true;
}

bool besthea::linear_algebra::linear_operator::mkl_fgmres_solve(
  const linear_operator & preconditioner, const vector_type & rhs,
  vector_type & solution, sc & relative_residual_error, lo & n_iterations,
  lo n_iterations_until_restart, bool trans, bool trans_preconditioner ) const {
  if ( _dim_domain != _dim_range || _dim_domain != rhs.size( )
    || _dim_domain != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  if ( n_iterations_until_restart == 0 ) {
    n_iterations_until_restart = n_iterations;
  }

  lo size = _dim_domain;

  lo rci;
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp( ( 2 * n_iterations_until_restart + 1 ) * size
    + n_iterations_until_restart * ( n_iterations_until_restart + 9 ) / 2 + 1 );
  sc * tmp_data = tmp.data( );

  vector_type tmp_1( size );
  vector_type tmp_2( size );
  vector_type rhs_copy( rhs );

  dfgmres_init(
    &size, solution.data( ), rhs_copy.data( ), &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "Failed to initialize MKL CG." << std::endl;
    return false;
  }

  bool silent = ( n_iterations_until_restart != ipar[ 4 ] );

  ipar[ 4 ] = n_iterations;  // maximum number of iterations
  ipar[ 7 ] = 1;             // perform the iteration stopping test
  ipar[ 8 ] = 1;             // do the residual stopping test
  ipar[ 9 ] = 0;             // do not request user stopping test
  ipar[ 10 ] = 1;            // preconditioned
  ipar[ 11 ] = 1;  // perform test for zero norm of generated direction
  ipar[ 14 ]
    = n_iterations_until_restart;  // number of iterations before restart
  if ( silent ) {
    ipar[ 6 ] = 0;  // disable the verbosity in case of non-default iterations
                    // until restart
  }

  dpar[ 0 ] = relative_residual_error;  // relative tolerance

  dfgmres_check(
    &size, solution.data( ), rhs_copy.data( ), &rci, ipar, dpar, tmp_data );
  if ( rci == -1100 ) {
    std::cout << "MKL parameters incorrect." << std::endl;
    return false;
  }

  while ( true ) {
    dfgmres(
      &size, solution.data( ), rhs_copy.data( ), &rci, ipar, dpar, tmp_data );

    if ( rci == 1 ) {  // apply operator
      tmp_1.copy_from_raw( _dim_domain, tmp_data + ipar[ 21 ] - 1 );
      apply( tmp_1, tmp_2, trans, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + ipar[ 22 ] - 1 );
      continue;
    } else if ( rci == 3 ) {  // apply preconditioner
      tmp_1.copy_from_raw( _dim_domain, tmp_data + ipar[ 21 ] - 1 );
      preconditioner.apply( tmp_1, tmp_2, trans_preconditioner, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + ipar[ 22 ] - 1 );
      continue;
    } else if ( rci == 0 ) {  // success
      dfgmres_get( &size, solution.data( ), rhs_copy.data( ), &rci, ipar, dpar,
        tmp_data, &iter );
      n_iterations = iter;
      relative_residual_error = dpar[ 4 ] / dpar[ 2 ];
      break;
    } else {
      std::cout << "Only RCI codes 0,1,3 supported." << std::endl;
      return false;
    }
  }

  return true;
}

sc besthea::linear_algebra::linear_operator::estimate_max_singular_value( )
  const {
  sc est_largest_sing_val = -1.0;

  vector_type iteration_vec( _dim_domain, false );
  iteration_vec.random_fill( 0.0, 1.0 );
  iteration_vec.scale( 1.0 / iteration_vec.norm( ) );
  vector_type prod_A( _dim_range, true );
  vector_type prod_A_adj( _dim_domain, true );

  // do 20 iterations of the power method
  for ( lo i = 0; i < 20; ++i ) {
    apply( iteration_vec, prod_A );
    apply( prod_A, prod_A_adj, true );
    est_largest_sing_val = std::sqrt( prod_A_adj.norm( ) );

    // prepare for next iteration step
    iteration_vec.copy( prod_A_adj );
    iteration_vec.scale(
      1.0 / ( est_largest_sing_val * est_largest_sing_val ) );
  }
  return est_largest_sing_val;
}
