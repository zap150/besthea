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

#include "besthea/block_linear_operator.h"

#include <mkl_rci.h>
#include <vector>

bool besthea::linear_algebra::block_linear_operator::mkl_cg_solve(
  const block_vector_type & rhs, block_vector_type & solution,
  sc & relative_residual_error, lo & n_iterations ) const {
  lo size = _dim_domain * _block_dim;

  if ( _dim_domain != _dim_range || size != rhs.size( )
    || size != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  lo rci;
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp( size * 3 );  // need _dim_domain * 4 for preconditioned
  sc * tmp_data = tmp.data( );

  block_vector_type tmp_1( _block_dim, _dim_domain );
  block_vector_type tmp_2( _block_dim, _dim_domain );

  vector_type rhs_contiguous( size );
  vector_type solution_contiguous( size );
  rhs.copy_to_vector( rhs_contiguous );
  solution.copy_to_vector( solution_contiguous );

  dcg_init( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci,
    ipar, dpar, tmp_data );
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

  dcg_check( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci,
    ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "MKL parameters incorrect." << std::endl;
    return false;
  }

  while ( true ) {
    dcg( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci, ipar,
      dpar, tmp_data );

    if ( rci == 1 ) {  // apply operator
      tmp_1.copy_from_raw( _block_dim, _dim_domain, tmp_data );
      apply( tmp_1, tmp_2, false, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + size );
      continue;
    } else if ( rci == 0 ) {  // success
      dcg_get( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci,
        ipar, dpar, tmp_data, &iter );
      solution.copy_from_vector( _block_dim, _dim_domain, solution_contiguous );
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

bool besthea::linear_algebra::block_linear_operator::mkl_cg_solve(
  const block_linear_operator & preconditioner, const block_vector_type & rhs,
  block_vector_type & solution, sc & relative_residual_error,
  lo & n_iterations ) const {
  lo size = _dim_domain * _block_dim;

  if ( _dim_domain != _dim_range || size != rhs.size( )
    || size != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  lo rci;
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp( size * 4 );
  sc * tmp_data = tmp.data( );

  block_vector_type tmp_1( _block_dim, _dim_domain );
  block_vector_type tmp_2( _block_dim, _dim_domain );

  vector_type rhs_contiguous( size );
  vector_type solution_contiguous( size );
  rhs.copy_to_vector( rhs_contiguous );
  solution.copy_to_vector( solution_contiguous );

  dcg_init( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci,
    ipar, dpar, tmp_data );
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

  dcg_check( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci,
    ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "MKL parameters incorrect." << std::endl;
    return false;
  }

  while ( true ) {
    dcg( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci, ipar,
      dpar, tmp_data );

    if ( rci == 1 ) {  // apply operator
      tmp_1.copy_from_raw( _block_dim, _dim_domain, tmp_data );
      apply( tmp_1, tmp_2, false, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + size );
      continue;
    } else if ( rci == 3 ) {  // apply preconditioner
      tmp_1.copy_from_raw( _block_dim, _dim_domain, tmp_data );
      preconditioner.apply( tmp_1, tmp_2, false, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + size );
      continue;
    } else if ( rci == 0 ) {  // success
      dcg_get( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci,
        ipar, dpar, tmp_data, &iter );
      solution.copy_from_vector( _block_dim, _dim_domain, solution_contiguous );
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

bool besthea::linear_algebra::block_linear_operator::mkl_fgmres_solve(
  const block_vector_type & rhs, block_vector_type & solution,
  sc & relative_residual_error, lo & n_iterations,
  lo n_iterations_until_restart, bool trans ) const {
  lo size = _dim_domain * _block_dim;

  if ( _dim_domain != _dim_range || size != rhs.size( )
    || size != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  if ( n_iterations_until_restart == 0 ) {
    n_iterations_until_restart = n_iterations;
  }

  lo rci;
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp( ( 2 * n_iterations_until_restart + 1 ) * size
    + n_iterations_until_restart * ( n_iterations_until_restart + 9 ) / 2 + 1 );
  sc * tmp_data = tmp.data( );

  block_vector_type tmp_1( _block_dim, _dim_domain );
  block_vector_type tmp_2( _block_dim, _dim_domain );

  vector_type rhs_contiguous( size );
  vector_type solution_contiguous( size );
  rhs.copy_to_vector( rhs_contiguous );
  solution.copy_to_vector( solution_contiguous );

  dfgmres_init( &size, solution_contiguous.data( ), rhs_contiguous.data( ),
    &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "Failed to initialize MKL FGMRES." << std::endl;
    return false;
  }

  ipar[ 0 ] = size;          // size of the problem
  ipar[ 4 ] = n_iterations;  // maximum number of iterations
  ipar[ 7 ] = 1;             // perform the iteration stopping test
  ipar[ 8 ] = 1;             // do the residual stopping test
  ipar[ 9 ] = 0;             // do not request user stopping test
  ipar[ 10 ] = 0;            // non-preconditioned
  ipar[ 11 ] = 1;  // perform test for zero norm of generated direction
  ipar[ 14 ]
    = n_iterations_until_restart;  // number of iterations before restart

  dpar[ 0 ] = relative_residual_error;  // relative tolerance

  dfgmres_check( &size, solution_contiguous.data( ), rhs_contiguous.data( ),
    &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "MKL parameters incorrect." << std::endl;
    return false;
  }

  while ( true ) {
    dfgmres( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci,
      ipar, dpar, tmp_data );

    if ( rci == 1 ) {  // apply operator
      tmp_1.copy_from_raw( _block_dim, _dim_domain, tmp_data + ipar[ 21 ] - 1 );
      apply( tmp_1, tmp_2, trans, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + ipar[ 22 ] - 1 );
      continue;
    } else if ( rci == 0 ) {  // success
      dfgmres_get( &size, solution_contiguous.data( ), rhs_contiguous.data( ),
        &rci, ipar, dpar, tmp_data, &iter );
      solution.copy_from_vector( _block_dim, _dim_domain, solution_contiguous );
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

bool besthea::linear_algebra::block_linear_operator::mkl_fgmres_solve(
  const block_linear_operator & preconditioner, const block_vector_type & rhs,
  block_vector_type & solution, sc & relative_residual_error, lo & n_iterations,
  lo n_iterations_until_restart, bool trans, bool trans_preconditioner ) const {
  lo size = _dim_domain * _block_dim;

  if ( _dim_domain != _dim_range || size != rhs.size( )
    || size != solution.size( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return false;
  }

  if ( n_iterations_until_restart == 0 ) {
    n_iterations_until_restart = n_iterations;
  }

  lo rci;
  lo iter;
  lo ipar[ 128 ];
  sc dpar[ 128 ];
  std::vector< sc > tmp( ( 2 * n_iterations_until_restart + 1 ) * size
    + n_iterations_until_restart * ( n_iterations_until_restart + 9 ) / 2 + 1 );
  sc * tmp_data = tmp.data( );

  block_vector_type tmp_1( _block_dim, _dim_domain );
  block_vector_type tmp_2( _block_dim, _dim_domain );

  vector_type rhs_contiguous( size );
  vector_type solution_contiguous( size );
  rhs.copy_to_vector( rhs_contiguous );
  solution.copy_to_vector( solution_contiguous );

  dfgmres_init( &size, solution_contiguous.data( ), rhs_contiguous.data( ),
    &rci, ipar, dpar, tmp_data );
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
  ipar[ 11 ] = 1;  // perform test for zero norm of generated direction
  ipar[ 14 ]
    = n_iterations_until_restart;  // number of iterations before restart

  dpar[ 0 ] = relative_residual_error;  // relative tolerance

  dfgmres_check( &size, solution_contiguous.data( ), rhs_contiguous.data( ),
    &rci, ipar, dpar, tmp_data );
  if ( rci ) {
    std::cout << "MKL parameters incorrect." << std::endl;
    return false;
  }

  while ( true ) {
    dfgmres( &size, solution_contiguous.data( ), rhs_contiguous.data( ), &rci,
      ipar, dpar, tmp_data );

    if ( rci == 1 ) {  // apply operator
      tmp_1.copy_from_raw( _block_dim, _dim_domain, tmp_data + ipar[ 21 ] - 1 );
      apply( tmp_1, tmp_2, trans, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + ipar[ 22 ] - 1 );
      continue;
    } else if ( rci == 3 ) {  // apply preconditioner
      tmp_1.copy_from_raw( _block_dim, _dim_domain, tmp_data + ipar[ 21 ] - 1 );
      preconditioner.apply( tmp_1, tmp_2, trans_preconditioner, 1.0, 0.0 );
      tmp_2.copy_to_raw( tmp_data + ipar[ 22 ] - 1 );
      continue;
    } else if ( rci == 0 ) {  // success
      dfgmres_get( &size, solution_contiguous.data( ), rhs_contiguous.data( ),
        &rci, ipar, dpar, tmp_data, &iter );
      solution.copy_from_vector( _block_dim, _dim_domain, solution_contiguous );
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

bool besthea::linear_algebra::block_linear_operator::gmres_solve(
  const block_vector_type & rhs, block_vector_type & solution,
  sc & relative_residual_error, lo & n_iterations,
  const block_linear_operator & prec, bool trans ) const {
  // initialize data
  lo max_it = n_iterations;
  n_iterations = 0;
  sc hs;
  lo block_size = rhs.get_block_size( );
  lo size_of_blocks = rhs.get_size_of_block( );
  block_vector_type r( rhs );
  std::vector< block_vector_type > V(
    max_it );  // orthogonalized search directions
  block_vector_type vs( rhs.get_block_size( ), rhs.get_size_of_block( ),
    true );  // new search direction
  block_vector_type vs_prec( rhs.get_block_size( ), rhs.get_size_of_block( ),
    true );  // auxiliary result of preconditioning
  std::vector< std::vector< sc > > H(
    max_it + 1 );  // Heesenberg matrix of the minimization problem
  std::vector< sc > gamma( max_it + 1 );  // rhs of minimization problem
  std::vector< sc > c( max_it + 1 );      // coeffs of Givens rotation
  std::vector< sc > s( max_it + 1 );      // coeffs of Givens rotation
  sc norm_vs;                             // h_k+1,k
  block_vector_type u_tilde( solution.get_block_size( ),
    solution.get_size_of_block( ), true );  // solution=prec*u_tilde
  sc gmres_eps = 1e-20;

  this->apply( solution, r, trans, -1.0, 1.0 );
  gamma[ 0 ] = r.norm( );

  if ( std::abs( gamma[ 0 ] ) < gmres_eps ) {
    return true;
  }

  V[ 0 ].copy( r );
  V[ 0 ].scale( 1.0 / gamma[ 0 ] );

  sc ref_error = relative_residual_error * gamma[ 0 ];
  lo k = 0;

  while ( std::abs( gamma[ k ] ) > ref_error && k < max_it ) {
    prec.apply( V[ k ], vs_prec, false, 1.0, 0.0 );
    this->apply( vs_prec, vs, trans, 1.0, 0.0 );

    H[ k ].resize( k + 1 );

    for ( lo i = 0; i < k + 1; ++i ) {
      H[ k ][ i ] = V[ i ].dot( vs );
      vs.add( V[ i ], -H[ k ][ i ] );
    }

    for ( lo i = 0; i < k; ++i ) {
      hs = c[ i ] * H[ k ][ i ] + s[ i ] * H[ k ][ i + 1 ];
      H[ k ][ i + 1 ] = c[ i ] * H[ k ][ i + 1 ] - s[ i ] * H[ k ][ i ];
      H[ k ][ i ] = hs;
    }

    norm_vs = vs.norm( );
    if ( norm_vs < gmres_eps ) {
      k++;
      break;
    }

    V[ k + 1 ].resize( block_size );
    V[ k + 1 ].resize_blocks( size_of_blocks );
    V[ k + 1 ].copy( vs );
    V[ k + 1 ].scale( 1.0 / norm_vs );

    // coefficients of next Givens rotation
    sc beta = std::sqrt( norm_vs * norm_vs + H[ k ][ k ] * H[ k ][ k ] );
    c[ k ] = H[ k ][ k ] / beta;
    s[ k ] = norm_vs / beta;

    // new Givens rotation
    H[ k ][ k ] = beta;
    if ( beta < gmres_eps ) {
      throw std::runtime_error( "Gmres aborted, matrix seems to be singular" );
    }

    gamma[ k + 1 ] = -s[ k ] * gamma[ k ];
    gamma[ k ] *= c[ k ];

    ++k;
  }

  if ( k == 0 ) {
    if ( !( std::abs( gamma[ k ] ) < ref_error ) ) {
      std::cout << "Gmres failed, stopped after " << k << " iterations!"
                << std::endl;
    }
    n_iterations = k;
    return;
  }

  // solve minimization problem by inverting upper triangle Hessenberg matrix
  vector_type z( k );
  for ( lo i = k - 1; i >= 0; i-- ) {
    sc sum = 0;
    for ( lo j = i + 1; j < k; ++j ) {
      sum += H[ j ][ i ] * z[ j ];
    }
    z[ i ] = ( gamma[ i ] - sum ) / H[ i ][ i ];
  }

  for ( lo i = 0; i < k; ++i ) {
    u_tilde.add( V[ i ], z[ i ] );
  }

  prec.apply( u_tilde, solution, false, 1.0, 1.0 );

  n_iterations = k;
}

bool besthea::linear_algebra::block_linear_operator::gmres_solve(
  const block_vector_type & rhs, block_vector_type & solution,
  sc & relative_residual_error, lo & n_iterations, bool trans ) const {
  // initialize data

  lo max_it = n_iterations;
  n_iterations = 0;
  sc hs;
  lo block_size = rhs.get_block_size( );
  lo size_of_blocks = rhs.get_size_of_block( );
  block_vector_type r( rhs );
  std::vector< block_vector_type > V(
    max_it + 1 );  // orthogonalized search directions
  block_vector_type vs( rhs.get_block_size( ), rhs.get_size_of_block( ),
    true );  // new search direction
  block_vector_type vs_prec( rhs.get_block_size( ), rhs.get_size_of_block( ),
    true );  // auxiliary result of preconditioning
  std::vector< std::vector< sc > > H(
    max_it + 1 );  // Heesenberg matrix of the minimization problem
  std::vector< sc > gamma( max_it + 1 );  // rhs of minimization problem
  std::vector< sc > c( max_it + 1 );      // coeffs of Givens rotation
  std::vector< sc > s( max_it + 1 );      // coeffs of Givens rotation
  sc norm_vs;                             // h_k+1,k
  block_vector_type u_tilde( solution.get_block_size( ),
    solution.get_size_of_block( ), true );  // solution=prec*u_tilde
  sc gmres_eps = 1e-20;

  this->apply( solution, r, trans, -1.0, 1.0 );
  gamma[ 0 ] = r.norm( );

  if ( gamma[ 0 ] == 0.0 ) {
    return true;
  }

  V[ 0 ].copy( r );
  V[ 0 ].scale( 1.0 / gamma[ 0 ] );

  sc ref_error = relative_residual_error * gamma[ 0 ];
  lo k = 0;

  while ( std::abs( gamma[ k ] ) > ref_error && k < max_it ) {
    this->apply( V[ k ], vs, trans, 1.0, 0.0 );

    H[ k ].resize( k + 1 );

    for ( lo i = 0; i < k + 1; ++i ) {
      H[ k ][ i ] = V[ i ].dot( vs );
      vs.add( V[ i ], -H[ k ][ i ] );
    }

    for ( lo i = 0; i < k; ++i ) {
      hs = c[ i ] * H[ k ][ i ] + s[ i ] * H[ k ][ i + 1 ];
      H[ k ][ i + 1 ] = c[ i ] * H[ k ][ i + 1 ] - s[ i ] * H[ k ][ i ];
      H[ k ][ i ] = hs;
    }

    norm_vs = vs.norm( );
    if ( norm_vs == 0.0 ) {
      k++;
      break;
    }

    V[ k + 1 ].resize( block_size );
    V[ k + 1 ].resize_blocks( size_of_blocks );
    V[ k + 1 ].copy( vs );
    V[ k + 1 ].scale( 1.0 / norm_vs );

    // coefficients of next Givens rotation
    sc beta = std::sqrt( norm_vs * norm_vs + H[ k ][ k ] * H[ k ][ k ] );
    c[ k ] = H[ k ][ k ] / beta;
    s[ k ] = norm_vs / beta;

    // new Givens rotation
    H[ k ][ k ] = beta;
    if ( beta < gmres_eps ) {
      throw std::runtime_error( "Gmres aborted, matrix seems to be singular" );
    }

    gamma[ k + 1 ] = -s[ k ] * gamma[ k ];
    gamma[ k ] *= c[ k ];

    ++k;
  }

  if ( k == 0 ) {
    if ( !( std::abs( gamma[ k ] ) < ref_error ) ) {
      std::cout << "Gmres failed, stopped after " << k << " iterations!"
                << std::endl;
    }
    n_iterations = k;
    return true;
  }

  // solve minimization problem by inverting upper triangle Hessenberg matrix
  vector_type z( k );
  for ( lo i = k - 1; i >= 0; i-- ) {
    sc sum = 0;
    for ( lo j = i + 1; j < k; ++j ) {
      sum += H[ j ][ i ] * z[ j ];
    }
    z[ i ] = ( gamma[ i ] - sum ) / H[ i ][ i ];
  }

  for ( lo i = 0; i < k; ++i ) {
    u_tilde.add( V[ i ], z[ i ] );
  }

  solution.add( u_tilde );

  n_iterations = k;
}

bool besthea::linear_algebra::block_linear_operator::gmres_solve(
  const distributed_block_vector_type & rhs,
  distributed_block_vector_type & solution, sc & relative_residual_error,
  lo & n_iterations, const block_linear_operator & prec, bool trans ) const {
  // initialize data
  lo max_it = n_iterations;
  n_iterations = 0;
  sc hs;
  lo block_size = rhs.get_block_size( );
  lo size_of_blocks = rhs.get_size_of_block( );
  distributed_block_vector_type r( rhs );
  std::vector< distributed_block_vector_type > V(
    max_it );  // orthogonalized search directions
  distributed_block_vector_type vs( rhs.get_block_size( ),
    rhs.get_size_of_block( ),
    true );  // new search direction
  distributed_block_vector_type vs_prec( rhs.get_block_size( ),
    rhs.get_size_of_block( ),
    true );  // auxiliary result of preconditioning
  std::vector< std::vector< sc > > H(
    max_it + 1 );  // Heesenberg matrix of the minimization problem
  std::vector< sc > gamma( max_it + 1 );  // rhs of minimization problem
  std::vector< sc > c( max_it + 1 );      // coeffs of Givens rotation
  std::vector< sc > s( max_it + 1 );      // coeffs of Givens rotation
  sc norm_vs;                             // h_k+1,k
  distributed_block_vector_type u_tilde( solution.get_block_size( ),
    solution.get_size_of_block( ), true );  // solution=prec*u_tilde
  sc gmres_eps = 1e-20;

  this->apply( solution, r, trans, -1.0, 1.0 );
  gamma[ 0 ] = r.norm( );

  if ( std::abs( gamma[ 0 ] ) < gmres_eps ) {
    return true;
  }

  V[ 0 ].copy( r );
  V[ 0 ].scale( 1.0 / gamma[ 0 ] );

  sc ref_error = relative_residual_error * gamma[ 0 ];
  lo k = 0;

  while ( std::abs( gamma[ k ] ) > ref_error && k < max_it ) {
    prec.apply( V[ k ], vs_prec, false, 1.0, 0.0 );
    this->apply( vs_prec, vs, trans, 1.0, 0.0 );

    H[ k ].resize( k + 1 );

    for ( lo i = 0; i < k + 1; ++i ) {
      H[ k ][ i ] = V[ i ].dot( vs );
      vs.add( V[ i ], -H[ k ][ i ] );
    }

    for ( lo i = 0; i < k; ++i ) {
      hs = c[ i ] * H[ k ][ i ] + s[ i ] * H[ k ][ i + 1 ];
      H[ k ][ i + 1 ] = c[ i ] * H[ k ][ i + 1 ] - s[ i ] * H[ k ][ i ];
      H[ k ][ i ] = hs;
    }

    norm_vs = vs.norm( );
    if ( norm_vs < gmres_eps ) {
      k++;
      break;
    }

    V[ k + 1 ].resize( block_size );
    V[ k + 1 ].resize_blocks( size_of_blocks );
    V[ k + 1 ].copy( vs );
    V[ k + 1 ].scale( 1.0 / norm_vs );

    // coefficients of next Givens rotation
    sc beta = std::sqrt( norm_vs * norm_vs + H[ k ][ k ] * H[ k ][ k ] );
    c[ k ] = H[ k ][ k ] / beta;
    s[ k ] = norm_vs / beta;

    // new Givens rotation
    H[ k ][ k ] = beta;
    if ( beta < gmres_eps ) {
      throw std::runtime_error( "Gmres aborted, matrix seems to be singular" );
    }

    gamma[ k + 1 ] = -s[ k ] * gamma[ k ];
    gamma[ k ] *= c[ k ];

    ++k;
  }

  if ( k == 0 ) {
    if ( !( std::abs( gamma[ k ] ) < ref_error ) ) {
      std::cout << "Gmres failed, stopped after " << k << " iterations!"
                << std::endl;
    }
    n_iterations = k;
    return;
  }

  // solve minimization problem by inverting upper triangle Hessenberg matrix
  vector_type z( k );
  for ( lo i = k - 1; i >= 0; i-- ) {
    sc sum = 0;
    for ( lo j = i + 1; j < k; ++j ) {
      sum += H[ j ][ i ] * z[ j ];
    }
    z[ i ] = ( gamma[ i ] - sum ) / H[ i ][ i ];
  }

  for ( lo i = 0; i < k; ++i ) {
    u_tilde.add( V[ i ], z[ i ] );
  }

  prec.apply( u_tilde, solution, false, 1.0, 1.0 );

  n_iterations = k;
}

bool besthea::linear_algebra::block_linear_operator::gmres_solve(
  const distributed_block_vector_type & rhs,
  distributed_block_vector_type & solution, sc & relative_residual_error,
  lo & n_iterations, bool trans ) const {
  // initialize data

  lo max_it = n_iterations;
  n_iterations = 0;
  sc hs;
  lo block_size = rhs.get_block_size( );
  lo size_of_blocks = rhs.get_size_of_block( );
  distributed_block_vector_type r( rhs );
  std::vector< distributed_block_vector_type > V(
    max_it + 1 );  // orthogonalized search directions
  distributed_block_vector_type vs( rhs.get_block_size( ),
    rhs.get_size_of_block( ),
    true );  // new search direction
  distributed_block_vector_type vs_prec( rhs.get_block_size( ),
    rhs.get_size_of_block( ),
    true );  // auxiliary result of preconditioning
  std::vector< std::vector< sc > > H(
    max_it + 1 );  // Heesenberg matrix of the minimization problem
  std::vector< sc > gamma( max_it + 1 );  // rhs of minimization problem
  std::vector< sc > c( max_it + 1 );      // coeffs of Givens rotation
  std::vector< sc > s( max_it + 1 );      // coeffs of Givens rotation
  sc norm_vs;                             // h_k+1,k
  distributed_block_vector_type u_tilde( solution.get_block_size( ),
    solution.get_size_of_block( ), true );  // solution=prec*u_tilde
  sc gmres_eps = 1e-20;

  this->apply( solution, r, trans, -1.0, 1.0 );
  gamma[ 0 ] = r.norm( );

  if ( gamma[ 0 ] == 0.0 ) {
    return true;
  }

  V[ 0 ].copy( r );
  V[ 0 ].scale( 1.0 / gamma[ 0 ] );

  sc ref_error = relative_residual_error * gamma[ 0 ];
  lo k = 0;

  while ( std::abs( gamma[ k ] ) > ref_error && k < max_it ) {
    this->apply( V[ k ], vs, trans, 1.0, 0.0 );

    H[ k ].resize( k + 1 );

    for ( lo i = 0; i < k + 1; ++i ) {
      H[ k ][ i ] = V[ i ].dot( vs );
      vs.add( V[ i ], -H[ k ][ i ] );
    }

    for ( lo i = 0; i < k; ++i ) {
      hs = c[ i ] * H[ k ][ i ] + s[ i ] * H[ k ][ i + 1 ];
      H[ k ][ i + 1 ] = c[ i ] * H[ k ][ i + 1 ] - s[ i ] * H[ k ][ i ];
      H[ k ][ i ] = hs;
    }

    norm_vs = vs.norm( );
    if ( norm_vs == 0.0 ) {
      k++;
      break;
    }

    V[ k + 1 ].resize( block_size );
    V[ k + 1 ].resize_blocks( size_of_blocks );
    V[ k + 1 ].copy( vs );
    V[ k + 1 ].scale( 1.0 / norm_vs );

    // coefficients of next Givens rotation
    sc beta = std::sqrt( norm_vs * norm_vs + H[ k ][ k ] * H[ k ][ k ] );
    c[ k ] = H[ k ][ k ] / beta;
    s[ k ] = norm_vs / beta;

    // new Givens rotation
    H[ k ][ k ] = beta;
    if ( beta < gmres_eps ) {
      throw std::runtime_error( "Gmres aborted, matrix seems to be singular" );
    }

    gamma[ k + 1 ] = -s[ k ] * gamma[ k ];
    gamma[ k ] *= c[ k ];

    ++k;
  }

  if ( k == 0 ) {
    if ( !( std::abs( gamma[ k ] ) < ref_error ) ) {
      std::cout << "Gmres failed, stopped after " << k << " iterations!"
                << std::endl;
    }
    n_iterations = k;
    return true;
  }

  // solve minimization problem by inverting upper triangle Hessenberg matrix
  vector_type z( k );
  for ( lo i = k - 1; i >= 0; i-- ) {
    sc sum = 0;
    for ( lo j = i + 1; j < k; ++j ) {
      sum += H[ j ][ i ] * z[ j ];
    }
    z[ i ] = ( gamma[ i ] - sum ) / H[ i ][ i ];
  }

  for ( lo i = 0; i < k; ++i ) {
    u_tilde.add( V[ i ], z[ i ] );
  }

  solution.add( u_tilde );

  n_iterations = k;
}
