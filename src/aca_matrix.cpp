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

#include "besthea/aca_matrix.h"

#define LR_FAILURE ( 0 )
#define LR_TRY ( 1 )
#define LR_SUCCESS ( 2 )

using besthea::linear_algebra::full_matrix;

besthea::linear_algebra::aca_matrix::aca_matrix( sc eps, lo max_rank )
  : _eps( eps ), _max_rank( max_rank ), _full_size( 0 ), _compressed_size( 0 ) {
}

void besthea::linear_algebra::aca_matrix::add_aca_matrix(
  full_matrix & new_matrix, mesh::general_spacetime_cluster * source,
  mesh::general_spacetime_cluster * target,
  sc svd_recompression_reference_value ) {
  full_matrix * u
    = new full_matrix( new_matrix.get_n_rows( ), new_matrix.get_n_columns( ) );
  full_matrix * v
    = new full_matrix( new_matrix.get_n_columns( ), new_matrix.get_n_rows( ) );

  _full_size += new_matrix.get_n_rows( ) * new_matrix.get_n_columns( );

  besthea::bem::kernel_from_matrix nf_matrix_kernel( &new_matrix );
  lo rank;

  if ( compute_low_rank_block( new_matrix.get_n_rows( ),
         new_matrix.get_n_columns( ), *u, *v, nf_matrix_kernel, rank, true,
         svd_recompression_reference_value ) ) {
    if ( rank != 0 ) {
      _u.push_back( u );
      _v.push_back( v );
      _sources_aca.push_back( source );
      _targets_aca.push_back( target );

      vector_type * y_tmp_1 = new vector_type( v->get_n_columns( ) );
      vector_type * y_tmp_2 = new vector_type( u->get_n_columns( ) );
      _y_tmp_1.push_back( y_tmp_1 );
      _y_tmp_2.push_back( y_tmp_2 );

      // std::cout << u->get_n_rows( ) << " " << u->get_n_columns( ) << " " <<
      // v->get_n_rows( ) << " " << v->get_n_columns( ) << " "<<
      // svd_recompression_reference_value <<std::endl;
      _compressed_size += ( u->get_n_rows( ) * u->get_n_columns( )
        + v->get_n_rows( ) * v->get_n_columns( ) );
      // std::cout << "Success" << std::endl;
    } else {
      //    std::cout << "Zero rank" << std::endl;
      delete u;
      delete v;
      _u.push_back( nullptr );
      _v.push_back( nullptr );
      _sources_aca.push_back( source );
      _targets_aca.push_back( target );
      _y_tmp_1.push_back( nullptr );
      _y_tmp_2.push_back( nullptr );
    }

    //     // todo: Delete me
    //     sc frob_norm = new_matrix.frobenius_norm( );
    //     new_matrix.multiply( *u, *v, false, true, 1.0, -1.0 );
    //     sc frob_error = new_matrix.frobenius_norm( );
    //     // todo: remove me
    // #pragma omp critical( debug )
    //     {
    //       std::cout << "rank: " << rank << ", ";
    //       std::cout << "target: ";
    //       target->print_short( );
    //       std::cout << "; source: ";
    //       source->print_short( );
    //       std::cout << "; abs. error: " << frob_error
    //                 << ", rel. error: " << frob_error / frob_norm <<
    //                 std::endl;
    //     }
  } else {
    //     // todo: remove me
    // #pragma omp critical( debug )
    //     {
    //       std::cout << "unsuccessful: ";
    //       std::cout << "target: ";
    //       target->print_short( );
    //       std::cout << "; source: ";
    //       source->print_short( );
    //       std::cout << ";" << std::endl;
    //     }
    full_matrix * full = new full_matrix( new_matrix );
    _full.push_back( full );
    _sources_full.push_back( source );
    _targets_full.push_back( target );

    _compressed_size += new_matrix.get_n_rows( ) * new_matrix.get_n_columns( );

    delete u;
    delete v;
  }
}

void besthea::linear_algebra::aca_matrix::add_full_matrix(
  full_matrix & new_matrix, mesh::general_spacetime_cluster * source,
  mesh::general_spacetime_cluster * target ) {
  // std::cout << "Full matrix" << std::endl;
  full_matrix * full = new full_matrix( new_matrix );
  _full.push_back( full );
  _sources_full.push_back( source );
  _targets_full.push_back( target );
  _full_size += new_matrix.get_n_rows( ) * new_matrix.get_n_columns( );
  _compressed_size += new_matrix.get_n_rows( ) * new_matrix.get_n_columns( );
}

template< typename T >
bool besthea::linear_algebra::aca_matrix::compute_low_rank_block( lo row_dim,
  lo col_dim, full_matrix & u, full_matrix & v,
  besthea::bem::low_rank_kernel< T > & kernel, lo & rank,
  bool enable_svd_recompression, sc svd_recompression_reference_value ) {
  bool ret_val = false;
  sc eps;
  int result = compute_aca_block< T >( row_dim, col_dim, u, v, kernel,
    enable_svd_recompression, svd_recompression_reference_value, eps, rank );

  if ( result == LR_SUCCESS ) {
    ret_val = true;
  }

  return ret_val;
}

template< typename T >
int besthea::linear_algebra::aca_matrix::compute_aca_block( lo row_dim,
  lo col_dim, full_matrix & u, full_matrix & v,
  besthea::bem::low_rank_kernel< T > & kernel, bool enable_svd_recompression,
  sc svd_recompression_reference_value, sc & eps, lo & rank ) {
  int ok = LR_FAILURE;      // status of the approximation
  lo i, j, ell;             // indices
  lo ik, jk;                // Pivot-Indices
  lo k;                     // current rank
  lo *Zj, *Zi;              // auxiliary variables
  sc old_eps;               // old accuracy
  sc error2 = 0.0;          // square of the Frobenius-norm of the update
  sc frobnr2 = 0.0;         // square of the Frobenius-norm of the block
  sc crit2;                 // auxiliary variable
  sc tmp, max, pmax = 1.0;  // auxiliary variables
  sc errU, errV;            // variables used for error computation
  sc scale;                 // auxiliary variable
  sc *p, *q;                // pointer to current rank-1-matrix
  sc * uext = u.data( );
  sc * vext = v.data( );

  old_eps = _eps;

  Zi = new lo[ row_dim ];
  Zj = new lo[ col_dim ];

  // initialize lists of indices
  for ( i = 0; i < row_dim; i++ ) Zi[ i ] = 1;
  for ( j = 0; j < col_dim; j++ ) Zj[ j ] = 1;

  // initialisation of indices
  k = 0;
  ik = 0;
  jk = 0;

  while ( 1 ) {
    p = uext + k * row_dim;  // memory address p = u_k
    // compute u_k
    {
      // generate a new column
      //      #pragma omp parallel for
      for ( i = 0; i < row_dim; i++ )
        if ( Zi[ i ] )
          p[ i ] = kernel.evaluate( i, jk );  // F(i, jk);

      // compute the residuum
      for ( ell = 0; ell < k; ell++ ) {
        scale = vext[ ell * col_dim + jk ];
        q = uext + ell * row_dim;
        for ( i = 0; i < row_dim; i++ )
          if ( Zi[ i ] )
            p[ i ] -= scale * q[ i ];
      }

      // compute the maximum (pivot)
      {
        max = 0.0;
        for ( i = 0; i < row_dim; i++ ) {
          if ( Zi[ i ] ) {
            tmp = std::abs( p[ i ] );

            if ( tmp > max ) {
              max = tmp;
              ik = i;
            }
          }
        }
        pmax = std::max( pmax, max );
      }
    }

    // check for zero column (TODO: currently hard coded)
    if ( max < 1.0e-30 * pmax ) {
      Zj[ jk ] = 0;
      for ( jk = 0; jk < col_dim && Zj[ jk ] == 0; jk++ ) {
      };

      if ( jk == col_dim )  // all columns considered
      {
        if ( enable_svd_recompression && k > 0 ) {
          recompression(
            u, v, row_dim, col_dim, svd_recompression_reference_value, k );
        }
        error2 = frobnr2 * old_eps * old_eps;

        if ( k * ( col_dim + row_dim ) > col_dim * row_dim ) {
          ok = LR_TRY;
        } else {
          ok = LR_SUCCESS;
        }

        break;
      }
    } else {
      // scale u_k
      scale = 1.0 / p[ ik ];
      for ( i = 0; i < row_dim; i++ )
        if ( Zi[ i ] )
          p[ i ] *= scale;

      q = vext + k * col_dim;  // memory address q = v_k

      // compute v_k
      {
        // generate new row
        //        #pragma omp parallel for
        for ( j = 0; j < col_dim; j++ )
          if ( Zj[ j ] )
            q[ j ] = kernel.evaluate( ik, j );  // F(ik, j);

        // compute the residuum
        for ( ell = 0; ell < k; ell++ ) {
          scale = uext[ ell * row_dim + ik ];
          p = vext + ell * col_dim;
          for ( j = 0; j < col_dim; j++ )
            if ( Zj[ j ] )
              q[ j ] -= p[ j ] * scale;
        }

        // exclude rows and columns in the future iterations.
        Zi[ ik ] = 0;
        Zj[ jk ] = 0;

        // compute the maximum (pivot)
        max = 0.0;

        for ( j = 0; j < col_dim; j++ ) {
          if ( Zj[ j ] ) {
            tmp = std::abs( q[ j ] );
            if ( tmp > max ) {
              max = tmp;
              jk = j;
            }
          }
        }
        pmax = std::max( pmax, max );
      }

      // compute the stopping criterion
      {
        for ( ell = 0; ell < k; ell++ ) {
          errU = cblas_ddot(
            row_dim, &uext[ ell * row_dim ], 1, &uext[ k * row_dim ], 1 );
          errV = cblas_ddot(
            col_dim, &vext[ k * col_dim ], 1, &vext[ ell * col_dim ], 1 );
          frobnr2 += 2.0 * errU * errV;
        }

        errU = cblas_ddot(
          row_dim, &uext[ k * row_dim ], 1, &uext[ k * row_dim ], 1 );
        errV = cblas_ddot(
          col_dim, &vext[ k * col_dim ], 1, &vext[ k * col_dim ], 1 );
        error2 = errU * errV;
        frobnr2 += error2;

        crit2 = ( _eps ) * (_eps) *frobnr2;
      }

      // increase the rank
      k++;

      // check the stopping criterion
      if ( error2 < crit2 ) {
        if ( enable_svd_recompression && k > 0 ) {
          recompression(
            u, v, row_dim, col_dim, svd_recompression_reference_value, k );
        }

        if ( k * ( col_dim + row_dim ) > col_dim * row_dim )
          ok = LR_TRY;
        else
          ok = LR_SUCCESS;

        break;
      } else {
        if ( k >= _max_rank ) {
          ok = LR_FAILURE;
          break;
        } else if ( k >= std::min( col_dim, row_dim ) ) {
          if ( enable_svd_recompression ) {
            recompression(
              u, v, row_dim, col_dim, svd_recompression_reference_value, k );
          }
          error2 = frobnr2 * old_eps * old_eps;
          if ( k * ( col_dim + row_dim ) > col_dim * row_dim )
            ok = LR_TRY;
          else
            ok = LR_SUCCESS;

          break;
        }
      }
    }
  }

  // resize u, v to achieved rank
  u.resize( row_dim, k, false );
  v.resize( col_dim, k, false );
  eps = sqrt( error2 / frobnr2 );
  rank = k;

  delete[] Zi;
  delete[] Zj;

  return ( ok );
}

void besthea::linear_algebra::aca_matrix::apply_aca_block( lou block,
  const vector_type & x, vector_type & y, bool trans, sc alpha,
  sc beta ) const {
  if ( _u[ block ] == nullptr ) {
    y.scale( beta );
    return;
  }
  if ( !trans ) {
    _v[ block ]->apply( x, *( _y_tmp_1[ block ] ), true, alpha, 0.0 );
    _u[ block ]->apply( *( _y_tmp_1[ block ] ), y, false, 1.0, beta );
  } else {
    _u[ block ]->apply( x, *( _y_tmp_2[ block ] ), true, alpha, 0.0 );
    _v[ block ]->apply( *( _y_tmp_2[ block ] ), y, false, 1.0, beta );
  }
}

void besthea::linear_algebra::aca_matrix::apply_full_block( lou block,
  const vector_type & x, vector_type & y, bool trans, sc alpha,
  sc beta ) const {
  _full[ block ]->apply( x, y, trans, alpha, beta );
}

void besthea::linear_algebra::aca_matrix::recompression( full_matrix & ue,
  full_matrix & ve, const lo rows, const lo cols,
  sc svd_recompression_refernce_value, lo & rank ) {
  int rk_svd_flops = 0;  // Floating-Point operations for recompression
  lo kmax;               // old rank
  long i, j, k;          // indices
  long lwork;            // size of workspace for LAPACK
  long info;             // returninfo of LAPACK-routines
  int minsize, maxsize;  // block sizes
  sc * r_work;           // workspace for LAPACK
  sc * sigma;            // singular values
  sc *atmp, *btmp;       // auxiliary matrices
  sc * rarb;             // R-matrix
  sc *tau1, *tau2;       // auxiliary factor for QR-decomposition
  sc *u, *v;             // auxiliary matrices for SVD
  sc * csigma;           // singular values
  sc * qr_work;          // Workspace for LAPACK

  sc * uext = ue.data( );
  sc * vext = ve.data( );

  minsize = std::min( rows, cols );
  maxsize = std::max( rows, cols );

  kmax = rank;

  atmp = (sc *) malloc( ( rows + cols ) * kmax * sizeof( sc ) );
  btmp = &atmp[ rows * kmax ];

  // copy U to auxiliary container
  for ( k = 0; k < kmax; k++ )
    for ( i = 0; i < rows; i++ ) atmp[ i + k * rows ] = uext[ i + k * rows ];

#ifdef DEBUG_COMPRESS
  printBlock< T >( "U", atmp, rows, kmax );
#endif

  // copy V to auxiliary container
  for ( k = 0; k < kmax; k++ )
    for ( j = 0; j < cols; j++ ) btmp[ j + k * cols ] = vext[ j + k * cols ];

#ifdef DEBUG_COMPRESS
  printBlock< T >( "V", btmp, cols, kmax );
#endif

  // allocate storage for auxiliary variables and workspace
  lwork = 8 * minsize + maxsize;
  qr_work = (sc *) malloc(
    ( lwork + rows + cols + minsize + 3 * kmax * kmax ) * sizeof( sc ) );
  r_work = (sc *) malloc( ( minsize + 5 * kmax * kmax ) * sizeof( sc ) );
  tau1 = &qr_work[ lwork ];
  tau2 = &qr_work[ lwork + rows ];
  csigma = &qr_work[ lwork + rows + cols ];
  rarb = &qr_work[ lwork + rows + cols + minsize ];
  u = &qr_work[ lwork + rows + cols + minsize + kmax * kmax ];
  v = &qr_work[ lwork + rows + cols + minsize + kmax * kmax + kmax * kmax ];
  sigma = &r_work[ 5 * kmax * kmax ];

  // QR-decomposition of U
  dgeqrf( &rows, &kmax, atmp, &rows, tau1, qr_work, &lwork, &info );
  rk_svd_flops += 2 * kmax * kmax * rows;

#ifdef DEBUG_COMPRESS
  if ( info != 0 ) {
    std::cout << "fatal error in recompression: RU" << std::endl;
    exit( 1 );
  }
  printBlockFill( "RU", atmp, rows, kmax, rows );
#endif

  // QR-decomposition of V
  dgeqrf( &cols, &kmax, btmp, &cols, tau2, qr_work, &lwork, &info );
  rk_svd_flops += 2 * kmax * kmax * cols;

#ifdef DEBUG_COMPRESS
  if ( info != 0 ) {
    std::cout << "fatal error in recompression: RV" << std::endl;
    exit( 1 );
  }
  printBlockFill( "RV", btmp, cols, kmax, cols );
#endif

  // determine RU*RV'  (RU is the R matrix in QR decomposition of U, RV similar)
  for ( i = 0; i < kmax * kmax; i++ ) rarb[ i ] = 0.0;

  for ( i = 0; i < kmax; i++ )
    for ( j = 0; j < kmax; j++ )
      for ( k = std::max( i, j ); k < kmax; k++ )
        // replaces for (k=0; k<kmax; k++) if ((k>=i) && (k>=j))
        rarb[ i + kmax * j ] += atmp[ i + k * rows ] * btmp[ j + k * cols ];

#ifdef DEBUG_COMPRESS
  printBlock< sc >( "R", rarb, kmax, kmax );
#endif

  // determine the matrix QU, i.e. Q in the QR decomposition of U
  dorgqr( &rows, &kmax, &kmax, atmp, &rows, tau1, qr_work, &lwork, &info );

#ifdef DEBUG_COMPRESS
  if ( info != 0 ) {
    std::cout << "fatal error in recompression: QU" << std::endl;
    exit( 1 );
  }
  printBlock< T >( "QU", atmp, rows, kmax );
#endif

  // determine the matrix QV, i.e. Q in the QR decomposition of V
  dorgqr( &cols, &kmax, &kmax, btmp, &cols, tau2, qr_work, &lwork, &info );

#ifdef DEBUG_COMPRESS
  if ( info != 0 ) {
    std::cout << "fatal error in recompression: QV" << std::endl;
    exit( 1 );
  }
  printBlock< sc >( "QV", btmp, cols, kmax );
#endif

  // sc diff = 0.0;  // auxiliary variable
  // for ( i = 0; i < kmax * kmax; i++ ) {
  //   if ( std::abs( rarb[ i ] ) < ( *eps ) * ( *eps ) )
  //     rarb[ i ] = 0.0;
  //   diff += std::abs( rarb[ i ] );
  // }

  // if ( diff < ( *eps ) * ( *eps ) ) {
  //   std::cout << "RECOMPRESSION: setting new rank to 0" << std::endl;
  //   *rank = 0;
  // } else
  {
    // determine the SVD of RU*RV'
    dgesvd( "A", "A", &kmax, &kmax, rarb, &kmax, sigma, u, &kmax, v, &kmax,
      qr_work, &lwork, &info );
    rk_svd_flops += 60 * kmax * kmax * kmax;

    for ( i = 0; i < kmax; i++ ) csigma[ i ] = (sc) sigma[ i ];

#ifdef DEBUG_COMPRESS
    if ( info != 0 ) {
      std::cout << "fatal error in recompression: SVN of R" << std::endl;
      exit( 1 );
    }
    printBlock< sc >( "UR", u, kmax, kmax );
    printBlock< sc >( "SR", sigma, 1, kmax );
    printBlockT< sc >( "VR", v, kmax, kmax );
#endif

    for ( i = 0; i < kmax; i++ ) {
      // scale UR with singular values
      cblas_dscal( kmax, csigma[ i ], &u[ i * kmax ], 1 );  // for MKL and cblas
      // dscal_ (&kmax, &sigma[i], &u[i*kmax], eins_);
      // copy VR
      cblas_dcopy(
        kmax, &v[ i ], kmax, &rarb[ i * kmax ], 1 );  // for MKL and cblas
      // dcopy_ (&kmax, &v[i], &kmax, &rarb[i*kmax], eins_);
    }

#ifdef DEBUG_COMPRESS
    printBlock< sc >( "URS", u, kmax, kmax );
#endif

    sc actual_reference_value = sigma[ 0 ];
    if ( svd_recompression_refernce_value > 0.0 ) {
      actual_reference_value = svd_recompression_refernce_value;
    }

    for ( k = 0; ( k < kmax ) && ( k < rows ) && ( k < cols )
          && ( sigma[ k ] > _eps * actual_reference_value );
          k++ ) {
    };
    // std::cout << "RECOMPRESSION: first singular value is " << sigma[ 0 ]
    //           << ", original rank " << kmax << ", new rank " << k;
    // if ( k < kmax ) {
    //   std::cout << ", first truncated svd: " << sigma[ k ];
    // }
    // std::cout << std::endl;

    rank = k;
    if ( k > 0 ) {
      // determine U = QU*URS
      cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, rows, k, kmax,
        1.0, atmp, rows, u, kmax, 0.0, uext, rows );
      // dgemm_ (ntrans, ntrans, &rows, &k, &kmax, deins_, atmp, &rows, u,
      // &kmax, mydnull_, U, &rows);

      // determine V = QV*VR
      cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, cols, k, kmax,
        1.0, btmp, cols, rarb, kmax, 0.0, vext, cols );
      // dgemm_ (ntrans, ntrans, &cols, &k ,&kmax, deins_, btmp, &cols, rarb,
      // &kmax, mydnull_, V, &cols);
      rk_svd_flops += 2 * ( rows + cols ) * k * k;
    }

#ifdef DEBUG_COMPRESS
    printBlock< T >( "UC", uext, rows, *rank );
    printBlock< T >( "VC", vext, cols, *rank );
#endif
  }

  free( qr_work );
  free( r_work );
  free( atmp );

  return;
}
