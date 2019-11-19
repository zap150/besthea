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

#include "besthea/sparse_matrix.h"

#include "Eigen/IterativeLinearSolvers"

besthea::linear_algebra::sparse_matrix::sparse_matrix( )
  : _data( ), _lu( ), _choleski( ) {
  _n_rows = 0;
  _n_columns = 0;
}

besthea::linear_algebra::sparse_matrix::sparse_matrix(
  const sparse_matrix & that )
  : _data( that._data ), _lu( ), _choleski( ) {
  _n_rows = that._n_rows;
  _n_columns = that._n_columns;
}

besthea::linear_algebra::sparse_matrix::sparse_matrix( los n_rows,
  los n_columns, std::vector< los > & row_indices,
  std::vector< los > & column_indices, std::vector< sc > & values )
  : _data( n_rows, n_columns ) {
  _n_rows = n_rows;
  _n_columns = n_columns;
  std::vector< Eigen::Triplet< sc, los > > triplet_list;
  triplet_list.reserve( row_indices.size( ) );

  for ( std::vector< los >::size_type i = 0; i < row_indices.size( ); ++i ) {
    triplet_list.push_back( Eigen::Triplet< sc, los >(
      row_indices[ i ], column_indices[ i ], values[ i ] ) );
  }
  _data.setFromTriplets( triplet_list.begin( ), triplet_list.end( ) );
  _data.makeCompressed( );
}

besthea::linear_algebra::sparse_matrix::~sparse_matrix( ) {
}

void besthea::linear_algebra::sparse_matrix::set_from_triplets( los n_rows,
  los n_columns, std::vector< los > & row_indices,
  std::vector< los > & column_indices, std::vector< sc > & values ) {
  _n_rows = n_rows;
  _n_columns = n_columns;
  _data.resize( n_rows, n_columns );
  std::vector< Eigen::Triplet< sc, los > > triplet_list;
  triplet_list.reserve( row_indices.size( ) );

  for ( std::vector< los >::size_type i = 0; i < row_indices.size( ); ++i ) {
    triplet_list.push_back( Eigen::Triplet< sc, los >(
      row_indices[ i ], column_indices[ i ], values[ i ] ) );
  }
  _data.setFromTriplets( triplet_list.begin( ), triplet_list.end( ) );
  _data.makeCompressed( );
}

void besthea::linear_algebra::sparse_matrix::set_from_triplets( los n_rows,
  los n_columns, std::vector< std::vector< los > > & row_indices,
  std::vector< std::vector< los > > & column_indices,
  std::vector< std::vector< sc > > & values ) {
  _n_rows = n_rows;
  _n_columns = n_columns;
  _data.resize( n_rows, n_columns );
  std::vector< Eigen::Triplet< sc, los > > triplet_list;
  triplet_list.reserve( row_indices.size( ) * row_indices.at( 0 ).size( ) );

  for ( std::vector< los >::size_type j = 0; j < row_indices.size( ); ++j ) {
    for ( std::vector< los >::size_type i = 0; i < row_indices.at( j ).size( );
          ++i ) {
      triplet_list.push_back(
        Eigen::Triplet< sc, los >( row_indices.at( j )[ i ],
          column_indices.at( j )[ i ], values.at( j )[ i ] ) );
    }
  }
  _data.setFromTriplets( triplet_list.begin( ), triplet_list.end( ) );
  _data.makeCompressed( );
}

void besthea::linear_algebra::sparse_matrix::apply(
  const vector & x, vector & y, bool trans, sc alpha, sc beta ) const {
  // converting raw arrays to Eigen type
  Eigen::Map< const Eigen::Matrix< sc, Eigen::Dynamic, 1 > > x2map(
    x.data( ), x.size( ) );
  Eigen::Map< Eigen::Matrix< sc, Eigen::Dynamic, 1 > > y2map(
    y.data( ), y.size( ) );

  if ( trans ) {
    y2map = beta * y2map + alpha * _data.transpose( ) * x2map;
  } else {
    y2map = beta * y2map + alpha * _data * x2map;
  }
}

void besthea::linear_algebra::sparse_matrix::eigen_cg_solve( const vector & rhs,
  vector & solution, sc & relative_residual_error, lo & n_iterations ) const {
  Eigen::ConjugateGradient< Eigen::SparseMatrix< sc, Eigen::ColMajor, los >,
    Eigen::Lower | Eigen::Upper >
    cg( _data );

  cg.setMaxIterations( static_cast< Eigen::Index >( n_iterations ) );
  cg.setTolerance( relative_residual_error );

  Eigen::Map< const Eigen::Matrix< sc, Eigen::Dynamic, 1 > > rhs_map(
    rhs.data( ), rhs.size( ) );
  Eigen::Map< Eigen::Matrix< sc, Eigen::Dynamic, 1 > > solution_map(
    solution.data( ), solution.size( ) );

  solution_map = cg.solve( rhs_map );

  relative_residual_error = static_cast< sc >( cg.error( ) );
  n_iterations = static_cast< lo >( cg.iterations( ) );
}

void besthea::linear_algebra::sparse_matrix::eigen_lu_decompose_and_solve(
  const vector & rhs, vector & solution ) {
  Eigen::Map< const Eigen::Matrix< sc, Eigen::Dynamic, 1 > > rhs_map(
    rhs.data( ), rhs.size( ) );
  Eigen::Map< Eigen::Matrix< sc, Eigen::Dynamic, 1 > > solution_map(
    solution.data( ), solution.size( ) );

  _lu.compute( _data );
  solution_map = _lu.solve( rhs_map );
}

void besthea::linear_algebra::sparse_matrix::eigen_lu_decompose( ) {
  _lu.compute( _data );
}

void besthea::linear_algebra::sparse_matrix::eigen_lu_solve(
  const vector & rhs, vector & solution ) {
  Eigen::Map< const Eigen::Matrix< sc, Eigen::Dynamic, 1 > > rhs_map(
    rhs.data( ), rhs.size( ) );
  Eigen::Map< Eigen::Matrix< sc, Eigen::Dynamic, 1 > > solution_map(
    solution.data( ), solution.size( ) );

  solution_map = _lu.solve( rhs_map );
}

void besthea::linear_algebra::sparse_matrix::eigen_cholesky_decompose_and_solve(
  const vector & rhs, vector & solution ) {
  Eigen::Map< const Eigen::Matrix< sc, Eigen::Dynamic, 1 > > rhs_map(
    rhs.data( ), rhs.size( ) );
  Eigen::Map< Eigen::Matrix< sc, Eigen::Dynamic, 1 > > solution_map(
    solution.data( ), solution.size( ) );

  _choleski.compute( _data );
  solution_map = _choleski.solve( rhs_map );
}

void besthea::linear_algebra::sparse_matrix::eigen_cholesky_decompose( ) {
  _choleski.compute( _data );
}

void besthea::linear_algebra::sparse_matrix::eigen_cholesky_solve(
  const vector & rhs, vector & solution ) {
  Eigen::Map< const Eigen::Matrix< sc, Eigen::Dynamic, 1 > > rhs_map(
    rhs.data( ), rhs.size( ) );
  Eigen::Map< Eigen::Matrix< sc, Eigen::Dynamic, 1 > > solution_map(
    solution.data( ), solution.size( ) );

  solution_map = _choleski.solve( rhs_map );
}
