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

besthea::linear_algebra::sparse_matrix::sparse_matrix( ) : _data( ) {
  _n_rows = 0;
  _n_columns = 0;
}

besthea::linear_algebra::sparse_matrix::sparse_matrix(
  const sparse_matrix & that )
  : _data( that._data ) {
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

  for ( lou i = 0; i < row_indices.size( ); ++i ) {
    triplet_list.push_back( Eigen::Triplet< sc, los >(
      row_indices[ i ], column_indices[ i ], values[ i ] ) );
  }
  _data.setFromTriplets( triplet_list.begin( ), triplet_list.end( ) );
  _data.makeCompressed( );
}

besthea::linear_algebra::sparse_matrix::~sparse_matrix( ) {
}

void besthea::linear_algebra::sparse_matrix::apply(
  const vector & x, vector & y, bool trans, sc alpha, sc beta ) const {
  // converting raw arrays to Eigen type
  typedef Eigen::Map< const Eigen::Matrix< sc, Eigen::Dynamic, 1 > > map_const;
  typedef Eigen::Map< Eigen::Matrix< sc, Eigen::Dynamic, 1 > > map;

  map_const x2map( x.data( ), x.size( ) );
  map y2map( y.data( ), y.size( ) );

  if ( trans ) {
    y2map = beta * y2map + alpha * _data.transpose( ) * x2map;
  } else {
    y2map = beta * y2map + alpha * _data * x2map;
  }
}
