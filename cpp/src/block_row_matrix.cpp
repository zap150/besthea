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

#include "besthea/block_row_matrix.h"

besthea::linear_algebra::block_row_matrix::block_row_matrix( )
  : block_row_linear_operator( 0, 0, 0 ), _data( ) {
}

besthea::linear_algebra::block_row_matrix::block_row_matrix(
  lo block_dim, lo n_rows, lo n_columns, std::initializer_list< sc > list )
  : block_row_linear_operator( block_dim, n_columns, n_rows ),
    _data( block_dim, matrix_type( n_rows, n_columns, list ) ) {
}

besthea::linear_algebra::block_row_matrix::block_row_matrix(
  lo block_dim, lo n_rows, lo n_columns, bool zero )
  : block_row_linear_operator( block_dim, n_columns, n_rows ),
    _data( block_dim, matrix_type( n_rows, n_columns, zero ) ) {
}

void besthea::linear_algebra::block_row_matrix::print(
  std::ostream & stream ) const {
  for ( const matrix_type & m : _data ) {
    m.print( );
  }
}

void besthea::linear_algebra::block_row_matrix::resize_blocks(
  lo n_rows, lo n_columns ) {
  for ( matrix_type & m : _data ) {
    m.resize( n_rows, n_columns );
  }
  _n_rows = n_rows;
  _n_columns = n_columns;
}

void besthea::linear_algebra::block_row_matrix::resize( lo block_dim ) {
  _data.resize( block_dim );
  _block_dim = block_dim;
}

void besthea::linear_algebra::block_row_matrix::apply( const vector_type & x,
  block_vector_type & y, bool trans, sc alpha, sc beta ) const {
  for ( lo i_block = 0; i_block < _block_dim; ++i_block ) {
    _data[ i_block ].apply( x, y.get_block( i_block ), trans, alpha, beta );
  }
}
