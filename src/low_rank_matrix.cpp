/*
Copyright (c) 2022, VSB - Technical University of Ostrava and Graz University of
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

#include "besthea/low_rank_matrix.h"

besthea::linear_algebra::low_rank_matrix::low_rank_matrix( ) : _u( ), _v( ) {
  this->_n_rows = 0;
  this->_n_columns = 0;
}

besthea::linear_algebra::low_rank_matrix::low_rank_matrix(
  const low_rank_matrix & that )
  : _u( that._u ), _v( that._v ) {
  this->_n_rows = that._n_rows;
  this->_n_columns = that._n_columns;
}

besthea::linear_algebra::low_rank_matrix::low_rank_matrix(
  full_matrix && u, full_matrix && v )
  : _u( std::move( u ) ), _v( std::move( v ) ) {
  this->_n_rows = u.get_n_rows( );
  this->_n_columns = v.get_n_rows( );
}

besthea::linear_algebra::low_rank_matrix::low_rank_matrix(
  const full_matrix & u, const full_matrix & v )
  : _u( u ), _v( v ) {
  this->_n_rows = u.get_n_rows( );
  this->_n_columns = v.get_n_rows( );
}

besthea::linear_algebra::low_rank_matrix::~low_rank_matrix( ) {
}

besthea::linear_algebra::low_rank_matrix &
besthea::linear_algebra::low_rank_matrix::operator=(
  low_rank_matrix && other ) {
  _u = std::move( other._u );
  _v = std::move( other._v );
  this->_n_rows = other._n_rows;
  this->_n_columns = other._n_columns;
  other._n_rows = 0;
  other._n_columns = 0;
  return *this;
}

void besthea::linear_algebra::low_rank_matrix::apply( const vector_type & x,
  vector_type & y, bool trans, sc alpha, sc beta ) const {
  vector_type y_intermediate( _u.get_n_columns( ) );
  // todo: check if it is beneficial to store y_intermediate once in the class
  if ( !trans ) {
    _v.apply( x, y_intermediate, true, alpha, 0.0 );
    _u.apply( y_intermediate, y, false, 1.0, beta );
  } else {
    _u.apply( x, y_intermediate, true, alpha, 0.0 );
    _v.apply( y_intermediate, y, false, 1.0, beta );
  }
}

void besthea::linear_algebra::low_rank_matrix::replace_matrices(
  full_matrix && u, full_matrix && v ) {
  _u = std::move( u );
  _v = std::move( v );
  this->_n_rows = _u.get_n_rows( );
  this->_n_columns = _v.get_n_rows( );
}

void besthea::linear_algebra::low_rank_matrix::replace_matrices(
  const full_matrix & u, const full_matrix & v ) {
  _u = u;
  _v = v;
  this->_n_rows = _u.get_n_rows( );
  this->_n_columns = _v.get_n_rows( );
}
