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

#include "besthea/block_vector.h"

#include "besthea/general_spacetime_cluster.h"

besthea::linear_algebra::block_vector::block_vector( )
  : _n_blocks( 0 ), _size( 0 ), _data( ) {
}

besthea::linear_algebra::block_vector::block_vector(
  lo n_blocks, std::initializer_list< sc > list )
  : _n_blocks( n_blocks ),
    _size( list.size( ) ),
    //    _data( n_blocks, list ) { // Why does this work??
    _data( n_blocks, vector_type( list ) ) {
}

besthea::linear_algebra::block_vector::block_vector(
  lo n_blocks, lo size, bool zero )
  : _n_blocks( n_blocks ),
    _size( size ),
    _data( n_blocks, vector_type( size, zero ) ) {
}

besthea::linear_algebra::block_vector::block_vector( const block_vector & that )
  : _n_blocks( that._n_blocks ), _size( that._size ), _data( that._data ) {
}

besthea::linear_algebra::block_vector::~block_vector( ) {
}

void besthea::linear_algebra::block_vector::print(
  std::ostream & stream ) const {
  for ( const vector_type & v : _data ) {
    v.print( stream );
  }
}

void besthea::linear_algebra::block_vector::copy( block_vector const & that ) {
  resize( that.get_n_blocks( ) );
  resize_blocks( that.get_size_of_block( ) );
  for ( lo i = 0; i < _n_blocks; ++i ) {
    _data[ i ].copy( that._data[ i ] );
  }
}

void besthea::linear_algebra::block_vector::copy_from_raw(
  lo n_blocks, lo size, const sc * data ) {
  if ( n_blocks != _n_blocks ) {
    resize( n_blocks );
  }
  if ( size != _size ) {
    resize_blocks( size, false );
  }
  for ( lo i = 0; i < n_blocks; ++i ) {
    _data[ i ].copy_from_raw( size, data + i * size );
  }
}

void besthea::linear_algebra::block_vector::copy_to_raw( sc * data ) const {
  for ( lo i = 0; i < _n_blocks; ++i ) {
    _data[ i ].copy_to_raw( data + i * _size );
  }
}

void besthea::linear_algebra::block_vector::copy_from_vector(
  lo n_blocks, lo size, const vector_type & data ) {
  if ( n_blocks != _n_blocks ) {
    resize( n_blocks );
  }
  if ( size != _size ) {
    resize_blocks( size, false );
  }
  for ( lo i = 0; i < n_blocks; ++i ) {
    _data[ i ].copy_from_raw( size, data.data( ) + i * size );
  }
}

void besthea::linear_algebra::block_vector::copy_to_vector(
  vector_type & data ) const {
  if ( data.size( ) != _n_blocks * _size ) {
    data.resize( _n_blocks * _size, false );
  }
  for ( lo i = 0; i < _n_blocks; ++i ) {
    _data[ i ].copy_to_raw( data.data( ) + i * _size );
  }
}
