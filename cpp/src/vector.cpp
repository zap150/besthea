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

#include "besthea/vector.h"

#include "besthea/blas_lapack_wrapper.h"

#include <algorithm>
#include <random>

besthea::linear_algebra::vector::vector( ) : _size( 0 ), _data( ) {
}

besthea::linear_algebra::vector::vector( const vector & that )
  : _size( that._size ), _data( that._data ) {
}

besthea::linear_algebra::vector::vector( std::initializer_list< sc > list )
  : _size( list.size( ) ), _data( list ) {
}

besthea::linear_algebra::vector::vector( lo size, bool zero )
  : _size( size ), _data( size ) {
  if ( zero ) {
    fill( 0.0 );
  }
}

besthea::linear_algebra::vector::~vector( ) {
}

void besthea::linear_algebra::vector::print( std::ostream & stream ) const {
  for ( lo i = 0; i < _size; ++i ) {
    stream << _data[ i ] << std::endl;
  }
  stream << std::endl;
}

void besthea::linear_algebra::vector::random_fill( sc lower, sc upper ) {
  std::random_device rd;
  std::mt19937 gen( rd( ) );
  std::uniform_real_distribution< sc > dis( lower, upper );
  std::generate(
    _data.begin( ), _data.end( ), [&gen, &dis]( ) { return dis( gen ); } );
}
