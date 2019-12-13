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

#include <besthea/coordinates.h>

template< std::size_t dimension >
besthea::linear_algebra::coordinates< dimension >::coordinates(
  const coordinates & that )
  : _dimension( that._dimension ) {
  std::copy( that.begin( ), that.end( ), this->begin( ) );
}

template< std::size_t dimension >
besthea::linear_algebra::coordinates< dimension >::coordinates(
  std::initializer_list< sc > list )
  : _dimension( list.size( ) ) {
  std::copy( list.begin( ), list.end( ), this->begin( ) );
}

template< std::size_t dimension >
besthea::linear_algebra::coordinates< dimension >::coordinates( bool zero )
  : _dimension( dimension ) {
  if ( zero ) {
    fill( 0.0 );
  }
}

template< std::size_t dimension >
besthea::linear_algebra::coordinates< dimension >::~coordinates( ) {
}

template< std::size_t dimension >
void besthea::linear_algebra::coordinates< dimension >::print(
  std::ostream & stream ) const {
  for ( lo i = 0; i < _dimension; ++i ) {
    stream << _data[ i ] << " ";
  }
  stream << std::endl;
}

template class besthea::linear_algebra::coordinates< 1 >;
template class besthea::linear_algebra::coordinates< 3 >;
template class besthea::linear_algebra::coordinates< 4 >;
