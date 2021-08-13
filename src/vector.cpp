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

#include "besthea/vector.h"

#include "besthea/basis_tetra_p1.h"
#include "besthea/fe_space.h"
#include "besthea/volume_space_cluster.h"

#include <algorithm>
#include <random>

besthea::linear_algebra::vector::vector( ) : _size( 0 ), _data( ) {
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

void besthea::linear_algebra::vector::print_h( std::ostream & stream ) const {
  for ( lo i = 0; i < _size; ++i ) {
    stream << _data[ i ] << " ";
  }
  stream << std::endl;
}

void besthea::linear_algebra::vector::random_fill( sc lower, sc upper ) {
  std::random_device rd;
  std::mt19937 gen( rd( ) );
  std::uniform_real_distribution< sc > dis( lower, upper );
  std::generate(
    _data.begin( ), _data.end( ), [ &gen, &dis ]( ) { return dis( gen ); } );
}

void besthea::linear_algebra::vector::copy_from_raw(
  lo size, const sc * data ) {
  if ( _size != size ) {
    resize( size, false );
  }
  std::copy( data, data + size, _data.data( ) );
}

void besthea::linear_algebra::vector::copy_to_raw( sc * data ) const {
  std::copy( _data.begin( ), _data.end( ), data );
}

template<>
void besthea::linear_algebra::vector::get_local_part<
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >(
  const besthea::mesh::volume_space_cluster * cluster,
  besthea::linear_algebra::vector & local_vector ) const {
  lo n_nodes = cluster->get_n_nodes( );
  std::vector< lo > local_2_global_nodes = cluster->get_local_2_global_nodes( );
  local_vector.resize( n_nodes, false );
  for ( lo i = 0; i < n_nodes; ++i ) {
    local_vector[ i ] = get( local_2_global_nodes[ i ] );
  }
}
