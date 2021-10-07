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

void besthea::linear_algebra::block_vector::copy_permute( const block_vector & that, sc alpha ) {
  resize_to_match_permute(that, false);

  constexpr lo tile_size = 128; // chosen experimentally, the best for double on 1 thread on Barbora
  lo bb_max = (that._n_blocks - 1) / tile_size;
  lo ii_max = (that._size - 1) / tile_size;

  for (lo bb = 0; bb < bb_max; bb++) {
    lo BB = bb * tile_size;
    for (lo ii = 0; ii < ii_max; ii++) {
      lo II = ii * tile_size;
      for (lo b = 0; b < tile_size; b++) {
        lo B = BB + b;
        for (lo i = 0; i < tile_size; i++) {
          lo I = II + i;
          sc val = that.get(B, I);
          this->set(I, B, alpha * val);
        }
      }
    }
    for(lo I = tile_size * ii_max; I < that._size; I++) {
      for (lo b = 0; b < tile_size; b++) {
        lo B = BB + b;
        sc val = that.get(B, I);
        this->set(I, B, alpha * val);
      }
    }
  }
  for (lo B = tile_size * bb_max; B < that._n_blocks; B++) {
    for(lo I = 0; I < that._size; I++) {
      sc val = that.get(B, I);
      this->set(I, B, alpha * val);
    }
  }



  // performance benefit of using mkl_domatcopy is very small and comsumes much more memory because of the required buffers
  // if the data in this class were stored in one contiguous buffer, mkl_domatcopy should be a much better option

  // sc * raw_data_in = new sc[_size * _block_size];
  // sc * raw_data_out = new sc[_size * _block_size];
  // that.copy_to_raw(raw_data_in);
  // mkl_domatcopy('C', 'T', that._size, that._block_size, alpha, raw_data_in, that._size, raw_data_out, that._block_size);
  // this->copy_from_raw(that._size, that._block_size, raw_data_out);    
  // delete[] raw_data_in;
  // delete[] raw_data_out;

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
  
void besthea::linear_algebra::block_vector::copy_from_raw_permute(
    lo block_size, lo size, const sc * data, sc alpha ) {
  resize( size );
  resize_blocks( block_size );

  constexpr lo tile_size = 128;
  lo bb_max = (block_size - 1) / tile_size;
  lo ii_max = (size - 1) / tile_size;

  for (lo bb = 0; bb < bb_max; bb++) {
    lo BB = bb * tile_size;
    for (lo ii = 0; ii < ii_max; ii++) {
      lo II = ii * tile_size;
      for (lo b = 0; b < tile_size; b++) {
        lo B = BB + b;
        for (lo i = 0; i < tile_size; i++) {
          lo I = II + i;
          sc val = data[B * size + I];
          this->set(I, B, alpha * val);
        }
      }
    }
    for(lo I = tile_size * ii_max; I < size; I++) {
      for (lo b = 0; b < tile_size; b++) {
        lo B = BB + b;
        sc val = data[B * size + I];
        this->set(I, B, alpha * val);
      }
    }
  }
  for (lo B = tile_size * bb_max; B < block_size; B++) {
    for(lo I = 0; I < size; I++) {
      sc val = data[B * size + I];
      this->set(I, B, alpha * val);
    }
  }

}

void besthea::linear_algebra::block_vector::copy_to_raw_permute( sc * data, sc alpha ) const {

  constexpr lo tile_size = 128;
  lo bb_max = (_n_blocks - 1) / tile_size;
  lo ii_max = (_size - 1) / tile_size;

  for (lo bb = 0; bb < bb_max; bb++) {
    lo BB = bb * tile_size;
    for (lo ii = 0; ii < ii_max; ii++) {
      lo II = ii * tile_size;
      for (lo b = 0; b < tile_size; b++) {
        lo B = BB + b;
        for (lo i = 0; i < tile_size; i++) {
          lo I = II + i;
          sc val = get(B, I);
          data[I * _n_blocks + B] = alpha * val;
        }
      }
    }
    for(lo I = tile_size * ii_max; I < _size; I++) {
      for (lo b = 0; b < tile_size; b++) {
        lo B = BB + b;
        sc val = get(B, I);
        data[I * _n_blocks + B] = alpha * val;
      }
    }
  }
  for (lo B = tile_size * bb_max; B < _n_blocks; B++) {
    for(lo I = 0; I < _size; I++) {
      sc val = get(B, I);
      data[I * _n_blocks + B] = alpha * val;
    }
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

void besthea::linear_algebra::block_vector::add_permute( const block_vector & that, sc alpha ) {
  constexpr lo tile_size = 128; // chosen experimentally, the best for double on 1 thread on Barbora
  lo bb_max = (that._n_blocks - 1) / tile_size;
  lo ii_max = (that._size - 1) / tile_size;

  for (lo bb = 0; bb < bb_max; bb++) {
    lo BB = bb * tile_size;
    for (lo ii = 0; ii < ii_max; ii++) {
      lo II = ii * tile_size;
      for (lo b = 0; b < tile_size; b++) {
        lo B = BB + b;
        for (lo i = 0; i < tile_size; i++) {
          lo I = II + i;
          sc val = that.get(B, I);
          this->add(I, B, alpha * val);
        }
      }
    }
    for(lo I = tile_size * ii_max; I < that._size; I++) {
      for (lo b = 0; b < tile_size; b++) {
        lo B = BB + b;
        sc val = that.get(B, I);
        this->add(I, B, alpha * val);
      }
    }
  }
  for (lo B = tile_size * bb_max; B < that._n_blocks; B++) {
    for(lo I = 0; I < that._size; I++) {
      sc val = that.get(B, I);
      this->add(I, B, alpha * val);
    }
  }

}
