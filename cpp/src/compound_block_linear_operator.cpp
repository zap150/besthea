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

#include "besthea/compound_block_linear_operator.h"

besthea::linear_algebra::compound_block_linear_operator::
  compound_block_linear_operator( )
  : _maximal_dimension( 0 ) {
  set_block_dim( 0 );
}

besthea::linear_algebra::compound_block_linear_operator::
  ~compound_block_linear_operator( ) {
}

void besthea::linear_algebra::compound_block_linear_operator::apply(
  const block_vector_type & x, block_vector_type & y, bool trans, sc alpha,
  sc beta ) const {
  lo size = _compound.size( );

  if ( size == 0 ) {
    return;
  } else if ( size == 1 ) {
    _compound[ 0 ]->apply(
      x, y, trans != _trans[ 0 ], alpha * _alpha[ 0 ], beta );
    return;
  }

  std::vector< block_vector_type > aux( 2 );
  aux[ 0 ].resize( x.get_block_size( ) );
  aux[ 1 ].resize( x.get_block_size( ) );
  aux[ 0 ].resize_blocks( _maximal_dimension, false );
  aux[ 1 ].resize_blocks( _maximal_dimension, false );
  const block_vector_type * src;
  block_vector_type * tgt;

  lo i;
  lo tgt_size;

  if ( !trans ) {
    tgt_size = ( !_trans[ 0 ] ) ? _compound[ 0 ]->get_dim_range( )
                                : _compound[ 0 ]->get_dim_domain( );
    tgt = &( aux[ 0 ] );
    tgt->resize_blocks( tgt_size, true );
    _compound[ 0 ]->apply( x, *tgt, _trans[ 0 ], alpha * _alpha[ 0 ], 0.0 );

    for ( i = 1; i < size - 1; ++i ) {
      src = &( aux[ ( i + 1 ) % 2 ] );
      tgt_size = ( !_trans[ i ] ) ? _compound[ i ]->get_dim_range( )
                                  : _compound[ i ]->get_dim_domain( );
      tgt = &( aux[ i % 2 ] );
      tgt->resize_blocks( tgt_size, true );
      _compound[ i ]->apply( *src, *tgt, _trans[ i ], _alpha[ i ], 0.0 );
    }
    src = &( aux[ ( i + 1 ) % 2 ] );
    _compound[ i ]->apply( *src, y, _trans[ i ], _alpha[ i ], beta );
  } else {
    tgt_size = ( !_trans[ size - 1 ] )
      ? _compound[ size - 1 ]->get_dim_domain( )
      : _compound[ size - 1 ]->get_dim_range( );
    tgt = &( aux[ 0 ] );
    tgt->resize_blocks( tgt_size, true );
    _compound[ size - 1 ]->apply(
      x, *tgt, !_trans[ size - 1 ], alpha * _alpha[ size - 1 ], 0.0 );
    lo ind;

    for ( i = 1; i < size - 1; ++i ) {
      ind = size - i - 1;
      src = &( aux[ ( i + 1 ) % 2 ] );
      tgt_size = ( !_trans[ ind ] ) ? _compound[ ind ]->get_dim_domain( )
                                    : _compound[ ind ]->get_dim_range( );
      tgt = &( aux[ i % 2 ] );
      tgt->resize_blocks( tgt_size, true );
      _compound[ ind ]->apply( *src, *tgt, !_trans[ ind ], _alpha[ ind ], 0.0 );
    }
    src = &( aux[ ( i + 1 ) % 2 ] );
    _compound[ 0 ]->apply( *src, y, !_trans[ 0 ], _alpha[ 0 ], beta );
  }
}

void besthea::linear_algebra::compound_block_linear_operator::push_back(
  const besthea::linear_algebra::block_linear_operator & op, bool trans,
  sc alpha ) {
  if ( _compound.size( ) > 1 && op.get_block_dim( ) != get_block_dim( ) ) {
    std::cout << "Check dimensions!" << std::endl;
    return;
  }

  _compound.push_back( &op );
  _trans.push_back( trans );
  _alpha.push_back( alpha );

  if ( _compound.size( ) == 1 ) {
    if ( !trans ) {
      _dim_domain = op.get_dim_domain( );
    } else {
      _dim_domain = op.get_dim_range( );
    }
    set_block_dim( op.get_block_dim( ) );
  }

  if ( !trans ) {
    _dim_range = op.get_dim_range( );
  } else {
    _dim_range = op.get_dim_domain( );
  }

  if ( op.get_dim_range( ) > _maximal_dimension ) {
    _maximal_dimension = op.get_dim_range( );
  }

  if ( op.get_dim_domain( ) > _maximal_dimension ) {
    _maximal_dimension = op.get_dim_domain( );
  }
}

bool besthea::linear_algebra::compound_block_linear_operator::is_valid( )
  const {
  lo size = _compound.size( );
  lo dim_range, dim_domain_next;

  for ( lo i = 0; i < size - 1; ++i ) {
    if ( !_compound[ i ] || !_compound[ i + 1 ] )
      return false;

    dim_range = ( !_trans[ i ] ) ? _compound[ i ]->get_dim_range( )
                                 : _compound[ i ]->get_dim_domain( );

    dim_domain_next = ( !_trans[ i + 1 ] )
      ? _compound[ i + 1 ]->get_dim_domain( )
      : _compound[ i + 1 ]->get_dim_range( );

    if ( dim_range != dim_domain_next )
      return false;
  }

  return true;
}
