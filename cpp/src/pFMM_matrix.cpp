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

#include "besthea/pFMM_matrix.h"

besthea::linear_algebra::sparse_matrix *
besthea::linear_algebra::pFMM_matrix::create_nearfield_matrix(
  lo test_idx, lo trial_idx, lo n_duplications ) {
  sparse_matrix_type * local_matrix = new sparse_matrix_type( );

  _nearfield_matrices.push_back( local_matrix );

  _nearfield_block_map.push_back( std::make_pair( test_idx, trial_idx ) );

  // duplicate diagonals
  for ( lo i = 1; i < n_duplications; ++i ) {
    _nearfield_matrices.push_back( local_matrix );
    _nearfield_block_map.push_back(
      std::make_pair( test_idx + i, trial_idx + i ) );
    _uniform = true;
  }

  return local_matrix;
}

besthea::linear_algebra::sparse_matrix *
besthea::linear_algebra::pFMM_matrix::create_farfield_matrix(
  lo test_idx, lo trial_idx ) {
  sparse_matrix_type * local_matrix = new sparse_matrix_type( );

  _farfield_matrices.push_back( local_matrix );

  _farfield_block_map.push_back( std::make_pair( test_idx, trial_idx ) );

  return local_matrix;
}

void besthea::linear_algebra::pFMM_matrix::apply( const block_vector_type & x,
  block_vector_type & y, bool trans, sc alpha, sc beta ) const {
  for ( lo i = 0; i < y.get_block_size( ); ++i ) {
    for ( lo j = 0; j < y.get_size_of_block( ); ++j ) {
      y.set( i, j, y.get( i, j ) * beta );
    }
  }

  sparse_matrix_type * current_block;

#pragma omp parallel
  {
    vector_type local_y( y.get_size_of_block( ) );
    // first, multiply by the nearfield blocks
#pragma omp for
    for ( lo i = 0; i < _nearfield_matrices.size( ); ++i ) {
      current_block = _nearfield_matrices.at( i );
      const std::pair< lo, lo > & indices = _nearfield_block_map.at( i );

      const vector_type & local_x = x.get_block( indices.second );

      current_block->apply( local_x, local_y, trans, alpha, 0.0 );
      for ( lo j = 0; j < local_y.size( ); ++j ) {
        y.add_atomic( indices.first, j, local_y[ j ] );
      }
    }

    // next, multiply by the farfield blocks
#pragma omp for
    for ( lo i = 0; i < _farfield_matrices.size( ); ++i ) {
      current_block = _farfield_matrices.at( i );
      const std::pair< lo, lo > & indices = _farfield_block_map.at( i );

      const vector_type & local_x = x.get_block( indices.second );

      current_block->apply( local_x, local_y, trans, alpha, 0.0 );
      for ( lo j = 0; j < local_y.size( ); ++j ) {
        y.add_atomic( indices.first, j, local_y[ j ] );
      }
    }
  }

  // lo matrix_idx = 0;
  //#pragma omp parallel

  //  for ( auto it = _nearfield_matrices.begin( );
  //        it != _nearfield_matrices.end( ); ++it ) {
  //    current_block = *it;
  //    const std::pair< lo, lo > & indices = _nearfield_block_map.at(
  //    matrix_idx );
  //
  //    const vector_type & local_x = x.get_block( indices.second );
  //    vector_type & local_y = y.get_block( indices.first );
  //
  //    current_block->apply( local_x, local_y, trans, alpha, 1.0 );
  //
  //    matrix_idx++;
  //  }
  // next, multiply the farfield blocks
  //  matrix_idx = 0;
  //  for ( auto it = _farfield_matrices.begin( ); it != _farfield_matrices.end(
  //  );
  //        ++it ) {
  //    current_block = *it;
  //    const std::pair< lo, lo > & indices = _farfield_block_map.at( matrix_idx
  //    );
  //
  //    const vector_type & local_x = x.get_block( indices.second );
  //    vector_type & local_y = y.get_block( indices.first );
  //
  //    current_block->apply( local_x, local_y, trans, alpha, 1.0 );
  //
  //    matrix_idx++;
  //  }
}
