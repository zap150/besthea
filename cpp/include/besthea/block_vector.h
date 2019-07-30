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

/** @file vector.h
 * @brief Vector of scalars.
 */

#ifndef INCLUDE_BESTHEA_BLOCK_VECTOR_H_
#define INCLUDE_BESTHEA_BLOCK_VECTOR_H_

#include "besthea/settings.h"
#include "besthea/vector.h"

#include <iostream>
#include <vector>

namespace besthea {
  namespace linear_algebra {
    class block_vector;
  }
}

/**
 *  Class representing a vector.
 */
class besthea::linear_algebra::block_vector {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  block_vector( );

  block_vector( const block_vector & that ) = delete;

  /**
   * Constructor with an initializer list.
   * @param[in] block_size Number of blocks.
   * @param[in] list Initializer list for vector.
   */
  block_vector( lo block_size, std::initializer_list< sc > list );

  /**
   * Constructing a vector of the given size.
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] zero Initialize to 0 if true.
   */
  block_vector( lo block_size, lo size, bool zero = true );

  ~block_vector( );

  /**
   * Returns a pointer to a single block.
   * @param[in] d Index of the block.
   */
  vector_type & get_block( lo d ) {
    return _data[ d ];
  }

  /**
   * Returns a pointer to a single block.
   * @param[in] d Index of the block.
   */
  const vector_type & get_block( lo d ) const {
    return _data[ d ];
  }

  /**
   * Resizes the block vector.
   * @param[in] block_size New size.
   */
  void resize( lo block_size ) {
    _data.resize( block_size );
    _block_size = block_size;
  }

  /**
   * Resizes the vector blocks.
   * @param[in] size New size.
   */
  void resize_blocks( lo size, bool zero = true ) {
    for ( vector_type & v : _data ) {
      v.resize( size, zero );
    }
    _size = size;
  }

  /*!
   * @brief Vector addition this += alpha * v.
   * @param[in] v
   * @param[in] alpha
   */
  void add( block_vector const & v, sc alpha = 1.0 ) {
    for ( lo i = 0; i < _block_size; ++i ) {
      _data[ i ].add( v._data[ i ], alpha );
    }
  }

  /*!
   * @brief Prints the vector.
   * @param[in] stream
   */
  void print( std::ostream & stream = std::cout ) const;

 protected:
  lo _block_size;                    //!< block size
  lo _size;                          //!< vector size
  std::vector< vector_type > _data;  //!< raw data
};

#endif /* INCLUDE_BESTHEA_BLOCK_VECTOR_H_ */
