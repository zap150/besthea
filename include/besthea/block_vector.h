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

/** @file vector.h
 * @brief Contains a class representing a block vector, i.e. a vector of scalars
 * partitioned into blocks.
 * @note updated documentation
 */

#ifndef INCLUDE_BESTHEA_BLOCK_VECTOR_H_
#define INCLUDE_BESTHEA_BLOCK_VECTOR_H_

#include "besthea/settings.h"
#include "besthea/vector.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace besthea {
  namespace linear_algebra {
    class block_vector;
  }
}

/**
 * Class representing a block vector, i.e. a vector of scalars partitioned into
 * blocks.
 */
class besthea::linear_algebra::block_vector {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Constructor.
   */
  block_vector( );

  /**
   * Copy constructor.
   * @param[in] that Vector to be copied.
   */
  block_vector( const block_vector & that );

  /**
   * Constructs a block vector with an initializer list. All @p n_blocks have
   * the same size and elements as the provided list.
   * @param[in] n_blocks Number of blocks.
   * @param[in] list Initializer list for vector.
   */
  block_vector( lo n_blocks, std::initializer_list< sc > list );

  /**
   * Constructs a vector with a given number of blocks of given size.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] zero Initialize to 0 if true.
   */
  block_vector( lo n_blocks, lo size, bool zero = true );

  ~block_vector( );

  /**
   * Returns a reference to a single block.
   * @param[in] d Index of the block.
   */
  vector_type & get_block( lo d ) {
    return _data[ d ];
  }

  /**
   * Returns a reference to a single block.
   * @param[in] d Index of the block.
   */
  const vector_type & get_block( lo d ) const {
    return _data[ d ];
  }

  /**
   * @brief Returns the i-th element of the d-th block.
   * @param[in] d Block index.
   * @param[in] i Element index.
   */
  sc get( lo d, lo i ) const {
    return _data[ d ][ i ];
  }

  /**
   * Returns the number of blocks.
   */
  lo get_n_blocks( ) const {
    return _n_blocks;
  }

  /**
   * Returns the size of a single block.
   */
  lo get_size_of_block( ) const {
    return _size;
  }

  /**
   * Returns the size of the whole vector, i.e. the total number of elements.
   */
  lo size( ) const {
    return _n_blocks * _size;
  }

  /**
   * Resizes the block vector by changing the number of blocks.
   * @param[in] n_blocks New number of blocks.
   */
  void resize( lo n_blocks ) {
    _data.resize( n_blocks );
    _data.shrink_to_fit( );
    _n_blocks = n_blocks;
  }

  /**
   * Resizes all blocks of the block vector.
   * @param[in] size New size of each block.
   * @param[in] zero If true, all blocks are filled with zeros.
   */
  void resize_blocks( lo size, bool zero = true ) {
    for ( vector_type & v : _data ) {
      v.resize( size, zero );
    }
    _size = size;
  }

  /**
   * Resizes this block vector and its blocks to match the dimensions of the original vector.
   * @param[in] other The original block vector with target dimensions.
   * @param[in] zero Initialize to 0 if true.
   */
  void resize_to_match( const block_vector & original, bool zero = true ) {
    resize(original.get_n_blocks());
    resize_blocks(original.get_size_of_block(), zero);
  }

  /**
   * Resizes this block vector and its blocks to match the permuted dimensions of the original vector.
   * @param[in] other The original block vector with permuted target dimensions.
   * @param[in] zero Initialize to 0 if true.
   */
  void resize_to_match_permute( const block_vector & original, bool zero = true ) {
    resize(original.get_size_of_block());
    resize_blocks(original.get_n_blocks(), zero);
  }

  /*!
   * @brief Sets the i-th element of the d-th block.
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be set.
   */
  void set( lo d, lo i, sc value ) {
    _data[ d ][ i ] = value;
  }

  /*!
   * @brief Adds a value atomically(!) to a single element of a single block.
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be added.
   */
  void add_atomic( lo d, lo i, sc value ) {
#pragma omp atomic update
    _data[ d ][ i ] += value;
  }

  /*!
   * @brief Adds a value to a single element of a single block.
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be added.
   */
  void add( lo d, lo i, sc value ) {
    _data[ d ][ i ] += value;
  }

  /*!
   * Copies data from another block vector.
   * @param[in] that Vector to be copied.
   */
  void copy( const block_vector & that );

  /*!
   * Copies data from another block vector, while permuting outer and inner block dimensions and rearranging the data.
   * @param[in] that Vector to be copied.
   * @param[in] alpha Scaling factor of data values.
   */
  void copy_permute( const block_vector & that, sc alpha = 1 );

  /*!
   * @brief Copies data from a raw array.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] data Array to copy from. Contains all elements, block by block.
   * @note If @p n_blocks and @p size are different from the member variables
   * @p _n_blocks and @p _size, respectively, the block vector is resized
   * appropriately.
   * @warning The source array has to contain at least @p n_blocks * @p size
   * elements.
   */
  void copy_from_raw( lo n_blocks, lo size, const sc * data );

  /*!
   * @brief Copies data to a raw array.
   * @param[in,out] data Array to copy to. Is filled with all elements, block by
   *                     block.
   * @warning The array's size has to be at least @p _n_blocks * @p _size.
   */
  void copy_to_raw( sc * data ) const;

  /*!
   * @brief Copies data from a raw vector while permuting outer and inner block dimensions and rearranging the data.
   * @param[in] block_size Number of blocks in the input.
   * @param[in] size Length of the input vector.
   * @param[in] data Array to copy from.
   * @param[in] alpha Scaling factor of data values.
   */
  void copy_from_raw_permute( lo block_size, lo size, const sc * data, sc alpha = 1.0 );

  /*!
   * @brief Copies data to a raw vector while permuting outer and inner block dimensions and rearranging the data.
   * @param[in] data Array to copy to.
   * @param[in] alpha Scaling factor of data values.
   */
  void copy_to_raw_permute( sc * data, sc alpha = 1.0 ) const;

  /*!
   * @brief Copies data from a raw vector.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] data Vector to copy from. Contains all elements, block by block.
   * @note If @p n_blocks and @p size are different from the member variables
   * @p _n_blocks and @p _size, respectively, the block vector is resized
   * appropriately.
   * @warning The source vector has to contain at least @p n_blocks * @p size
   * elements.
   */
  void copy_from_vector( lo n_blocks, lo size, const vector_type & data );

  /*!
   * @brief Copies data to a raw vector.
   * @param[in,out] data Vector to copy to. Is filled with all elements, block
   *                     by block.
   * @warning The target vector has to contain at least
   * @p _n_blocks * @p _size elements.
   */
  void copy_to_vector( vector_type & data ) const;

  /*!
   * @brief Vector addition: this += alpha * v.
   * @param[in] v Block vector with the same number and size of blocks.
   * @param[in] alpha Scaling factor.
   */
  void add( block_vector const & v, sc alpha = 1.0 ) {
    for ( lo i = 0; i < _n_blocks; ++i ) {
      _data[ i ].add( v._data[ i ], alpha );
    }
  }

  /*!
   * @brief Vector addition this += alpha * v, where v is permuted.
   * @param[in] v
   * @param[in] alpha
   */
  void add_permute( block_vector const & v, sc alpha = 1.0 );

  /*!
   * @brief Vector addition this += alpha * v, where v is raw data.
   * @param[in] v
   * @param[in] alpha
   */
  void add_from_raw( const sc * v, sc alpha = 1.0 ) {
    for (lo b = 0; b < _n_blocks; b++) {
      const sc * v_b = v + b * _size;
      _data[ b ].add_from_raw(v_b, alpha);
    }
  }

  /*!
   * @brief Fills the block vector with the given value.
   * @param[in] value Value to fill the blocks with.
   */
  void fill( sc value ) {
    for ( lo i = 0; i < _n_blocks; ++i ) {
      _data[ i ].fill( value );
    }
  }

  /*!
   * @brief Returns the euclidean dot product.
   * @param[in] v Second block vector for dot product.
   * @warning Dimension of the second block vector have to agree!
   */
  sc dot( block_vector const & v ) const {
    sc val = 0.0;
    for ( lo i = 0; i < _n_blocks; ++i ) {
      val += _data[ i ].dot( v.get_block( i ) );
    }

    return val;
  }

  /*!
   * @brief Returns the Euclidean norm of the vector.
   * @return Euclidean norm of the vector.
   */
  sc norm( ) const {
    return std::sqrt( this->dot( *this ) );
  }

  /*!
   * @brief Scales the vector by a given scalar.
   * @param[in] alpha Scaling factor.
   */
  void scale( sc alpha ) {
    for ( auto & it : _data ) {
      it.scale( alpha );
    }
  }

  /*!
   * @brief Prints the vector.
   * @param[in] stream  Stream into which the vector is printed.
   */
  void print( std::ostream & stream = std::cout ) const;

  /**
   * Prints info on the object.
   */
  void print_info( ) const {
    std::cout << "besthea::linear_algebra::block_vector" << std::endl;
    std::cout << "  number of blocks: " << _data.size( ) << std::endl;
    std::cout << "  dimension of each block: " << _data[ 0 ].size( )
              << std::endl;
  }

 protected:
  lo _n_blocks;                      //!< number of blocks
  lo _size;                          //!< size of each block.
  std::vector< vector_type > _data;  //!< raw data
};

#endif /* INCLUDE_BESTHEA_BLOCK_VECTOR_H_ */
