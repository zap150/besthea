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

//#include "besthea/fast_spacetime_be_space.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

#include <iostream>
#include <vector>

namespace besthea {
  namespace linear_algebra {
    class block_vector;
  }
}

namespace besthea {
  namespace mesh {
    class spacetime_cluster;
    class general_spacetime_cluster;
  }
}

namespace besthea {
  namespace bem {
    template< class basis_type >
    class fast_spacetime_be_space;
  }
}

/**
 *  Class representing a vector.
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

  /*!
   * @brief Returns the i-th element of the d-th block.
   * @param[in] d Block index.
   * @param[in] i Element index.
   */
  sc get( lo d, lo i ) const {
    return _data[ d ][ i ];
  }

  /**
   * Returns the block dimension.
   */
  lo get_block_size( ) const {
    return _block_size;
  }

  /**
   * Returns the dimension of a single block
   */
  lo get_size_of_block( ) const {
    return _size;
  }

  /**
   * Returns the dimension of the whole vector.
   */
  lo size( ) const {
    return _block_size * _size;
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
   * @param[in] zero Initialize to 0 if true.
   */
  void resize_blocks( lo size, bool zero = true ) {
    for ( vector_type & v : _data ) {
      v.resize( size, zero );
    }
    _size = size;
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
   * @brief Adds atomically(!) to a single position of a vector.
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be added.
   */
  void add_atomic( lo d, lo i, sc value ) {
#pragma omp atomic update
    _data[ d ][ i ] += value;
  }

  /*!
   * @brief Adds to a single position of a vector.
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be added.
   */
  void add( lo d, lo i, sc value ) {
    _data[ d ][ i ] += value;
  }

  /*!
   * @brief Copies data from a raw vector.
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] data Array to copy from.
   */
  void copy_from_raw( lo block_size, lo size, const sc * data );

  /*!
   * @brief Copies data to a raw vector.
   * @param[in] data Array to copy to.
   */
  void copy_to_raw( sc * data ) const;

  /*!
   * @brief Copies data from a raw vector.
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] data Array to copy from.
   */
  void copy_from_vector( lo block_size, lo size, const vector_type & data );

  /*!
   * @brief Copies data to a raw vector.
   * @param[in] data Array to copy to.
   */
  void copy_to_vector( vector_type & data ) const;

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
   * @brief Fills the block vector with the given value.
   * @param[in] value
   */
  void fill( sc value ) {
    for ( lo i = 0; i < _block_size; ++i ) {
      _data[ i ].fill( value );
    }
  }

  /*!
   * Gets local part of a block vector corresponding to dofs in a spacetime
   * cluster.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_vector Local part of block vector.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   * @warning The local vector must have the correct size.
   * @note The local vector is not a block vector anymore, but a contiguous
   *       vector.
   */
  template< class space_type >
  void get_local_part( besthea::mesh::spacetime_cluster * cluster,
    besthea::linear_algebra::vector & local_vector ) const;

  /*!
   * Gets local part of a block vector corresponding to dofs in a spacetime
   * cluster.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_vector Local part of block vector.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   * @warning The local vector must have the correct size.
   * @note The local vector is not a block vector anymore, but a contiguous
   *       vector.
   */
  template< class space_type >
  void get_local_part( besthea::mesh::general_spacetime_cluster * cluster,
    besthea::linear_algebra::vector & local_vector ) const;

  /*!
   * Adds local vector to appropriate positions of a block vector. The positions
   * are determined by the dofs in a spacetime cluster.
   * @param[in] cluster  Cluster determining the positions in the
   block_vector
   * to which the local vector is added.
   * @param[in] local_vector Local part of block vector to be added.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   * @note The entries in the local vector are ordered according to the
   ordering
   *       of the time elements and spatial dofs in the spacetime cluster (time
   *       step after time step).
   */
  template< class space_type >
  void add_local_part( besthea::mesh::spacetime_cluster * cluster,
    const besthea::linear_algebra::vector & local_vector );

  /*!
   * Adds local vector to appropriate positions of a block vector. The positions
   * are determined by the dofs in a spacetime cluster.
   * @param[in] cluster Cluster determining the positions in the
                        block_vector to which the local vector is added.
   * @param[in] local_vector Local part of block vector to be added.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   * @note  The entries in the local vector are ordered according to the
   *        ordering of the time elements and spatial dofs in the spacetime
   *        cluster (time step after time step).
   */
  template< class space_type >
  void add_local_part( besthea::mesh::general_spacetime_cluster * cluster,
    const besthea::linear_algebra::vector & local_vector );

  /*!
   * @brief Prints the vector.
   * @param[in] stream
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
  lo _block_size;                    //!< block size
  lo _size;                          //!< vector size
  std::vector< vector_type > _data;  //!< raw data
};

#endif /* INCLUDE_BESTHEA_BLOCK_VECTOR_H_ */
