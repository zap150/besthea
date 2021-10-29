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
 * @brief Vector of scalars.
 */

#ifndef INCLUDE_BESTHEA_VECTOR_H_
#define INCLUDE_BESTHEA_VECTOR_H_

#include "besthea/settings.h"

#include <iostream>
#include <mkl.h>
#include <vector>

namespace besthea {
  namespace linear_algebra {
    class vector;
  }
}

namespace besthea {
  namespace mesh {
    class volume_space_cluster;
  }
}

// TODO: float version of cblas routines!

/**
 *  Class representing a vector.
 */
class besthea::linear_algebra::vector {
 public:
  vector( );

  /**
   * Constructor with an initializer list.
   * @param[in] list Initializer list for std::vector.
   */
  vector( std::initializer_list< sc > list );

  /**
   * Constructing a vector of the given size.
   * @param[in] size Length of the vector.
   * @param[in] zero Initialize to 0 if true.
   */
  vector( lo size, bool zero = true );

  /**
   * Destructor
   */
  ~vector( );

  /**
   * Prints the vector.
   * @param[in] stream
   */
  void print( std::ostream & stream = std::cout ) const;

  /**
   * Prints the vector horizontally.
   * @param[in] stream
   */
  void print_h( std::ostream & stream = std::cout ) const;

  /*!
   * @brief Fills the vector with the given value.
   * @param[in] value
   */
  void fill( sc value ) {
    std::fill( _data.begin( ), _data.end( ), value );
  }

  /**
   * Resizes the vector.
   * @param[in] size New size.
   * @param[in] zero Initialize to 0 if true.
   */
  void resize( lo size, bool zero = true ) {
    _data.resize( size );
    _data.shrink_to_fit( );
    if ( zero ) {
      fill( 0.0 );
    }
    _size = size;
  }

  /**
   * Fills the vector with random numbers (uniform distribution).
   * @param[in] lower Lower bound.
   * @param[in] upper Upper bound.
   */
  void random_fill( sc lower, sc upper );

  /*!
   * @brief Returns the i-th element of the vector.
   * @param[in] i
   */
  sc get( lo i ) const {
    return _data[ i ];
  }

  /*!
   * @brief Sets the i-th element of the vector.
   * @param[in] i Element index.
   * @param[in] value Value to be set.
   */
  void set( lo i, sc value ) {
    _data[ i ] = value;
  }

  /*!
   * @brief Copies data from a raw vector.
   * @param[in] size Data size.
   * @param[in] data Array to copy from.
   */
  void copy_from_raw( lo size, const sc * data );

  /*!
   * @brief Copies data to a raw vector.
   * @param[in] data Array to copy to.
   */
  void copy_to_raw( sc * data ) const;

  /*!
   * @brief Overloads the [] operator.
   * @param[in] i Index.
   */
  sc & operator[]( lo i ) {
    return _data[ i ];
  }

  /*!
   * @brief Overloads the () operator.
   * @param[in] i Index.
   */
  sc operator( )( lo i ) const {
    return _data[ i ];
  }

  /*!
   * @brief Overloads the () operator.
   * @param[in] i Index.
   */
  sc & operator( )( lo i ) {
    return _data[ i ];
  }

  /*!
   * @brief Overloads the [] operator.
   * @param[in] i Index.
   */
  sc operator[]( lo i ) const {
    return _data[ i ];
  }

  /*!
   * @brief Returns the raw data.
   */
  sc * data( ) {
    return _data.data( );
  }

  /*!
   * @brief Returns the raw data.
   */
  const sc * data( ) const {
    return _data.data( );
  }

  /*!
   * @brief Returns the euclidean dot product.
   * @param[in] v
   */
  sc dot( vector const & v ) const {
    return cblas_ddot( _size, _data.data( ), 1, v._data.data( ), 1 );
  }

  /*!
   * @brief The euclidean norm.
   */
  sc norm( ) {
    return cblas_dnrm2( _size, _data.data( ), 1 );
  }

  /*!
   * @brief Vector addition this += alpha * v.
   * @param[in] v
   * @param[in] alpha
   */
  void add( vector const & v, sc alpha = 1.0 ) {
    cblas_daxpy( _size, alpha, v._data.data( ), 1, _data.data( ), 1 );
  }

  /*!
   * @brief Vector addition this += alpha * v, where v is raw data.
   * @param[in] v
   * @param[in] alpha
   */
  void add_from_raw( const sc * v, sc alpha = 1.0 ) {
    cblas_daxpy( _size, alpha, v, 1, _data.data( ), 1 );
  }

  /*!
   * @brief Scales the vecotr's element with alpha.
   * @param[in] alpha
   */
  void scale( sc alpha ) {
    cblas_dscal( _size, alpha, _data.data( ), 1 );
  }

  /*!
   * @brief Adds atomically to a single position of a vector.
   * @param[in] i Index of an element.
   * @param[in] val Value to be added.
   */
  void add_atomic( lo i, sc val ) {
#pragma omp atomic update
    _data.data( )[ i ] += val;
  }

  /*!
   * @brief Size of the vector.
   */
  lo size( ) const {
    return _size;
  }

  /*!
   * @brief Copies data from another vector.
   * @param[in] that Reference to the vector to be copied.
   */
  void copy( vector const & that ) {
    _size = that._size;
    _data = that._data;
  }

  /*!
   * Gets the local part of a vector corresponding to the dofs in a volume space
   * cluster.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_vector Local part of the vector.
   * @tparam space_type @ref besthea::bem::fe_space representing basis
   * functions. It determines the DOFs.
   * @warning The local vector must have the correct size.
   */
  template< class space_type >
  void get_local_part( const besthea::mesh::volume_space_cluster * cluster,
    besthea::linear_algebra::vector & local_vector ) const;

 protected:
  lo _size;                                                //!< vector size
  std::vector< sc, besthea::allocator_type< sc > > _data;  //!< raw data
};

#endif /* INCLUDE_BESTHEA_VECTOR_H_ */
