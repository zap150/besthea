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

/** @file coordinates.h
 * @brief Coordinates of an n-dimensional point or vector.
 */

#ifndef INCLUDE_BESTHEA_COORDINATES_H_
#define INCLUDE_BESTHEA_COORDINATES_H_

#include "besthea/settings.h"

#include <array>
#include <cmath>
#include <iostream>

namespace besthea {
  namespace linear_algebra {
    template< std::size_t dimension >
    class coordinates;
  }
}

/**
 *  Class representing an n-dimensional point or vector.
 */
template< std::size_t dimension >
class besthea::linear_algebra::coordinates {
 public:
  /**
   * Copy constructor.
   * @param[in] that Coordinates to be deep copied.
   */
  coordinates( const coordinates & that );

  /**
   * Constructor with an initializer list.
   * @param[in] list Initializer list for std::array.
   */
  coordinates( std::initializer_list< sc > list );

  /**
   * Constructing coordinates of the given size.
   * @param[in] zero Initialize to 0 if true.
   */
  coordinates( bool zero = true );

  /**
   * Destructor
   */
  ~coordinates( );

  /**
   * Returns a pointer to the raw data.
   */
  sc * data( ) {
    return begin( );
  }

  /**
   * Returns a const pointer to the raw data.
   */
  const sc * data( ) const {
    return begin( );
  }

  /**
   * Returns a pointer to the first element.
   */
  sc * begin( ) {
    return &( _data[ 0 ] );
  }

  /**
   * Returns a pointer to the element following the last element.
   */
  sc * end( ) {
    return &( _data[ 0 ] ) + dimension;
  }

  /**
   * Returns a const pointer to the first element.
   */
  const sc * begin( ) const {
    return &( _data[ 0 ] );
  }

  /**
   * Returns a const pointer to the element following the last element.
   */
  const sc * end( ) const {
    return &( _data[ 0 ] ) + dimension;
  }

  /*!
   * @brief Fills the coordinates with the given value.
   * @param[in] value
   */
  void fill( sc value ) {
    std::fill( begin( ), end( ), value );
  }

  /**
   * Prints the coordinates.
   * @param[in] stream
   */
  void print( std::ostream & stream = std::cout ) const;

  /*!
   * @brief Returns the i-th element of the coordinates.
   * @param[in] i
   */
  sc get( lo i ) const {
    return _data[ i ];
  }

  /*!
   * @brief Sets the i-th element of the coordinates.
   * @param[in] i Element index.
   * @param[in] value Value to be set.
   */
  void set( lo i, sc value ) {
    _data[ i ] = value;
  }

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
   * @brief Returns the euclidean dot product.
   * @param[in] c
   */
  sc dot( coordinates< dimension > const & c ) const {
    sc value = 0.0;
    for ( lo i = 0; i < _dimension; ++i ) {
      value += c._data[ i ] * _data[ i ];
    }
    return value;
  }

  /*!
   * @brief The euclidean norm.
   */
  sc norm( ) {
    sc value = 0.0;
    for ( lo i = 0; i < _dimension; ++i ) {
      value += _data[ i ] * _data[ i ];
    }
    value = std::sqrt( value );
    return value;
  }

  /*!
   * @brief The squared euclidean norm.
   */
  sc norm_squared( ) {
    sc value = 0.0;
    for ( lo i = 0; i < _dimension; ++i ) {
      value += _data[ i ] * _data[ i ];
    }
    return value;
  }

  /*!
   * @brief Coordinate addition this += alpha * v.
   * @param[in] c
   * @param[in] alpha
   */
  void add( coordinates< dimension > const & c, sc alpha = 1.0 ) {
    for ( lo i = 0; i < _dimension; ++i ) {
      _data[ i ] += alpha * c._data[ i ];
    }
  }

 protected:
  lo _dimension;                                //!< coordinates dimension
  alignas( DATA_ALIGN ) sc _data[ dimension ];  //!< raw data
};

#endif /* INCLUDE_BESTHEA_COORDINATES_H_ */
