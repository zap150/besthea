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

/** @file indices.h
 * @brief Indices to and array.
 */

#ifndef INCLUDE_BESTHEA_INDICES_H_
#define INCLUDE_BESTHEA_INDICES_H_

#include "besthea/settings.h"

#include <iostream>

namespace besthea {
  namespace linear_algebra {
    template< std::size_t dimension >
    class indices;
  }
}

/**
 *  Class representing indices to an array.
 */
template< std::size_t dimension >
class besthea::linear_algebra::indices {
 public:
  indices( const indices & that ) = delete;

  /**
   * Constructor.
   */
  explicit indices( bool zero = true );

  /**
   * Constructor with an initializer list.
   * @param[in] list Initializer list.
   */
  indices( std::initializer_list< lo > list );

  /**
   * Destructor
   */
  ~indices( );

  /**
   * Returns a pointer to the raw data.
   */
  lo * data( ) {
    return begin( );
  }

  /**
   * Returns a const pointer to the raw data.
   */
  const lo * data( ) const {
    return begin( );
  }

  /**
   * Returns a pointer to the first element.
   */
  lo * begin( ) {
    return &( _data[ 0 ] );
  }

  /**
   * Returns a pointer to the element following the last element.
   */
  lo * end( ) {
    return &( _data[ 0 ] ) + dimension;
  }

  /**
   * Returns a const pointer to the first element.
   */
  const lo * begin( ) const {
    return &( _data[ 0 ] );
  }

  /**
   * Returns a const pointer to the element following the last element.
   */
  const lo * end( ) const {
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
   * Prints the indices.
   * @param[in] stream
   */
  void print( std::ostream & stream = std::cout ) const;

  /*!
   * @brief Returns the i-th element of the indices.
   * @param[in] i
   */
  lo get( lo i ) const {
    return _data[ i ];
  }

  /*!
   * @brief Sets the i-th element of the indices.
   * @param[in] i Element index.
   * @param[in] value Value to be set.
   */
  void set( lo i, lo value ) {
    _data[ i ] = value;
  }

  /*!
   * @brief Overloads the [] operator.
   * @param[in] i Index.
   */
  lo operator[]( lo i ) const {
    return _data[ i ];
  }

  /*!
   * @brief Overloads the [] operator.
   * @param[in] i Index.
   */
  lo & operator[]( lo i ) {
    return _data[ i ];
  }

  /*!
   * @brief Overloads the () operator.
   * @param[in] i Index.
   */
  lo operator( )( lo i ) const {
    return _data[ i ];
  }

  /*!
   * @brief Overloads the () operator.
   * @param[in] i Index.
   */
  lo & operator( )( lo i ) {
    return _data[ i ];
  }

 protected:
  lo _dimension;                                //!< indices dimension
  alignas( DATA_ALIGN ) lo _data[ dimension ];  //!< raw data
};

#endif /* INCLUDE_BESTHEA_INDICES_H_ */
