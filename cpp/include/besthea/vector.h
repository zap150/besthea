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

#ifndef INCLUDE_BESTHEA_VECTOR_H_
#define INCLUDE_BESTHEA_VECTOR_H_

#include "besthea/settings.h"

#include <iostream>
#include <vector>

namespace besthea {
  namespace linear_algebra {
    class vector;
  }
}

/**
 *  Class representing a vector of scalars.
 */
class besthea::linear_algebra::vector {
 public:
  vector( );

  vector( const vector & that );

  vector( sc size );

  ~vector( );

  /*!
   * @brief Prints the vector.
   * @param[in] stream
   */
  void print( std::ostream & stream = std::cout ) const;

  /*!
   * @brief Fills the vector with random numbers (uniform distribution).
   * @param[in] lower Lower bound.
   * @param[in] upper Upper bound.
   */
  void random_fill( sc lower, sc upper );

 protected:
  lo _size;
  std::vector< sc > _data;
};

#endif /* INCLUDE_BESTHEA_VECTOR_H_ */
