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

/** @file uniform_spacetime_be_identity.h
 * @brief Discretized identity operator.
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_IDENTITY_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_IDENTITY_H_

#include "besthea/block_matrix.h"
#include "besthea/sparse_matrix.h"
#include "besthea/uniform_spacetime_be_space.h"

namespace besthea {
  namespace bem {
    template< class test_space_type, class trial_space_type >
    class uniform_spacetime_be_identity;
  }
}

/**
 *  Class representing a boundary element identity operator.
 */
template< class test_space_type, class trial_space_type >
class besthea::bem::uniform_spacetime_be_identity
  : public besthea::linear_algebra::block_matrix {
 public:
  using matrix_type = besthea::linear_algebra::sparse_matrix;  //!< Matrix type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.

  /**
   * Constructor.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   * @param[in] order_regular Triangle quadrature order for regular quadrature.
   */
  uniform_spacetime_be_identity( test_space_type & test_space,
    trial_space_type & trial_space, int order_regular = 4 );

  ~uniform_spacetime_be_identity( );

  /**
   * Assembles the identity matrix.
   */
  void assemble( );

  /**
   * Assembles the identity matrix.
   */
  void assemble( matrix_type & global_matrix ) const;

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  void apply( const block_vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

 private:
  /**
   * Assembles the triplets for the sparse identity matrix.
   * @param[in] ii Row indices.
   * @param[in] jj Column indices.
   * @param[in] vv Values.
   */
  void assemble_triplets( std::vector< los > & ii, std::vector< los > & jj,
    std::vector< sc > & vv ) const;

  matrix_type _data;  //!< Raw matrix data.

  const test_space_type * _test_space;  //!< Boundary element test space.

  const trial_space_type * _trial_space;  //!< Boundary element trial space.

  int _order_regular;  //!< Triangle quadrature order for the regular integrals.
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_IDENTITY_H_ */
