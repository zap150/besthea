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

/** @file compound_linear_operator.h
 * @brief Class for a compound linear operator.
 */

#ifndef INCLUDE_BESTHEA_COMPOUND_LINEAR_OPERATOR_H_
#define INCLUDE_BESTHEA_COMPOUND_LINEAR_OPERATOR_H_

#include "besthea/linear_operator.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

#include <vector>

namespace besthea {
  namespace linear_algebra {
    class compound_linear_operator;
  }
}

/**
 *  Class representing a compound linear operator.
 */
class besthea::linear_algebra::compound_linear_operator
  : public besthea::linear_algebra::linear_operator {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Constructor.
   */
  compound_linear_operator( );

  /**
   * Destructor.
   */
  virtual ~compound_linear_operator( );

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const vector_type & x, vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const override;

  /**
   * Adds a linear operator to the compound.
   * @param[in] op Linear operator.
   * @param[in] trans Determines whether to apply transposed.
   * @param[in] alpha Multiplicative factor.
   */
  void push_back( const besthea::linear_algebra::linear_operator & op,
    bool trans = false, sc alpha = 1.0 );

  /**
   * Returns true if dimensions match.
   */
  bool is_valid( ) const;

 protected:
  std::vector< const besthea::linear_algebra::linear_operator * >
    _compound;                 //!< Vector of operators.
  std::vector< bool > _trans;  //!< Transposition of individual operators.
  std::vector< sc > _alpha;    //!< Multiplicative factors.

  lo _maximal_dimension;  //!< Maximal dimension of all operators.
};

#endif /* INCLUDE_BESTHEA_COMPOUND_LINEAR_OPERATOR_H_ */
