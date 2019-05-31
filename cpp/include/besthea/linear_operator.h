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

/** @file linear_operator.h
 * @brief Parent class for linear operators.
 */

#ifndef INCLUDE_BESTHEA_LINEAR_OPERATOR_H_
#define INCLUDE_BESTHEA_LINEAR_OPERATOR_H_

#include "besthea/settings.h"
#include "besthea/vector.h"

namespace besthea {
  namespace linear_algebra {
    class linear_operator;
  }
}

/**
 *  Class representing a linear operator.
 */
class besthea::linear_algebra::linear_operator {
  using vector = besthea::linear_algebra::vector;

 public:
  /*!
   * @brief y = beta * y + alpha * this * x.
   * @param[in] A
   * @param[in] x
   * @param[in,out] y
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( vector const & x, vector & y, bool trans = false,
    sc alpha = 1.0, sc beta = 0.0 )
    = 0;

 protected:
  lo dim_domain;
  lo dim_range;
};

#endif /* INCLUDE_BESTHEA_LINEAR_OPERATOR_H_ */
