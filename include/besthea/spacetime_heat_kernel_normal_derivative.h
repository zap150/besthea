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

/** @file spacetime_heat_kernel_normal_derivative.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_HEAT_KERNEL_NORMAL_DERIVATIVE_H_
#define INCLUDE_BESTHEA_SPACETIME_HEAT_KERNEL_NORMAL_DERIVATIVE_H_

#include "besthea/settings.h"
#include "besthea/spacetime_kernel.h"

#include <vector>

namespace besthea {
  namespace bem {
    class spacetime_heat_kernel_normal_derivative;
  }
}

/**
 *  Class representing a spacetime heat kernel.
 */
class besthea::bem::spacetime_heat_kernel_normal_derivative
  : public besthea::bem::spacetime_kernel<
      besthea::bem::spacetime_heat_kernel_normal_derivative > {
 public:
  /**
   * Constructor.
   * @param[in] alpha Heat conductivity.
   */
  spacetime_heat_kernel_normal_derivative( sc alpha )
    : _alpha( alpha ), _alpha_sqrt_alpha( alpha * std::sqrt( alpha ) ) {
  }

  /**
   * Destructor.
   */
  virtual ~spacetime_heat_kernel_normal_derivative( ) {
  }

  /**
   * Evaluates the kernel.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] ttau `t-tau`.
   */
#pragma omp declare simd uniform( this, nx, ny, ttau ) simdlen( DATA_WIDTH )
  sc do_evaluate( sc xy1, sc xy2, sc xy3, [[maybe_unused]] const sc * nx,
    const sc * ny, sc ttau ) const {
    sc value = 0.0;

    if ( ttau > _eps ) {
      sc norm2 = xy1 * xy1 + xy2 * xy2 + xy3 * xy3;
      sc dot = xy1 * ny[ 0 ] + xy2 * ny[ 1 ] + xy3 * ny[ 2 ];

      value = dot
        / ( _sixteen * _pi_sqrt_pi * _alpha_sqrt_alpha * ttau * ttau
          * std::sqrt( ttau ) )
        * std::exp( -norm2 / ( _four * _alpha * ttau ) );
    }

    return value;
  }

 protected:
  sc _alpha;  //!< Heat conductivity.

  sc _alpha_sqrt_alpha;  //!< Auxiliary variable

  const sc _pi_sqrt_pi{ M_PI * std::sqrt( M_PI ) };  //!< Auxiliary variable
  static constexpr sc _four{ 4.0 };                  //!< Auxiliary variable
  static constexpr sc _sixteen{ 16.0 };              //!< Auxiliary variable
};

#endif /* INCLUDE_BESTHEA_SPACETIME_HEAT_KERNEL_NORMAL_DERIVATIVE_H_ */