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

/** @file spacetime_heat_initial_kernel_antiderivative.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_HEAT_INITIAL_M0_KERNEL_ANTIDERIVATIVE_H_
#define INCLUDE_BESTHEA_SPACETIME_HEAT_INITIAL_M0_KERNEL_ANTIDERIVATIVE_H_

#include <besthea/spacetime_heat_initial_kernel_antiderivative.h>

#include "besthea/settings.h"

#include <vector>

namespace besthea {
  namespace bem {
    class spacetime_heat_initial_m0_kernel_antiderivative;
  }
}

/**
 *  Class representing a first and second antiderivative of the double-layer
 * spacetime kernel.
 */
class besthea::bem::spacetime_heat_initial_m0_kernel_antiderivative
  : public besthea::bem::spacetime_heat_initial_kernel_antiderivative<
      spacetime_heat_initial_m0_kernel_antiderivative > {
 public:
  /**
   * Constructor.
   * @param[in] alpha Heat conductivity.
   */
  spacetime_heat_initial_m0_kernel_antiderivative( sc alpha )
    : spacetime_heat_initial_kernel_antiderivative<
      spacetime_heat_initial_m0_kernel_antiderivative >( alpha ) {
  }

  /**
   * Destructor.
   */
  virtual ~spacetime_heat_initial_m0_kernel_antiderivative( ) {
  }

  /**
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] t `t`.
   */
#pragma omp declare simd uniform( this, nx, t ) simdlen( DATA_WIDTH )
  sc do_anti_t_regular(
    sc xy1, sc xy2, sc xy3, [[maybe_unused]] const sc * nx, sc t ) const {
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
    sc sqrt_d = std::sqrt( t );

    sc value = -std::erf( norm / ( _two * _sqrt_alpha * sqrt_d ) )
      / ( _four * _pi * _alpha * norm );

    return value;
  }

  /**
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   */
#pragma omp declare simd uniform( this, nx ) simdlen( DATA_WIDTH )
  sc do_anti_t_limit(
    sc xy1, sc xy2, sc xy3, [[maybe_unused]] const sc * nx ) const {
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );

    sc value = -_one / ( _four * _pi * _alpha * norm );

    return value;
  }
};

#endif /* INCLUDE_BESTHEA_SPACETIME_HEAT_INITIAL_M0_KERNEL_ANTIDERIVATIVE_H_ \
        */
