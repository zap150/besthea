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

/** @file uniform_spacetime_heat_kernel_antiderivative.h
 * @brief Kernel for uniform_spacetime_tensor_mesh.h.
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_SL_KERNEL_ANTIDERIVATIVE_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_SL_KERNEL_ANTIDERIVATIVE_H_

#include "besthea/settings.h"
#include "besthea/uniform_spacetime_heat_kernel_antiderivative.h"

#include <vector>

namespace besthea {
  namespace bem {
    class uniform_spacetime_heat_sl_kernel_antiderivative;
  }
}

class besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative
  : public besthea::bem::uniform_spacetime_heat_kernel_antiderivative<
      uniform_spacetime_heat_sl_kernel_antiderivative > {
 public:
  uniform_spacetime_heat_sl_kernel_antiderivative( ) = delete;

  /**
   * Constructor.
   * @param[in] ht Time step.
   * @param[in] alpha Heat conductivity.
   */
  uniform_spacetime_heat_sl_kernel_antiderivative( sc ht, sc alpha )
    : uniform_spacetime_heat_kernel_antiderivative<
      uniform_spacetime_heat_sl_kernel_antiderivative >( ht, alpha ) {
  }

  /**
   * Destructor.
   */
  virtual ~uniform_spacetime_heat_sl_kernel_antiderivative( ) {
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] delta Difference of time intervals.
   */
#pragma omp declare simd uniform( nx, ny, delta ) simdlen( DATA_WIDTH )
  sc do_anti_tau_anti_t(
    sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny, lo delta );

  /**
   * Evaluates the first antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] delta Difference of time intervals.
   */
#pragma omp declare simd uniform( nx, ny, delta ) simdlen( DATA_WIDTH )
  sc do_anti_tau(
    sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny, lo delta );
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_SL_KERNEL_ANTIDERIVATIVE_H_ \
        */
