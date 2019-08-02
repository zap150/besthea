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

/** @file uniform_spacetime_kernel_antiderivative.h
 * @brief Kernel for uniform_spacetime_tensor_mesh.h.
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_KERNEL_ANTIDERIVATIVE_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_KERNEL_ANTIDERIVATIVE_H_

#include "besthea/settings.h"

#include <vector>

namespace besthea {
  namespace bem {
    template< class derived_type >
    class uniform_spacetime_kernel_antiderivative;
  }
}

/**
 *  Class representing a first and second antiderivative of a spacetime kernel.
 */
template< class derived_type >
class besthea::bem::uniform_spacetime_kernel_antiderivative {
 public:
  /**
   * Constructor.
   * @param[in] timestep Timestep.
   */
  uniform_spacetime_kernel_antiderivative( sc timestep )
    : _timestep( timestep ) {
  }

  /**
   * Destructor.
   */
  virtual ~uniform_spacetime_kernel_antiderivative( ) {
  }

  /**
   * Returns this cast to the descendant's type.
   */
  derived_type * derived( ) {
    return static_cast< derived_type * >( this );
  }

  /**
   * Returns this cast to the descendant's type.
   */
  const derived_type * derived( ) const {
    return static_cast< const derived_type * >( this );
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   */
#pragma omp declare simd uniform( ny, scaled_delta ) simdlen( DATA_WIDTH )
  sc anti_tau_anti_t( sc xy1, sc xy2, sc xy3, const sc * ny, sc scaled_delta ) const {
    return derived( )->do_anti_tau_anti_t( xy1, xy2, xy3, ny, scaled_delta );
  }

  /**
   * Evaluates the first antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] ny Normal in the `y` variable.
   */
#pragma omp declare simd uniform( ny ) simdlen( DATA_WIDTH )
  sc anti_tau_limit( sc xy1, sc xy2, sc xy3, const sc * ny ) const {
    return derived( )->do_anti_tau_limit( xy1, xy2, xy3, ny );
  }

  /**
   * Evaluates the first antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   */
#pragma omp declare simd uniform( ny, scaled_delta ) simdlen( DATA_WIDTH )
  sc anti_tau_regular(
    sc xy1, sc xy2, sc xy3, const sc * ny, sc scaled_delta ) const {
    return derived( )->do_anti_tau_regular( xy1, xy2, xy3, ny, scaled_delta );
  }

 protected:
  sc _timestep;  //!< Time step.
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_KERNEL_ANTIDERIVATIVE_H_ */
