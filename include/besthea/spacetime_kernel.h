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

/** @file spacetime_kernel.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_KERNEL_H_
#define INCLUDE_BESTHEA_SPACETIME_KERNEL_H_

#include "besthea/settings.h"

namespace besthea {
  namespace bem {
    template< class derived_type >
    class spacetime_kernel;
  }
}

/**
 *  Class representing a spacetime kernel.
 */
template< class derived_type >
class besthea::bem::spacetime_kernel {
 public:
  /**
   * Constructor.
   */
  spacetime_kernel( ) {
  }

  /**
   * Destructor.
   */
  virtual ~spacetime_kernel( ) {
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
   * Evaluates the kernel.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] ttau `t-tau`.
   */
#pragma omp declare simd uniform( this, nx, ny, ttau ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc evaluate(
    sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny, sc ttau ) const {
    return derived( )->do_evaluate( xy1, xy2, xy3, nx, ny, ttau );
  }

 protected:
  static constexpr sc _eps{ 1e-12 };  //!< Auxiliary variable
};

#endif /* INCLUDE_BESTHEA_SPACETIME_KERNEL_H_ */
