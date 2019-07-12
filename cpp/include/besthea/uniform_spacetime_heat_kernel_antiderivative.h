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

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_KERNEL_ANTIDERIVATIVE_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_KERNEL_ANTIDERIVATIVE_H_

#include "besthea/settings.h"
#include "besthea/uniform_spacetime_kernel_antiderivative.h"

#include <cmath>
#include <vector>

namespace besthea {
  namespace bem {
    template< class derived_type >
    class uniform_spacetime_heat_kernel_antiderivative;
  }
}

/**
 *  Class representing a first and second antiderivative of a spacetime heat
 * kernel.
 */
template< class derived_type >
class besthea::bem::uniform_spacetime_heat_kernel_antiderivative
  : public besthea::bem::uniform_spacetime_kernel_antiderivative<
      derived_type > {
 public:
  uniform_spacetime_heat_kernel_antiderivative( ) = delete;

  /**
   * Constructor.
   * @param[in] timestep Time step.
   * @param[in] alpha Heat conductivity.
   */
  uniform_spacetime_heat_kernel_antiderivative( sc timestep, sc alpha )
    : uniform_spacetime_kernel_antiderivative< derived_type >( timestep ),
      _alpha( alpha ),
      _sqrt_alpha( std::sqrt( alpha ) ) {
  }

  /**
   * Destructor.
   */
  virtual ~uniform_spacetime_heat_kernel_antiderivative( ) {
  }

 protected:
  sc _alpha;  //!< Heat conductivity.

  sc _sqrt_alpha;  //!< Auxiliary variable

  const sc _pi{ M_PI };                    //!< Auxiliary variable
  const sc _sqrt_pi{ std::sqrt( M_PI ) };  //!< Auxiliary variable
  const sc _one{ 1.0 };                    //!< Auxiliary variable
  const sc _two{ 2.0 };                    //!< Auxiliary variable
  const sc _four{ 4.0 };                   //!< Auxiliary variable
  const sc _eight{ 8.0 };                  //!< Auxiliary variable

  const sc _eps{ 1e-12 };  //!< Auxiliary variable
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_KERNEL_ANTIDERIVATIVE_H_ */
