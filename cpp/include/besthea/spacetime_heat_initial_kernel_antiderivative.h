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

#ifndef INCLUDE_BESTHEA_SPACETIME_HEAT_INITIAL_KERNEL_ANTIDERIVATIVE_H_
#define INCLUDE_BESTHEA_SPACETIME_HEAT_INITIAL_KERNEL_ANTIDERIVATIVE_H_

#include <besthea/spacetime_initial_kernel_antiderivative.h>

#include "besthea/settings.h"

#include <cmath>

namespace besthea {
  namespace bem {
    template< class derived_type >
    class spacetime_heat_initial_kernel_antiderivative;
  }
}

/**
 *  Class representing a first antiderivative of an initial spacetime heat
 * kernel.
 */
template< class derived_type >
class besthea::bem::spacetime_heat_initial_kernel_antiderivative
  : public besthea::bem::spacetime_initial_kernel_antiderivative<
      derived_type > {
 public:
  /**
   * Constructor.
   * @param[in] alpha Heat conductivity.
   */
  spacetime_heat_initial_kernel_antiderivative( sc alpha )
    : _alpha( alpha ),
      _sqrt_alpha( std::sqrt( alpha ) ),
      _alpha2( alpha * alpha ) {
  }

  /**
   * Destructor.
   */
  virtual ~spacetime_heat_initial_kernel_antiderivative( ) {
  }

 protected:
  sc _alpha;  //!< Heat conductivity.

  sc _sqrt_alpha;  //!< Auxiliary variable
  sc _alpha2;      //!< Auxiliary variable

  const sc _pi{ M_PI };                    //!< Auxiliary variable
  const sc _sqrt_pi{ std::sqrt( M_PI ) };  //!< Auxiliary variable
  const sc _zero{ 0.0 };                   //!< Auxiliary variable
  const sc _one{ 1.0 };                    //!< Auxiliary variable
  const sc _two{ 2.0 };                    //!< Auxiliary variable
  const sc _four{ 4.0 };                   //!< Auxiliary variable
  const sc _eight{ 8.0 };                  //!< Auxiliary variable

  const sc _eps{ 1e-12 };  //!< Auxiliary variable
};

#endif /* INCLUDE_BESTHEA_SPACETIME_HEAT_INITIAL_KERNEL_ANTIDERIVATIVE_H_ */
