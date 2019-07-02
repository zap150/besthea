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

#include "besthea/uniform_spacetime_heat_sl_kernel_antiderivative.h"

#include "besthea/settings.h"

#include <cmath>

#pragma omp declare simd uniform( nx, ny, delta ) simdlen( DATA_WIDTH )
sc besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative::
  anti_tau_anti_t(
    sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny, lo delta ) {
  sc value = 0.0;
  sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
  sc sqrt_d = std::sqrt( delta );
  sc sqrt_pi_a = sqrt( M_PI * _alpha );

  if ( delta > 0 ) {
    if ( norm > 0.0 ) {  //  delta > 0, norm > 0
      value = ( delta / ( 4.0 * M_PI * _alpha * norm )
                + norm / ( 8.0 * M_PI * _alpha * _alpha ) )
          * std::erf( norm / ( 2.0 * sqrt_d * std::sqrt( _alpha ) ) )
        + sqrt_d / ( 4.0 * M_PI * _alpha * sqrt_pi_a )
          * std::exp( -( norm * norm ) / ( 4.0 * delta * _alpha ) );
    } else {  //  delta > 0, limit for norm -> 0
      value = sqrt_d / ( 2 * M_PI * _alpha * sqrt_pi_a );
    }
  } else {  // limit for delta -> 0, assuming norm > 0
    value = norm / ( 8.0 * M_PI * _alpha * _alpha );
  }

  return value;
}

#pragma omp declare simd uniform( nx, ny, delta ) simdlen( DATA_WIDTH )
sc besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative::anti_tau(
  sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny, lo delta ) {
  sc value = 0.0;
  sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );

  if ( delta > 0 ) {
    value = std::erf( norm / sqrt( 4.0 * _alpha * delta ) )
      / ( 4.0 * M_PI * _alpha * norm );
  } else {
    value = 1.0 / ( 4.0 * M_PI * _alpha * norm );
  }

  return value;
}
