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

#pragma omp declare simd uniform( nx, ny, scaled_delta ) simdlen( DATA_WIDTH )
sc besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative::
  do_anti_tau_anti_t(
    sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny, sc scaled_delta ) {
  sc value;
  sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
  sc sqrt_d = std::sqrt( scaled_delta );

  if ( scaled_delta > _eps ) {
    if ( norm > _eps ) {  //  delta > 0, norm > 0
      value = ( scaled_delta / ( _four * _pi * _alpha * norm )
                + norm / ( _eight * _pi * _alpha * _alpha ) )
          * std::erf( norm / ( _two * sqrt_d * _sqrt_alpha ) )
        + sqrt_d / ( _four * _pi * _alpha * _sqrt_pi * _sqrt_alpha )
          * std::exp( -( norm * norm ) / ( _four * scaled_delta * _alpha ) );
    } else {  //  delta > 0, limit for norm -> 0
      value = sqrt_d / ( _two * _pi * _alpha * _sqrt_pi * _sqrt_alpha );
    }
  } else {  // limit for delta -> 0, assuming norm > 0
    value = norm / ( _eight * _pi * _alpha * _alpha );
  }

  return value;
}

#pragma omp declare simd uniform( nx, ny, scaled_delta ) simdlen( DATA_WIDTH )
sc besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative::do_anti_tau(
  sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny, sc scaled_delta ) {
  sc value;
  sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );

  if ( scaled_delta > _eps ) {  //  delta > 0, norm > 0
    value = std::erf( norm / std::sqrt( _four * _alpha * scaled_delta ) )
      / ( _four * _pi * _alpha * norm );
  } else {  // limit for delta -> 0, assuming norm > 0
    value = _one / ( _four * _pi * _alpha * norm );
  }

  return value;
}

#pragma omp declare simd uniform( nx, ny ) simdlen( DATA_WIDTH )
sc besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative::
  do_anti_tau_limit( sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny ) {
  sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );

  sc value = _one / ( _four * _pi * _alpha * norm );

  return value;
}
