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

/** @file spacetime_heat_sl_kernel_antiderivative.h
 * @brief Kernel for uniform_spacetime_tensor_mesh.h.
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_HEAT_SL_KERNEL_ANTIDERIVATIVE_H_
#define INCLUDE_BESTHEA_SPACETIME_HEAT_SL_KERNEL_ANTIDERIVATIVE_H_

#include <besthea/spacetime_heat_kernel_antiderivative.h>

#include "besthea/settings.h"

#include <vector>

namespace besthea {
  namespace bem {
    class spacetime_heat_sl_kernel_antiderivative;
  }
}

/**
 *  Class representing a first and second antiderivative of the single-layer
 * spacetime kernel.
 */
class besthea::bem::spacetime_heat_sl_kernel_antiderivative
  : public besthea::bem::spacetime_heat_kernel_antiderivative<
      spacetime_heat_sl_kernel_antiderivative > {
 public:
  /**
   * Constructor.
   * @param[in] alpha Heat conductivity.
   */
  spacetime_heat_sl_kernel_antiderivative( sc alpha )
    : spacetime_heat_kernel_antiderivative<
      spacetime_heat_sl_kernel_antiderivative >( alpha ) {
  }

  /**
   * Destructor.
   */
  virtual ~spacetime_heat_sl_kernel_antiderivative( ) {
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] ttau `t-tau`.
   */
#pragma omp declare simd uniform( this, nx, ny, ttau ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc do_anti_tau_anti_t( sc xy1, sc xy2, sc xy3, [[maybe_unused]] const sc * nx,
    [[maybe_unused]] const sc * ny, sc ttau ) const {
    sc value;
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
    sc sqrt_d = std::sqrt( ttau );

    if ( ttau > _eps ) {
      if ( norm > _eps ) {  //  ttau > 0, norm > 0
        value = ( ttau / ( _four * _pi * _alpha * norm )
                  + norm / ( _eight * _pi * _alpha2 ) )
            * std::erf( norm / ( _two * sqrt_d * _sqrt_alpha ) )
          + sqrt_d / ( _four * _pi * _alpha * _sqrt_pi * _sqrt_alpha )
            * std::exp( -( norm * norm ) / ( _four * ttau * _alpha ) );
      } else {  //  ttau > 0, limit for norm -> 0
        value = sqrt_d / ( _two * _pi * _alpha * _sqrt_pi * _sqrt_alpha );
      }
    } else {  // limit for ttau -> 0, assuming norm > 0
      value = norm / ( _eight * _pi * _alpha2 );
    }

    return value;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] ttau `t-tau`.
   */
#pragma omp declare simd uniform( this, nx, ny, ttau ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc do_anti_tau_anti_t_regular_in_time( sc xy1, sc xy2, sc xy3,
    [[maybe_unused]] const sc * nx, [[maybe_unused]] const sc * ny,
    sc ttau ) const {
    sc value;
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
    sc sqrt_d = std::sqrt( ttau );

    if ( norm > _eps ) {  //  delta > 0, norm > 0
      value = ( ttau / ( _four * _pi * _alpha * norm )
                + norm / ( _eight * _pi * _alpha2 ) )
          * std::erf( norm / ( _two * sqrt_d * _sqrt_alpha ) )
        + sqrt_d / ( _four * _pi * _alpha * _sqrt_pi * _sqrt_alpha )
          * std::exp( -( norm * norm ) / ( _four * ttau * _alpha ) );
    } else {  //  delta > 0, limit for norm -> 0
      value = sqrt_d / ( _two * _pi * _alpha * _sqrt_pi * _sqrt_alpha );
    }

    return value;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] ttau `t-tau`.
   */
#pragma omp declare simd uniform( this, nx, ny, ttau ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc do_anti_tau_anti_t_regular_in_time_regular_in_space( sc xy1, sc xy2,
    sc xy3, [[maybe_unused]] const sc * nx, [[maybe_unused]] const sc * ny,
    sc ttau ) const {
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
    sc sqrt_d = std::sqrt( ttau );

    //  ttau > 0, norm > 0
    sc value = ( ttau / ( _four * _pi * _alpha * norm )
                 + norm / ( _eight * _pi * _alpha2 ) )
        * std::erf( norm / ( _two * sqrt_d * _sqrt_alpha ) )
      + sqrt_d / ( _four * _pi * _alpha * _sqrt_pi * _sqrt_alpha )
        * std::exp( -( norm * norm ) / ( _four * ttau * _alpha ) );

    return value;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   */
#pragma omp declare simd uniform( this, nx, ny ) simdlen( BESTHEA_SIMD_WIDTH )
  sc do_anti_tau_anti_t_limit_in_time_regular_in_space( sc xy1, sc xy2, sc xy3,
    [[maybe_unused]] const sc * nx, [[maybe_unused]] const sc * ny ) const {
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );

    // limit for ttau -> 0, assuming norm > 0
    sc value = norm / ( _eight * _pi * _alpha2 );

    return value;
  }

  /**
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] ttau `t-tau`.
   */
#pragma omp declare simd uniform( this, nx, ny, ttau ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc do_anti_tau_regular( sc xy1, sc xy2, sc xy3,
    [[maybe_unused]] const sc * nx, [[maybe_unused]] const sc * ny,
    sc ttau ) const {
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
    sc sqrt_d = std::sqrt( ttau );

    sc value = std::erf( norm / ( _two * _sqrt_alpha * sqrt_d ) )
      / ( _four * _pi * _alpha * norm );

    return value;
  }

  /**
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   */
#pragma omp declare simd uniform( this, nx, ny ) simdlen( BESTHEA_SIMD_WIDTH )
  sc do_anti_tau_limit( sc xy1, sc xy2, sc xy3, [[maybe_unused]] const sc * nx,
    [[maybe_unused]] const sc * ny ) const {
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );

    sc value = _one / ( _four * _pi * _alpha * norm );

    return value;
  }

  /**
   * Evaluates the definite integral over the same time interval.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] t0 Start of interval.
   * @param[in] t1 End of interval.
   */
#pragma omp declare simd uniform( this, nx, ny, t0, t1 ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc do_definite_integral_over_same_interval(
    sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny, sc t0, sc t1 ) const {
    sc value = ( t1 - t0 ) * do_anti_tau_limit( xy1, xy2, xy3, nx, ny )
      - do_anti_tau_anti_t_regular_in_time( xy1, xy2, xy3, nx, ny, t1 - t0 )
      + do_anti_tau_anti_t_limit_in_time_regular_in_space(
        xy1, xy2, xy3, nx, ny );

    return value;
  }

  /**
   * Evaluates the definite integral over the different time intervals.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] t0 Start of interval in `t`.
   * @param[in] t1 End of interval in `t`.
   * @param[in] tau0 Start of interval in `tau`.
   * @param[in] tau1 End of interval in `tau`.
   */
#pragma omp declare simd uniform( this, nx, ny, t0, t1, tau0, tau1 ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc do_definite_integral_over_different_intervals( sc xy1, sc xy2, sc xy3,
    const sc * nx, const sc * ny, sc t0, sc t1, sc tau0, sc tau1 ) const {
    sc value = do_anti_tau_anti_t( xy1, xy2, xy3, nx, ny, t1 - tau1 )
      - do_anti_tau_anti_t( xy1, xy2, xy3, nx, ny, t1 - tau0 )
      - do_anti_tau_anti_t( xy1, xy2, xy3, nx, ny, t0 - tau1 )
      + do_anti_tau_anti_t( xy1, xy2, xy3, nx, ny, t0 - tau0 );

    return value;
  }
};

#endif /* INCLUDE_BESTHEA_SPACETIME_HEAT_SL_KERNEL_ANTIDERIVATIVE_H_ \
        */
