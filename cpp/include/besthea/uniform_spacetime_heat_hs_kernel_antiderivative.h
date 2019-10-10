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

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_HS_KERNEL_ANTIDERIVATIVE_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_HS_KERNEL_ANTIDERIVATIVE_H_

#include "besthea/settings.h"
#include "besthea/uniform_spacetime_heat_kernel_antiderivative.h"

#include <vector>

namespace besthea {
  namespace bem {
    class uniform_spacetime_heat_hs_kernel_antiderivative;
  }
}

/**
 *  Class representing a first and second antiderivative of the hypersingular
 * spacetime kernel.
 */
class besthea::bem::uniform_spacetime_heat_hs_kernel_antiderivative
  : public besthea::bem::uniform_spacetime_heat_kernel_antiderivative<
      uniform_spacetime_heat_hs_kernel_antiderivative > {
 public:
  /**
   * Constructor.
   * @param[in] timestep Time step.
   * @param[in] alpha Heat conductivity.
   */
  uniform_spacetime_heat_hs_kernel_antiderivative( sc timestep, sc alpha )
    : uniform_spacetime_heat_kernel_antiderivative<
      uniform_spacetime_heat_hs_kernel_antiderivative >( timestep, alpha ) {
  }

  /**
   * Destructor.
   */
  virtual ~uniform_spacetime_heat_hs_kernel_antiderivative( ) {
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   */
#pragma omp declare simd uniform( this, ny, scaled_delta ) simdlen( DATA_WIDTH )
  sc do_anti_tau_anti_t(
    sc xy1, sc xy2, sc xy3, const sc * ny, sc scaled_delta ) const {
    return _zero;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   */
#pragma omp declare simd uniform( this, ny, scaled_delta ) simdlen( DATA_WIDTH )
  sc do_anti_tau_anti_t_regular_in_time(
    sc xy1, sc xy2, sc xy3, const sc * ny, sc scaled_delta ) const {
    return _zero;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   */
#pragma omp declare simd uniform( this, ny, scaled_delta ) simdlen( DATA_WIDTH )
  sc do_anti_tau_anti_t_regular_in_time_regular_in_space(
    sc xy1, sc xy2, sc xy3, const sc * ny, sc scaled_delta ) const {
    return _zero;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] ny Normal in the `y` variable.
   */
#pragma omp declare simd uniform( this, ny ) simdlen( DATA_WIDTH )
  sc do_anti_tau_anti_t_limit_in_time_regular_in_space(
    sc xy1, sc xy2, sc xy3, const sc * ny ) const {
    return _zero;
  }

  /**
   * ONLY NEEDED FOR POTENTIALS!
   * Evaluates the first antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   */
#pragma omp declare simd uniform( this, ny, scaled_delta ) simdlen( DATA_WIDTH )
  sc do_anti_tau_regular(
    sc xy1, sc xy2, sc xy3, const sc * ny, sc scaled_delta ) const {
    return _zero;
  }

/**
 * Evaluates the first antiderivative.
 * @param[in] xy1 First coordinate of `x - y`.
 * @param[in] xy2 Second coordinate of `x - y`.
 * @param[in] xy3 Third coordinate of `x - y`.
 * @param[in] ny Normal in the `y` variable.
 */
#pragma omp declare simd uniform( this, ny ) simdlen( DATA_WIDTH )
  sc do_anti_tau_limit( sc xy1, sc xy2, sc xy3, const sc * ny ) const {
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );

    sc value = _one / ( _four * _pi * _alpha * norm );

    value *= _alpha2;
    return value;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   * @param[out] value1 Return value for anti_tau_anti_t part.
   * @param[out] value2 Return value for anti_t part.
   */
#pragma omp declare simd uniform( this, nx, ny, scaled_delta ) \
  simdlen( DATA_WIDTH )
  void anti_tau_anti_t_and_anti_t( sc xy1, sc xy2, sc xy3, const sc * nx,
    const sc * ny, sc scaled_delta, sc * value1, sc * value2 ) const {
    sc dot = nx[ 0 ] * ny[ 0 ] + nx[ 1 ] * ny[ 1 ] + nx[ 2 ] * ny[ 2 ];
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
    sc sqrt_d = std::sqrt( scaled_delta );
    sc erf_value;
    sc four_pi_alpha_norm;
    sc pi_alpha_sqrtpi_sqrta;

    if ( scaled_delta > _eps ) {
      pi_alpha_sqrtpi_sqrta = _pi * _alpha * _sqrt_pi * _sqrt_alpha;
      if ( norm > _eps ) {  //  delta > 0, norm > 0
        erf_value = std::erf( norm / ( _two * sqrt_d * _sqrt_alpha ) );
        four_pi_alpha_norm = _four * _pi * _alpha * norm;
        *value1 = ( scaled_delta / four_pi_alpha_norm
                    + norm / ( _eight * _pi * _alpha2 ) )
            * erf_value
          + sqrt_d / ( _four * _pi * _alpha * _sqrt_pi * _sqrt_alpha )
            * std::exp( -( norm * norm ) / ( _four * scaled_delta * _alpha ) );

        *value2 = erf_value / four_pi_alpha_norm;
      } else {  //  delta > 0, limit for norm -> 0
        *value1 = sqrt_d / ( _two * pi_alpha_sqrtpi_sqrta );
        *value2 = _one / ( _four * pi_alpha_sqrtpi_sqrta * sqrt_d );
      }
    } else {  // limit for delta -> 0, assuming norm > 0
      *value1 = norm / ( _eight * _pi * _alpha2 );
      *value2 = _one / ( _four * _pi * _alpha * norm );
    }

    *value1 *= _alpha2;
    *value2 *= dot * _alpha;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   * @param[out] value1 Return value for anti_tau_anti_t part.
   * @param[out] value2 Return value for anti_t part.
   */
#pragma omp declare simd uniform( this, nx, ny, scaled_delta ) \
  simdlen( DATA_WIDTH )
  void anti_tau_anti_t_and_anti_t_regular_in_time( sc xy1, sc xy2, sc xy3,
    const sc * nx, const sc * ny, sc scaled_delta, sc * value1,
    sc * value2 ) const {
    sc dot = nx[ 0 ] * ny[ 0 ] + nx[ 1 ] * ny[ 1 ] + nx[ 2 ] * ny[ 2 ];
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
    sc sqrt_d = std::sqrt( scaled_delta );

    if ( norm > _eps ) {  //  delta > 0, norm > 0
      sc erf_value = std::erf( norm / ( _two * sqrt_d * _sqrt_alpha ) );
      sc four_pi_alpha_norm = _four * _pi * _alpha * norm;
      *value1 = ( scaled_delta / four_pi_alpha_norm
                  + norm / ( _eight * _pi * _alpha2 ) )
          * erf_value
        + sqrt_d / ( _four * _pi * _alpha * _sqrt_pi * _sqrt_alpha )
          * std::exp( -( norm * norm ) / ( _four * scaled_delta * _alpha ) );

      *value2 = erf_value / four_pi_alpha_norm;
    } else {  //  delta > 0, limit for norm -> 0
      sc pi_alpha_sqrtpi_sqrta = _pi * _alpha * _sqrt_pi * _sqrt_alpha;
      *value1 = sqrt_d / ( _two * pi_alpha_sqrtpi_sqrta );

      *value2 = _one / ( _four * pi_alpha_sqrtpi_sqrta * sqrt_d );
    }

    *value1 *= _alpha2;
    *value2 *= dot * _alpha;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[in] scaled_delta Difference of time intervals.
   * @param[out] value1 Return value for anti_tau_anti_t part.
   * @param[out] value2 Return value for anti_t part.
   */
#pragma omp declare simd uniform( this, nx, ny, scaled_delta ) \
  simdlen( DATA_WIDTH )
  void anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space( sc xy1,
    sc xy2, sc xy3, const sc * nx, const sc * ny, sc scaled_delta, sc * value1,
    sc * value2 ) const {
    sc dot = nx[ 0 ] * ny[ 0 ] + nx[ 1 ] * ny[ 1 ] + nx[ 2 ] * ny[ 2 ];
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
    sc sqrt_d = std::sqrt( scaled_delta );
    sc erf_value = std::erf( norm / ( _two * sqrt_d * _sqrt_alpha ) );
    sc four_pi_alpha_norm = _four * _pi * _alpha * norm;

    //  delta > 0, norm > 0
    *value1 = ( scaled_delta / four_pi_alpha_norm
                + norm / ( _eight * _pi * _alpha2 ) )
        * erf_value
      + sqrt_d / ( _four * _pi * _alpha * _sqrt_pi * _sqrt_alpha )
        * std::exp( -( norm * norm ) / ( _four * scaled_delta * _alpha ) );

    *value2 = erf_value / four_pi_alpha_norm;

    *value1 *= _alpha2;
    *value2 *= dot * _alpha;
  }

  /**
   * Evaluates the second antiderivative.
   * @param[in] xy1 First coordinate of `x - y`.
   * @param[in] xy2 Second coordinate of `x - y`.
   * @param[in] xy3 Third coordinate of `x - y`.
   * @param[in] nx Normal in the `x` variable.
   * @param[in] ny Normal in the `y` variable.
   * @param[out] value1 Return value for anti_tau_anti_t part.
   * @param[out] value2 Return value for anti_t part.
   */
#pragma omp declare simd uniform( this, nx, ny ) simdlen( DATA_WIDTH )
  void anti_tau_anti_t_and_anti_t_limit_in_time_regular_in_space( sc xy1,
    sc xy2, sc xy3, const sc * nx, const sc * ny, sc * value1,
    sc * value2 ) const {
    sc dot = nx[ 0 ] * ny[ 0 ] + nx[ 1 ] * ny[ 1 ] + nx[ 2 ] * ny[ 2 ];
    sc norm = std::sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );

    // limit for delta -> 0, assuming norm > 0
    *value1 = norm / ( _eight * _pi * _alpha2 );
    *value2 = _one / ( _four * _pi * _alpha * norm );

    *value1 *= _alpha2;
    *value2 *= dot * _alpha;
  }
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_HEAT_HS_KERNEL_ANTIDERIVATIVE_H_ \
        */
