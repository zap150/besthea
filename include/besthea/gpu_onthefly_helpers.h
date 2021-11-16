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

/** @file gpu_onthefly_helpers.h
 * @brief Helper structs for onthefly classes.
 */

#ifndef INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_
#define INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_

#include "besthea/settings.h"

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace besthea::bem::onthefly::helpers {
  template< int quadr_order >
  struct quadrature_reference_raw;

  template< int quadr_order >
  struct quadrature_nodes_raw;

  struct heat_kernel_parameters;

  struct gpu_apply_vectors_data;

  class apply_load_distribution;

  struct timer_collection;

  struct gpu_threads_per_block;

  /*!
   * Translates quadrature order to quadrature size -- number of quadrature
   * nodes.
   */
#ifdef __NVCC__
  __host__ __device__
#endif
    constexpr int
    qo2qs( int quadr_order ) {
    // this needs to be changed too if quadrature rules are changed in future
    switch ( quadr_order ) {
      case 5:
        return 49;
      case 4:
        return 36;
      case 2:
        return 9;
      case 1:
      default:
        return 1;
    }
  }

  extern bool is_gpu_quadr_order5_initialized;
  extern bool is_gpu_quadr_order4_initialized;
  extern bool is_gpu_quadr_order2_initialized;
  extern bool is_gpu_quadr_order1_initialized;

}

/*!
 *  Struct containing reference quadrature nodes and quadrature weights as raw
 * data.
 */
template< int quadr_order >
struct besthea::bem::onthefly::helpers::quadrature_reference_raw {
  sc _x1_ref[ qo2qs( quadr_order ) ];
  sc _x2_ref[ qo2qs( quadr_order ) ];
  sc _y1_ref[ qo2qs( quadr_order ) ];
  sc _y2_ref[ qo2qs( quadr_order ) ];
  sc _w[ qo2qs( quadr_order ) ];
};

/*!
 *  Struct containing mapped quadrature nodes as raw data.
 */
template< int quadr_order >
struct besthea::bem::onthefly::helpers::quadrature_nodes_raw {
  sc xs[ qo2qs( quadr_order ) ];
  sc ys[ qo2qs( quadr_order ) ];
  sc zs[ qo2qs( quadr_order ) ];
};

/*!
 *  Struct containing parameters of heat kernel and other auxiliary variables.
 */
struct besthea::bem::onthefly::helpers::heat_kernel_parameters {
  sc alpha;
  sc sqrt_alpha;
  sc alpha_2;
  sc pi;
  sc sqrt_pi;
  heat_kernel_parameters( sc alpha_ ) {
    this->alpha = alpha_;
    sqrt_alpha = std::sqrt( alpha_ );
    alpha_2 = alpha_ * alpha_;
    pi = M_PI;
    sqrt_pi = std::sqrt( M_PI );
  }
};

#define CUDA_CHECK( err )                                                  \
  if ( err != cudaSuccess ) {                                              \
    std::cerr << "CUDA error " << err << " '" << cudaGetErrorString( err ) \
              << "':\n"                                                    \
              << "  in file '" << __FILE__ << "'\n"                        \
              << "  in function '" << __func__ << "'\n"                    \
              << "  on line " << __LINE__ << "'\n";                        \
    throw std::runtime_error( "BESTHEA Exception: cuda error" );           \
  }

#endif /* INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_ */
