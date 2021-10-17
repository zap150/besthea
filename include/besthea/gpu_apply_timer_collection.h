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

/** @file gpu_apply_timer_collection.h
 * @brief Class taking care of CPU-GPU laod distribution.
 */

#ifndef INCLUDE_BESTHEA_GPU_APPLY_TIMER_COLLECTION_H_
#define INCLUDE_BESTHEA_GPU_APPLY_TIMER_COLLECTION_H_

#include "besthea/time_measurer.h"
#include "besthea/time_measurer_cuda.h"

#include <vector>


namespace besthea::bem::onthefly {
  class gpu_apply_timer_collection;
}

/*!
 *  Struct containing several timers used in GPU onthefly matrix apply.
 */
struct besthea::bem::onthefly::gpu_apply_timer_collection {
  /*!
   * Constructor
   * @param[in] n_gpus Number of GPU devices
   */
  gpu_apply_timer_collection( int n_gpus );

  /*!
   * Prints all the timing information
   */
  void print_all( );

  /*!
   * Prints the provided timing information for all GPUs
   */
  void print_timers(
    std::vector< besthea::tools::time_measurer_cuda > & timers );

  /*!
   * Returns the execution time of the constant part on the CPU
   */
  double get_cpu_time_const( );

  /*!
   * Returns the execution time of the part that scales with number of elements
   * on the CPU
   */
  double get_cpu_time_scaling( );

  /*!
   * Returns the execution time of the constant part on the GPU
   */
  double get_gpu_time_const( );

  /*!
   * Returns the execution time of the part that scales with number of elements
   * on the GPU
   */
  double get_gpu_time_scaling( );

  /*!
   * Returns the total execution time on the GPU
   */
  double get_gpu_time_all( );

  std::vector< besthea::tools::time_measurer_cuda > gpu_all;
  std::vector< besthea::tools::time_measurer_cuda > gpu_copyin;
  std::vector< besthea::tools::time_measurer_cuda > gpu_compute;
  std::vector< besthea::tools::time_measurer_cuda > gpu_copyout;
  besthea::tools::time_measurer cpu_scalein;
  besthea::tools::time_measurer cpu_treg_sreg;
  besthea::tools::time_measurer cpu_treg_ssng;
  besthea::tools::time_measurer cpu_tsng;
  besthea::tools::time_measurer cpu_all;
  besthea::tools::time_measurer combined;
};

#endif /* INCLUDE_BESTHEA_GPU_APPLY_TIMER_COLLECTION_H_ */
