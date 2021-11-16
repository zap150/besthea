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

/** @file gpu_apply_load_distribution.h
 * @brief Class taking care of CPU-GPU laod distribution.
 */

#ifndef INCLUDE_BESTHEA_GPU_APPLY_LOAD_DISTRIBUTION_H_
#define INCLUDE_BESTHEA_GPU_APPLY_LOAD_DISTRIBUTION_H_

#include "besthea/settings.h"

namespace besthea::bem::onthefly {
  class gpu_apply_load_distribution;
}

/*!
 *  Class taking care of CPU-GPU load distribution
 */
class besthea::bem::onthefly::gpu_apply_load_distribution {
 public:
  /*!
   * Constructor
   * @param[in] n_gpus_ Number of GPU devices
   * @param[in] n_elems_ Number of spatial elements
   * @param[in] gpu_chunk_size_ Partition the work on GPU to multiples of
   * this number
   * @param[in] use_cpu_ Whether to assign work to CPU or use only GPU
   */
  gpu_apply_load_distribution(
    int n_gpus_, lo n_elems_, lo gpu_chunk_size_, bool use_cpu_ );

  /*!
   * Adapts the load distribution according to the provided timing results
   * @param[in] cpu_time_const Execution time of the constant part of CPU work
   * @param[in] cpu_time_scaling Execution time of the part that scales with
   * number of elements on the CPU
   * @param[in] gpu_time_const Execution time of the constant part of GPU work
   * @param[in] gpu_time_scaling Execution time of the part that scales with
   * number of elements on the GPU
   * @param[in] inertia Give this much weight to the previous distribution
   * and 1-inertia weight to the newly calculated distribution
   */
  void adapt( double cpu_time_const, double cpu_time_scaling,
    double gpu_time_const, double gpu_time_scaling, double inertia = 0.0 );

  /*!
   * Prints the load distribution
   */
  void print( );

  /*!
   * Returns the start (inclusive) of the range of spatial elements for the CPU
   */
  lo get_cpu_begin( ) const {
    return n_elems - cpu_n_tst_elems;
  }

  /*!
   * Returns the end (exclusive) of the range of spatial elements for the CPU
   */
  lo get_cpu_end( ) const {
    return n_elems;
  }

  /*!
   * Returns the number of GPUs this instance is working with
   */
  lo get_cpu_count( ) const {
    return cpu_n_tst_elems;
  }

  /*!
   * Returns the start (inclusive) of the range of spatial elements
   * for the gpu_idx-th GPU
   */
  lo get_gpu_begin( int gpu_idx ) const {
    return gpu_i_tst_begins[ gpu_idx ];
  }

  /*!
   * Returns the end (exclusive) of the range of spatial elements
   * for the gpu_idx-th GPU
   */
  lo get_gpu_end( int gpu_idx ) const {
    return gpu_i_tst_begins[ gpu_idx + 1 ];
  }

  /*!
   * Returns the number of spatial elements for the gpu_idx-th GPU
   */
  lo get_gpu_count( int gpu_idx ) const {
    return get_gpu_end( gpu_idx ) - get_gpu_begin( gpu_idx );
  }

  /*!
   * Returns the total number of spatial elements assigned for the GPUs
   */
  lo get_gpu_count_total( ) const {
    return n_elems - cpu_n_tst_elems;
  }

 private:
  /*!
   * Calculates the number of elements assigned to the CPU from the calculated
   * optimal value. Mainly handles the edge cases.
   */
  lo adjust_cpu_count( double suggested ) const;
  /*!
   * Updates the ranges of elements assigned to the GPUs based on the current
   * number of elements assigned to the CPU
   */
  void update_gpu_begins( );

 private:
  lo cpu_n_tst_elems;             //!< Number of elements assigned to the CPU
  double cpu_n_tst_elems_target;  //!< The intended number of elements assigned
                                  //!< to the CPU
  std::vector< lo > gpu_i_tst_begins;  //!< Indexes dividing the elements to
                                       //!< ranges for the GPUs.
  lo n_elems;                          //!< Total number of spatial elements
  lo gpu_chunk_size;  //!< Partition the work on GPU to multiples of this number
  int n_gpus;         //<! Number of GPUs
  bool use_cpu;       //<! Whether to assign work to CPU or use only GPU
};

#endif /* INCLUDE_BESTHEA_GPU_APPLY_LOAD_DISTRIBUTION_H_ */
