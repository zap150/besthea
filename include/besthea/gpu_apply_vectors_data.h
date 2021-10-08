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

/** @file gpu_apply_vectors_data.h
 * @brief Class taking care of CPU-GPU laod distribution.
 */

#ifndef INCLUDE_BESTHEA_GPU_APPLY_VECTORS_DATA_H_
#define INCLUDE_BESTHEA_GPU_APPLY_VECTORS_DATA_H_

#include "besthea/settings.h"

namespace besthea::bem::onthefly {
    class gpu_apply_vectors_data;
}

/*!
 *  Struct containing CPU and GPU resident vectors data.
 */
struct besthea::bem::onthefly::gpu_apply_vectors_data {
  /*!
   * Constructor
   */
  gpu_apply_vectors_data( );

  /*!
   * Copy constructor - deleted
   */
  gpu_apply_vectors_data( const gpu_apply_vectors_data & that ) = delete;
  
  /*!
   * Destructor
   */
  ~gpu_apply_vectors_data( );

  /*!
   * Allocates the data vectors
   * @param[in] n_gpus Number of GPU devices
   * @param[in] x_block_count Number of blocks in vector x
   * @param[in] x_size_of_block Size of block in vector x
   * @param[in] y_block_count Number of blocks in vector y
   * @param[in] y_size_of_block Size of block in vector y
   */
  void allocate( int n_gpus, lo x_block_count, lo x_size_of_block,
    lo y_block_count, lo y_size_of_block );
    
  /*!
   * Frees the memory of the vectors
   */
  void free( );
  
  sc * h_x;                                //<! raw data on host
  std::vector< sc * > h_y;                 //<! raw data on host
  std::vector< sc * > d_x, d_y;            //<! raw data on device
  std::vector< size_t > pitch_x, pitch_y;  //<! pitch in bytes
  std::vector< lo > ld_x, ld_y;            //<! leading dimension in elements
};

#endif /* INCLUDE_BESTHEA_GPU_APPLY_VECTORS_DATA_H_ */
