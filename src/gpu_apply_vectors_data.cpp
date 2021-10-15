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

#include "besthea/gpu_apply_vectors_data.h"

#include <cuda_runtime.h>
#include "besthea/gpu_onthefly_helpers.h"



besthea::bem::onthefly::gpu_apply_vectors_data::
  gpu_apply_vectors_data()
  : h_x(nullptr) {
}



besthea::bem::onthefly::gpu_apply_vectors_data::
  ~gpu_apply_vectors_data() {

  free();
}



void besthea::bem::onthefly::gpu_apply_vectors_data::
  allocate(int n_gpus,
  lo x_block_count, lo x_size_of_block, lo y_block_count, lo y_size_of_block) {

  h_y.resize(n_gpus);
  d_x.resize(n_gpus);
  d_y.resize(n_gpus);
  pitch_x.resize(n_gpus);
  pitch_y.resize(n_gpus);
  ld_x.resize(n_gpus);
  ld_y.resize(n_gpus);

  CUDA_CHECK(cudaMallocHost(&h_x, x_block_count * x_size_of_block * sizeof(sc)));

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    CUDA_CHECK(cudaSetDevice(gpu_idx));

    CUDA_CHECK(cudaMallocHost(&h_y[gpu_idx], y_block_count * y_size_of_block * sizeof(sc)));

    CUDA_CHECK(cudaMallocPitch(&d_x[gpu_idx], &pitch_x[gpu_idx],
      x_size_of_block * sizeof(sc), x_block_count));
    CUDA_CHECK(cudaMallocPitch(&d_y[gpu_idx], &pitch_y[gpu_idx],
      y_size_of_block * sizeof(sc), y_block_count));
    
    ld_x[gpu_idx] = pitch_x[gpu_idx] / sizeof(sc);
    ld_y[gpu_idx] = pitch_y[gpu_idx] / sizeof(sc);
  }

}



void besthea::bem::onthefly::gpu_apply_vectors_data::free() {

  if(h_x != nullptr) {
    CUDA_CHECK(cudaFreeHost(h_x));
  }
  for(unsigned int i = 0; i < h_y.size(); i++) {
    CUDA_CHECK(cudaSetDevice(i));

    CUDA_CHECK(cudaFreeHost(h_y[i]));
    CUDA_CHECK(cudaFree(d_x[i]));
    CUDA_CHECK(cudaFree(d_y[i]));
  }

  h_x = nullptr;
  h_y.clear();
  d_x.clear();
  d_y.clear();
  pitch_x.clear();
  pitch_y.clear();
  ld_x.clear();
  ld_y.clear();

}
