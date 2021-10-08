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

#include "besthea/gpu_apply_load_distribution.h"

#include <cmath>
#include <omp.h>



besthea::bem::onthefly::gpu_apply_load_distribution::
  gpu_apply_load_distribution(int n_gpus_, lo n_elems_, lo gpu_chunk_size_, bool use_cpu_) {

  this->n_gpus = n_gpus_;
  this->n_elems = n_elems_;
  this->gpu_chunk_size = gpu_chunk_size_;
  this->use_cpu = use_cpu_;
  
  this->cpu_n_tst_elems_target = (double)std::min((lo)omp_get_max_threads(), n_elems / 2);
  this->cpu_n_tst_elems = adjust_cpu_count(this->cpu_n_tst_elems_target);

  this->gpu_i_tst_begins.resize(n_gpus+1);
  update_gpu_begins();

}



void besthea::bem::onthefly::gpu_apply_load_distribution::adapt(
  double cpu_time_const, double cpu_time_scaling,
  double gpu_time_const, double gpu_time_scaling, double inertia) {

  if(besthea::settings::output_verbosity.onthefly_loadbalance >= 2) {
    printf("BESTHEA Info: onthefly load balancing: before adapt: %6ld %12.6f\n",
      cpu_n_tst_elems, cpu_n_tst_elems_target);
  }

  if(cpu_n_tst_elems == 0) {
    this->cpu_n_tst_elems_target = 0;
  } else {
    lo gpu_n_tst_elems = n_elems - cpu_n_tst_elems;
    double cpu_n_elems_ideal;

    if(gpu_n_tst_elems > 0) {
      double time_per_elem_gpu = gpu_time_scaling / gpu_n_tst_elems;
      double time_per_elem_cpu = cpu_time_scaling / cpu_n_tst_elems;

      cpu_n_elems_ideal =
          (n_elems * time_per_elem_gpu - cpu_time_const + gpu_time_const)
          /
          (time_per_elem_cpu + time_per_elem_gpu);
    }
    else {
      cpu_n_elems_ideal = (double)n_elems;
    }
      
    this->cpu_n_tst_elems_target =
      cpu_n_tst_elems_target * inertia + cpu_n_elems_ideal * (1 - inertia);
  }

  this->cpu_n_tst_elems = adjust_cpu_count(cpu_n_tst_elems_target);

  this->update_gpu_begins();

  if(besthea::settings::output_verbosity.onthefly_loadbalance >= 2) {
    printf("BESTHEA Info: onthefly load balancing: after adapt:  %6ld %12.6f\n",
      cpu_n_tst_elems, cpu_n_tst_elems_target);
  }

}



void besthea::bem::onthefly::gpu_apply_load_distribution::print() {
  printf("BESTHEA Info: onthefly load balancing: total %ld CPU %ld GPU %ld:",
    n_elems, get_cpu_count(), get_gpu_count_total());
  for(unsigned int i = 0; i < gpu_i_tst_begins.size(); i++)
    printf(" %ld", gpu_i_tst_begins[i]);
  printf("\n");
}



lo besthea::bem::onthefly::gpu_apply_load_distribution::
  adjust_cpu_count(double suggested) const {
  
  if(!use_cpu)
    return 0;

  suggested = std::clamp(suggested, 0.0, (double)n_elems);

  lo suggested_gpu_elems = (lo)std::ceil(n_elems - suggested);

  lo chunked_gpu_elems = 
    ((suggested_gpu_elems - 1) / gpu_chunk_size + 1) * gpu_chunk_size;
  
  if( chunked_gpu_elems >= n_elems)
    chunked_gpu_elems = (n_elems / gpu_chunk_size) * gpu_chunk_size;
  
  return n_elems - chunked_gpu_elems;
}



void besthea::bem::onthefly::gpu_apply_load_distribution::
  update_gpu_begins() {

  lo gpus_n_tst_elems = n_elems - cpu_n_tst_elems;

  for(int gpu_idx = 0; gpu_idx <= n_gpus; gpu_idx++) {
    gpu_i_tst_begins[gpu_idx] =
      (((gpus_n_tst_elems * gpu_idx) / n_gpus) / gpu_chunk_size) * gpu_chunk_size;
  }

}
