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

#include "besthea/gpu_onthefly_helpers.h"

#include <cuda_runtime.h>





bool besthea::bem::onthefly::helpers::is_gpu_quadr_order5_initialized = false;
bool besthea::bem::onthefly::helpers::is_gpu_quadr_order4_initialized = false;
bool besthea::bem::onthefly::helpers::is_gpu_quadr_order2_initialized = false;
bool besthea::bem::onthefly::helpers::is_gpu_quadr_order1_initialized = false;














besthea::bem::onthefly::helpers::gpu_apply_vectors_data::
  gpu_apply_vectors_data()
  : h_x(nullptr) {
}



besthea::bem::onthefly::helpers::gpu_apply_vectors_data::
  ~gpu_apply_vectors_data() {

  free();
}



void besthea::bem::onthefly::helpers::gpu_apply_vectors_data::
  allocate(int n_gpus,
  lo x_block_count, lo x_size_of_block, lo y_block_count, lo y_size_of_block) {

  h_y.resize(n_gpus);
  d_x.resize(n_gpus);
  d_y.resize(n_gpus);
  pitch_x.resize(n_gpus);
  pitch_y.resize(n_gpus);
  ld_x.resize(n_gpus);
  ld_y.resize(n_gpus);

  cudaMallocHost(&h_x, x_block_count * x_size_of_block * sizeof(sc));

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    cudaSetDevice(gpu_idx);

    cudaMallocHost(&h_y[gpu_idx], y_block_count * y_size_of_block * sizeof(sc));

    cudaMallocPitch(&d_x[gpu_idx], &pitch_x[gpu_idx],
      x_size_of_block * sizeof(sc), x_block_count);
    cudaMallocPitch(&d_y[gpu_idx], &pitch_y[gpu_idx],
      y_size_of_block * sizeof(sc), y_block_count);
    
    ld_x[gpu_idx] = pitch_x[gpu_idx] / sizeof(sc);
    ld_y[gpu_idx] = pitch_y[gpu_idx] / sizeof(sc);
  }

}


void besthea::bem::onthefly::helpers::gpu_apply_vectors_data::free() {

  if(h_x != nullptr) {
    cudaFreeHost(h_x);
  }
  for(unsigned int i = 0; i < h_y.size(); i++) {
    cudaSetDevice(i);

    cudaFreeHost(h_y[i]);
    cudaFree(d_x[i]);
    cudaFree(d_y[i]);
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















besthea::bem::onthefly::helpers::apply_load_distribution::
  apply_load_distribution(int n_gpus_, lo n_elems_, lo gpu_chunk_size_, bool use_cpu_) {

  this->n_gpus = n_gpus_;
  this->n_elems = n_elems_;
  this->gpu_chunk_size = gpu_chunk_size_;
  this->use_cpu = use_cpu_;
  
  this->cpu_n_tst_elems_target = (double)std::min((lo)omp_get_max_threads(), n_elems / 2);
  this->cpu_n_tst_elems = adjust_cpu_count(this->cpu_n_tst_elems_target);

  this->gpu_i_tst_begins.resize(n_gpus+1);
  update_gpu_begins();

}



void besthea::bem::onthefly::helpers::apply_load_distribution::
  update_gpu_begins() {

  lo gpus_n_tst_elems = n_elems - cpu_n_tst_elems;

  for(int gpu_idx = 0; gpu_idx <= n_gpus; gpu_idx++) {
    gpu_i_tst_begins[gpu_idx] =
      (((gpus_n_tst_elems * gpu_idx) / n_gpus) / gpu_chunk_size) * gpu_chunk_size;
  }

}



void besthea::bem::onthefly::helpers::apply_load_distribution::adapt(
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



lo besthea::bem::onthefly::helpers::apply_load_distribution::
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



void besthea::bem::onthefly::helpers::apply_load_distribution::print() {
  printf("BESTHEA Info: onthefly load balancing: total %ld CPU %ld GPU %ld:",
    n_elems, get_cpu_count(), get_gpu_count_total());
  for(unsigned int i = 0; i < gpu_i_tst_begins.size(); i++)
    printf(" %ld", gpu_i_tst_begins[i]);
  printf("\n");
}
















besthea::bem::onthefly::helpers::timer_collection::
  timer_collection(int n_gpus) {

  gpu_all.resize(n_gpus);
  gpu_copyin.resize(n_gpus);
  gpu_compute.resize(n_gpus);
  gpu_copyout.resize(n_gpus);

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    cudaSetDevice(gpu_idx);
    gpu_all[gpu_idx].init(gpu_idx, 0);
    gpu_copyin[gpu_idx].init(gpu_idx, 0);
    gpu_compute[gpu_idx].init(gpu_idx, 0);
    gpu_copyout[gpu_idx].init(gpu_idx, 0);
  }
}



void besthea::bem::onthefly::helpers::timer_collection::print_all() {
  printf("BESTHEA Info: time gpu_copyin:   ");
  print_timers(gpu_copyin);
  printf("BESTHEA Info: time gpu_compute:  ");
  print_timers(gpu_compute);
  printf("BESTHEA Info: time gpu_copyout:  ");
  print_timers(gpu_copyout);
  printf("BESTHEA Info: time gpu_all:      ");
  print_timers(gpu_all);
  printf("BESTHEA Info: time cpu_scalein:  %10.6f\n", cpu_scalein.get_time());
  printf("BESTHEA Info: time cpu_regular:  %10.6f\n", cpu_treg_sreg.get_time());
  printf("BESTHEA Info: time cpu_singular: %10.6f\n", cpu_treg_ssng.get_time());
  printf("BESTHEA Info: time cpu_delta0:   %10.6f\n", cpu_tsng.get_time());
  printf("BESTHEA Info: time cpu_all:      %10.6f\n", cpu_all.get_time());
  printf("BESTHEA Info: time gpu_max:      %10.6f\n", get_gpu_time_all());
  printf("BESTHEA Info: time combined:     %10.6f\n", combined.get_time());
}



void besthea::bem::onthefly::helpers::timer_collection::
  print_timers(std::vector<besthea::tools::time_measurer_cuda> & timers) {
  for(unsigned int i = 0; i < timers.size(); i++) {
    printf("%10.6f  ", timers[i].get_time());
  }
  printf("\n");
}



double besthea::bem::onthefly::helpers::timer_collection::
  get_cpu_time_const() {
  return cpu_treg_ssng.get_time() + cpu_tsng.get_time();
}

double besthea::bem::onthefly::helpers::timer_collection::
  get_cpu_time_scaling() {
  return cpu_treg_sreg.get_time();
}

double besthea::bem::onthefly::helpers::timer_collection::
  get_gpu_time_const() {

  double max_time_copyin = -1;
  for(unsigned int i = 0; i < gpu_copyin.size(); i++) {
    max_time_copyin = std::max(max_time_copyin, gpu_copyin[i].get_time());
  }
  double max_time_copyout = -1;
  for(unsigned int i = 0; i < gpu_copyout.size(); i++) {
    max_time_copyout = std::max(max_time_copyout, gpu_copyout[i].get_time());
  }
  return max_time_copyin + max_time_copyout;
}

double besthea::bem::onthefly::helpers::timer_collection::
  get_gpu_time_scaling() {

  double max_time_compute = -1;
  for(unsigned int i = 0; i < gpu_compute.size(); i++) {
    max_time_compute = std::max(max_time_compute, gpu_compute[i].get_time());
  }
  return max_time_compute;
}

double besthea::bem::onthefly::helpers::timer_collection::
  get_gpu_time_all() {
    
  double max_time_gpu_all = -1;
  for(unsigned int i = 0; i < gpu_all.size(); i++) {
    max_time_gpu_all = std::max(max_time_gpu_all, gpu_all[i].get_time());
  }
  return max_time_gpu_all;
}






