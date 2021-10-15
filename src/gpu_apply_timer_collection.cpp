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

#include "besthea/gpu_apply_timer_collection.h"



besthea::bem::onthefly::gpu_apply_timer_collection::
  gpu_apply_timer_collection(int n_gpus) {

  gpu_all.resize(n_gpus);
  gpu_copyin.resize(n_gpus);
  gpu_compute.resize(n_gpus);
  gpu_copyout.resize(n_gpus);

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    CUDA_CHECK(cudaSetDevice(gpu_idx));
    gpu_all[gpu_idx].init(0);
    gpu_copyin[gpu_idx].init(0);
    gpu_compute[gpu_idx].init(0);
    gpu_copyout[gpu_idx].init(0);
  }
}



void besthea::bem::onthefly::gpu_apply_timer_collection::print_all() {
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



void besthea::bem::onthefly::gpu_apply_timer_collection::
  print_timers(std::vector<besthea::tools::time_measurer_cuda> & timers) {
  for(unsigned int i = 0; i < timers.size(); i++) {
    printf("%10.6f  ", timers[i].get_time());
  }
  printf("\n");
}



double besthea::bem::onthefly::gpu_apply_timer_collection::
  get_cpu_time_const() {
  return cpu_treg_ssng.get_time() + cpu_tsng.get_time();
}



double besthea::bem::onthefly::gpu_apply_timer_collection::
  get_cpu_time_scaling() {
  return cpu_treg_sreg.get_time();
}



double besthea::bem::onthefly::gpu_apply_timer_collection::
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



double besthea::bem::onthefly::gpu_apply_timer_collection::
  get_gpu_time_scaling() {

  double max_time_compute = -1;
  for(unsigned int i = 0; i < gpu_compute.size(); i++) {
    max_time_compute = std::max(max_time_compute, gpu_compute[i].get_time());
  }
  return max_time_compute;
}



double besthea::bem::onthefly::gpu_apply_timer_collection::
  get_gpu_time_all() {
    
  double max_time_gpu_all = -1;
  for(unsigned int i = 0; i < gpu_all.size(); i++) {
    max_time_gpu_all = std::max(max_time_gpu_all, gpu_all[i].get_time());
  }
  return max_time_gpu_all;
}
