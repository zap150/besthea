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

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/gpu_onthefly_helpers.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_be_matrix_onthefly_gpu.h"
#include "besthea/uniform_spacetime_be_space.h"

#include <array>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace besthea::mesh;
using namespace besthea::bem;
using namespace besthea::bem::onthefly;
using namespace besthea::bem::onthefly::helpers;

// clang-format off
// I tried to keep the max-80-column formatting as I could, but sometimes it is
//   just much better to violate this rule. Also the formatter collapses
//   multiple empty lines into one, which reduces clarity and readibility
//   of the code. Therefore I decided to turn off formatting for this file.



__constant__ __device__ quadrature_reference_raw<5> c_quadr_reference_order5;
__constant__ __device__ quadrature_reference_raw<4> c_quadr_reference_order4;
__constant__ __device__ quadrature_reference_raw<2> c_quadr_reference_order2;
__constant__ __device__ quadrature_reference_raw<1> c_quadr_reference_order1;

template<int quadr_order> __host__ __device__
const quadrature_reference_raw<quadr_order> & c_get_quadr_reference();

template<> const quadrature_reference_raw<5> & c_get_quadr_reference<5>() {
  return c_quadr_reference_order5;
}
template<> const quadrature_reference_raw<4> & c_get_quadr_reference<4>() {
  return c_quadr_reference_order4;
}
template<> const quadrature_reference_raw<2> & c_get_quadr_reference<2>() {
  return c_quadr_reference_order2;
}
template<> const quadrature_reference_raw<1> & c_get_quadr_reference<1>() {
  return c_quadr_reference_order1;
}

constexpr dim3 tpb_V[5][6] = { // tpb_V[gpu_kernel_version][quadrature_order]
  {dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0)},
  {dim3(  0,  0),     dim3( 64,  1),     dim3( 64,  1),     dim3(  0,  0),     dim3(256,  1),     dim3(256,  1)},
  {dim3(  0,  0),     dim3( 64,  1),     dim3(128,  1),     dim3(  0,  0),     dim3(128,  1),     dim3(128,  1)},
  {dim3(  0,  0),     dim3(  8, 16),     dim3(  2, 32),     dim3(  0,  0),     dim3(  8, 16),     dim3(  8, 16)},
  {dim3(  0,  0),     dim3(  2, 32),     dim3(  8, 16),     dim3(  0,  0),     dim3(  8, 16),     dim3(  8, 16)}
};
constexpr dim3 tpb_K[5][6] = {
  {dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0)},
  {dim3(  0,  0),     dim3(128,  1),     dim3(128,  1),     dim3(  0,  0),     dim3(128,  1),     dim3(128,  1)},
  {dim3(  0,  0),     dim3(128,  1),     dim3(128,  1),     dim3(  0,  0),     dim3(128,  1),     dim3(128,  1)},
  {dim3(  0,  0),     dim3(  8, 16),     dim3(  8, 16),     dim3(  0,  0),     dim3(  8, 16),     dim3(  8, 16)},
  {dim3(  0,  0),     dim3(  1, 32),     dim3(  4, 16),     dim3(  0,  0),     dim3(  4, 16),     dim3(  8, 16)}
};
constexpr dim3 tpb_Kt[5][6] = {
  {dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0)},
  {dim3(  0,  0),     dim3(128,  1),     dim3( 32,  1),     dim3(  0,  0),     dim3(128,  1),     dim3(128,  1)},
  {dim3(  0,  0),     dim3( 64,  1),     dim3(128,  1),     dim3(  0,  0),     dim3(128,  1),     dim3(128,  1)},
  {dim3(  0,  0),     dim3(  8, 16),     dim3(  8, 16),     dim3(  0,  0),     dim3(  8, 16),     dim3(  8, 16)},
  {dim3(  0,  0),     dim3(  1, 32),     dim3(  1, 32),     dim3(  0,  0),     dim3(  4, 16),     dim3(  8, 16)}
};
constexpr dim3 tpb_D[5][6] = {
  {dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0),     dim3(  0,  0)},
  {dim3(  0,  0),     dim3(128,  1),     dim3( 64,  1),     dim3(  0,  0),     dim3( 64,  1),     dim3( 64,  1)},
  {dim3(  0,  0),     dim3( 64,  1),     dim3( 64,  1),     dim3(  0,  0),     dim3( 64,  1),     dim3( 64,  1)},
  {dim3(  0,  0),     dim3(  8, 16),     dim3(  8, 16),     dim3(  0,  0),     dim3(  8,  8),     dim3(  4,  8)},
  {dim3(  0,  0),     dim3(  2, 32),     dim3(  2, 32),     dim3(  0,  0),     dim3(  2, 16),     dim3(  4, 16)}
};

constexpr dim3 get_tpb(
  [[maybe_unused]] const spacetime_heat_sl_kernel_antiderivative * _hka,
  [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
  [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
  int version, int quadrature_order) {
    return tpb_V[version][quadrature_order];
}
constexpr dim3 get_tpb(
  [[maybe_unused]] const spacetime_heat_dl_kernel_antiderivative * _hka,
  [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
  [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
  int version, int quadrature_order) {
    return tpb_K[version][quadrature_order];
}
constexpr dim3 get_tpb(
  [[maybe_unused]] const spacetime_heat_adl_kernel_antiderivative * _hka,
  [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
  [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
  int version, int quadrature_order) {
    return tpb_Kt[version][quadrature_order];
}
constexpr dim3 get_tpb(
  [[maybe_unused]] const spacetime_heat_hs_kernel_antiderivative * _hka,
  [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
  [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
  int version, int quadrature_order) {
    return tpb_D[version][quadrature_order];
}















template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu
  <kernel_type, test_space_type, trial_space_type>::
  uniform_spacetime_be_matrix_onthefly_gpu( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  const uniform_spacetime_tensor_mesh_gpu & gpu_mesh_,
  int order_singular, int order_regular, int gpu_kernel_version_,
  bool loadbalancing_use_cpu_ )
  : uniform_spacetime_be_matrix_onthefly_cpu
      <kernel_type, test_space_type, trial_space_type>(
        kernel, test_space, trial_space, order_singular, order_regular ),
    gpu_mesh(&gpu_mesh_),
    n_gpus(gpu_mesh_.get_n_gpus()),
    gpu_kernel_version(gpu_kernel_version_),
    load_distr(nullptr),
    loadbalancing_use_cpu(loadbalancing_use_cpu_) {

  // quadrature inited in base class constructor

  if(n_gpus == 0) {
    if(besthea::settings::output_verbosity.warnings >= 1) {
      std::cerr << "BESTHEA Warning: Trying to use gpu version of onthefly "
        "matrix class, but no cuda-capable devices were found. "
        "Using cpu-only version.\n";
    }
  }

  bool isCcOk = true;
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    int ccVerMajor, ccVerMinor;
    CUDA_CHECK(cudaDeviceGetAttribute(&ccVerMajor,
      cudaDevAttrComputeCapabilityMajor, gpu_idx));
    if(ccVerMajor < 6) {
      CUDA_CHECK(cudaDeviceGetAttribute(&ccVerMinor,
        cudaDevAttrComputeCapabilityMinor, gpu_idx));
      std::cerr << "BESTHEA Error: gpu with index " << gpu_idx
        << " has insufficient compute capability ("
        << ccVerMajor << "." << ccVerMinor
        << "). Compute capability >= 6.0 is required.\n";
      isCcOk = false;
    }
  }
  if(!isCcOk) {
    throw std::runtime_error("BESTHEA Exception: "
      "insufficient gpu compute capability");
  }

  if(gpu_kernel_version < 1 || gpu_kernel_version > 4) {
    if(besthea::settings::output_verbosity.warnings >= 1) {
      std::cerr << "BESTHEA Warning: invalid value of gpu_kernel_version="
        << gpu_kernel_version << ", using default gpu_kernel_version=2.\n";
    }
    this->gpu_kernel_version = 2;
  }

  if(n_gpus > 0) {
    init_gpu_data();
  }

}



template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu
  <kernel_type, test_space_type, trial_space_type>::
  ~uniform_spacetime_be_matrix_onthefly_gpu( ) {

  delete this->load_distr;

}






template<class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu
  <kernel_type, test_space_type, trial_space_type>::
  init_gpu_data() {

  switch(gpu_kernel_version) {
    case 4:
      // x_perm, y
      vectors_data.allocate(n_gpus,
        this->get_dim_domain(), this->get_block_dim(),
        this->get_block_dim(), this->get_dim_range()
      );
      break;
    case 3:
      // x_perm, y
      vectors_data.allocate(n_gpus,
        this->get_dim_domain(), this->get_block_dim(),
        this->get_block_dim(), this->get_dim_range()
      );
      break;
    case 2:
    default:
      // x_perm, y
      vectors_data.allocate(n_gpus,
        this->get_dim_domain(), this->get_block_dim(),
        this->get_block_dim(), this->get_dim_range()
      );
      break;
    case 1:
      // x, y_perm
      vectors_data.allocate(n_gpus,
        this->get_block_dim(), this->get_dim_domain(),
        this->get_dim_range(), this->get_block_dim()
      );
      break;
  }


  lo gpu_chunk_size;
  switch(gpu_kernel_version) {
    case 4:
      gpu_chunk_size = get_tpb(this->_kernel, this->_test_space, this->_trial_space, 4, this->_order_regular).x;
      break;
    case 3:
      gpu_chunk_size = get_tpb(this->_kernel, this->_test_space, this->_trial_space, 3, this->_order_regular).x;
      break;
    case 2:
    default:
      gpu_chunk_size = get_tpb(this->_kernel, this->_test_space, this->_trial_space, 2, this->_order_regular).y;
      break;
    case 1:
      gpu_chunk_size = get_tpb(this->_kernel, this->_test_space, this->_trial_space, 1, this->_order_regular).y;
      break;
  }
  this->load_distr = new gpu_apply_load_distribution(
    n_gpus, gpu_mesh->get_metadata().n_elems,
    gpu_chunk_size, loadbalancing_use_cpu
  );



  switch(this->_order_regular) {
    case 5:
      if(!is_gpu_quadr_order5_initialized) {
        init_gpu_quadrature_memory<5>();
        is_gpu_quadr_order5_initialized = true;
      }
      break;
    case 4:
      if(!is_gpu_quadr_order4_initialized) {
        init_gpu_quadrature_memory<4>();
        is_gpu_quadr_order4_initialized = true;
      }
      break;
    case 2:
      if(!is_gpu_quadr_order2_initialized) {
        init_gpu_quadrature_memory<2>();
        is_gpu_quadr_order2_initialized = true;
      }
      break;
    case 1:
    default:
      if(!is_gpu_quadr_order1_initialized) {
        init_gpu_quadrature_memory<1>();
        is_gpu_quadr_order1_initialized = true;
      }
      break;
  }
}







template<class kernel_type, class test_space_type, class trial_space_type >
template<int quadr_order>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu
  <kernel_type, test_space_type, trial_space_type>::
  init_gpu_quadrature_memory() const {

  quadrature_reference_raw<quadr_order> quadr_ref_tmp;
  std::copy(this->quadr_reference._x1_ref[0].begin(), this->quadr_reference._x1_ref[0].end(), quadr_ref_tmp._x1_ref);
  std::copy(this->quadr_reference._x2_ref[0].begin(), this->quadr_reference._x2_ref[0].end(), quadr_ref_tmp._x2_ref);
  std::copy(this->quadr_reference._y1_ref[0].begin(), this->quadr_reference._y1_ref[0].end(), quadr_ref_tmp._y1_ref);
  std::copy(this->quadr_reference._y2_ref[0].begin(), this->quadr_reference._y2_ref[0].end(), quadr_ref_tmp._y2_ref);
  std::copy(this->quadr_reference._w[0].begin(), this->quadr_reference._w[0].end(), quadr_ref_tmp._w);


  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    CUDA_CHECK(cudaSetDevice(gpu_idx));

    switch(quadr_order) {
      case 5:
        CUDA_CHECK(cudaMemcpyToSymbol(c_quadr_reference_order5, &quadr_ref_tmp, sizeof(quadr_ref_tmp)));
        break;
      case 4:
        CUDA_CHECK(cudaMemcpyToSymbol(c_quadr_reference_order4, &quadr_ref_tmp, sizeof(quadr_ref_tmp)));
        break;
      case 2:
        CUDA_CHECK(cudaMemcpyToSymbol(c_quadr_reference_order2, &quadr_ref_tmp, sizeof(quadr_ref_tmp)));
        break;
      case 1:
      default:
        CUDA_CHECK(cudaMemcpyToSymbol(c_quadr_reference_order1, &quadr_ref_tmp, sizeof(quadr_ref_tmp)));
        break;
    }

  }

}













template<int tpbx>
__device__ void d_reduce_sum(volatile sc * shmem_vals) {

  // assuming tpbx is power of 2
  // assuming 1D block
  // assuming number of values is equal to tpbx
  // assuming tpbx is equal to blockDim.x

  int curr_thread_count = tpbx / 2;
  int tid = threadIdx.x;
  while(curr_thread_count > 32) {
    if(tid < curr_thread_count) {
      shmem_vals[tid] += shmem_vals[tid ^ curr_thread_count];
    }
    __syncthreads();
    curr_thread_count /= 2;
  }

  if(tid < 32) {
    if(tpbx >= 64) shmem_vals[tid] += shmem_vals[tid ^ 32];
    if(tpbx >= 32) shmem_vals[tid] += shmem_vals[tid ^ 16];
    if(tpbx >= 16) shmem_vals[tid] += shmem_vals[tid ^  8];
    if(tpbx >=  8) shmem_vals[tid] += shmem_vals[tid ^  4];
    if(tpbx >=  4) shmem_vals[tid] += shmem_vals[tid ^  2];
    if(tpbx >=  2) shmem_vals[tid] += shmem_vals[tid ^  1];
  }

  __syncthreads();

}



template<int tpbx>
__device__ void d_reduce_sum_multiple(volatile sc * shmem_vals1,
  volatile sc * shmem_vals2, volatile sc * shmem_vals3) {

  // assuming tpbx is power of 2
  // assuming 1D block
  // assuming number of values is equal to tpbx
  // assuming tpbx is equal to blockDim.x

  int curr_thread_count = tpbx / 2;
  int tid = threadIdx.x;
  while(curr_thread_count > 32) {
    if(tid < curr_thread_count) {
      shmem_vals1[tid] += shmem_vals1[tid ^ curr_thread_count];
      shmem_vals2[tid] += shmem_vals2[tid ^ curr_thread_count];
      shmem_vals3[tid] += shmem_vals3[tid ^ curr_thread_count];
    }
    __syncthreads();
    curr_thread_count /= 2;
  }

  if(tid < 32) {
    if(tpbx >= 64) shmem_vals1[tid] += shmem_vals1[tid ^ 32];
    if(tpbx >= 32) shmem_vals1[tid] += shmem_vals1[tid ^ 16];
    if(tpbx >= 16) shmem_vals1[tid] += shmem_vals1[tid ^  8];
    if(tpbx >=  8) shmem_vals1[tid] += shmem_vals1[tid ^  4];
    if(tpbx >=  4) shmem_vals1[tid] += shmem_vals1[tid ^  2];
    if(tpbx >=  2) shmem_vals1[tid] += shmem_vals1[tid ^  1];
    //
    if(tpbx >= 64) shmem_vals2[tid] += shmem_vals2[tid ^ 32];
    if(tpbx >= 32) shmem_vals2[tid] += shmem_vals2[tid ^ 16];
    if(tpbx >= 16) shmem_vals2[tid] += shmem_vals2[tid ^  8];
    if(tpbx >=  8) shmem_vals2[tid] += shmem_vals2[tid ^  4];
    if(tpbx >=  4) shmem_vals2[tid] += shmem_vals2[tid ^  2];
    if(tpbx >=  2) shmem_vals2[tid] += shmem_vals2[tid ^  1];
    //
    if(tpbx >= 64) shmem_vals3[tid] += shmem_vals3[tid ^ 32];
    if(tpbx >= 32) shmem_vals3[tid] += shmem_vals3[tid ^ 16];
    if(tpbx >= 16) shmem_vals3[tid] += shmem_vals3[tid ^  8];
    if(tpbx >=  8) shmem_vals3[tid] += shmem_vals3[tid ^  4];
    if(tpbx >=  4) shmem_vals3[tid] += shmem_vals3[tid ^  2];
    if(tpbx >=  2) shmem_vals3[tid] += shmem_vals3[tid ^  1];
  }

  __syncthreads();

}



template<int tpbx, int tpby>
__device__ void d_reduce_sum_2d_y(volatile sc * shmem_vals) {
  // sums the values along the y dimension
  // assuming tpbx and tpby is power of 2
  // assuming tpbx = blockDim.x and tpby = blockDim.y

  int curr_thread_count = (tpbx * tpby) / 2;
  int tid = threadIdx.y * tpbx + threadIdx.x;

  while(curr_thread_count >= tpbx) {
    if(tid < curr_thread_count) {
      shmem_vals[tid] += shmem_vals[tid ^ curr_thread_count];
    }
    __syncthreads();
    curr_thread_count /= 2;
  }

}



template<int tpbx, int tpby>
__device__ void d_reduce_sum_2d_y_multiple(volatile sc * shmem_vals1,
  volatile sc * shmem_vals2, volatile sc * shmem_vals3) {
    // sums the values along the y dimension
    // assuming tpbx and tpby is power of 2
    // assuming tpbx = blockDim.x and tpby = blockDim.y

  int curr_thread_count = (tpbx * tpby) / 2;
  int tid = threadIdx.y * tpbx + threadIdx.x;

  while(curr_thread_count >= tpbx) {
    if(tid < curr_thread_count) {
      shmem_vals1[tid] += shmem_vals1[tid ^ curr_thread_count];
      shmem_vals2[tid] += shmem_vals2[tid ^ curr_thread_count];
      shmem_vals3[tid] += shmem_vals3[tid ^ curr_thread_count];
    }
    __syncthreads();
    curr_thread_count /= 2;
  }

}

















// 0 shared vertices, 0 rot_test, 0 rot_trial
template<int quadr_order>
__device__ void d_triangles_to_geometry_000_tst_shmem( lo i_tst,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data & mesh_data,
  quadrature_nodes_raw<quadr_order> & shmem_quadr_nodes_tst ) {
  const lo * tst_elem_nodes = mesh_data.d_element_nodes + 3 * i_tst;

  const sc * x1 = mesh_data.d_node_coords + 3 * tst_elem_nodes[0];
  const sc * x2 = mesh_data.d_node_coords + 3 * tst_elem_nodes[1];
  const sc * x3 = mesh_data.d_node_coords + 3 * tst_elem_nodes[2];

  const sc * x1_ref = c_get_quadr_reference<quadr_order>()._x1_ref;
  const sc * x2_ref = c_get_quadr_reference<quadr_order>()._x2_ref;

  constexpr int quadr_size = qo2qs(quadr_order);
  for ( lo i = threadIdx.x; i < quadr_size; i += blockDim.x ) {
    shmem_quadr_nodes_tst.xs[ i ] = x1[ 0 ]
      + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ];
    shmem_quadr_nodes_tst.ys[ i ] = x1[ 1 ]
      + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ];
    shmem_quadr_nodes_tst.zs[ i ] = x1[ 2 ]
      + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ];
  }
}



// 0 shared vertices, 0 rot_test, 0 rot_trial
template<int quadr_order>
__device__ void d_triangles_to_geometry_000_tst( lo i_tst,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data & mesh_data,
  quadrature_nodes_raw<quadr_order> & shmem_quadr_nodes_tst ) {
  const lo * tst_elem_nodes = mesh_data.d_element_nodes + 3 * i_tst;

  const sc * x1 = mesh_data.d_node_coords + 3 * tst_elem_nodes[0];
  const sc * x2 = mesh_data.d_node_coords + 3 * tst_elem_nodes[1];
  const sc * x3 = mesh_data.d_node_coords + 3 * tst_elem_nodes[2];

  const sc * x1_ref = c_get_quadr_reference<quadr_order>()._x1_ref;
  const sc * x2_ref = c_get_quadr_reference<quadr_order>()._x2_ref;

  constexpr int quadr_size = qo2qs(quadr_order);
  for ( lo i = 0; i < quadr_size; ++i ) {
    shmem_quadr_nodes_tst.xs[ i ] = x1[ 0 ]
      + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ];
    shmem_quadr_nodes_tst.ys[ i ] = x1[ 1 ]
      + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ];
    shmem_quadr_nodes_tst.zs[ i ] = x1[ 2 ]
      + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ];
  }
}



// 0 shared vertices, 0 rot_test, 0 rot_trial
template<int quadr_order>
__device__ void d_triangles_to_geometry_000_trl( lo i_trl,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data & mesh_data,
  quadrature_nodes_raw<quadr_order> & quadr_nodes_trl ) {

  const lo * trl_elem_nodes = mesh_data.d_element_nodes + 3 * i_trl;

  const sc * y1 = mesh_data.d_node_coords + 3 * trl_elem_nodes[0];
  const sc * y2 = mesh_data.d_node_coords + 3 * trl_elem_nodes[1];
  const sc * y3 = mesh_data.d_node_coords + 3 * trl_elem_nodes[2];

  const sc * y1_ref = c_get_quadr_reference<quadr_order>()._y1_ref;
  const sc * y2_ref = c_get_quadr_reference<quadr_order>()._y2_ref;
  constexpr int quadr_size = qo2qs(quadr_order);
  for ( lo i = 0; i < quadr_size; ++i ) {
    quadr_nodes_trl.xs[ i ] = y1[ 0 ]
      + ( y2[ 0 ] - y1[ 0 ] ) * y1_ref[ i ]
      + ( y3[ 0 ] - y1[ 0 ] ) * y2_ref[ i ];
    quadr_nodes_trl.ys[ i ] = y1[ 1 ]
      + ( y2[ 1 ] - y1[ 1 ] ) * y1_ref[ i ]
      + ( y3[ 1 ] - y1[ 1 ] ) * y2_ref[ i ];
    quadr_nodes_trl.zs[ i ] = y1[ 2 ]
      + ( y2[ 2 ] - y1[ 2 ] ) * y1_ref[ i ]
      + ( y3[ 2 ] - y1[ 2 ] ) * y2_ref[ i ];
  }
}















__device__ sc d_sl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
  sc xy1, sc xy2, sc xy3, sc ttau, sc ttau_sqrt,
  const heat_kernel_parameters & kp ) {

  constexpr sc _two = 2.0;
  constexpr sc _four = 4.0;
  constexpr sc _eight = 8.0;

  sc norm = sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
  sc &sqrt_d = ttau_sqrt;

  //  ttau > 0, norm > 0
  sc value = ( ttau / ( _four * kp.pi * kp.alpha * norm )
               + norm / ( _eight * kp.pi * kp.alpha_2 ) )
      * erf( norm / ( _two * sqrt_d * kp.sqrt_alpha ) )
    + sqrt_d / ( _four * kp.pi * kp.alpha * kp.sqrt_pi * kp.sqrt_alpha )
      * exp( -( norm * norm ) / ( _four * ttau * kp.alpha ) );

  return value;
}



__device__ sc d_dl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
  sc xy1, sc xy2, sc xy3, const sc * ny, sc ttau, sc ttau_sqrt,
  const heat_kernel_parameters & kp ) {

  constexpr sc _one = 1.0;
  constexpr sc _two = 2.0;
  constexpr sc _four = 4.0;

  sc norm2 = xy1 * xy1 + xy2 * xy2 + xy3 * xy3;
  sc norm = sqrt( norm2 );
  sc dot = xy1 * ny[ 0 ] + xy2 * ny[ 1 ] + xy3 * ny[ 2 ];
  sc &sqrt_d = ttau_sqrt;

  //  ttau > 0, norm > 0
  sc value = -dot / ( _four * kp.pi * norm )
             * ( ( _one / ( _two * kp.alpha ) - ttau / norm2 )
             * erf( norm / ( _two * sqrt_d * kp.sqrt_alpha ) )
           + sqrt_d / ( kp.sqrt_pi * kp.sqrt_alpha * norm )
             * exp( -norm2 / ( _four * kp.alpha * ttau ) ) );

  return value;
}



__device__ sc d_adl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
  sc xy1, sc xy2, sc xy3, const sc * nx, sc ttau, sc ttau_sqrt,
  const heat_kernel_parameters & kp ) {

  constexpr sc _one = 1.0;
  constexpr sc _two = 2.0;
  constexpr sc _four = 4.0;

  sc norm2 = xy1 * xy1 + xy2 * xy2 + xy3 * xy3;
  sc norm = sqrt( norm2 );
  sc dot = -xy1 * nx[ 0 ] - xy2 * nx[ 1 ] - xy3 * nx[ 2 ];
  sc &sqrt_d = ttau_sqrt;

  //  ttau > 0, norm > 0
  sc value = -dot / ( _four * kp.pi * norm )
             * ( ( _one / ( _two * kp.alpha ) - ttau / norm2 )
             * erf( norm / ( _two * sqrt_d * kp.sqrt_alpha ) )
           + sqrt_d / ( kp.sqrt_pi * kp.sqrt_alpha * norm )
             * exp( -norm2 / ( _four * kp.alpha * ttau ) ) );

  return value;
}



__device__ void d_hs_kernel_do_anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space(
  sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny,
  sc ttau, sc ttau_sqrt, sc * value1, sc * value2,
  const heat_kernel_parameters & kp) {

  constexpr sc _two = 2.0;
  constexpr sc _four = 4.0;
  constexpr sc _eight = 8.0;

  sc dot = nx[ 0 ] * ny[ 0 ] + nx[ 1 ] * ny[ 1 ] + nx[ 2 ] * ny[ 2 ];
  sc norm = sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
  sc sqrt_d = ttau_sqrt;
  sc erf_value = erf( norm / ( _two * sqrt_d * kp.sqrt_alpha ) );
  sc four_pi_alpha_norm = _four * kp.pi * kp.alpha * norm;

  //  ttau > 0, norm > 0
  *value1 = (ttau / four_pi_alpha_norm + norm / (_eight * kp.pi * kp.alpha_2))
      * erf_value
    + sqrt_d / ( _four * kp.pi * kp.alpha * kp.sqrt_pi * kp.sqrt_alpha )
      * exp( -( norm * norm ) / ( _four * ttau * kp.alpha ) );

  *value2 = erf_value / four_pi_alpha_norm;

  *value1 *= kp.alpha_2;
  *value2 *= dot * kp.alpha;

}







__device__ void d_basis_tri_p1_evaluate_curl_00(
  lo i_elem, const sc * n, bool swap, sc * curls,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data & mesh_data) {

  const lo * elem_nodes = mesh_data.d_element_nodes + 3 * i_elem;
  const sc * x1rot = mesh_data.d_node_coords + 3 * elem_nodes[0];
  const sc * x2rot = mesh_data.d_node_coords + 3 * elem_nodes[1];
  const sc * x3rot = mesh_data.d_node_coords + 3 * elem_nodes[2];

  // first two rows of R^\trans, third is n
  sc a11 = x2rot[ 0 ] - x1rot[ 0 ];
  sc a12 = x2rot[ 1 ] - x1rot[ 1 ];
  sc a13 = x2rot[ 2 ] - x1rot[ 2 ];
  sc a21 = x3rot[ 0 ] - x1rot[ 0 ];
  sc a22 = x3rot[ 1 ] - x1rot[ 1 ];
  sc a23 = x3rot[ 2 ] - x1rot[ 2 ];

  // determinant to invert the matrix
  sc det = n[ 0 ] * ( a12 * a23 - a13 * a22 )
         + n[ 1 ] * ( a13 * a21 - a11 * a23 )
         + n[ 2 ] * ( a11 * a22 - a21 * a12 );

  // gradients in actual triangle
  // R^{-\trans} * [1;0;0]
  sc g21 =  n[ 2 ] * a22 - n[ 1 ] * a23;
  sc g22 = -n[ 2 ] * a21 + n[ 0 ] * a23;
  sc g23 =  n[ 1 ] * a21 - n[ 0 ] * a22;
  // n x gradient
  curls[ 3 ] = ( n[ 1 ] * g23 - n[ 2 ] * g22 ) / det;
  curls[ 4 ] = ( n[ 2 ] * g21 - n[ 0 ] * g23 ) / det;
  curls[ 5 ] = ( n[ 0 ] * g22 - n[ 1 ] * g21 ) / det;

  // R^{-\trans} * [0;1;0]
  sc g31 = -n[ 2 ] * a12 + n[ 1 ] * a13;
  sc g32 = n[ 2 ] * a11 - n[ 0 ] * a13;
  sc g33 = -n[ 1 ] * a11 + n[ 0 ] * a12;
  // n x gradient
  curls[ 6 ] = ( n[ 1 ] * g33 - n[ 2 ] * g32 ) / det;
  curls[ 7 ] = ( n[ 2 ] * g31 - n[ 0 ] * g33 ) / det;
  curls[ 8 ] = ( n[ 0 ] * g32 - n[ 1 ] * g31 ) / det;

  // R^{-\trans} * [-1;-1;0]
  // n x gradient
  curls[ 0 ] = ( -n[ 1 ] * ( g23 + g33 ) + n[ 2 ] * ( g22 + g32 ) ) / det;
  curls[ 1 ] = ( -n[ 2 ] * ( g21 + g31 ) + n[ 0 ] * ( g23 + g33 ) ) / det;
  curls[ 2 ] = ( -n[ 0 ] * ( g22 + g32 ) + n[ 1 ] * ( g21 + g31 ) ) / det;

}














template<int quadr_order>
__device__ void d_get_local_contributions_treg_sreg_sl_p0_p0(sc * values_out,
  lo delta, lo i_test, lo i_trial,
  const quadrature_nodes_raw<quadr_order> & quadr_nodes_tst,
  const quadrature_nodes_raw<quadr_order> & quadr_nodes_trl,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata & mesh_metadata,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data & mesh_data,
  const heat_kernel_parameters & kp) {
  sc timestep = mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const sc test_area = mesh_data.d_element_areas[i_test];
  const sc trial_area = mesh_data.d_element_areas[i_trial];

  const sc * w = c_get_quadr_reference<quadr_order>()._w;


  sc value = 0;
  constexpr int quadr_size = qo2qs(quadr_order);
  for ( lo i_quad = 0; i_quad < quadr_size; ++i_quad ) {
    value += d_sl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
      quadr_nodes_tst.xs[ i_quad ] - quadr_nodes_trl.xs[ i_quad ],
      quadr_nodes_tst.ys[ i_quad ] - quadr_nodes_trl.ys[ i_quad ],
      quadr_nodes_tst.zs[ i_quad ] - quadr_nodes_trl.zs[ i_quad ],
      ttau, sqrt_ttau, kp ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  *values_out = value * multiplier;
  return;

}















template<int quadr_order>
__device__ void d_get_local_contributions_treg_sreg_dl_p0_p1(sc * values_out,
  lo delta, lo i_test, lo i_trial,
  const quadrature_nodes_raw<quadr_order> & quadr_nodes_tst,
  const quadrature_nodes_raw<quadr_order> & quadr_nodes_trl,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata & mesh_metadata,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data & mesh_data,
  const heat_kernel_parameters & kp) {
  sc timestep = mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const sc * ny = mesh_data.d_element_normals + 3 * i_trial;

  const sc test_area = mesh_data.d_element_areas[i_test];
  const sc trial_area = mesh_data.d_element_areas[i_trial];

  const sc * w = c_get_quadr_reference<quadr_order>()._w;
  const sc * y1_ref = c_get_quadr_reference<quadr_order>()._y1_ref;
  const sc * y2_ref = c_get_quadr_reference<quadr_order>()._y2_ref;


  sc kernel;
  sc value1, value2, value3;

  value1 = value2 = value3 = 0.0;

  constexpr int quadr_size = qo2qs(quadr_order);
  for ( lo i_quad = 0; i_quad < quadr_size; ++i_quad ) {
    kernel = d_dl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
      quadr_nodes_tst.xs[ i_quad ] - quadr_nodes_trl.xs[ i_quad ],
      quadr_nodes_tst.ys[ i_quad ] - quadr_nodes_trl.ys[ i_quad ],
      quadr_nodes_tst.zs[ i_quad ] - quadr_nodes_trl.zs[ i_quad ],
      ny, ttau, sqrt_ttau, kp ) * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  values_out[0] = value1 * multiplier;
  values_out[1] = value2 * multiplier;
  values_out[2] = value3 * multiplier;
  return;

}















template<int quadr_order>
__device__ void d_get_local_contributions_treg_sreg_adl_p1_p0(sc * values_out,
  lo delta, lo i_test, lo i_trial,
  const quadrature_nodes_raw<quadr_order> & quadr_nodes_tst,
  const quadrature_nodes_raw<quadr_order> & quadr_nodes_trl,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata & mesh_metadata,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data & mesh_data,
  const heat_kernel_parameters & kp) {
  sc timestep = mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const sc * nx = mesh_data.d_element_normals + 3 * i_test;

  const sc test_area = mesh_data.d_element_areas[i_test];
  const sc trial_area = mesh_data.d_element_areas[i_trial];

  const sc * w = c_get_quadr_reference<quadr_order>()._w;
  const sc * x1_ref = c_get_quadr_reference<quadr_order>()._x1_ref;
  const sc * x2_ref = c_get_quadr_reference<quadr_order>()._x2_ref;


  sc kernel;
  sc value1, value2, value3;

  value1 = value2 = value3 = 0.0;

  constexpr int quadr_size = qo2qs(quadr_order);
  for ( lo i_quad = 0; i_quad < quadr_size; ++i_quad ) {
    kernel = d_adl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
      quadr_nodes_tst.xs[ i_quad ] - quadr_nodes_trl.xs[ i_quad ],
      quadr_nodes_tst.ys[ i_quad ] - quadr_nodes_trl.ys[ i_quad ],
      quadr_nodes_tst.zs[ i_quad ] - quadr_nodes_trl.zs[ i_quad ],
      nx, ttau, sqrt_ttau, kp ) * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ] );
    value2 += kernel * x1_ref[ i_quad ];
    value3 += kernel * x2_ref[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  values_out[0] = value1 * multiplier;
  values_out[1] = value2 * multiplier;
  values_out[2] = value3 * multiplier;
  return;

}















template<int quadr_order>
__device__ void d_get_local_contributions_treg_sreg_hs_p1_p1(sc * values_out,
  lo delta, lo i_test, lo i_trial,
  const quadrature_nodes_raw<quadr_order> & quadr_nodes_tst,
  const quadrature_nodes_raw<quadr_order> & quadr_nodes_trl,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata & mesh_metadata,
  const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data & mesh_data,
  const heat_kernel_parameters & kp) {
  sc timestep = mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);
  const sc * nx = mesh_data.d_element_normals + 3 * i_test;
  const sc * ny = mesh_data.d_element_normals + 3 * i_trial;

  const sc test_area = mesh_data.d_element_areas[i_test];
  const sc trial_area = mesh_data.d_element_areas[i_trial];

  const sc * w = c_get_quadr_reference<quadr_order>()._w;
  const sc * x1_ref = c_get_quadr_reference<quadr_order>()._x1_ref;
  const sc * x2_ref = c_get_quadr_reference<quadr_order>()._x2_ref;
  const sc * y1_ref = c_get_quadr_reference<quadr_order>()._y1_ref;
  const sc * y2_ref = c_get_quadr_reference<quadr_order>()._y2_ref;


  sc test_curls[ 9 ], trial_curls[ 9 ];
  sc phi1x, phi1y;
  sc kernel1, kernel2;
  lo test_curl_offset, trial_curl_offset;
  sc curl_dot[ 9 ];

  d_basis_tri_p1_evaluate_curl_00( i_test, nx, false, test_curls, mesh_data );
  d_basis_tri_p1_evaluate_curl_00( i_trial, ny, true, trial_curls, mesh_data );
  for ( lo i_loc_test = 0; i_loc_test < 3; ++i_loc_test ) {
    test_curl_offset = 3 * i_loc_test;
    for ( lo i_loc_trial = 0; i_loc_trial < 3; ++i_loc_trial ) {
      trial_curl_offset = 3 * i_loc_trial;
      curl_dot[ i_loc_trial * 3 + i_loc_test ]
        = test_curls[ test_curl_offset     ] * trial_curls[ trial_curl_offset     ]
        + test_curls[ test_curl_offset + 1 ] * trial_curls[ trial_curl_offset + 1 ]
        + test_curls[ test_curl_offset + 2 ] * trial_curls[ trial_curl_offset + 2 ];
    }
  }

  for(lo j = 0; j < 9; j++) values_out[j] = 0;
  sc &value11 = values_out[0];   sc &value12 = values_out[1];   sc &value13 = values_out[2];
  sc &value21 = values_out[3];   sc &value22 = values_out[4];   sc &value23 = values_out[5];
  sc &value31 = values_out[6];   sc &value32 = values_out[7];   sc &value33 = values_out[8];

  constexpr int quadr_size = qo2qs(quadr_order);
  for ( lo i_quad = 0; i_quad < quadr_size; ++i_quad ) {
    d_hs_kernel_do_anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space(
      quadr_nodes_tst.xs[ i_quad ] - quadr_nodes_trl.xs[ i_quad ],
      quadr_nodes_tst.ys[ i_quad ] - quadr_nodes_trl.ys[ i_quad ],
      quadr_nodes_tst.zs[ i_quad ] - quadr_nodes_trl.zs[ i_quad ],
      nx, ny, ttau, sqrt_ttau, &kernel1, &kernel2, kp );

    phi1x = (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ];
    phi1y = (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ];

    value11 += ( kernel1 * curl_dot[ 0 ] + kernel2 * phi1x            * phi1y )            * w[ i_quad ];
    value21 += ( kernel1 * curl_dot[ 1 ] + kernel2 * x1_ref[ i_quad ] * phi1y )            * w[ i_quad ];
    value31 += ( kernel1 * curl_dot[ 2 ] + kernel2 * x2_ref[ i_quad ] * phi1y )            * w[ i_quad ];
    value12 += ( kernel1 * curl_dot[ 3 ] + kernel2 * phi1x            * y1_ref[ i_quad ] ) * w[ i_quad ];
    value22 += ( kernel1 * curl_dot[ 4 ] + kernel2 * x1_ref[ i_quad ] * y1_ref[ i_quad ] ) * w[ i_quad ];
    value32 += ( kernel1 * curl_dot[ 5 ] + kernel2 * x2_ref[ i_quad ] * y1_ref[ i_quad ] ) * w[ i_quad ];
    value13 += ( kernel1 * curl_dot[ 6 ] + kernel2 * phi1x            * y2_ref[ i_quad ] ) * w[ i_quad ];
    value23 += ( kernel1 * curl_dot[ 7 ] + kernel2 * x1_ref[ i_quad ] * y2_ref[ i_quad ] ) * w[ i_quad ];
    value33 += ( kernel1 * curl_dot[ 8 ] + kernel2 * x2_ref[ i_quad ] * y2_ref[ i_quad ] ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  for(lo j = 0; j < 9; j++) values_out[j] *= multiplier;
  return;

}















template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver1
  ( [[maybe_unused]] const spacetime_heat_sl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
    const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, and loops through all the blocks

  constexpr int tpbx = tpb_V[1][quadr_order].x;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ volatile sc shmem_y_vals[tpbx];

  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const unsigned int &tid = threadIdx.x;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x;

  shmem_y_vals[tid] = 0;
  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  sc matrix_val;
  sc val_prev;
  sc val_curr;
  sc val_next;

  quadrature_nodes_raw<quadr_order> quadr_nodes_trl;

  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {
    d_triangles_to_geometry_000_trl(i_trl, mesh_data, quadr_nodes_trl);

    const lo &row = i_tst;
    const lo &col = i_trl;

    val_curr = 0;
    val_next = 0;
    for (lo delta = 0; delta < n_blocks; delta++) {
      val_prev = val_curr;
      val_curr = val_next;
      d_get_local_contributions_treg_sreg_sl_p0_p0(&val_next,
        delta+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
        mesh_metadata, mesh_data, kp);

      matrix_val = ((i_tst == i_trl) ? (0) : (-val_prev + 2*val_curr - val_next));

      lo max_block = n_blocks - delta;
      for (lo block = 0; block < max_block; block++) {
        lo block_row = delta + block;
        lo block_col = block;
        shmem_y_vals[tid] = matrix_val * x[block_col * ld_x + col];
        __syncthreads();
        d_reduce_sum<tpbx>(shmem_y_vals);
        if(tid == 0)
          y_perm[block_row + ld_y_perm * row] += alpha * shmem_y_vals[0];
      }
    }

    shmem_y_vals[tid] = 0;
    __syncthreads();

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver1
  ( [[maybe_unused]] const spacetime_heat_dl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
    const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, and loops through all the blocks

  constexpr int tpbx = tpb_K[1][quadr_order].x;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_y_vals[tpbx];

  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const unsigned int &tid = threadIdx.x;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x;

  shmem_y_vals[tid] = 0;
  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  sc matrix_vals[3] = {0,0,0};
  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {
    d_triangles_to_geometry_000_trl(i_trl, mesh_data, quadr_nodes_trl);

    const lo &row = i_tst;
    const lo * cols = mesh_data.d_element_nodes + 3 * i_trl;

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    for (lo delta = 0; delta < n_blocks; delta++) {
      vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
      vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
      d_get_local_contributions_treg_sreg_dl_p0_p1(vals_next,
        delta+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
        mesh_metadata, mesh_data, kp);

      matrix_vals[0] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0]));
      matrix_vals[1] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
      matrix_vals[2] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));

      lo max_block = n_blocks - delta;
      for (lo block = 0; block < max_block; block++) {
        lo block_row = delta + block;
        lo block_col = block;
        shmem_y_vals[tid]  = matrix_vals[0] * x[block_col * ld_x + cols[0]];
        shmem_y_vals[tid] += matrix_vals[1] * x[block_col * ld_x + cols[1]];
        shmem_y_vals[tid] += matrix_vals[2] * x[block_col * ld_x + cols[2]];
        __syncthreads();
        d_reduce_sum<tpbx>(shmem_y_vals);
        if(tid == 0)
          y_perm[block_row + ld_y_perm * row] += alpha * shmem_y_vals[0];
      }
    }

    shmem_y_vals[tid] = 0;
    __syncthreads();

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver1
  ( [[maybe_unused]] const spacetime_heat_adl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
    const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_Kt[1][quadr_order].x;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_y_vals_data[3 * tpbx];
  sc * shmem_y_vals[3];
  shmem_y_vals[0] = shmem_y_vals_data + 0 * blockDim.x;
  shmem_y_vals[1] = shmem_y_vals_data + 1 * blockDim.x;
  shmem_y_vals[2] = shmem_y_vals_data + 2 * blockDim.x;

  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo i_tst = i_tst_begin + blockIdx.x;

  shmem_y_vals[0][tid] = 0;
  shmem_y_vals[1][tid] = 0;
  shmem_y_vals[2][tid] = 0;
  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  sc matrix_vals[3];
  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {
    d_triangles_to_geometry_000_trl(i_trl, mesh_data, quadr_nodes_trl);

    const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;
    const lo &col = i_trl;

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    for (lo delta = 0; delta < n_blocks; delta++) {
      vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
      vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
      d_get_local_contributions_treg_sreg_adl_p1_p0(vals_next,
        delta+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
        mesh_metadata, mesh_data, kp);

      matrix_vals[0] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0]));
      matrix_vals[1] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
      matrix_vals[2] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));

      lo max_block = n_blocks - delta;
      for (lo block = 0; block < max_block; block++) {
        lo block_row = delta + block;
        lo block_col = block;
        sc x_val = x[block_col * ld_x + col];
        shmem_y_vals[0][tid] = matrix_vals[0] * x_val;
        shmem_y_vals[1][tid] = matrix_vals[1] * x_val;
        shmem_y_vals[2][tid] = matrix_vals[2] * x_val;
        __syncthreads();
        d_reduce_sum_multiple<tpbx>(shmem_y_vals[0], shmem_y_vals[1], shmem_y_vals[2]);
        if(tid < 3)
          atomicAdd(&y_perm[block_row + ld_y_perm * rows[tid]], alpha * shmem_y_vals[tid][0]);
      }
    }

    shmem_y_vals[0][tid] = 0;
    shmem_y_vals[1][tid] = 0;
    shmem_y_vals[2][tid] = 0;
    __syncthreads();

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver1
  ( [[maybe_unused]] const spacetime_heat_hs_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
    const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_D[1][quadr_order].x;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_y_vals_data[3 * tpbx];
  sc * shmem_y_vals[3];
  shmem_y_vals[0] = shmem_y_vals_data + 0 * blockDim.x;
  shmem_y_vals[1] = shmem_y_vals_data + 1 * blockDim.x;
  shmem_y_vals[2] = shmem_y_vals_data + 2 * blockDim.x;

  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo i_tst = i_tst_begin + blockIdx.x;

  shmem_y_vals[0][tid] = 0;
  shmem_y_vals[1][tid] = 0;
  shmem_y_vals[2][tid] = 0;
  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  sc matrix_vals[9];
  sc vals_prev[9];
  sc vals_curr[9];
  sc vals_next[9];
  sc x_vals[3];

  quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {
    d_triangles_to_geometry_000_trl(i_trl, mesh_data, quadr_nodes_trl);

    const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;
    const lo * cols = mesh_data.d_element_nodes + 3 * i_trl;

    for(lo j = 0; j < 9; j++) vals_curr[j] = 0;
    for(lo j = 0; j < 9; j++) vals_next[j] = 0;

    for (lo delta = 0; delta < n_blocks; delta++) {
      for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
      for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
      d_get_local_contributions_treg_sreg_hs_p1_p1(vals_next,
        delta+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
        mesh_metadata, mesh_data, kp);

      for(lo j = 0; j < 9; j++)
        matrix_vals[j] = ((i_tst == i_trl) ? (0) : (-vals_prev[j] + 2*vals_curr[j] - vals_next[j]));

      lo max_block = n_blocks - delta;
      for (lo block = 0; block < max_block; block++) {
        lo block_row = delta + block;
        lo block_col = block;
        x_vals[0] = x[block_col * ld_x + cols[0]];
        x_vals[1] = x[block_col * ld_x + cols[1]];
        x_vals[2] = x[block_col * ld_x + cols[2]];
        for(lo r = 0; r < 3; r++)
          shmem_y_vals[r][tid] = 0;
        for(lo r = 0; r < 3; r++)
          for(lo c = 0; c < 3; c++)
            shmem_y_vals[r][tid] += matrix_vals[3*r+c] * x_vals[c];
        __syncthreads();
        d_reduce_sum_multiple<tpbx>(shmem_y_vals[0], shmem_y_vals[1], shmem_y_vals[2]);
        if(tid < 3)
          atomicAdd(&y_perm[block_row + ld_y_perm * rows[tid]], alpha * shmem_y_vals[tid][0]);
      }
    }

    shmem_y_vals[0][tid] = 0;
    shmem_y_vals[1][tid] = 0;
    shmem_y_vals[2][tid] = 0;
    __syncthreads();

  }

}
















template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver2
  ( [[maybe_unused]] const spacetime_heat_sl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements,
  //   then is assigned to a block and loops through all the trial elements

  constexpr int tpbx = tpb_V[2][quadr_order].x;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_matrix_vals[tpbx];

  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo i_tst = i_tst_begin + blockIdx.x;
  const lo &row = i_tst;

  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  sc val_prev;
  sc val_curr;
  sc val_next;

  for(lo i = threadIdx.x; i < n_elems; i += blockDim.x) {
    d_triangles_to_geometry_000_trl(i, mesh_data, quadr_nodes_trl);

    val_curr = 0;
    val_next = 0;

    lo curr_active_threads =
      (i >= (n_elems / blockDim.x) * blockDim.x) ? (n_elems % blockDim.x) : blockDim.x;

    for(lo delta = 0; delta < n_blocks; delta++) {
      // each thread calculates value corresponding to its i (i_trl)
      {
        lo &i_trl = i;
        val_prev = val_curr;
        val_curr = val_next;
        d_get_local_contributions_treg_sreg_sl_p0_p0(&val_next,
          delta+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
          mesh_metadata, mesh_data, kp);
        shmem_matrix_vals[tid] = ((i_tst == i_trl) ? (0) : (-val_prev + 2*val_curr - val_next));
        __syncthreads();
      }

      // now the thread logic is changed, each thread takes one (or more) blocks
      // in this delta and loops through all of the current trial elements (columns)
      {
        lo max_block = n_blocks - delta;
        for(lo block = threadIdx.x; block < max_block; block += curr_active_threads) {
          lo block_row = delta + block;
          lo block_col = block;
          sc y_val = 0;
          for(lo j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.x) * blockDim.x + j;
            lo &col = i_trl;
            sc x_val = x_perm[block_col + ld_x_perm * col];
            y_val += shmem_matrix_vals[j] * x_val;
          }
          y_val *= alpha;
          y[block_row * ld_y + row] += y_val;
        }
        __syncthreads();
      }

    }

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver2
  ( [[maybe_unused]] const spacetime_heat_dl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements,
  //   then is assigned to a block and loops through all the trial elements

  constexpr int tpbx = tpb_K[2][quadr_order].x;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_matrix_vals_data[3 * tpbx];
  sc * shmem_matrix_vals[3];
  shmem_matrix_vals[0] = shmem_matrix_vals_data + 0 * blockDim.x;
  shmem_matrix_vals[1] = shmem_matrix_vals_data + 1 * blockDim.x;
  shmem_matrix_vals[2] = shmem_matrix_vals_data + 2 * blockDim.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo i_tst = i_tst_begin + blockIdx.x;
  const lo &row = i_tst;

  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  for(lo i = threadIdx.x; i < n_elems; i += blockDim.x) {
    d_triangles_to_geometry_000_trl(i, mesh_data, quadr_nodes_trl);

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    lo curr_active_threads =
      (i >= (n_elems / blockDim.x) * blockDim.x) ? (n_elems % blockDim.x) : blockDim.x;

    for(lo delta = 0; delta < n_blocks; delta++) {
      // each thread calculates value corresponding to its i (i_trl)
      {
        lo &i_trl = i;
        vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
        vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
        d_get_local_contributions_treg_sreg_dl_p0_p1(vals_next,
          delta+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
          mesh_metadata, mesh_data, kp);
        shmem_matrix_vals[0][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0]));
        shmem_matrix_vals[1][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
        shmem_matrix_vals[2][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));
        __syncthreads();
      }

      // now the thread logic is changed, each thread takes one (or more) blocks
      // in this delta and loops through all of the current trial elements (columns)
      {
        lo max_block = n_blocks - delta;
        for(lo block = threadIdx.x; block < max_block; block += curr_active_threads) {
          lo block_row = delta + block;
          lo block_col = block;
          sc y_val = 0;
          for(lo j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.x) * blockDim.x + j;
            lo * cols = mesh_data.d_element_nodes + 3 * i_trl;
            y_val += shmem_matrix_vals[0][j] * x_perm[block_col + ld_x_perm * cols[0]];
            y_val += shmem_matrix_vals[1][j] * x_perm[block_col + ld_x_perm * cols[1]];
            y_val += shmem_matrix_vals[2][j] * x_perm[block_col + ld_x_perm * cols[2]];
          }
          y_val *= alpha;
          y[block_row * ld_y + row] += y_val;
        }
        __syncthreads();
      }

    }

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver2
  ( [[maybe_unused]] const spacetime_heat_adl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, then is assigned
  //   to a block and loops through all the trial elements

  constexpr int tpbx = tpb_Kt[2][quadr_order].x;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_matrix_vals_data[3 * tpbx];
  sc * shmem_matrix_vals[3];
  shmem_matrix_vals[0] = shmem_matrix_vals_data + 0 * blockDim.x;
  shmem_matrix_vals[1] = shmem_matrix_vals_data + 1 * blockDim.x;
  shmem_matrix_vals[2] = shmem_matrix_vals_data + 2 * blockDim.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo i_tst = i_tst_begin + blockIdx.x;
  const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;

  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  for(lo i = threadIdx.x; i < n_elems; i += blockDim.x) {
    d_triangles_to_geometry_000_trl(i, mesh_data, quadr_nodes_trl);

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    lo curr_active_threads =
      (i >= (n_elems / blockDim.x) * blockDim.x) ? (n_elems % blockDim.x) : blockDim.x;

    for(lo delta = 0; delta < n_blocks; delta++) {
      // each thread calculates value corresponding to its i (i_trl)
      {
        lo &i_trl = i;
        vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
        vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
        d_get_local_contributions_treg_sreg_adl_p1_p0(vals_next,
          delta+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
          mesh_metadata, mesh_data, kp);
        shmem_matrix_vals[0][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0]));
        shmem_matrix_vals[1][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
        shmem_matrix_vals[2][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));
        __syncthreads();
      }

      // now the thread logic is changed, each thread takes one (or more) blocks
      // in this delta and loops through all of the current trial elements (columns)
      {
        lo max_block = n_blocks - delta;
        for(lo block = threadIdx.x; block < max_block; block += curr_active_threads) {
          lo block_row = delta + block;
          lo block_col = block;
          sc y_vals[3] = {0,0,0};
          for(lo j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.x) * blockDim.x + j;
            lo &col = i_trl;
            sc x_val = x_perm[block_col + ld_x_perm * col];
            y_vals[0] += shmem_matrix_vals[0][j] * x_val;
            y_vals[1] += shmem_matrix_vals[1][j] * x_val;
            y_vals[2] += shmem_matrix_vals[2][j] * x_val;
          }
          atomicAdd(&y[block_row * ld_y + rows[0]], alpha * y_vals[0]);
          atomicAdd(&y[block_row * ld_y + rows[1]], alpha * y_vals[1]);
          atomicAdd(&y[block_row * ld_y + rows[2]], alpha * y_vals[2]);
        }
        __syncthreads();
      }

    }

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver2
  ( [[maybe_unused]] const spacetime_heat_hs_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, then is assigned
  //   to a block and loops through all the trial elements

  constexpr int tpbx = tpb_D[2][quadr_order].x;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_matrix_vals_data[9 * tpbx];
  sc * shmem_matrix_vals[9];
  for(lo j = 0; j < 9; j++)
    shmem_matrix_vals[j] = shmem_matrix_vals_data + j * blockDim.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo i_tst = i_tst_begin + blockIdx.x;
  const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;

  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  sc vals_prev[9];
  sc vals_curr[9];
  sc vals_next[9];

  for(lo i = threadIdx.x; i < n_elems; i += blockDim.x) {
    d_triangles_to_geometry_000_trl(i, mesh_data, quadr_nodes_trl);

    for(lo j = 0; j < 9; j++) vals_curr[j] = 0;
    for(lo j = 0; j < 9; j++) vals_next[j] = 0;

    lo curr_active_threads =
      (i >= (n_elems / blockDim.x) * blockDim.x) ? (n_elems % blockDim.x) : blockDim.x;

    for(lo delta = 0; delta < n_blocks; delta++) {
      // each thread calculates value corresponding to its i (i_trl)
      {
        lo &i_trl = i;
        for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
        for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
        d_get_local_contributions_treg_sreg_hs_p1_p1(vals_next,
          delta+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
          mesh_metadata, mesh_data, kp);
        for(lo j = 0; j < 9; j++)
          shmem_matrix_vals[j][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[j] + 2*vals_curr[j] - vals_next[j]));
        __syncthreads();
      }

      // now the thread logic is changed, each thread takes one (or more) blocks
      // in this delta and loops through all of the current trial elements (columns)
      {
        lo max_block = n_blocks - delta;
        for(lo block = threadIdx.x; block < max_block; block += curr_active_threads) {
          lo block_row = delta + block;
          lo block_col = block;
          sc y_vals[3] = {0,0,0};
          for(lo j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.x) * blockDim.x + j;
            lo * cols = mesh_data.d_element_nodes + 3 * i_trl;
            sc x_vals[3];
            x_vals[0] = x_perm[block_col + ld_x_perm * cols[0]];
            x_vals[1] = x_perm[block_col + ld_x_perm * cols[1]];
            x_vals[2] = x_perm[block_col + ld_x_perm * cols[2]];
            for(lo r = 0; r < 3; r++)
             for(lo c = 0; c < 3; c++)
              y_vals[r] += shmem_matrix_vals[3*r+c][j] * x_vals[c];
          }
          atomicAdd(&y[block_row * ld_y + rows[0]], alpha * y_vals[0]);
          atomicAdd(&y[block_row * ld_y + rows[1]], alpha * y_vals[1]);
          atomicAdd(&y[block_row * ld_y + rows[2]], alpha * y_vals[2]);
        }
        __syncthreads();
      }

    }

  }

}














template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver3
  ( [[maybe_unused]] const spacetime_heat_sl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_V[3][quadr_order].x;
  constexpr int tpby = tpb_V[3][quadr_order].y;
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst[tpbx];
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_trl[tpby];
  __shared__ sc shmem_y_vals[tpbx * tpby];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x * blockDim.x + threadIdx.x;
  shmem_y_vals[tid] = 0;
  if(tid < blockDim.x)
    d_triangles_to_geometry_000_tst(i_tst_begin + blockIdx.x * blockDim.x + tid,
      mesh_data, shmem_quadr_nodes_tst[tid]);
  __syncthreads();

  sc matrix_val;
  sc val_prev;
  sc val_curr;
  sc val_next;

  for(lo i_trl = threadIdx.y; i_trl < n_elems; i_trl += blockDim.y) {
    if(tid < blockDim.y)
      d_triangles_to_geometry_000_trl((i_trl / blockDim.y) * blockDim.y + tid,
      mesh_data, shmem_quadr_nodes_trl[tid]);
    __syncthreads();

    const lo &row = i_tst;
    const lo &col = i_trl;

    val_curr = 0;
    val_next = 0;

    for(lo delta = 0; delta < n_blocks; delta++) {
      val_prev = val_curr;
      val_curr = val_next;
      d_get_local_contributions_treg_sreg_sl_p0_p0(&val_next,
        delta+1, i_tst, i_trl,
        shmem_quadr_nodes_tst[threadIdx.x], shmem_quadr_nodes_trl[threadIdx.y],
        mesh_metadata, mesh_data, kp);
      matrix_val = ((i_tst == i_trl) ? (0) : (-val_prev + 2*val_curr - val_next));
      lo max_block = n_blocks - delta;
      for(lo block = 0; block < max_block; block++) {
        lo block_row = delta + block;
        lo block_col = block;
        shmem_y_vals[tid] = matrix_val * x_perm[block_col + ld_x_perm * col];
        __syncthreads();
        d_reduce_sum_2d_y<tpbx,tpby>(shmem_y_vals);
        if(tid < blockDim.x)
          y[block_row * ld_y + row] += alpha * shmem_y_vals[tid];
        __syncthreads();
      }
    }

    shmem_y_vals[tid] = 0;
    __syncthreads();

  }
}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver3
  ( [[maybe_unused]] const spacetime_heat_dl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_K[3][quadr_order].x;
  constexpr int tpby = tpb_K[3][quadr_order].y;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst[tpbx];
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_trl[tpby];
  __shared__ sc shmem_y_vals[tpbx * tpby];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x * blockDim.x + threadIdx.x;
  shmem_y_vals[tid] = 0;
  if(tid < blockDim.x)
    d_triangles_to_geometry_000_tst(i_tst_begin + blockIdx.x * blockDim.x + tid,
      mesh_data, shmem_quadr_nodes_tst[tid]);
  __syncthreads();

  sc matrix_vals[3];
  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  for(lo i_trl = threadIdx.y; i_trl < n_elems; i_trl += blockDim.y) {
    if(tid < blockDim.y)
      d_triangles_to_geometry_000_trl((i_trl / blockDim.y) * blockDim.y + tid,
      mesh_data, shmem_quadr_nodes_trl[tid]);
    __syncthreads();

    const lo &row = i_tst;
    const lo * cols = mesh_data.d_element_nodes + 3 * i_trl;

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    for(lo delta = 0; delta < n_blocks; delta++) {
      vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
      vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
      d_get_local_contributions_treg_sreg_dl_p0_p1(vals_next,
        delta+1, i_tst, i_trl,
        shmem_quadr_nodes_tst[threadIdx.x], shmem_quadr_nodes_trl[threadIdx.y],
        mesh_metadata, mesh_data, kp);

      matrix_vals[0] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0]));
      matrix_vals[1] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
      matrix_vals[2] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));

      lo max_block = n_blocks - delta;
      for(lo block = 0; block < max_block; block++) {
        lo block_row = delta + block;
        lo block_col = block;
        shmem_y_vals[tid]  = matrix_vals[0] * x_perm[block_col + ld_x_perm * cols[0]];
        shmem_y_vals[tid] += matrix_vals[1] * x_perm[block_col + ld_x_perm * cols[1]];
        shmem_y_vals[tid] += matrix_vals[2] * x_perm[block_col + ld_x_perm * cols[2]];
        __syncthreads();
        d_reduce_sum_2d_y<tpbx,tpby>(shmem_y_vals);
        if(tid < blockDim.x)
          y[block_row * ld_y + row] += alpha * shmem_y_vals[tid];
        __syncthreads();
      }
    }

    shmem_y_vals[tid] = 0;
    __syncthreads();

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver3
  ( [[maybe_unused]] const spacetime_heat_adl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_Kt[3][quadr_order].x;
  constexpr int tpby = tpb_Kt[3][quadr_order].y;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst[tpbx];
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_trl[tpby];
  __shared__ sc shmem_y_vals_data[3 * tpbx * tpby];
  sc * shmem_y_vals[3];
  shmem_y_vals[0] = shmem_y_vals_data + 0 * blockDim.x * blockDim.y;
  shmem_y_vals[1] = shmem_y_vals_data + 1 * blockDim.x * blockDim.y;
  shmem_y_vals[2] = shmem_y_vals_data + 2 * blockDim.x * blockDim.y;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x * blockDim.x + threadIdx.x;
  shmem_y_vals[0][tid] = 0;
  shmem_y_vals[1][tid] = 0;
  shmem_y_vals[2][tid] = 0;
  if(tid < blockDim.x)
    d_triangles_to_geometry_000_tst(i_tst_begin + blockIdx.x * blockDim.x + tid,
      mesh_data, shmem_quadr_nodes_tst[tid]);
  __syncthreads();

  sc matrix_vals[3];
  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  for(lo i_trl = threadIdx.y; i_trl < n_elems; i_trl += blockDim.y) {
    if(tid < blockDim.y)
      d_triangles_to_geometry_000_trl((i_trl / blockDim.y) * blockDim.y + tid,
        mesh_data, shmem_quadr_nodes_trl[tid]);
    __syncthreads();

    const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;
    const lo &col = i_trl;

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    for(lo delta = 0; delta < n_blocks; delta++) {
      vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
      vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
      d_get_local_contributions_treg_sreg_adl_p1_p0(vals_next,
        delta+1, i_tst, i_trl,
        shmem_quadr_nodes_tst[threadIdx.x], shmem_quadr_nodes_trl[threadIdx.y],
        mesh_metadata, mesh_data, kp);

      matrix_vals[0] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0]));
      matrix_vals[1] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
      matrix_vals[2] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));

      lo max_block = n_blocks - delta;
      for(lo block = 0; block < max_block; block++) {
        lo block_row = delta + block;
        lo block_col = block;
        sc x_val = x_perm[block_col + ld_x_perm * col];
        shmem_y_vals[0][tid] = matrix_vals[0] * x_val;
        shmem_y_vals[1][tid] = matrix_vals[1] * x_val;
        shmem_y_vals[2][tid] = matrix_vals[2] * x_val;
        __syncthreads();
        d_reduce_sum_2d_y_multiple<tpbx,tpby>(shmem_y_vals[0], shmem_y_vals[1], shmem_y_vals[2]);
        if(threadIdx.y < 3)
          atomicAdd(&y[block_row * ld_y + rows[threadIdx.y]], alpha * shmem_y_vals[threadIdx.y][threadIdx.x]);
        __syncthreads(); // dont understand why this needs to be there, but it needs to
      }
    }

    shmem_y_vals[0][tid] = 0;
    shmem_y_vals[1][tid] = 0;
    shmem_y_vals[2][tid] = 0;
    __syncthreads();

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver3
  ( [[maybe_unused]] const spacetime_heat_hs_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_D[3][quadr_order].x;
  constexpr int tpby = tpb_D[3][quadr_order].y;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst[tpbx];
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_trl[tpby];
  __shared__ sc shmem_y_vals_data[3 * tpbx * tpby];
  sc * shmem_y_vals[3];
  shmem_y_vals[0] = shmem_y_vals_data + 0 * blockDim.x * blockDim.y;
  shmem_y_vals[1] = shmem_y_vals_data + 1 * blockDim.x * blockDim.y;
  shmem_y_vals[2] = shmem_y_vals_data + 2 * blockDim.x * blockDim.y;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x * blockDim.x + threadIdx.x;
  shmem_y_vals[0][tid] = 0;
  shmem_y_vals[1][tid] = 0;
  shmem_y_vals[2][tid] = 0;
  if(tid < blockDim.x)
    d_triangles_to_geometry_000_tst(i_tst_begin + blockIdx.x * blockDim.x + tid,
      mesh_data, shmem_quadr_nodes_tst[tid]);
  __syncthreads();

  sc matrix_vals[9];
  sc vals_prev[9];
  sc vals_curr[9];
  sc vals_next[9];
  sc x_vals[3];

  for(lo i_trl = threadIdx.y; i_trl < n_elems; i_trl += blockDim.y) {
    if(tid < blockDim.y)
      d_triangles_to_geometry_000_trl((i_trl / blockDim.y) * blockDim.y + tid,
      mesh_data, shmem_quadr_nodes_trl[tid]);
    __syncthreads();

    const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;
    const lo * cols = mesh_data.d_element_nodes + 3 * i_trl;

    for(lo j = 0; j < 9; j++) vals_curr[j] = 0;
    for(lo j = 0; j < 9; j++) vals_next[j] = 0;

    for(lo delta = 0; delta < n_blocks; delta++) {
      for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
      for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
      d_get_local_contributions_treg_sreg_hs_p1_p1(vals_next,
        delta+1, i_tst, i_trl,
        shmem_quadr_nodes_tst[threadIdx.x], shmem_quadr_nodes_trl[threadIdx.y],
        mesh_metadata, mesh_data, kp);

      for(lo j = 0; j < 9; j++) matrix_vals[j] = ((i_tst == i_trl) ? (0) : (-vals_prev[j] + 2*vals_curr[j] - vals_next[j]));

      lo max_block = n_blocks - delta;
      for(lo block = 0; block < max_block; block++) {
        lo block_row = delta + block;
        lo block_col = block;
        x_vals[0] = x_perm[block_col + ld_x_perm * cols[0]];
        x_vals[1] = x_perm[block_col + ld_x_perm * cols[1]];
        x_vals[2] = x_perm[block_col + ld_x_perm * cols[2]];
        for(lo r = 0; r < 3; r++)
          shmem_y_vals[r][tid] = 0;
        for(lo r = 0; r < 3; r++)
          for(lo c = 0; c < 3; c++)
            shmem_y_vals[r][tid] += matrix_vals[3*r+c] * x_vals[c];
        __syncthreads();
        d_reduce_sum_2d_y_multiple<tpbx,tpby>(shmem_y_vals[0], shmem_y_vals[1], shmem_y_vals[2]);
        if(threadIdx.y < 3)
          atomicAdd(&y[block_row * ld_y + rows[threadIdx.y]], alpha * shmem_y_vals[threadIdx.y][threadIdx.x]);
        __syncthreads();
      }
    }

    shmem_y_vals[0][tid] = 0;
    shmem_y_vals[1][tid] = 0;
    shmem_y_vals[2][tid] = 0;
    __syncthreads();

  }

}















template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver4
  ( [[maybe_unused]] const spacetime_heat_sl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_V[4][quadr_order].x;
  constexpr int tpby = tpb_V[4][quadr_order].y;
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst[tpbx];
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_trl[tpby];
  __shared__ sc shmem_matrix_vals[tpbx * tpby];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const lo &row = i_tst;
  if(tid < blockDim.x)
    d_triangles_to_geometry_000_tst(i_tst_begin + blockIdx.x * blockDim.x + tid,
      mesh_data, shmem_quadr_nodes_tst[tid]);
  __syncthreads();

  sc val_prev;
  sc val_curr;
  sc val_next;

  for(lo i = threadIdx.y; i < n_elems; i += blockDim.y) {
    if(tid < blockDim.y)
      d_triangles_to_geometry_000_trl((i / blockDim.y) * blockDim.y + tid,
      mesh_data, shmem_quadr_nodes_trl[tid]);
    __syncthreads();

    val_curr = 0;
    val_next = 0;

    lo curr_active_threads = (i >= (n_elems / blockDim.y) * blockDim.y) ? (n_elems % blockDim.y) : blockDim.y;

    for(lo delta = 0; delta < n_blocks; delta++) {
      {
        lo &i_trl = i;
        val_prev = val_curr;
        val_curr = val_next;
        d_get_local_contributions_treg_sreg_sl_p0_p0(&val_next,
          delta+1, i_tst, i_trl,
          shmem_quadr_nodes_tst[threadIdx.x], shmem_quadr_nodes_trl[threadIdx.y],
          mesh_metadata, mesh_data, kp);
        shmem_matrix_vals[tid] = ((i_tst == i_trl) ? (0) : (-val_prev + 2*val_curr - val_next));
        __syncthreads();
      }
      {
        lo max_block = n_blocks - delta;
        for(lo block = threadIdx.y; block < max_block; block += curr_active_threads) {
          lo block_row = delta + block;
          lo block_col = block;
          sc y_val = 0.0;
          for(int j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.y) * blockDim.y + j;
            lo &col = i_trl;
            sc x_val = x_perm[block_col + ld_x_perm * col];
            y_val += shmem_matrix_vals[j * blockDim.x + threadIdx.x] * x_val;
          }
          y_val *= alpha;
          y[block_row * ld_y + row] += y_val;
        }
        __syncthreads();
      }

    }

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver4
  ( [[maybe_unused]] const spacetime_heat_dl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_K[4][quadr_order].x;
  constexpr int tpby = tpb_K[4][quadr_order].y;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst[tpbx];
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_trl[tpby];
  __shared__ sc shmem_matrix_vals_data[3 * tpbx * tpby];
  sc * shmem_matrix_vals[3];
  shmem_matrix_vals[0] = shmem_matrix_vals_data + 0 * blockDim.x * blockDim.y;
  shmem_matrix_vals[1] = shmem_matrix_vals_data + 1 * blockDim.x * blockDim.y;
  shmem_matrix_vals[2] = shmem_matrix_vals_data + 2 * blockDim.x * blockDim.y;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const lo &row = i_tst;
  if(tid < blockDim.x)
    d_triangles_to_geometry_000_tst(i_tst_begin + blockIdx.x * blockDim.x + tid,
      mesh_data, shmem_quadr_nodes_tst[tid]);
  __syncthreads();

  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  for(lo i = threadIdx.y; i < n_elems; i += blockDim.y) {
    if(tid < blockDim.y)
      d_triangles_to_geometry_000_trl((i / blockDim.y) * blockDim.y + tid,
      mesh_data, shmem_quadr_nodes_trl[tid]);
    __syncthreads();

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    lo curr_active_threads = (i >= (n_elems / blockDim.y) * blockDim.y) ? (n_elems % blockDim.y) : blockDim.y;

    for(lo delta = 0; delta < n_blocks; delta++) {
      {
        lo &i_trl = i;
        vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
        vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
        d_get_local_contributions_treg_sreg_dl_p0_p1(vals_next,
          delta+1, i_tst, i_trl,
          shmem_quadr_nodes_tst[threadIdx.x], shmem_quadr_nodes_trl[threadIdx.y],
          mesh_metadata, mesh_data, kp);
        shmem_matrix_vals[0][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0]));
        shmem_matrix_vals[1][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
        shmem_matrix_vals[2][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));
        __syncthreads();
      }
      {
        lo max_block = n_blocks - delta;
        for(lo block = threadIdx.y; block < max_block; block += curr_active_threads) {
          lo block_row = delta + block;
          lo block_col = block;
          sc y_val = 0.0;
          for(int j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.y) * blockDim.y + j;
            lo * cols = mesh_data.d_element_nodes + 3 * i_trl;
            y_val += shmem_matrix_vals[0][j * blockDim.x + threadIdx.x] * x_perm[block_col + ld_x_perm * cols[0]];
            y_val += shmem_matrix_vals[1][j * blockDim.x + threadIdx.x] * x_perm[block_col + ld_x_perm * cols[1]];
            y_val += shmem_matrix_vals[2][j * blockDim.x + threadIdx.x] * x_perm[block_col + ld_x_perm * cols[2]];
          }
          y_val *= alpha;
          y[block_row * ld_y + row] += y_val;
        }
        __syncthreads();
      }
    }

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver4
  ( [[maybe_unused]] const spacetime_heat_adl_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p0 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_Kt[4][quadr_order].x;
  constexpr int tpby = tpb_Kt[4][quadr_order].y;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst[tpbx];
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_trl[tpby];
  __shared__ sc shmem_matrix_vals_data[3 * tpbx * tpby];
  sc * shmem_matrix_vals[3];
  shmem_matrix_vals[0] = shmem_matrix_vals_data + 0 * blockDim.x * blockDim.y;
  shmem_matrix_vals[1] = shmem_matrix_vals_data + 1 * blockDim.x * blockDim.y;
  shmem_matrix_vals[2] = shmem_matrix_vals_data + 2 * blockDim.x * blockDim.y;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;
  if(tid < blockDim.x)
    d_triangles_to_geometry_000_tst(i_tst_begin + blockIdx.x * blockDim.x + tid,
      mesh_data, shmem_quadr_nodes_tst[tid]);
  __syncthreads();

  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  for(lo i = threadIdx.y; i < n_elems; i += blockDim.y) {
    if(tid < blockDim.y)
      d_triangles_to_geometry_000_trl((i / blockDim.y) * blockDim.y + tid,
      mesh_data, shmem_quadr_nodes_trl[tid]);
    __syncthreads();

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    lo curr_active_threads = (i >= (n_elems / blockDim.y) * blockDim.y) ? (n_elems % blockDim.y) : blockDim.y;

    for(lo delta = 0; delta < n_blocks; delta++) {
      {
        lo &i_trl = i;
        vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
        vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
        d_get_local_contributions_treg_sreg_adl_p1_p0(vals_next,
          delta+1, i_tst, i_trl,
          shmem_quadr_nodes_tst[threadIdx.x], shmem_quadr_nodes_trl[threadIdx.y],
          mesh_metadata, mesh_data, kp);
        shmem_matrix_vals[0][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0]));
        shmem_matrix_vals[1][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
        shmem_matrix_vals[2][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));
        __syncthreads();
      }
      {
        lo max_block = n_blocks - delta;
        for(lo block = threadIdx.y; block < max_block; block += curr_active_threads) {
          lo block_row = delta + block;
          lo block_col = block;
          sc y_vals[3] = {0,0,0};
          for(int j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.y) * blockDim.y + j;
            lo &col = i_trl;
            sc x_val = x_perm[block_col + ld_x_perm * col];
            y_vals[0] += shmem_matrix_vals[0][j * blockDim.x + threadIdx.x] * x_val;
            y_vals[1] += shmem_matrix_vals[1][j * blockDim.x + threadIdx.x] * x_val;
            y_vals[2] += shmem_matrix_vals[2][j * blockDim.x + threadIdx.x] * x_val;
          }
          atomicAdd(&y[block_row * ld_y + rows[0]], alpha * y_vals[0]);
          atomicAdd(&y[block_row * ld_y + rows[1]], alpha * y_vals[1]);
          atomicAdd(&y[block_row * ld_y + rows[2]], alpha * y_vals[2]);
        }
        __syncthreads();
      }
    }

  }

}



template< int quadr_order >
__global__ void g_apply_gpu_treg_sreg_ver4
  ( [[maybe_unused]] const spacetime_heat_hs_kernel_antiderivative * _hka,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _tst_space,
    [[maybe_unused]] const uniform_spacetime_be_space< basis_tri_p1 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y, lo ld_y, sc alpha, lo i_tst_begin,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_metadata mesh_metadata,
    const uniform_spacetime_tensor_mesh_gpu::mesh_raw_data mesh_data,
    const heat_kernel_parameters kp) {

  constexpr int tpbx = tpb_D[4][quadr_order].x;
  constexpr int tpby = tpb_D[4][quadr_order].y;

  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst[tpbx];
  __shared__ quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_trl[tpby];
  __shared__ sc shmem_matrix_vals_data[9 * tpbx * tpby];
  sc * shmem_matrix_vals[9];
  for(lo j = 0; j < 9; j++) shmem_matrix_vals[j] = shmem_matrix_vals_data + j * blockDim.x * blockDim.y;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const lo &n_blocks = mesh_metadata.n_temporal_elements;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo i_tst = i_tst_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;
  if(tid < blockDim.x)
    d_triangles_to_geometry_000_tst(i_tst_begin + blockIdx.x * blockDim.x + tid,
      mesh_data, shmem_quadr_nodes_tst[tid]);
  __syncthreads();

  sc vals_prev[9];
  sc vals_curr[9];
  sc vals_next[9];

  for(lo i = threadIdx.y; i < n_elems; i += blockDim.y) {
    if(tid < blockDim.y)
      d_triangles_to_geometry_000_trl((i / blockDim.y) * blockDim.y + tid,
      mesh_data, shmem_quadr_nodes_trl[tid]);
    __syncthreads();

    for(lo j = 0; j < 9; j++) vals_curr[j] = 0;
    for(lo j = 0; j < 9; j++) vals_next[j] = 0;

    lo curr_active_threads = (i >= (n_elems / blockDim.y) * blockDim.y) ? (n_elems % blockDim.y) : blockDim.y;

    for(lo delta = 0; delta < n_blocks; delta++) {
      {
        lo &i_trl = i;
        for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
        for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
        d_get_local_contributions_treg_sreg_hs_p1_p1(vals_next,
          delta+1, i_tst, i_trl,
          shmem_quadr_nodes_tst[threadIdx.x], shmem_quadr_nodes_trl[threadIdx.y],
          mesh_metadata, mesh_data, kp);
        for(lo j = 0; j < 9; j++)
          shmem_matrix_vals[j][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[j] + 2*vals_curr[j] - vals_next[j]));
        __syncthreads();
      }
      {
        lo max_block = n_blocks - delta;
        for(lo block = threadIdx.y; block < max_block; block += curr_active_threads) {
          lo block_row = delta + block;
          lo block_col = block;
          sc y_vals[3] = {0,0,0};
          for(int j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.y) * blockDim.y + j;
            lo * cols = mesh_data.d_element_nodes + 3 * i_trl;
            sc x_vals[3];
            x_vals[0] = x_perm[block_col + ld_x_perm * cols[0]];
            x_vals[1] = x_perm[block_col + ld_x_perm * cols[1]];
            x_vals[2] = x_perm[block_col + ld_x_perm * cols[2]];
            for(lo r = 0; r < 3; r++)
              for(lo c = 0; c < 3; c++)
                y_vals[r] += shmem_matrix_vals[3*r+c][j * blockDim.x + threadIdx.x] * x_vals[c];
          }
          atomicAdd(&y[block_row * ld_y + rows[0]], alpha * y_vals[0]);
          atomicAdd(&y[block_row * ld_y + rows[1]], alpha * y_vals[1]);
          atomicAdd(&y[block_row * ld_y + rows[2]], alpha * y_vals[2]);
        }
        __syncthreads();
      }
    }

  }

}














template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu
  <kernel_type, test_space_type, trial_space_type>::
  apply( const block_vector_type & x, block_vector_type & y, bool trans, sc alpha, sc beta ) const {

  if(n_gpus == 0) {
    uniform_spacetime_be_matrix_onthefly_cpu
      <kernel_type, test_space_type, trial_space_type>::
      apply(x, y, trans, alpha, beta);
    return;
  }

  if(trans) {
    if(besthea::settings::output_verbosity.warnings >= 1) {
      std::cerr << "BESTHEA Warning: transposed onthefly matrices are not "
        "supported. Apply operation will not be performed.\n";
    }
    return;
  }

  if(x.get_n_blocks() != this->get_block_dim() ||
     x.get_size_of_block() != this->get_dim_domain()) {
    std::cerr << "BESTHEA Error: x block vector dimension ("
      << x.get_n_blocks() << "*" << x.get_size_of_block()
      << ") does not match block matrix domain dimension ("
      << this->get_block_dim() << "*" << this->get_dim_domain()
      << ").\n";
    throw std::runtime_error("BESTHEA Exception: "
      "incompatible matrix and vector dimensions");
  }

  if(y.get_n_blocks() != this->get_block_dim() ||
     y.get_size_of_block() != this->get_dim_range()) {
    std::cerr << "BESTHEA Error: y block vector dimension ("
      << y.get_n_blocks() << "*" << y.get_size_of_block()
      << ") does not match block matrix range dimension ("
      << this->get_block_dim() << "*" << this->get_dim_range()
      << "). Apply will not be performed.\n";
    throw std::runtime_error("BESTHEA Exception: "
      "incompatible matrix and vector dimensions");
  }
  gpu_apply_timer_collection timers(n_gpus);

  timers.combined.start();



  block_vector_type y_perm;
  block_vector_type x_perm;

  timers.cpu_scalein.start();
  x_perm.copy_permute(x);
  y_perm.resize_to_match_permute(y, true);
  y.scale(beta);
  timers.cpu_scalein.stop();

  if(besthea::settings::output_verbosity.onthefly_loadbalance >= 1) {
    load_distr->print();
  }

  if(load_distr->get_gpu_count_total() > 0) {
    switch(gpu_kernel_version) {
      case 4:
        this->apply_gpu_treg_sreg_begin(x_perm, y, alpha, timers);
        break;
      case 3:
        this->apply_gpu_treg_sreg_begin(x_perm, y, alpha, timers);
        break;
      case 2:
      default:
        this->apply_gpu_treg_sreg_begin(x_perm, y, alpha, timers);
        break;
      case 1:
        this->apply_gpu_treg_sreg_begin(x, y_perm, alpha, timers);
        break;
    }
  }
  timers.cpu_all.start();
  if(load_distr->get_cpu_count() > 0) {
    timers.cpu_treg_sreg.start();
    this->apply_cpu_treg_sreg(x_perm, y_perm, alpha,
      load_distr->get_cpu_begin(), load_distr->get_cpu_end());
    timers.cpu_treg_sreg.stop();
  }
  timers.cpu_treg_ssng.start();
  this->apply_cpu_treg_ssng(x_perm, y_perm, alpha);
  timers.cpu_treg_ssng.stop();
  timers.cpu_tsng.start();
  this->apply_cpu_tsng(x_perm, y_perm, alpha);
  timers.cpu_tsng.stop();
  timers.cpu_all.stop();

  if(load_distr->get_gpu_count_total() > 0) {
    switch(gpu_kernel_version) {
      case 4:
        this->apply_gpu_treg_sreg_finalize(y);
        break;
      case 3:
        this->apply_gpu_treg_sreg_finalize(y);
        break;
      case 2:
      default:
        this->apply_gpu_treg_sreg_finalize(y);
        break;
      case 1:
        this->apply_gpu_treg_sreg_finalize(y_perm);
        break;
    }
  }

  y.add_permute(y_perm);



  timers.combined.stop();

  if(besthea::settings::output_verbosity.timers >= 1) {
    if(besthea::settings::output_verbosity.timers >= 2) {
      timers.print_all();
    }
    std::cout << "BESTHEA Info: apply elapsed time = "
      << timers.combined.get_elapsed_time_in_seconds() << " seconds\n";
  }


  load_distr->adapt(
    timers.get_cpu_time_const(),
    timers.get_cpu_time_scaling(),
    timers.get_gpu_time_const(),
    timers.get_gpu_time_scaling()
  );

}





template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu
  <kernel_type, test_space_type, trial_space_type>::
  apply_gpu_treg_sreg_begin( const block_vector_type & x,
    const block_vector_type & y, sc alpha,
    gpu_apply_timer_collection & timers ) const {
  // this function does not really care if x or y are permuted or not

  x.copy_to_raw(vectors_data.h_x);

  // just submit all the work to the gpus, it will be executed asynchronously



  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    CUDA_CHECK(cudaSetDevice(gpu_idx));

    timers.gpu_all[gpu_idx].start_submit();

    timers.gpu_copyin[gpu_idx].start_submit();
    CUDA_CHECK(cudaMemcpy2DAsync(
      vectors_data.d_x[gpu_idx], vectors_data.pitch_x[gpu_idx],
      vectors_data.h_x, x.get_size_of_block() * sizeof(sc),
      x.get_size_of_block() * sizeof(sc), x.get_n_blocks(),
      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset2DAsync(
      vectors_data.d_y[gpu_idx], vectors_data.pitch_y[gpu_idx],
      0, y.get_size_of_block() * sizeof(sc), y.get_n_blocks()));
    timers.gpu_copyin[gpu_idx].stop_submit();
  }



  heat_kernel_parameters kp(this->_kernel->get_alpha());

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    CUDA_CHECK(cudaSetDevice(gpu_idx));
    lo gpu_tst_elem_begin = load_distr->get_gpu_begin(gpu_idx);
    lo gpu_tst_elem_count = load_distr->get_gpu_count(gpu_idx);

    if(gpu_tst_elem_count == 0)
      continue;

    timers.gpu_compute[gpu_idx].start_submit();
    switch(gpu_kernel_version) {
      case 4: {
        dim3 blockSize = get_tpb(this->_kernel, this->_test_space, this->_trial_space, 4, this->_order_regular);
        dim3 gridSize(gpu_tst_elem_count / blockSize.x);
        switch(this->_order_regular) {
          case 5:
            g_apply_gpu_treg_sreg_ver4 <5> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 4:
            g_apply_gpu_treg_sreg_ver4 <4> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 2:
            g_apply_gpu_treg_sreg_ver4 <2> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 1:
          default:
            g_apply_gpu_treg_sreg_ver4 <1> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
        }
        break;
      }

      case 3: {
        dim3 blockSize = get_tpb(this->_kernel, this->_test_space, this->_trial_space, 3, this->_order_regular);
        dim3 gridSize(gpu_tst_elem_count / blockSize.x);
        switch(this->_order_regular) {
          case 5:
            g_apply_gpu_treg_sreg_ver3 <5> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 4:
            g_apply_gpu_treg_sreg_ver3 <4> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 2:
            g_apply_gpu_treg_sreg_ver3 <2> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 1:
          default:
            g_apply_gpu_treg_sreg_ver3 <1> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
        }
        break;
      }

      case 2:
      default: {
        dim3 blockSize = get_tpb(this->_kernel, this->_test_space, this->_trial_space, 2, this->_order_regular);
        dim3 gridSize(gpu_tst_elem_count);
        switch(this->_order_regular) {
          case 5:
            g_apply_gpu_treg_sreg_ver2 <5> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 4:
            g_apply_gpu_treg_sreg_ver2 <4> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 2:
            g_apply_gpu_treg_sreg_ver2 <2> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 1:
          default:
            g_apply_gpu_treg_sreg_ver2 <1> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
        }
        break;
      }

      case 1: {
        dim3 blockSize = get_tpb(this->_kernel, this->_test_space, this->_trial_space, 1, this->_order_regular);
        dim3 gridSize(gpu_tst_elem_count);
        switch(this->_order_regular) {
          case 5:
            g_apply_gpu_treg_sreg_ver1 <5> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 4:
            g_apply_gpu_treg_sreg_ver1 <4> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 2:
            g_apply_gpu_treg_sreg_ver1 <2> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
          case 1:
          default:
            g_apply_gpu_treg_sreg_ver1 <1> < < < gridSize, blockSize > > >
              (this->_kernel, this->_test_space, this->_trial_space, vectors_data.d_x[gpu_idx], vectors_data.ld_x[gpu_idx], vectors_data.d_y[gpu_idx], vectors_data.ld_y[gpu_idx], alpha, gpu_tst_elem_begin, gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
            break;
        }
        break;
      }

    }

    // cant wrap kernel launch in macro
    CUDA_CHECK(cudaGetLastError());

    timers.gpu_compute[gpu_idx].stop_submit();
  }



  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    CUDA_CHECK(cudaSetDevice(gpu_idx));

    timers.gpu_copyout[gpu_idx].start_submit();
    CUDA_CHECK(cudaMemcpy2DAsync(vectors_data.h_y[gpu_idx],
      y.get_size_of_block() * sizeof(sc),
      vectors_data.d_y[gpu_idx],
      vectors_data.pitch_y[gpu_idx],
      y.get_size_of_block() * sizeof(sc),
      y.get_n_blocks(),
      cudaMemcpyDeviceToHost));
    timers.gpu_copyout[gpu_idx].stop_submit();

    timers.gpu_all[gpu_idx].stop_submit();
  }

}





template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu
  <kernel_type, test_space_type, trial_space_type>::
  apply_gpu_treg_sreg_finalize( block_vector_type & y ) const {
  // this function does not really care if y is permuted or not

  // wait for all submitted work to finish
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    CUDA_CHECK(cudaSetDevice(gpu_idx));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

#pragma omp parallel for
  for(lo b = 0; b < y.get_n_blocks(); b++) {
    vector_type &y_block = y.get_block(b);

    for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
      const sc * h_y_block = vectors_data.h_y[gpu_idx] + b * y.get_size_of_block();
      y_block.add_from_raw(h_y_block, 1.0);
    }
  }

}
















template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
    besthea::bem::spacetime_heat_sl_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
    besthea::bem::spacetime_heat_dl_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
    besthea::bem::spacetime_heat_adl_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
    besthea::bem::spacetime_heat_hs_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

// clang-format on
