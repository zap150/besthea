
#include "besthea/uniform_spacetime_be_onthefly_matrix_gpu.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/timer.h"

#include <iostream>
#include <vector>
#include <cuda_runtime.h>



__constant__ __device__ besthea::onthefly::quadrature_reference_raw<5> c_quadr_reference_order5;
__constant__ __device__ besthea::onthefly::quadrature_reference_raw<4> c_quadr_reference_order4;
__constant__ __device__ besthea::onthefly::quadrature_reference_raw<2> c_quadr_reference_order2;
__constant__ __device__ besthea::onthefly::quadrature_reference_raw<1> c_quadr_reference_order1;

template<int quadr_order> __host__ __device__ const besthea::onthefly::quadrature_reference_raw<quadr_order> & c_get_quadr_reference();
template<> const besthea::onthefly::quadrature_reference_raw<5> & c_get_quadr_reference<5>() { return c_quadr_reference_order5; }
template<> const besthea::onthefly::quadrature_reference_raw<4> & c_get_quadr_reference<4>() { return c_quadr_reference_order4; }
template<> const besthea::onthefly::quadrature_reference_raw<2> & c_get_quadr_reference<2>() { return c_quadr_reference_order2; }
template<> const besthea::onthefly::quadrature_reference_raw<1> & c_get_quadr_reference<1>() { return c_quadr_reference_order1; }



template< class kernel_type, class test_space_type, class trial_space_type >
besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  uniform_spacetime_be_onthefly_matrix_gpu( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular,
  const besthea::onthefly::gpu_uniform_spacetime_tensor_mesh & gpu_mesh,
  int gpu_kernel_version )
  : uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>(kernel, test_space, trial_space, order_singular, order_regular),
    gpu_mesh(&gpu_mesh),
    n_gpus(gpu_mesh.get_n_gpus()),
    gpu_kernel_version(gpu_kernel_version) {

  // quadrature inited in base class constructor

  if(gpu_kernel_version < 1 || gpu_kernel_version > 2) {
    std::cerr << "Warning: invalid value of gpu_kernel_version=" << gpu_kernel_version << ", using default gpu_kernel_version=2\n";
    this->gpu_kernel_version = 2;
  }
  
  switch(order_regular) {
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

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::~uniform_spacetime_be_onthefly_matrix_gpu( ) {

}







template<class kernel_type, class test_space_type, class trial_space_type >
template<int quadr_order>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  init_gpu_quadrature_memory() const {

  besthea::onthefly::quadrature_reference_raw<quadr_order> quadr_ref_tmp;
  std::copy(this->quadr_reference._x1_ref[0].begin(), this->quadr_reference._x1_ref[0].end(), quadr_ref_tmp._x1_ref);
  std::copy(this->quadr_reference._x2_ref[0].begin(), this->quadr_reference._x2_ref[0].end(), quadr_ref_tmp._x2_ref);
  std::copy(this->quadr_reference._y1_ref[0].begin(), this->quadr_reference._y1_ref[0].end(), quadr_ref_tmp._y1_ref);
  std::copy(this->quadr_reference._y2_ref[0].begin(), this->quadr_reference._y2_ref[0].end(), quadr_ref_tmp._y2_ref);
  std::copy(this->quadr_reference._w[0].begin(), this->quadr_reference._w[0].end(), quadr_ref_tmp._w);

  

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    cudaSetDevice(gpu_idx);

    switch(quadr_order) {
      case 5:
        cudaMemcpyToSymbol(c_quadr_reference_order5, &quadr_ref_tmp, sizeof(quadr_ref_tmp));
        break;
      case 4:
        cudaMemcpyToSymbol(c_quadr_reference_order4, &quadr_ref_tmp, sizeof(quadr_ref_tmp));
        break;
      case 2:
        cudaMemcpyToSymbol(c_quadr_reference_order2, &quadr_ref_tmp, sizeof(quadr_ref_tmp));
        break;
      case 1:
      default:
        cudaMemcpyToSymbol(c_quadr_reference_order1, &quadr_ref_tmp, sizeof(quadr_ref_tmp));
        break;
    }

  }

}















__device__ void d_reduce_sum(volatile sc * shmem_vals, sc * add_result_to, sc output_scaling_factor) {

  // assuming number of threads is power of 2
  // assuming 1D block
  // assuming number of values is equal to blockDim

  // TODO: optimize better, maybe use only one warp

  int thread_count = blockDim.x / 2;
  int tid = threadIdx.x;
  
  while(thread_count > 32) {
    if(tid < thread_count) {
      shmem_vals[tid] += shmem_vals[tid + thread_count];
    }
    __syncthreads();
    thread_count /= 2;
  }

  if(tid < 32) {
    shmem_vals[tid] += shmem_vals[tid + 32];
    shmem_vals[tid] += shmem_vals[tid + 16];
    shmem_vals[tid] += shmem_vals[tid +  8];
    shmem_vals[tid] += shmem_vals[tid +  4];
    shmem_vals[tid] += shmem_vals[tid +  2];
    shmem_vals[tid] += shmem_vals[tid +  1];

    if(tid == 0)
      *add_result_to += shmem_vals[0] * output_scaling_factor;
  }
  
  __syncthreads();

}



template<int quadr_order>
__device__ void d_triangles_to_geometry_000_tst_shmem( // 0 shared vertices, 0 rot_test, 0 rot_trial
    lo i_tst, const besthea::onthefly::mesh_raw_data & mesh_data,
    besthea::onthefly::quadrature_nodes_raw<quadr_order> & shmem_quadr_nodes_tst ) {
  
  const lo * tst_elem_nodes = mesh_data.d_element_nodes + 3 * i_tst;

  const sc * x1 = mesh_data.d_node_coords + 3 * tst_elem_nodes[0];
  const sc * x2 = mesh_data.d_node_coords + 3 * tst_elem_nodes[1];
  const sc * x3 = mesh_data.d_node_coords + 3 * tst_elem_nodes[2];

  const sc * x1_ref = c_get_quadr_reference<quadr_order>()._x1_ref;
  const sc * x2_ref = c_get_quadr_reference<quadr_order>()._x2_ref;

  constexpr int quadr_size = besthea::onthefly::qo2qs(quadr_order);
  for ( lo i = threadIdx.x; i < quadr_size; i += blockDim.x ) {
    shmem_quadr_nodes_tst.xs[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
                                            + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ];
    shmem_quadr_nodes_tst.ys[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
                                            + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ];
    shmem_quadr_nodes_tst.zs[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
                                            + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ];
  }
}



template<int quadr_order>
__device__ void d_triangles_to_geometry_000_trl( // 0 shared vertices, 0 rot_test, 0 rot_trial
    lo i_trl, const besthea::onthefly::mesh_raw_data & mesh_data,
    besthea::onthefly::quadrature_nodes_raw<quadr_order> & quadr_nodes_trl ) {

  const lo * trl_elem_nodes = mesh_data.d_element_nodes + 3 * i_trl;

  const sc * y1 = mesh_data.d_node_coords + 3 * trl_elem_nodes[0];
  const sc * y2 = mesh_data.d_node_coords + 3 * trl_elem_nodes[1];
  const sc * y3 = mesh_data.d_node_coords + 3 * trl_elem_nodes[2];

  const sc * y1_ref = c_get_quadr_reference<quadr_order>()._y1_ref;
  const sc * y2_ref = c_get_quadr_reference<quadr_order>()._y2_ref;
  
  constexpr int quadr_size = besthea::onthefly::qo2qs(quadr_order);
  for ( lo i = 0; i < quadr_size; ++i ) {
    quadr_nodes_trl.xs[ i ] = y1[ 0 ] + ( y2[ 0 ] - y1[ 0 ] ) * y1_ref[ i ]
                                      + ( y3[ 0 ] - y1[ 0 ] ) * y2_ref[ i ];
    quadr_nodes_trl.ys[ i ] = y1[ 1 ] + ( y2[ 1 ] - y1[ 1 ] ) * y1_ref[ i ]
                                      + ( y3[ 1 ] - y1[ 1 ] ) * y2_ref[ i ];
    quadr_nodes_trl.zs[ i ] = y1[ 2 ] + ( y2[ 2 ] - y1[ 2 ] ) * y1_ref[ i ]
                                      + ( y3[ 2 ] - y1[ 2 ] ) * y2_ref[ i ];
  }
}















__device__ sc d_sl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
  sc xy1, sc xy2, sc xy3, sc ttau, sc ttau_sqrt,
  const besthea::onthefly::heat_kernel_parameters & kp ) {

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
  const besthea::onthefly::heat_kernel_parameters & kp ) {

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



__device__ void d_kernel_do_anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space(
  sc xy1, sc xy2, sc xy3, const sc * nx, const sc * ny,
  sc ttau, sc ttau_sqrt, sc * value1, sc * value2,
  const besthea::onthefly::heat_kernel_parameters & kp) {

  constexpr sc _two = 2.0;
  constexpr sc _four = 4.0;
  constexpr sc _eight = 8.0;

  sc dot = nx[ 0 ] * ny[ 0 ] + nx[ 1 ] * ny[ 1 ] + nx[ 2 ] * ny[ 2 ];
  sc norm = sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
  sc sqrt_d = ttau_sqrt;
  sc erf_value = erf( norm / ( _two * sqrt_d * kp.sqrt_alpha ) );
  sc four_pi_alpha_norm = _four * kp.pi * kp.alpha * norm;

  //  ttau > 0, norm > 0
  *value1 = ( ttau / four_pi_alpha_norm + norm / ( _eight * kp.pi * kp.alpha_2 ) )
      * erf_value
    + sqrt_d / ( _four * kp.pi * kp.alpha * kp.sqrt_pi * kp.sqrt_alpha )
      * exp( -( norm * norm ) / ( _four * ttau * kp.alpha ) );

  *value2 = erf_value / four_pi_alpha_norm;

  *value1 *= kp.alpha_2;
  *value2 *= dot * kp.alpha;

}







__device__ void d_basis_tri_p1_evaluate_curl_00(
  lo i_elem, const sc * n, bool swap, sc * curls,
  const besthea::onthefly::mesh_raw_data & mesh_data) {

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
__device__ void d_get_values_regular_sl_p0_p0(sc * values_out, lo delta, lo i_test, lo i_trial,
  const besthea::onthefly::quadrature_nodes_raw<quadr_order> & quadr_nodes_tst,
  besthea::onthefly::quadrature_nodes_raw<quadr_order> & quadr_nodes_trl,
  const besthea::onthefly::mesh_raw_metadata & mesh_metadata,
  const besthea::onthefly::mesh_raw_data & mesh_data,
  const besthea::onthefly::heat_kernel_parameters & kp) {

  if(besthea::onthefly::quick_matrix_vals) {
    values_out[0] = (sc)(i_test + 2*delta + 3*i_trial);
    return;
  }
  
  sc timestep = mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const sc test_area = mesh_data.d_element_areas[i_test];
  const sc trial_area = mesh_data.d_element_areas[i_trial];

  d_triangles_to_geometry_000_trl( i_trial, mesh_data, quadr_nodes_trl );

  const sc * w = c_get_quadr_reference<quadr_order>()._w;


  sc value = 0;
  constexpr int quadr_size = besthea::onthefly::qo2qs(quadr_order);
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
__device__ void d_get_values_regular_dl_p0_p1(sc * values_out, lo delta, lo i_test, lo i_trial,
  const besthea::onthefly::quadrature_nodes_raw<quadr_order> & quadr_nodes_tst,
  besthea::onthefly::quadrature_nodes_raw<quadr_order> & quadr_nodes_trl,
  const besthea::onthefly::mesh_raw_metadata & mesh_metadata,
  const besthea::onthefly::mesh_raw_data & mesh_data,
  const besthea::onthefly::heat_kernel_parameters & kp) {
    
  if(besthea::onthefly::quick_matrix_vals) {
    for(int j = 0; j < 3; j++) values_out[j] = (sc)((j + 1) * (i_test + 2*delta) + 3*i_trial);
    return;
  }

  sc timestep = mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const sc * ny = mesh_data.d_element_normals + 3 * i_trial;

  const sc test_area = mesh_data.d_element_areas[i_test];
  const sc trial_area = mesh_data.d_element_areas[i_trial];

  d_triangles_to_geometry_000_trl( i_trial, mesh_data, quadr_nodes_trl );

  const sc * w = c_get_quadr_reference<quadr_order>()._w;
  const sc * y1_ref = c_get_quadr_reference<quadr_order>()._y1_ref;
  const sc * y2_ref = c_get_quadr_reference<quadr_order>()._y2_ref;


  sc kernel;
  sc value1, value2, value3;

  value1 = value2 = value3 = 0.0;

  constexpr int quadr_size = besthea::onthefly::qo2qs(quadr_order);
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
__device__ void d_get_values_regular_hs_p1_p1(sc * values_out, lo delta, lo i_test, lo i_trial,
  const besthea::onthefly::quadrature_nodes_raw<quadr_order> & quadr_nodes_tst,
  besthea::onthefly::quadrature_nodes_raw<quadr_order> & quadr_nodes_trl,
  const besthea::onthefly::mesh_raw_metadata & mesh_metadata,
  const besthea::onthefly::mesh_raw_data & mesh_data,
  const besthea::onthefly::heat_kernel_parameters & kp) {
  
  if(besthea::onthefly::quick_matrix_vals) {
    for(int j = 0; j < 9; j++) values_out[j] = (sc)((j + 1) * (i_test + 2*delta) + 3*i_trial);
    return;
  }

  sc timestep = mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);
  
  const sc * nx = mesh_data.d_element_normals + 3 * i_test;
  const sc * ny = mesh_data.d_element_normals + 3 * i_trial;

  const sc test_area = mesh_data.d_element_areas[i_test];
  const sc trial_area = mesh_data.d_element_areas[i_trial];

  d_triangles_to_geometry_000_trl( i_trial, mesh_data, quadr_nodes_trl );

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
        = test_curls[ test_curl_offset     ] * trial_curls[ trial_curl_offset ]
        + test_curls[ test_curl_offset + 1 ] * trial_curls[ trial_curl_offset + 1 ]
        + test_curls[ test_curl_offset + 2 ] * trial_curls[ trial_curl_offset + 2 ];
    }
  }

  for(lo j = 0; j < 9; j++) values_out[j] = 0;
  sc &value11 = values_out[0];   sc &value12 = values_out[1];   sc &value13 = values_out[2];
  sc &value21 = values_out[3];   sc &value22 = values_out[4];   sc &value23 = values_out[5];
  sc &value31 = values_out[6];   sc &value32 = values_out[7];   sc &value33 = values_out[8];

  constexpr int quadr_size = besthea::onthefly::qo2qs(quadr_order);
  for ( lo i_quad = 0; i_quad < quadr_size; ++i_quad ) {
    d_kernel_do_anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space(
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
__global__ void g_apply_regular
  ( [[maybe_unused]] besthea::bem::spacetime_heat_sl_kernel_antiderivative * _hka,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > * _tst_space,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > * _trl_space,
    const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const besthea::onthefly::mesh_raw_metadata mesh_metadata,
    const besthea::onthefly::mesh_raw_data mesh_data,
    const besthea::onthefly::heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, and loops through all the blocks

  __shared__ besthea::onthefly::quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_y_vals[besthea::onthefly::gpu_threads_per_block];

  const lo &n_blocks = mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const unsigned int &tid = threadIdx.x;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo &i_tst = i_tst_begin + blockIdx.x;

  shmem_y_vals[tid] = 0;
  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  sc matrix_val;
  sc val_prev;
  sc val_curr;
  sc val_next;

  besthea::onthefly::quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {

    const lo &row = i_tst;
    const lo &col = i_trl;

    val_curr = 0; // delta=0 and 0s wiil be done by the cpu
    val_next = 0;
    
    for (lo diag = 0; diag < n_blocks; diag++) {
      val_prev = val_curr;
      val_curr = val_next;
      d_get_values_regular_sl_p0_p0(&val_next, diag+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl, mesh_metadata, mesh_data, kp);

      matrix_val = ((i_tst == i_trl) ? (0) : (-val_prev + 2*val_curr - val_next)); // singular will be done by the cpu

      lo max_block = n_blocks - diag;
      for (lo block = 0; block < max_block; block++) {
        lo block_row = diag + block;
        lo block_col = block;
        shmem_y_vals[tid] = matrix_val * x[ ld_x * block_col + col ];
        __syncthreads();
        d_reduce_sum(shmem_y_vals, &y_perm[row * ld_y_perm + block_row], alpha);
      }
    }

    shmem_y_vals[tid] = 0;
    __syncthreads();

  }

}



template< int quadr_order >
__global__ void g_apply_regular
  ( [[maybe_unused]] besthea::bem::spacetime_heat_dl_kernel_antiderivative * _hka,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > * _tst_space,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > * _trl_space,
    const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const besthea::onthefly::mesh_raw_metadata mesh_metadata,
    const besthea::onthefly::mesh_raw_data mesh_data,
    const besthea::onthefly::heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, and loops through all the blocks

  __shared__ besthea::onthefly::quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_y_vals[besthea::onthefly::gpu_threads_per_block];

  const lo &n_blocks = mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const unsigned int &tid = threadIdx.x;
  const lo &n_elems = mesh_metadata.n_elems;
  const lo &i_tst = i_tst_begin + blockIdx.x;

  shmem_y_vals[tid] = 0;
  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  sc matrix_vals[3] = {0,0,0};
  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  besthea::onthefly::quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {

    const lo &row = i_tst;
    const lo * cols = mesh_data.d_element_nodes + 3 * i_trl;

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0; // delta=0 and 0s wiil be done by the cpu
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    for (lo diag = 0; diag < n_blocks; diag++) {
      vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
      vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
      d_get_values_regular_dl_p0_p1(vals_next, diag+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl, mesh_metadata, mesh_data, kp);

      matrix_vals[0] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0])); // singular will be done by the cpu
      matrix_vals[1] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
      matrix_vals[2] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));

      lo max_block = n_blocks - diag;
      for (lo block = 0; block < max_block; block++) {
        lo block_row = diag + block;
        lo block_col = block;
        shmem_y_vals[tid]  = matrix_vals[0] * x[ ld_x * block_col + cols[0]];
        shmem_y_vals[tid] += matrix_vals[1] * x[ ld_x * block_col + cols[1]];
        shmem_y_vals[tid] += matrix_vals[2] * x[ ld_x * block_col + cols[2]];
        __syncthreads();
        d_reduce_sum(shmem_y_vals, &y_perm[row * ld_y_perm + block_row], alpha);
      }
    }

    shmem_y_vals[tid] = 0;
    __syncthreads();

  }

}



template< int quadr_order >
__global__ void g_apply_regular
  ( [[maybe_unused]] besthea::bem::spacetime_heat_hs_kernel_antiderivative * _hka,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > * _tst_space,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > * _trl_space,
    const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const besthea::onthefly::mesh_raw_metadata mesh_metadata,
    const besthea::onthefly::mesh_raw_data mesh_data,
    const besthea::onthefly::heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, and loops through all the blocks

  __shared__ besthea::onthefly::quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;

  const lo &n_blocks = mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const lo &n_elems = mesh_metadata.n_elems;
  const lo &i_tst = i_tst_begin + blockIdx.x;

  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  sc matrix_vals[9];
  sc vals_prev[9];
  sc vals_curr[9];
  sc vals_next[9];
  sc x_vals[3];
  sc y_vals[3];

  besthea::onthefly::quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {
    if(i_tst == i_trl)
      continue;

    const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;
    const lo * cols = mesh_data.d_element_nodes + 3 * i_trl;

    for(lo j = 0; j < 9; j++) vals_curr[j] = 0; // delta=0 and 0s wiil be done by the cpu
    for(lo j = 0; j < 9; j++) vals_next[j] = 0;

    for (lo diag = 0; diag < n_blocks; diag++) {
      for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
      for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
      d_get_values_regular_hs_p1_p1(vals_next, diag+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl, mesh_metadata, mesh_data, kp);

      for(lo j = 0; j < 9; j++) matrix_vals[j] = ((i_tst == i_trl) ? (0) : (-vals_prev[j] + 2*vals_curr[j] - vals_next[j])); // singular will be done by the cpu

      lo max_block = n_blocks - diag;
      for (lo block = 0; block < max_block; block++) {
        lo block_row = diag + block;
        lo block_col = block;
        x_vals[0] = alpha * x[ ld_x * block_col + cols[0]];
        x_vals[1] = alpha * x[ ld_x * block_col + cols[1]];
        x_vals[2] = alpha * x[ ld_x * block_col + cols[2]];
        y_vals[0] = y_vals[1] = y_vals[2] = 0;
        for(lo r = 0; r < 3; r++) for(lo c = 0; c < 3; c++) y_vals[r] += matrix_vals[3*r+c] * x_vals[c]; // TODO: might use tensor unit for this
        atomicAdd(&y_perm[rows[0] * ld_y_perm + block_row], y_vals[0]);
        atomicAdd(&y_perm[rows[1] * ld_y_perm + block_row], y_vals[1]);
        atomicAdd(&y_perm[rows[2] * ld_y_perm + block_row], y_vals[2]);
        
      }
    }

  }

}
















template< int quadr_order >
__global__ void g_apply_regular_ver2
  ( [[maybe_unused]] besthea::bem::spacetime_heat_sl_kernel_antiderivative * _hka,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > * _tst_space,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const besthea::onthefly::mesh_raw_metadata mesh_metadata,
    const besthea::onthefly::mesh_raw_data mesh_data,
    const besthea::onthefly::heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, then is assigned to a block and loops through all the trial elements
                      
  __shared__ besthea::onthefly::quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_matrix_vals[besthea::onthefly::gpu_threads_per_block];
  
  const lo &n_blocks = mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo &i_tst = i_tst_begin + blockIdx.x;
  const lo &row = i_tst;

  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  besthea::onthefly::quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  sc val_prev;
  sc val_curr;
  sc val_next;

  for(lo i = threadIdx.x; i < n_elems; i += blockDim.x) {
    val_curr = 0;
    val_next = 0;

    lo curr_active_threads = (i >= (n_elems / blockDim.x) * blockDim.x) ? (n_elems % blockDim.x) : blockDim.x;

    for(lo diag = 0; diag < n_blocks; diag++) {
      // each thread calculates value corresponding to its i (i_trl)
      {
        lo &i_trl = i;
        val_prev = val_curr;
        val_curr = val_next;
        d_get_values_regular_sl_p0_p0(&val_next, diag+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl, mesh_metadata, mesh_data, kp);
        shmem_matrix_vals[tid] = ((i_tst == i_trl) ? (0) : (-val_prev + 2*val_curr - val_next)); // singular will be done by the cpu
        __syncthreads();
      }

      // now the thread logic is changed, each thread takes one (or more) blocks in this diag and loops through all of the current trial elements (columns)
      {
        lo max_block = n_blocks - diag;
        for(lo block = threadIdx.x; block < max_block; block += curr_active_threads) {
          lo block_row = diag + block;
          lo block_col = block;
          sc y_val = 0;
          for(lo j = 0; j < curr_active_threads; j++) {
            lo col = (i / blockDim.x) * blockDim.x + j;
            sc x_val = x_perm[col * ld_x_perm + block_col];
            y_val += shmem_matrix_vals[j] * x_val;
          }
          y_val *= alpha;
          y_perm[row * ld_y_perm + block_row] += y_val;
        }
        __syncthreads();
      }

    }

  }
  

}



template< int quadr_order >
__global__ void g_apply_regular_ver2
  ( [[maybe_unused]] besthea::bem::spacetime_heat_dl_kernel_antiderivative * _hka,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > * _tst_space,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const besthea::onthefly::mesh_raw_metadata mesh_metadata,
    const besthea::onthefly::mesh_raw_data mesh_data,
    const besthea::onthefly::heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, then is assigned to a block and loops through all the trial elements

  __shared__ besthea::onthefly::quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_matrix_vals_data[3 * besthea::onthefly::gpu_threads_per_block];
  sc * shmem_matrix_vals[3];
  shmem_matrix_vals[0] = shmem_matrix_vals_data + 0 * blockDim.x;
  shmem_matrix_vals[1] = shmem_matrix_vals_data + 1 * blockDim.x;
  shmem_matrix_vals[2] = shmem_matrix_vals_data + 2 * blockDim.x;
  
  const lo &n_blocks = mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo &i_tst = i_tst_begin + blockIdx.x;
  const lo &row = i_tst;

  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  besthea::onthefly::quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  for(lo i = threadIdx.x; i < n_elems; i += blockDim.x) {
    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    lo curr_active_threads = (i >= (n_elems / blockDim.x) * blockDim.x) ? (n_elems % blockDim.x) : blockDim.x;

    for(lo diag = 0; diag < n_blocks; diag++) {
      // each thread calculates value corresponding to its i (i_trl)
      {
        lo &i_trl = i;
        vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
        vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
        d_get_values_regular_dl_p0_p1(vals_next, diag+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl, mesh_metadata, mesh_data, kp);
        shmem_matrix_vals[0][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[0] + 2*vals_curr[0] - vals_next[0])); // singular will be done by the cpu
        shmem_matrix_vals[1][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[1] + 2*vals_curr[1] - vals_next[1]));
        shmem_matrix_vals[2][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[2] + 2*vals_curr[2] - vals_next[2]));
        __syncthreads();
      }

      // now the thread logic is changed, each thread takes one (or more) blocks in this diag and loops through all of the current trial elements (columns)
      {
        lo max_block = n_blocks - diag;
        for(lo block = threadIdx.x; block < max_block; block += curr_active_threads) {
          lo block_row = diag + block;
          lo block_col = block;
          sc y_val = 0;
          for(lo j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.x) * blockDim.x + j;
            lo * cols = mesh_data.d_element_nodes + 3 * i_trl;
            y_val += shmem_matrix_vals[0][j] * x_perm[cols[0] * ld_x_perm + block_col];
            y_val += shmem_matrix_vals[1][j] * x_perm[cols[1] * ld_x_perm + block_col];
            y_val += shmem_matrix_vals[2][j] * x_perm[cols[2] * ld_x_perm + block_col];
          }
          y_val *= alpha;
          y_perm[row * ld_y_perm + block_row] += y_val;
        }
        __syncthreads();
      }

    }

  }

}



template< int quadr_order >
__global__ void g_apply_regular_ver2
  ( [[maybe_unused]] besthea::bem::spacetime_heat_hs_kernel_antiderivative * _hka,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > * _tst_space,
    [[maybe_unused]] besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > * _trl_space,
    const sc * x_perm, lo ld_x_perm, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin,
    const besthea::onthefly::mesh_raw_metadata mesh_metadata,
    const besthea::onthefly::mesh_raw_data mesh_data,
    const besthea::onthefly::heat_kernel_parameters kp) {

  // each block handles one test element
  // each thread handles one or more trial elements, then is assigned to a block and loops through all the trial elements

  __shared__ besthea::onthefly::quadrature_nodes_raw<quadr_order> shmem_quadr_nodes_tst;
  __shared__ sc shmem_matrix_vals_data[9 * besthea::onthefly::gpu_threads_per_block];
  sc * shmem_matrix_vals[9];
  for(lo j = 0; j < 9; j++) shmem_matrix_vals[j] = shmem_matrix_vals_data + j * blockDim.x;
  
  const lo &n_blocks = mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const lo &n_elems = mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;
  const lo &i_tst = i_tst_begin + blockIdx.x;
  const lo * rows = mesh_data.d_element_nodes + 3 * i_tst;

  d_triangles_to_geometry_000_tst_shmem(i_tst, mesh_data, shmem_quadr_nodes_tst);
  __syncthreads();

  besthea::onthefly::quadrature_nodes_raw<quadr_order> quadr_nodes_trl;


  sc vals_prev[9];
  sc vals_curr[9];
  sc vals_next[9];

  for(lo i = threadIdx.x; i < n_elems; i += blockDim.x) {
    for(lo j = 0; j < 9; j++) vals_curr[j] = 0;
    for(lo j = 0; j < 9; j++) vals_next[j] = 0;

    lo curr_active_threads = (i >= (n_elems / blockDim.x) * blockDim.x) ? (n_elems % blockDim.x) : blockDim.x;

    for(lo diag = 0; diag < n_blocks; diag++) {
      // each thread calculates value corresponding to its i (i_trl)
      {
        lo &i_trl = i;
        for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
        for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
        d_get_values_regular_hs_p1_p1(vals_next, diag+1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl, mesh_metadata, mesh_data, kp);
        for(lo j = 0; j < 9; j++) shmem_matrix_vals[j][tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[j] + 2*vals_curr[j] - vals_next[j])); // singular will be done by the cpu
        __syncthreads();
      }

      // now the thread logic is changed, each thread takes one (or more) blocks in this diag and loops through all of the current trial elements (columns)
      {
        lo max_block = n_blocks - diag;
        for(lo block = threadIdx.x; block < max_block; block += curr_active_threads) {
          lo block_row = diag + block;
          lo block_col = block;
          sc y_vals[3] = {0,0,0};
          for(lo j = 0; j < curr_active_threads; j++) {
            lo i_trl = (i / blockDim.x) * blockDim.x + j;
            lo * cols = mesh_data.d_element_nodes + 3 * i_trl;
            sc x_vals[3];
            x_vals[0] = x_perm[cols[0] * ld_x_perm + block_col];
            x_vals[1] = x_perm[cols[1] * ld_x_perm + block_col];
            x_vals[2] = x_perm[cols[2] * ld_x_perm + block_col];
            for(lo r = 0; r < 3; r++) for(lo c = 0; c < 3; c++) y_vals[r] += shmem_matrix_vals[3*r+c][j] * x_vals[c];
          }
          y_vals[0] *= alpha;
          y_vals[1] *= alpha;
          y_vals[2] *= alpha;
          atomicAdd(&y_perm[rows[0] * ld_y_perm + block_row],  y_vals[0]);
          atomicAdd(&y_perm[rows[1] * ld_y_perm + block_row],  y_vals[1]);
          atomicAdd(&y_perm[rows[2] * ld_y_perm + block_row],  y_vals[2]);
        }
        __syncthreads();
      }

    }

  }

}














template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  apply_regular_gpu_begin( const block_vector_type & x, const block_vector_type & y, sc alpha, std::vector<apply_regular_gpu_tmp_data> & tmp_data ) const {

  // permuted vectors or not, the data are copied in the same way
  // for gpu_kernel_version 1, x must NOT be permuted, y must be permuted
  // for gpu_kernel_version 2, both x and y must be permuted

  std::vector<lo> gpu_i_tst_begins(n_gpus+1);
  for(int gpu_idx = 0; gpu_idx <= n_gpus; gpu_idx++) {
    gpu_i_tst_begins[gpu_idx] = (gpu_mesh->get_metadata().n_elems * gpu_idx) / n_gpus;
  }
  std::vector<lo> gpu_n_tst_elems(n_gpus);
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    gpu_n_tst_elems[gpu_idx] = gpu_i_tst_begins[gpu_idx+1] - gpu_i_tst_begins[gpu_idx];
  }
    
  tmp_data.resize(n_gpus);

  heat_kernel_parameters kp(this->_kernel->get_alpha());

  // TODO: test if better direct copy from block_vector to gpu
  sc * x_raw;
  cudaMallocHost(&x_raw, x.size() * sizeof(*x_raw));
  x.copy_to_raw(x_raw);
  
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    sc * &d_x = tmp_data[gpu_idx].d_x;
    sc * &d_y = tmp_data[gpu_idx].d_y;
    size_t &pitch_x = tmp_data[gpu_idx].pitch_x;
    size_t &pitch_y = tmp_data[gpu_idx].pitch_y;
    lo &ld_x = tmp_data[gpu_idx].ld_x;
    lo &ld_y = tmp_data[gpu_idx].ld_y;

    cudaSetDevice(gpu_idx);

    cudaMallocPitch(&d_x, &pitch_x, x.get_size_of_block() * sizeof(*d_x), x.get_block_size());
    cudaMallocPitch(&d_y, &pitch_y, y.get_size_of_block() * sizeof(*d_y), y.get_block_size());
    ld_x = pitch_x / sizeof(*d_x); // assuming the pitch is a multiple of element size
    ld_y = pitch_y / sizeof(*d_y);

    // TODO: async, streams
    cudaMemcpy2D(d_x, pitch_x, x_raw, x.get_size_of_block() * sizeof(*x_raw), x.get_size_of_block() * sizeof(*x_raw), x.get_block_size(), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, pitch_y * y.get_block_size());

    int gridSize = gpu_n_tst_elems[gpu_idx];
    int blockSize = 256; // number of gpu threads per block



    if(gpu_kernel_version == 1) {

      switch(this->_order_regular) {
        case 5:
          g_apply_regular <5> <<< gridSize, blockSize >>>
            (this->_kernel, this->_test_space, this->_trial_space, d_x, ld_x, d_y, ld_y, alpha, gpu_i_tst_begins[gpu_idx], gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
          break;
        case 4:
          g_apply_regular <4> <<< gridSize, blockSize >>>
            (this->_kernel, this->_test_space, this->_trial_space, d_x, ld_x, d_y, ld_y, alpha, gpu_i_tst_begins[gpu_idx], gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
          break;
        case 2:
          g_apply_regular <2> <<< gridSize, blockSize >>>
            (this->_kernel, this->_test_space, this->_trial_space, d_x, ld_x, d_y, ld_y, alpha, gpu_i_tst_begins[gpu_idx], gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
          break;
        case 1:
        default:
          g_apply_regular <1> <<< gridSize, blockSize >>>
            (this->_kernel, this->_test_space, this->_trial_space, d_x, ld_x, d_y, ld_y, alpha, gpu_i_tst_begins[gpu_idx], gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
          break;
      }

    } else if(gpu_kernel_version == 2) {

      switch(this->_order_regular) {
        case 5:
          g_apply_regular_ver2 <5> <<< gridSize, blockSize >>>
            (this->_kernel, this->_test_space, this->_trial_space, d_x, ld_x, d_y, ld_y, alpha, gpu_i_tst_begins[gpu_idx], gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
          break;
        case 4:
          g_apply_regular_ver2 <4> <<< gridSize, blockSize >>>
            (this->_kernel, this->_test_space, this->_trial_space, d_x, ld_x, d_y, ld_y, alpha, gpu_i_tst_begins[gpu_idx], gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
          break;
        case 2:
          g_apply_regular_ver2 <2> <<< gridSize, blockSize >>>
            (this->_kernel, this->_test_space, this->_trial_space, d_x, ld_x, d_y, ld_y, alpha, gpu_i_tst_begins[gpu_idx], gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
          break;
        case 1:
        default:
          g_apply_regular_ver2 <1> <<< gridSize, blockSize >>>
            (this->_kernel, this->_test_space, this->_trial_space, d_x, ld_x, d_y, ld_y, alpha, gpu_i_tst_begins[gpu_idx], gpu_mesh->get_metadata(), gpu_mesh->get_per_gpu_data()[gpu_idx], kp);
          break;
      }

    }

  }

  cudaFreeHost(x_raw);

}



template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  apply_regular_gpu_finish( block_vector_type & y, std::vector<apply_regular_gpu_tmp_data> & tmp_data ) const {

  sc * y_raw;
  cudaMallocHost(&y_raw, y.size() * sizeof(*y_raw));

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    sc * &d_x = tmp_data[gpu_idx].d_x;
    sc * &d_y = tmp_data[gpu_idx].d_y;
    //size_t &pitch_x = tmp_data[gpu_idx].pitch_x;
    size_t &pitch_y = tmp_data[gpu_idx].pitch_y;
    //lo &ld_x = tmp_data[gpu_idx].ld_x;
    //lo &ld_y = tmp_data[gpu_idx].ld_y;

    cudaSetDevice(gpu_idx);

    cudaDeviceSynchronize();

    cudaMemcpy2D(y_raw, y.get_size_of_block() * sizeof(*y_raw), d_y, pitch_y, y.get_size_of_block() * sizeof(*y_raw), y.get_block_size(), cudaMemcpyDeviceToHost);
    
    y.add_from_raw(y_raw);

    cudaFree(d_x);
    cudaFree(d_y);
  }

  cudaFreeHost(y_raw);

  // TODO: error checking

}















template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  apply( const block_vector_type & x, block_vector_type & y, bool trans, sc alpha, sc beta ) const {

  if(trans) {
    std::cerr << "I dont support trans matrices\n";
    return;
  }
  
  block_vector_type y_perm;
  block_vector_type x_perm;

  y_perm.copy_permute(y, beta);
  x_perm.copy_permute(x, alpha);

  std::vector<besthea::onthefly::apply_regular_gpu_tmp_data> tmp_data;

  if(gpu_kernel_version == 1)
    this->apply_regular_gpu_begin(x, y_perm, alpha, tmp_data);
  else if(gpu_kernel_version == 2)
    this->apply_regular_gpu_begin(x_perm, y_perm, 1, tmp_data);
  
  this->apply_singular(x_perm, y_perm);
  this->apply_delta0(x_perm, y_perm);
  this->apply_regular_gpu_finish(y_perm, tmp_data);


  y.copy_permute(y_perm);

}












template class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;

template class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

  template class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
    besthea::bem::spacetime_heat_hs_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;




