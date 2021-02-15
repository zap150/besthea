
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
#include <cuda_runtime.h>



__constant__ __device__ besthea::onthefly::mesh_raw_metadata c_mesh_metadata; // c for constant
__constant__ __device__ besthea::onthefly::mesh_raw_data c_mesh_data;
__constant__ __device__ besthea::onthefly::quadrature_wrapper_readonly_regular_raw c_my_quadrature;
__constant__ __device__ besthea::onthefly::heat_kernel_parameters c_kernel_params;





template< class kernel_type, class test_space_type, class trial_space_type >
besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  uniform_spacetime_be_onthefly_matrix_gpu( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular )
  : uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>(kernel, test_space, trial_space, order_singular, order_regular) {

  // quadrature inited in base class constructor

  cudaGetDeviceCount(&n_gpus);

  // TODO: error messages when no cuda-capable gpu found

  besthea::mesh::triangular_surface_mesh * mesh = this->_test_space->get_mesh()->get_spatial_surface_mesh();
  mesh_metadata.timestep = this->_test_space->get_mesh()->get_timestep();
  mesh_metadata.n_temporal_elements = this->_test_space->get_mesh()->get_n_temporal_elements();
  mesh_metadata.n_elems = mesh->get_n_elements();
  mesh_metadata.n_nodes = mesh->get_n_nodes();

  per_gpu_mesh_data.resize(n_gpus);
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {

    cudaSetDevice(gpu_idx);

    mesh_raw_data &gpu_mesh_data = per_gpu_mesh_data[gpu_idx];

    cudaMalloc(&gpu_mesh_data.d_element_areas,       mesh_metadata.n_elems * sizeof(*gpu_mesh_data.d_element_areas));
    cudaMalloc(&gpu_mesh_data.d_node_coords,     3 * mesh_metadata.n_nodes * sizeof(*gpu_mesh_data.d_node_coords));
    cudaMalloc(&gpu_mesh_data.d_element_nodes,   3 * mesh_metadata.n_elems * sizeof(*gpu_mesh_data.d_element_nodes));
    cudaMalloc(&gpu_mesh_data.d_element_normals, 3 * mesh_metadata.n_elems * sizeof(*gpu_mesh_data.d_element_normals));

    cudaMemcpy(gpu_mesh_data.d_element_areas,   mesh->get_all_areas().data(),        mesh_metadata.n_elems * sizeof(*gpu_mesh_data.d_element_areas), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mesh_data.d_node_coords,     mesh->get_all_nodes().data(),    3 * mesh_metadata.n_nodes * sizeof(*gpu_mesh_data.d_node_coords),   cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mesh_data.d_element_nodes,   mesh->get_all_elements().data(), 3 * mesh_metadata.n_elems * sizeof(*gpu_mesh_data.d_element_nodes), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mesh_data.d_element_normals, mesh->get_all_normals().data(),  3 * mesh_metadata.n_elems * sizeof(*gpu_mesh_data.d_element_normals), cudaMemcpyHostToDevice);
  }
  
  init_gpu_constant_memory();

}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::~uniform_spacetime_be_onthefly_matrix_gpu( ) {

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {

    cudaSetDevice(gpu_idx);

    mesh_raw_data &gpu_mesh_data = per_gpu_mesh_data[gpu_idx];

    cudaFree(gpu_mesh_data.d_element_areas);
    cudaFree(gpu_mesh_data.d_node_coords);
    cudaFree(gpu_mesh_data.d_element_nodes);
    cudaFree(gpu_mesh_data.d_element_normals);
  }

}








template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  init_gpu_constant_memory() const {

  besthea::onthefly::quadrature_wrapper_readonly_regular_raw my_quadr_tmp;
  std::copy(this->my_quadrature._x1_ref[0].begin(), this->my_quadrature._x1_ref[0].end(), my_quadr_tmp._x1_ref);
  std::copy(this->my_quadrature._x2_ref[0].begin(), this->my_quadrature._x2_ref[0].end(), my_quadr_tmp._x2_ref);
  std::copy(this->my_quadrature._y1_ref[0].begin(), this->my_quadrature._y1_ref[0].end(), my_quadr_tmp._y1_ref);
  std::copy(this->my_quadrature._y2_ref[0].begin(), this->my_quadrature._y2_ref[0].end(), my_quadr_tmp._y2_ref);
  std::copy(this->my_quadrature._w[0].begin(), this->my_quadrature._w[0].end(), my_quadr_tmp._w);
  my_quadr_tmp._size = this->my_quadrature._sizes[0];

  besthea::onthefly::heat_kernel_parameters kernel_params_tmp;
  sc alpha = this->_kernel->get_alpha();
  kernel_params_tmp.alpha = alpha;
  kernel_params_tmp.sqrt_alpha = std::sqrt(alpha);
  kernel_params_tmp.alpha_2 = alpha * alpha;
  kernel_params_tmp.pi = M_PI;
  kernel_params_tmp.sqrt_pi = std::sqrt(M_PI);


  cudaMemcpyToSymbol(c_mesh_metadata, &mesh_metadata, sizeof(mesh_metadata));
  cudaMemcpyToSymbol(c_my_quadrature, &my_quadr_tmp, sizeof(my_quadr_tmp));
  cudaMemcpyToSymbol(c_kernel_params, &kernel_params_tmp, sizeof(kernel_params_tmp));
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    cudaSetDevice(gpu_idx);
    cudaMemcpyToSymbol(c_mesh_data, &per_gpu_mesh_data[gpu_idx], sizeof(per_gpu_mesh_data[gpu_idx]));
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



__device__ void d_triangles_to_geometry_000( // 0 shared vertices, 0 rot_test, 0 rot_trial
    const sc * x1, const sc * x2, const sc * x3,
    const sc * y1, const sc * y2, const sc * y3,
    besthea::onthefly::quadrature_wrapper_changing_regular_raw & quadr_changing ) {

  const sc * x1_ref = c_my_quadrature._x1_ref;
  const sc * x2_ref = c_my_quadrature._x2_ref;
  const sc * y1_ref = c_my_quadrature._y1_ref;
  const sc * y2_ref = c_my_quadrature._y2_ref;

  lo size = c_my_quadrature._size;

  for ( lo i = 0; i < size; ++i ) {
    quadr_changing._x1[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
                                      + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ];
    quadr_changing._x2[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
                                      + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ];
    quadr_changing._x3[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
                                      + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ];
  }

  for ( lo i = 0; i < size; ++i ) {
    quadr_changing._y1[ i ] = y1[ 0 ] + ( y2[ 0 ] - y1[ 0 ] ) * y1_ref[ i ]
                                      + ( y3[ 0 ] - y1[ 0 ] ) * y2_ref[ i ];
    quadr_changing._y2[ i ] = y1[ 1 ] + ( y2[ 1 ] - y1[ 1 ] ) * y1_ref[ i ]
                                      + ( y3[ 1 ] - y1[ 1 ] ) * y2_ref[ i ];
    quadr_changing._y3[ i ] = y1[ 2 ] + ( y2[ 2 ] - y1[ 2 ] ) * y1_ref[ i ]
                                      + ( y3[ 2 ] - y1[ 2 ] ) * y2_ref[ i ];
  }
}



__device__ sc d_sl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
  sc xy1, sc xy2, sc xy3, sc ttau, sc ttau_sqrt ) {

  besthea::onthefly::heat_kernel_parameters &kp = c_kernel_params;
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
  sc xy1, sc xy2, sc xy3, const sc * ny, sc ttau, sc ttau_sqrt ) {

  besthea::onthefly::heat_kernel_parameters &kp = c_kernel_params;
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
  sc ttau, sc ttau_sqrt, sc * value1, sc * value2) {

  besthea::onthefly::heat_kernel_parameters &kp = c_kernel_params;
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
  lo i_elem, const sc * n, bool swap, sc * curls) {

  const lo * elem_nodes = c_mesh_data.d_element_nodes + 3 * i_elem;
  const sc * x1rot = c_mesh_data.d_node_coords + 3 * elem_nodes[0];
  const sc * x2rot = c_mesh_data.d_node_coords + 3 * elem_nodes[1];
  const sc * x3rot = c_mesh_data.d_node_coords + 3 * elem_nodes[2];

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














__device__ void d_get_values_regular_sl_p0_p0(sc * values_out, lo delta, lo i_test, lo i_trial,
  besthea::onthefly::quadrature_wrapper_changing_regular_raw & quadr_changing) {

  if(besthea::onthefly::quick_matrix_vals) {
    values_out[0] = (sc)(i_test + 2*delta + 3*i_trial);
    return;
  }
  
  sc timestep = c_mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const lo * tst_elem_nodes = c_mesh_data.d_element_nodes + 3 * i_test;
  const lo * trl_elem_nodes = c_mesh_data.d_element_nodes + 3 * i_trial;

  const sc * x1 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[0];
  const sc * x2 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[1];
  const sc * x3 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[2];
  const sc * y1 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[0];
  const sc * y2 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[1];
  const sc * y3 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[2];

  const sc test_area = c_mesh_data.d_element_areas[i_test];
  const sc trial_area = c_mesh_data.d_element_areas[i_trial];

  d_triangles_to_geometry_000( x1, x2, x3, y1, y2, y3, quadr_changing );

  const sc * w = c_my_quadrature._w;
  const sc * x1_mapped = quadr_changing._x1;
  const sc * x2_mapped = quadr_changing._x2;
  const sc * x3_mapped = quadr_changing._x3;
  const sc * y1_mapped = quadr_changing._y1;
  const sc * y2_mapped = quadr_changing._y2;
  const sc * y3_mapped = quadr_changing._y3;

  lo size = c_my_quadrature._size;


  sc value = 0;
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    value += d_sl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
            x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
            x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
            x3_mapped[ i_quad ] - y3_mapped[ i_quad ], 
            ttau, sqrt_ttau ) * w[ i_quad ];
  }

  value *= test_area * trial_area;
  *values_out = value;
  return;

}















__device__ void d_get_values_regular_dl_p0_p1(sc * values_out, lo delta, lo i_test, lo i_trial,
  besthea::onthefly::quadrature_wrapper_changing_regular_raw & quadr_changing) {
    
  if(besthea::onthefly::quick_matrix_vals) {
    for(int j = 0; j < 3; j++) values_out[j] = (sc)((j + 1) * (i_test + 2*delta) + 3*i_trial);
    return;
  }

  sc timestep = c_mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const lo * tst_elem_nodes = c_mesh_data.d_element_nodes + 3 * i_test;
  const lo * trl_elem_nodes = c_mesh_data.d_element_nodes + 3 * i_trial;

  const sc * x1 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[0];
  const sc * x2 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[1];
  const sc * x3 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[2];
  const sc * y1 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[0];
  const sc * y2 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[1];
  const sc * y3 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[2];
  const sc * ny = c_mesh_data.d_element_normals + 3 * i_trial;

  const sc test_area = c_mesh_data.d_element_areas[i_test];
  const sc trial_area = c_mesh_data.d_element_areas[i_trial];

  d_triangles_to_geometry_000( x1, x2, x3, y1, y2, y3, quadr_changing );

  const sc * w = c_my_quadrature._w;
  const sc * x1_mapped = quadr_changing._x1;
  const sc * x2_mapped = quadr_changing._x2;
  const sc * x3_mapped = quadr_changing._x3;
  const sc * y1_mapped = quadr_changing._y1;
  const sc * y2_mapped = quadr_changing._y2;
  const sc * y3_mapped = quadr_changing._y3;
  const sc * y1_ref = c_my_quadrature._y1_ref;
  const sc * y2_ref = c_my_quadrature._y2_ref;

  lo size = c_my_quadrature._size;


  sc kernel;
  sc value1, value2, value3;

  value1 = value2 = value3 = 0.0;

  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel = d_dl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
          x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
          x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
          x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
          ny, ttau, sqrt_ttau ) * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc factor = test_area * trial_area;
  values_out[0] = value1 * factor;
  values_out[1] = value2 * factor;
  values_out[2] = value3 * factor;
  return;

}















__device__ void d_get_values_regular_hs_p1_p1(sc * values_out, lo delta, lo i_test, lo i_trial,
  besthea::onthefly::quadrature_wrapper_changing_regular_raw & quadr_changing) {
  
  if(besthea::onthefly::quick_matrix_vals) {
    for(int j = 0; j < 9; j++) values_out[j] = (sc)((j + 1) * (i_test + 2*delta) + 3*i_trial);
    return;
  }

  sc timestep = c_mesh_metadata.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const lo * tst_elem_nodes = c_mesh_data.d_element_nodes + 3 * i_test;
  const lo * trl_elem_nodes = c_mesh_data.d_element_nodes + 3 * i_trial;

  const sc * x1 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[0];
  const sc * x2 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[1];
  const sc * x3 = c_mesh_data.d_node_coords + 3 * tst_elem_nodes[2];
  const sc * y1 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[0];
  const sc * y2 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[1];
  const sc * y3 = c_mesh_data.d_node_coords + 3 * trl_elem_nodes[2];
  const sc * nx = c_mesh_data.d_element_normals + 3 * i_test;
  const sc * ny = c_mesh_data.d_element_normals + 3 * i_trial;

  const sc test_area = c_mesh_data.d_element_areas[i_test];
  const sc trial_area = c_mesh_data.d_element_areas[i_trial];

  d_triangles_to_geometry_000( x1, x2, x3, y1, y2, y3, quadr_changing );

  const sc * w = c_my_quadrature._w;
  const sc * x1_mapped = quadr_changing._x1;
  const sc * x2_mapped = quadr_changing._x2;
  const sc * x3_mapped = quadr_changing._x3;
  const sc * y1_mapped = quadr_changing._y1;
  const sc * y2_mapped = quadr_changing._y2;
  const sc * y3_mapped = quadr_changing._y3;
  const sc * x1_ref = c_my_quadrature._x1_ref;
  const sc * x2_ref = c_my_quadrature._x2_ref;
  const sc * y1_ref = c_my_quadrature._y1_ref;
  const sc * y2_ref = c_my_quadrature._y2_ref;

  lo size = c_my_quadrature._size;


  sc test_curls[ 9 ], trial_curls[ 9 ];
  sc phi1x, phi1y;
  sc kernel1, kernel2;
  lo test_curl_offset, trial_curl_offset;
  sc curl_dot[ 9 ];

  d_basis_tri_p1_evaluate_curl_00( i_test, nx, false, test_curls );
  d_basis_tri_p1_evaluate_curl_00( i_trial, ny, true, trial_curls );
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

  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    d_kernel_do_anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nx, ny, ttau, sqrt_ttau, &kernel1, &kernel2 );

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

  sc factor = test_area * trial_area;
  for(lo j = 0; j < 9; j++) values_out[j] *= factor;
  return;

}














__global__ void apply_regular_sl_p0_p0(const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin) {

  // each block handles one test element
  // each thread handles one or more trial elements, and loops through all the blocks

  extern __shared__ sc shmem_y_vals[]; // requires shared memory size (in bytes) to be specified while calling this kernel. needs sizeof(sc)*blockDim.x bytes

  lo &n_blocks = c_mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const unsigned int &tid = threadIdx.x;
  lo &n_elems = c_mesh_metadata.n_elems;

  shmem_y_vals[tid] = 0;
  __syncthreads();

  sc matrix_val;
  sc val_prev;
  sc val_curr;
  sc val_next;

  besthea::onthefly::quadrature_wrapper_changing_regular_raw quadr_changing;

  const lo &i_tst = i_tst_begin + blockIdx.x;

  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {

    const lo &row = i_tst;
    const lo &col = i_trl;

    val_curr = 0; // delta=0 and 0s wiil be done by the cpu
    val_next = 0;
    
    for (lo diag = 0; diag < n_blocks; diag++) {
      val_prev = val_curr;
      val_curr = val_next;
      d_get_values_regular_sl_p0_p0(&val_next, diag+1, i_tst, i_trl, quadr_changing);

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



__global__ void apply_regular_dl_p0_p1(const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin) {

  // each block handles one test element
  // each thread handles one or more trial elements, and loops through all the blocks

  extern __shared__ sc shmem_y_vals[]; // requires shared memory size (in bytes) to be specified while calling this kernel. needs sizeof(sc)*blockDim.x bytes

  lo &n_blocks = c_mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const unsigned int &tid = threadIdx.x;
  lo &n_elems = c_mesh_metadata.n_elems;

  shmem_y_vals[tid] = 0;
  __syncthreads();

  sc matrix_vals[3] = {0,0,0};
  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  besthea::onthefly::quadrature_wrapper_changing_regular_raw quadr_changing;

  const lo &i_tst = i_tst_begin + blockIdx.x;

  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {

    const lo &row = i_tst;
    const lo * cols = c_mesh_data.d_element_nodes + 3 * i_trl;

    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0; // delta=0 and 0s wiil be done by the cpu
    vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

    for (lo diag = 0; diag < n_blocks; diag++) {
      vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
      vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
      d_get_values_regular_dl_p0_p1(vals_next, diag+1, i_tst, i_trl, quadr_changing);

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



__global__ void apply_regular_hs_p1_p1(const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin) {

  // each block handles one test element
  // each thread handles one or more trial elements, and loops through all the blocks

  lo &n_blocks = c_mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  lo &n_elems = c_mesh_metadata.n_elems;

  sc matrix_vals[9];
  sc vals_prev[9];
  sc vals_curr[9];
  sc vals_next[9];
  sc x_vals[3];
  sc y_vals[3];

  besthea::onthefly::quadrature_wrapper_changing_regular_raw quadr_changing;

  const lo &i_tst = i_tst_begin + blockIdx.x;

  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {
    if(i_tst == i_trl)
      continue;

    const lo * rows = c_mesh_data.d_element_nodes + 3 * i_tst;
    const lo * cols = c_mesh_data.d_element_nodes + 3 * i_trl;

    for(lo j = 0; j < 9; j++) vals_curr[j] = 0; // delta=0 and 0s wiil be done by the cpu
    for(lo j = 0; j < 9; j++) vals_next[j] = 0;

    for (lo diag = 0; diag < n_blocks; diag++) {
      for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
      for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
      d_get_values_regular_hs_p1_p1(vals_next, diag+1, i_tst, i_trl, quadr_changing);

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














__global__ void apply_regular_sl_p0_p0_ver2(const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha, lo i_tst_begin) {

  // each block handles one test element
  // each thread handles one or more trial elements, then is assigned to a block and loops through all the deltas

  extern __shared__ sc shmem[]; // requires shared memory size (in bytes) to be specified while calling this kernel. needs 4*sizeof(sc)*blockDim.x bytes
  sc * matrix_vals = shmem + 0 * blockDim.x;
  sc * vals_prev = shmem + 1 * blockDim.x;
  sc * vals_curr = shmem + 2 * blockDim.x;
  sc * vals_next = shmem + 3 * blockDim.x;

  lo &n_blocks = c_mesh_metadata.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  lo &n_elems = c_mesh_metadata.n_elems;
  const unsigned int &tid = threadIdx.x;

  besthea::onthefly::quadrature_wrapper_changing_regular_raw quadr_changing;

  const lo &i_tst = i_tst_begin + blockIdx.x;
  const lo &row = i_tst;

  for(lo i = threadIdx.x; i < n_elems; i += blockDim.x) {
    vals_curr[tid] = 0;
    vals_next[tid] = 0;
    __syncthreads();

    lo curr_active_threads = (i >= (n_elems / blockDim.x) * blockDim.x) ? (n_elems % blockDim.x) : blockDim.x;

    for(lo diag = 0; diag < n_blocks; diag++) {
      // each thread calculates value corresponding to its col
      {
        lo &i_trl = i;
        vals_prev[tid] = vals_curr[tid];
        vals_curr[tid] = vals_next[tid];
        d_get_values_regular_sl_p0_p0(&vals_next[tid], diag+1, i_tst, i_trl, quadr_changing);
        matrix_vals[tid] = ((i_tst == i_trl) ? (0) : (-vals_prev[tid] + 2*vals_curr[tid] - vals_next[tid])); // singular will be done by the cpu
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
            sc x_val = x[block_col * ld_x + col];
            y_val += matrix_vals[j] * x_val;
          }
          y_val *= alpha;
          y_perm[row * ld_y_perm + block_row] += y_val;
        }
        __syncthreads();
      }

    }

  }
  

}














template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_regular_gpu_begin( const block_vector_type & x, block_vector_type & y_perm, sc alpha, apply_regular_gpu_tmp_data & tmp_data ) const {

  lo n_elems = mesh_metadata.n_elems;
  lo n_blocks = mesh_metadata.n_temporal_elements;
  
  sc * &x_raw = tmp_data.x_raw;
  sc * &y_perm_raw = tmp_data.y_perm_raw;

  std::vector<lo> gpu_i_tst_begins(n_gpus+1);
  for(int gpu_idx = 0; gpu_idx <= n_gpus; gpu_idx++) {
    gpu_i_tst_begins[gpu_idx] = (n_elems * gpu_idx) / n_gpus;
  }
  std::vector<lo> gpu_n_tst_elems(n_gpus);
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    gpu_n_tst_elems[gpu_idx] = gpu_i_tst_begins[gpu_idx+1] - gpu_i_tst_begins[gpu_idx];
  }
    


  cudaMallocHost(&x_raw, x.size() * sizeof(*x_raw));
  cudaMallocHost(&y_perm_raw, y_perm.size() * sizeof(*y_perm_raw));
  // TODO: test if better direct copy from block_vector to gpu
  x.copy_to_raw(x_raw);
  
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    sc * &d_x = tmp_data.per_gpu_data[gpu_idx].d_x;
    sc * &d_y_perm = tmp_data.per_gpu_data[gpu_idx].d_y_perm;
    size_t &pitch_x = tmp_data.per_gpu_data[gpu_idx].pitch_x;
    size_t &pitch_y_perm = tmp_data.per_gpu_data[gpu_idx].pitch_y_perm;
    lo &ld_x = tmp_data.per_gpu_data[gpu_idx].ld_x;
    lo &ld_y_perm = tmp_data.per_gpu_data[gpu_idx].ld_y_perm;

    cudaSetDevice(gpu_idx);

    cudaMallocPitch(&d_x, &pitch_x, n_elems * sizeof(*d_x), n_blocks);
    cudaMallocPitch(&d_y_perm, &pitch_y_perm, n_blocks * sizeof(*d_y_perm), n_elems);
    ld_x = pitch_x / sizeof(*d_x); // assuming the pitch is a multiple of element size
    ld_y_perm = pitch_y_perm / sizeof(*d_y_perm);

    // TODO: async, streams
    cudaMemcpy2D(d_x, pitch_x, x_raw, n_elems * sizeof(*x_raw), n_elems * sizeof(*x_raw), n_blocks, cudaMemcpyHostToDevice);
    cudaMemset(d_y_perm, 0, pitch_y_perm * n_elems);



    int gridSize = gpu_n_tst_elems[gpu_idx];
    int blockSize = 256; // number of gpu threads per block
    int shmemSize = blockSize * sizeof(sc);

    apply_regular_sl_p0_p0<<< gridSize, blockSize, shmemSize >>>(d_x, ld_x, d_y_perm, ld_y_perm, alpha, gpu_i_tst_begins[gpu_idx]);
    //apply_regular_sl_p0_p0_ver2<<< gridSize, blockSize, 4*shmemSize >>>(d_x, ld_x, d_y_perm, ld_y_perm, alpha, gpu_i_tst_begins[gpu_idx]);

  }

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_regular_gpu_finish( block_vector_type & y_perm, apply_regular_gpu_tmp_data & tmp_data ) const {

  lo n_elems = mesh_metadata.n_elems;
  lo n_blocks = mesh_metadata.n_temporal_elements;

  sc * &x_raw = tmp_data.x_raw;
  sc * &y_perm_raw = tmp_data.y_perm_raw;

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    sc * &d_x = tmp_data.per_gpu_data[gpu_idx].d_x;
    sc * &d_y_perm = tmp_data.per_gpu_data[gpu_idx].d_y_perm;
    //size_t &pitch_x = tmp_data.per_gpu_data[gpu_idx].pitch_x;
    size_t &pitch_y_perm = tmp_data.per_gpu_data[gpu_idx].pitch_y_perm;
    //lo &ld_x = tmp_data.per_gpu_data[gpu_idx].ld_x;
    //lo &ld_y_perm = tmp_data.per_gpu_data[gpu_idx].ld_y_perm;

    cudaSetDevice(gpu_idx);

    cudaDeviceSynchronize();

    cudaMemcpy2D(y_perm_raw, n_blocks * sizeof(*y_perm_raw), d_y_perm, pitch_y_perm, n_blocks * sizeof(*y_perm_raw), n_elems, cudaMemcpyDeviceToHost);
    
    y_perm.add_from_raw(y_perm_raw);

    cudaFree(d_x);
    cudaFree(d_y_perm);
  }

  cudaFreeHost(x_raw);
  cudaFreeHost(y_perm_raw);

  // TODO: error checking

}














template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_regular_gpu_begin( const block_vector_type & x, block_vector_type & y_perm, sc alpha, apply_regular_gpu_tmp_data & tmp_data ) const {

  lo n_elems = mesh_metadata.n_elems;
  lo n_nodes = mesh_metadata.n_nodes;
  lo n_blocks = mesh_metadata.n_temporal_elements;
  
  sc * &x_raw = tmp_data.x_raw;
  sc * &y_perm_raw = tmp_data.y_perm_raw;

  std::vector<lo> gpu_i_tst_begins(n_gpus+1);
  for(int gpu_idx = 0; gpu_idx <= n_gpus; gpu_idx++) {
    gpu_i_tst_begins[gpu_idx] = (n_elems * gpu_idx) / n_gpus;
  }
  std::vector<lo> gpu_n_tst_elems(n_gpus);
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    gpu_n_tst_elems[gpu_idx] = gpu_i_tst_begins[gpu_idx+1] - gpu_i_tst_begins[gpu_idx];
  }



  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    sc * &d_x = tmp_data.per_gpu_data[gpu_idx].d_x;
    sc * &d_y_perm = tmp_data.per_gpu_data[gpu_idx].d_y_perm;
    size_t &pitch_x = tmp_data.per_gpu_data[gpu_idx].pitch_x;
    size_t &pitch_y_perm = tmp_data.per_gpu_data[gpu_idx].pitch_y_perm;
    lo &ld_x = tmp_data.per_gpu_data[gpu_idx].ld_x;
    lo &ld_y_perm = tmp_data.per_gpu_data[gpu_idx].ld_y_perm;

    cudaSetDevice(gpu_idx);

    cudaMallocHost(&x_raw, x.size() * sizeof(*x_raw));
    cudaMallocHost(&y_perm_raw, y_perm.size() * sizeof(*y_perm_raw));

    cudaMallocPitch(&d_x, &pitch_x, n_nodes * sizeof(*d_x), n_blocks);
    cudaMallocPitch(&d_y_perm, &pitch_y_perm, n_blocks * sizeof(*d_y_perm), n_elems);
    ld_x = pitch_x / sizeof(*d_x); // assuming the pitch is a multiple of element size
    ld_y_perm = pitch_y_perm / sizeof(*d_y_perm);

    x.copy_to_raw(x_raw);

    cudaMemcpy2D(d_x, pitch_x, x_raw, n_nodes * sizeof(*x_raw), n_nodes * sizeof(*x_raw), n_blocks, cudaMemcpyHostToDevice);
    cudaMemset(d_y_perm, 0, pitch_y_perm * n_elems);


      
    int gridSize = gpu_n_tst_elems[gpu_idx];
    int blockSize = 256; // number of gpu threads per block
    int shmemSize = blockSize * sizeof(sc);  

    apply_regular_dl_p0_p1<<< gridSize, blockSize, shmemSize >>>(d_x, ld_x, d_y_perm, ld_y_perm, alpha, gpu_i_tst_begins[gpu_idx]);
  }

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_regular_gpu_finish( block_vector_type & y_perm, apply_regular_gpu_tmp_data & tmp_data ) const {

  lo n_elems = mesh_metadata.n_elems;
  //lo n_nodes = mesh_metadata.n_nodes;
  lo n_blocks = mesh_metadata.n_temporal_elements;

  sc * &x_raw = tmp_data.x_raw;
  sc * &y_perm_raw = tmp_data.y_perm_raw;

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    sc * &d_x = tmp_data.per_gpu_data[gpu_idx].d_x;
    sc * &d_y_perm = tmp_data.per_gpu_data[gpu_idx].d_y_perm;
    //size_t &pitch_x = tmp_data.per_gpu_data[gpu_idx].pitch_x;
    size_t &pitch_y_perm = tmp_data.per_gpu_data[gpu_idx].pitch_y_perm;
    //lo &ld_x = tmp_data.per_gpu_data[gpu_idx].ld_x;
    //lo &ld_y_perm = tmp_data.per_gpu_data[gpu_idx].ld_y_perm;

    cudaSetDevice(gpu_idx);

    cudaDeviceSynchronize();

    cudaMemcpy2D(y_perm_raw, n_blocks * sizeof(*y_perm_raw), d_y_perm, pitch_y_perm, n_blocks * sizeof(*y_perm_raw), n_elems, cudaMemcpyDeviceToHost);

    y_perm.add_from_raw(y_perm_raw);

    cudaFree(d_x);
    cudaFree(d_y_perm);
  }

  cudaFreeHost(x_raw);
  cudaFreeHost(y_perm_raw);

}














template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_regular_gpu_begin( const block_vector_type & x, block_vector_type & y_perm, sc alpha, apply_regular_gpu_tmp_data & tmp_data ) const {

  lo n_elems = mesh_metadata.n_elems;
  lo n_nodes = mesh_metadata.n_nodes;
  lo n_blocks = mesh_metadata.n_temporal_elements;
  
  sc * &x_raw = tmp_data.x_raw;
  sc * &y_perm_raw = tmp_data.y_perm_raw;

  std::vector<lo> gpu_i_tst_begins(n_gpus+1);
  for(int gpu_idx = 0; gpu_idx <= n_gpus; gpu_idx++) {
    gpu_i_tst_begins[gpu_idx] = (n_elems * gpu_idx) / n_gpus;
  }
  std::vector<lo> gpu_n_tst_elems(n_gpus);
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    gpu_n_tst_elems[gpu_idx] = gpu_i_tst_begins[gpu_idx+1] - gpu_i_tst_begins[gpu_idx];
  }



  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    sc * &d_x = tmp_data.per_gpu_data[gpu_idx].d_x;
    sc * &d_y_perm = tmp_data.per_gpu_data[gpu_idx].d_y_perm;
    size_t &pitch_x = tmp_data.per_gpu_data[gpu_idx].pitch_x;
    size_t &pitch_y_perm = tmp_data.per_gpu_data[gpu_idx].pitch_y_perm;
    lo &ld_x = tmp_data.per_gpu_data[gpu_idx].ld_x;
    lo &ld_y_perm = tmp_data.per_gpu_data[gpu_idx].ld_y_perm;

    cudaSetDevice(gpu_idx);

    cudaMallocHost(&x_raw, x.size() * sizeof(*x_raw));
    cudaMallocHost(&y_perm_raw, y_perm.size() * sizeof(*y_perm_raw));

    cudaMallocPitch(&d_x, &pitch_x, n_nodes * sizeof(*d_x), n_blocks);
    cudaMallocPitch(&d_y_perm, &pitch_y_perm, n_blocks * sizeof(*d_y_perm), n_nodes);
    ld_x = pitch_x / sizeof(*d_x); // assuming the pitch is a multiple of element size
    ld_y_perm = pitch_y_perm / sizeof(*d_y_perm);

    x.copy_to_raw(x_raw);

    cudaMemcpy2D(d_x, pitch_x, x_raw, n_nodes * sizeof(*x_raw), n_nodes * sizeof(*x_raw), n_blocks, cudaMemcpyHostToDevice);
    cudaMemset(d_y_perm, 0, pitch_y_perm * n_nodes);



    int gridSize = gpu_n_tst_elems[gpu_idx];
    int blockSize = 256; // number of gpu threads per block

    apply_regular_hs_p1_p1<<< gridSize, blockSize >>>(d_x, ld_x, d_y_perm, ld_y_perm, alpha, gpu_i_tst_begins[gpu_idx]);
  }

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_regular_gpu_finish( block_vector_type & y_perm, apply_regular_gpu_tmp_data & tmp_data ) const {

  //lo n_elems = mesh_metadata.n_elems;
  lo n_nodes = mesh_metadata.n_nodes;
  lo n_blocks = mesh_metadata.n_temporal_elements;

  sc * &x_raw = tmp_data.x_raw;
  sc * &y_perm_raw = tmp_data.y_perm_raw;

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    sc * &d_x = tmp_data.per_gpu_data[gpu_idx].d_x;
    sc * &d_y_perm = tmp_data.per_gpu_data[gpu_idx].d_y_perm;
    //size_t &pitch_x = tmp_data.per_gpu_data[gpu_idx].pitch_x;
    size_t &pitch_y_perm = tmp_data.per_gpu_data[gpu_idx].pitch_y_perm;
    //lo &ld_x = tmp_data.per_gpu_data[gpu_idx].ld_x;
    //lo &ld_y_perm = tmp_data.per_gpu_data[gpu_idx].ld_y_perm;

    cudaSetDevice(gpu_idx);

    cudaDeviceSynchronize();

    cudaMemcpy2D(y_perm_raw, n_blocks * sizeof(*y_perm_raw), d_y_perm, pitch_y_perm, n_blocks * sizeof(*y_perm_raw), n_nodes, cudaMemcpyDeviceToHost);

    y_perm.add_from_raw(y_perm_raw);

    cudaFree(d_x);
    cudaFree(d_y_perm);
  }
  
  cudaFreeHost(x_raw);
  cudaFreeHost(y_perm_raw);

}















template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  apply( const block_vector_type & x, block_vector_type & y, bool trans, sc alpha, sc beta ) const {

  if(trans) {
    std::cerr << "I dont support trans matrices\n";
    return;
  }

  // TODO: try the x_perm gpu algorithm
  
  // permuting the vector y should prevent false sharing and improve data locality
  // permuting the vector x should improve data locality
  block_vector_type y_perm;
  block_vector_type x_perm;

  y_perm.copy_permute(y, beta);
  x_perm.copy_permute(x, alpha);

  besthea::onthefly::apply_regular_gpu_tmp_data tmp_data;
  tmp_data.per_gpu_data.resize(n_gpus);

  this->apply_regular_gpu_begin(x, y_perm, alpha, tmp_data);
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




