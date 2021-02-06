
#include "besthea/uniform_spacetime_be_onthefly_matrix_gpu.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"

#include <iostream>
#include <cuda_runtime.h>



__constant__ __device__ besthea::onthefly::uniform_spacetime_tensor_mesh_raw c_mesh; // c for constant
__constant__ __device__ besthea::onthefly::quadrature_wrapper_readonly_regular_raw c_my_quadrature;
__constant__ __device__ besthea::onthefly::kernel_parameters c_kernel_params;





template< class kernel_type, class test_space_type, class trial_space_type >
besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::
  uniform_spacetime_be_onthefly_matrix_gpu( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular )
  : uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>(kernel, test_space, trial_space, order_singular, order_regular) {

  // quadrature inited in base class constructor

  besthea::mesh::triangular_surface_mesh * mesh = this->_test_space->get_mesh()->get_spatial_surface_mesh();
  mesh_raw.n_temporal_elements = this->_test_space->get_mesh()->get_n_temporal_elements();
  mesh_raw.timestep = this->_test_space->get_mesh()->get_timestep();
  mesh_raw.surf_mesh.n_elems = mesh->get_n_elements();
  mesh_raw.surf_mesh.n_nodes = mesh->get_n_nodes();

  cudaMalloc(&mesh_raw.surf_mesh.d_element_areas,       mesh_raw.surf_mesh.n_elems * sizeof(*mesh_raw.surf_mesh.d_element_areas));
  cudaMalloc(&mesh_raw.surf_mesh.d_node_coords,     3 * mesh_raw.surf_mesh.n_nodes * sizeof(*mesh_raw.surf_mesh.d_node_coords));
  cudaMalloc(&mesh_raw.surf_mesh.d_element_nodes,   3 * mesh_raw.surf_mesh.n_elems * sizeof(*mesh_raw.surf_mesh.d_element_nodes));
  cudaMalloc(&mesh_raw.surf_mesh.d_element_normals, 3 * mesh_raw.surf_mesh.n_elems * sizeof(*mesh_raw.surf_mesh.d_element_normals));

  cudaMemcpy(mesh_raw.surf_mesh.d_element_areas,   mesh->get_all_areas().data(),        mesh_raw.surf_mesh.n_elems * sizeof(*mesh_raw.surf_mesh.d_element_areas), cudaMemcpyHostToDevice);
  cudaMemcpy(mesh_raw.surf_mesh.d_node_coords,     mesh->get_all_nodes().data(),    3 * mesh_raw.surf_mesh.n_nodes * sizeof(*mesh_raw.surf_mesh.d_node_coords),   cudaMemcpyHostToDevice);
  cudaMemcpy(mesh_raw.surf_mesh.d_element_nodes,   mesh->get_all_elements().data(), 3 * mesh_raw.surf_mesh.n_elems * sizeof(*mesh_raw.surf_mesh.d_element_nodes), cudaMemcpyHostToDevice);
  cudaMemcpy(mesh_raw.surf_mesh.d_element_normals, mesh->get_all_normals().data(),  3 * mesh_raw.surf_mesh.n_elems * sizeof(*mesh_raw.surf_mesh.d_element_normals), cudaMemcpyHostToDevice);

  init_gpu_constant_memory();

}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::~uniform_spacetime_be_onthefly_matrix_gpu( ) {

  cudaFree(mesh_raw.surf_mesh.d_element_areas);
  cudaFree(mesh_raw.surf_mesh.d_node_coords);
  cudaFree(mesh_raw.surf_mesh.d_element_nodes);
  cudaFree(mesh_raw.surf_mesh.d_element_normals);

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
  cudaMemcpyToSymbol(c_my_quadrature, &my_quadr_tmp, sizeof(my_quadr_tmp));



  cudaMemcpyToSymbol(c_mesh, &this->mesh_raw, sizeof(this->mesh_raw));



  besthea::onthefly::kernel_parameters kernel_params_tmp;
  sc alpha = this->_kernel->get_alpha();
  kernel_params_tmp.alpha = alpha;
  kernel_params_tmp.sqrt_alpha = std::sqrt(alpha);
  kernel_params_tmp.alpha_2 = alpha * alpha;
  kernel_params_tmp.pi = M_PI;
  kernel_params_tmp.sqrt_pi = std::sqrt(M_PI);
  cudaMemcpyToSymbol(c_kernel_params, &kernel_params_tmp, sizeof(kernel_params_tmp));

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

  sc norm = sqrt( xy1 * xy1 + xy2 * xy2 + xy3 * xy3 );
  sc &sqrt_d = ttau_sqrt;

  besthea::onthefly::kernel_parameters &kp = c_kernel_params;
  constexpr sc _two = 2.0;
  constexpr sc _four = 4.0;
  constexpr sc _eight = 8.0;

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

  sc norm2 = xy1 * xy1 + xy2 * xy2 + xy3 * xy3;
  sc norm = sqrt( norm2 );
  sc dot = xy1 * ny[ 0 ] + xy2 * ny[ 1 ] + xy3 * ny[ 2 ];
  sc &sqrt_d = ttau_sqrt;

  besthea::onthefly::kernel_parameters &kp = c_kernel_params;
  constexpr sc _one = 1.0;
  constexpr sc _two = 2.0;
  constexpr sc _four = 4.0;

  //  ttau > 0, norm > 0
  sc value = -dot / ( _four * kp.pi * norm )
    * ( ( _one / ( _two * kp.alpha ) - ttau / norm2 )
        * erf( norm / ( _two * sqrt_d * kp.sqrt_alpha ) )
      + sqrt_d / ( kp.sqrt_pi * kp.sqrt_alpha * norm )
        * exp( -norm2 / ( _four * kp.alpha * ttau ) ) );

  return value;
}














__device__ void d_get_values_regular_sl_p0_p0(sc * values_out, lo delta, lo i_test, lo i_trial,
  besthea::onthefly::quadrature_wrapper_changing_regular_raw & quadr_changing) {
  
  sc timestep = c_mesh.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const lo * tst_elem_nodes = c_mesh.surf_mesh.d_element_nodes + 3 * i_test;
  const lo * trl_elem_nodes = c_mesh.surf_mesh.d_element_nodes + 3 * i_trial;

  const sc * x1 = c_mesh.surf_mesh.d_node_coords + 3 * tst_elem_nodes[0];
  const sc * x2 = c_mesh.surf_mesh.d_node_coords + 3 * tst_elem_nodes[1];
  const sc * x3 = c_mesh.surf_mesh.d_node_coords + 3 * tst_elem_nodes[2];
  const sc * y1 = c_mesh.surf_mesh.d_node_coords + 3 * trl_elem_nodes[0];
  const sc * y2 = c_mesh.surf_mesh.d_node_coords + 3 * trl_elem_nodes[1];
  const sc * y3 = c_mesh.surf_mesh.d_node_coords + 3 * trl_elem_nodes[2];

  const sc test_area = c_mesh.surf_mesh.d_element_areas[i_test];
  const sc trial_area = c_mesh.surf_mesh.d_element_areas[i_trial];

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
    
  sc timestep = c_mesh.timestep;

  sc ttau = timestep * delta;
  sc sqrt_ttau = sqrt(ttau);

  const lo * tst_elem_nodes = c_mesh.surf_mesh.d_element_nodes + 3 * i_test;
  const lo * trl_elem_nodes = c_mesh.surf_mesh.d_element_nodes + 3 * i_trial;

  const sc * x1 = c_mesh.surf_mesh.d_node_coords + 3 * tst_elem_nodes[0];
  const sc * x2 = c_mesh.surf_mesh.d_node_coords + 3 * tst_elem_nodes[1];
  const sc * x3 = c_mesh.surf_mesh.d_node_coords + 3 * tst_elem_nodes[2];
  const sc * y1 = c_mesh.surf_mesh.d_node_coords + 3 * trl_elem_nodes[0];
  const sc * y2 = c_mesh.surf_mesh.d_node_coords + 3 * trl_elem_nodes[1];
  const sc * y3 = c_mesh.surf_mesh.d_node_coords + 3 * trl_elem_nodes[2];
  const sc * ny = c_mesh.surf_mesh.d_element_normals + 3 * i_trial;

  const sc test_area = c_mesh.surf_mesh.d_element_areas[i_test];
  const sc trial_area = c_mesh.surf_mesh.d_element_areas[i_trial];

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














__global__ void apply_regular_sl_p0_p0(const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha) {

  // each block handles one test element (inner row of a block matrix)
  // each thread handles one (or more if not enough threads) entry of the row (trial element), and loops through all the blocks

  extern __shared__ sc shmem_y_vals[]; // requires shared memory size (in bytes) to be specified while calling this kernel

  lo &n_blocks = c_mesh.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const unsigned int &tid = threadIdx.x;
  lo &n_elems = c_mesh.surf_mesh.n_elems;

  shmem_y_vals[tid] = 0;
  __syncthreads();

  sc matrix_val;
  sc val_prev;
  sc val_curr;
  sc val_next;

  besthea::onthefly::quadrature_wrapper_changing_regular_raw quadr_changing;

  const lo &i_tst = blockIdx.x;

  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {

    const lo &row = i_tst;
    const lo &col = i_trl;

    val_prev = 0; // delta=0 and 0s wiil be done by the cpu
    val_curr = 0;
    d_get_values_regular_sl_p0_p0(&val_next, 1, i_tst, i_trl, quadr_changing);
    
    matrix_val = ((i_tst == i_trl) ? (0) : (val_prev + val_curr - val_next)); // singular will be done by the cpu

    for (lo block = 0; block < n_blocks; block++) {
      shmem_y_vals[tid] = matrix_val * x[ ld_x * block + col ];
      __syncthreads();
      d_reduce_sum(shmem_y_vals, &y_perm[row * ld_y_perm + block], alpha);
    }

    for (lo diag = 1; diag < n_blocks; diag++) {
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



__global__ void apply_regular_dl_p0_p1(const sc * x, lo ld_x, sc * y_perm, lo ld_y_perm, sc alpha) {

  // each block handles one test element (inner row of a block matrix)
  // each thread handles one (or more if not enough threads) entry of the row (trial element), and loops through all the blocks

  extern __shared__ sc shmem_y_vals[]; // requires shared memory size (in bytes) to be specified while calling this kernel

  lo &n_blocks = c_mesh.n_temporal_elements; // number of blocks of matrix, not gpu threadblocks
  const unsigned int &tid = threadIdx.x;
  lo &n_elems = c_mesh.surf_mesh.n_elems;

  shmem_y_vals[tid] = 0;
  __syncthreads();

  sc matrix_vals[3] = {0,0,0};
  sc vals_prev[3];
  sc vals_curr[3];
  sc vals_next[3];

  besthea::onthefly::quadrature_wrapper_changing_regular_raw quadr_changing;

  const lo &i_tst = blockIdx.x;

  for (lo i_trl = threadIdx.x; i_trl < n_elems; i_trl += blockDim.x) {

    const lo &row = i_tst;
    const lo * cols = c_mesh.surf_mesh.d_element_nodes + 3 * i_trl;

    vals_prev[0] = 0;   vals_prev[1] = 0;   vals_prev[2] = 0; // delta=0 and 0s wiil be done by the cpu
    vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
    d_get_values_regular_dl_p0_p1(vals_next, 1, i_tst, i_trl, quadr_changing);
    
    matrix_vals[0] = ((i_tst == i_trl) ? (0) : (-vals_next[0])); // singular will be done by the cpu
    matrix_vals[1] = ((i_tst == i_trl) ? (0) : (-vals_next[1]));
    matrix_vals[2] = ((i_tst == i_trl) ? (0) : (-vals_next[2]));

    for (lo block = 0; block < n_blocks; block++) {
      shmem_y_vals[tid]  = matrix_vals[0] * x[ ld_x * block + cols[0]];
      shmem_y_vals[tid] += matrix_vals[1] * x[ ld_x * block + cols[1]];
      shmem_y_vals[tid] += matrix_vals[2] * x[ ld_x * block + cols[2]];
      __syncthreads();
      d_reduce_sum(shmem_y_vals, &y_perm[row * ld_y_perm + block], alpha);
    }

    for (lo diag = 1; diag < n_blocks; diag++) {
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















template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_regular( const block_vector_type & x, block_vector_type & y_perm, sc alpha ) const {

  lo n_elems = mesh_raw.surf_mesh.n_elems; // size of block of everything here
  lo n_blocks = mesh_raw.n_temporal_elements;

  
  sc *x_raw;
  sc *y_perm_raw;
  sc *d_x;
  sc *d_y_perm;
  size_t pitch_x, pitch_y_perm; // pitch in bytes
  lo ld_x, ld_y_perm; // leading dimension in elements
  int gridSize = n_elems;
  int blockSize = 256; // number of gpu threads per block
  int shmemSize = blockSize * sizeof(sc);

#pragma omp parallel
  {

#pragma omp master
    { // cuda computations preparation and start

      cudaMallocHost(&x_raw, x.size() * sizeof(*x_raw));
      cudaMallocHost(&y_perm_raw, y_perm.size() * sizeof(*y_perm_raw));

      cudaMallocPitch(&d_x, &pitch_x, n_elems * sizeof(*d_x), n_blocks);
      cudaMallocPitch(&d_y_perm, &pitch_y_perm, n_blocks * sizeof(*d_y_perm), n_elems);
      ld_x = pitch_x / sizeof(*d_x); // assuming the pitch is a multiple of element size
      ld_y_perm = pitch_y_perm / sizeof(*d_y_perm);

      // TODO: test if better direct copy from block_vector to gpu
      x.copy_to_raw(x_raw);

      // TODO: async, streams
      cudaMemcpy2D(d_x, pitch_x, x_raw, n_elems * sizeof(*x_raw), n_elems * sizeof(*x_raw), n_blocks, cudaMemcpyHostToDevice);
      cudaMemset(d_y_perm, 0, pitch_y_perm * n_elems);

      apply_regular_sl_p0_p0<<< gridSize, blockSize, shmemSize >>>(d_x, ld_x, d_y_perm, ld_y_perm, alpha);

    }

    // TODO: pute cpu computations for singular and delta0 here


#pragma omp single
    {
      cudaDeviceSynchronize();

      cudaMemcpy2D(y_perm_raw, n_blocks * sizeof(*y_perm_raw), d_y_perm, pitch_y_perm, n_blocks * sizeof(*y_perm_raw), n_elems, cudaMemcpyDeviceToHost);

      cudaFreeHost(x_raw);
      cudaFree(d_x);
      cudaFree(d_y_perm);
    }

  }

  y_perm.add_from_raw(y_perm_raw);

  cudaFreeHost(y_perm_raw);


  cudaError_t err = cudaGetLastError();
  if(err != 0) {
    std::cout << "Cuda error " << err << ": " << cudaGetErrorString(err) << "\n";
  }


  return;

}





template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_regular( const block_vector_type & x, block_vector_type & y_perm, sc alpha ) const {

    lo n_elems = mesh_raw.surf_mesh.n_elems;
    lo n_nodes = mesh_raw.surf_mesh.n_nodes;
    lo n_blocks = mesh_raw.n_temporal_elements;
    
    sc *x_raw;
    sc *y_perm_raw;
    sc *d_x;
    sc *d_y_perm;
    size_t pitch_x, pitch_y_perm; // pitch in bytes
    lo ld_x, ld_y_perm; // leading dimension in elements
    int gridSize = n_elems;
    int blockSize = 256; // number of gpu threads per block
    int shmemSize = blockSize * sizeof(sc);
  
  #pragma omp parallel
    {
  
  #pragma omp master
      { // cuda computations preparation and start
  
        cudaMallocHost(&x_raw, x.size() * sizeof(*x_raw));
        cudaMallocHost(&y_perm_raw, y_perm.size() * sizeof(*y_perm_raw));
  
        cudaMallocPitch(&d_x, &pitch_x, n_nodes * sizeof(*d_x), n_blocks);
        cudaMallocPitch(&d_y_perm, &pitch_y_perm, n_blocks * sizeof(*d_y_perm), n_elems);
        ld_x = pitch_x / sizeof(*d_x); // assuming the pitch is a multiple of element size
        ld_y_perm = pitch_y_perm / sizeof(*d_y_perm);
  
        // TODO: test if better direct copy from block_vector to gpu
        x.copy_to_raw(x_raw);
  
        // TODO: async, streams
        cudaMemcpy2D(d_x, pitch_x, x_raw, n_nodes * sizeof(*x_raw), n_nodes * sizeof(*x_raw), n_blocks, cudaMemcpyHostToDevice);
        cudaMemset(d_y_perm, 0, pitch_y_perm * n_elems);
  
        apply_regular_dl_p0_p1<<< gridSize, blockSize, shmemSize >>>(d_x, ld_x, d_y_perm, ld_y_perm, alpha);
  
      }
  
      // TODO: pute cpu computations for singular and delta0 here
  
  
  #pragma omp single
      {
        cudaDeviceSynchronize();
  
        cudaMemcpy2D(y_perm_raw, n_blocks * sizeof(*y_perm_raw), d_y_perm, pitch_y_perm, n_blocks * sizeof(*y_perm_raw), n_elems, cudaMemcpyDeviceToHost);
  
        cudaFreeHost(x_raw);
        cudaFree(d_x);
        cudaFree(d_y_perm);
      }
  
    }
  
    y_perm.add_from_raw(y_perm_raw);
  
    cudaFreeHost(y_perm_raw);
  
  
    cudaError_t err = cudaGetLastError();
    if(err != 0) {
      std::cout << "Cuda error " << err << ": " << cudaGetErrorString(err) << "\n";
    }
  
  
    return;

}










template class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
  

template class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;




