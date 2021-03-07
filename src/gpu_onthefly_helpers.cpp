
#include "besthea/gpu_onthefly_helpers.h"

#include <cuda_runtime.h>





bool besthea::onthefly::is_gpu_quadr_order5_initialized = false;
bool besthea::onthefly::is_gpu_quadr_order4_initialized = false;
bool besthea::onthefly::is_gpu_quadr_order2_initialized = false;
bool besthea::onthefly::is_gpu_quadr_order1_initialized = false;















besthea::onthefly::apply_regular_gpu_tmp_data::apply_regular_gpu_tmp_data() :
  h_x(nullptr) {
}



besthea::onthefly::apply_regular_gpu_tmp_data::~apply_regular_gpu_tmp_data() {
  free();
}



void besthea::onthefly::apply_regular_gpu_tmp_data::allocate(int n_gpus, lo x_block_count, lo x_size_of_block, lo y_block_count, lo y_size_of_block, lo n_elems) {

  h_y.resize(n_gpus);
  d_x.resize(n_gpus);
  d_y.resize(n_gpus);
  pitch_x.resize(n_gpus);
  pitch_y.resize(n_gpus);
  ld_x.resize(n_gpus);
  ld_y.resize(n_gpus);

  cudaMallocHost(&h_x, x_block_count * x_size_of_block * sizeof(sc));

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    cudaMallocHost(&h_y[gpu_idx], y_block_count * y_size_of_block * sizeof(sc));

    cudaMallocPitch(&d_x[gpu_idx], &pitch_x[gpu_idx], x_size_of_block * sizeof(sc), x_block_count);
    cudaMallocPitch(&d_y[gpu_idx], &pitch_y[gpu_idx], y_size_of_block * sizeof(sc), y_block_count);
    
    ld_x[gpu_idx] = pitch_x[gpu_idx] / sizeof(sc);
    ld_y[gpu_idx] = pitch_y[gpu_idx] / sizeof(sc);
  }


  gpu_i_tst_begins.resize(n_gpus+1);
  for(int gpu_idx = 0; gpu_idx <= n_gpus; gpu_idx++) {
    gpu_i_tst_begins[gpu_idx] = (n_elems * gpu_idx) / n_gpus;
  }

}


void besthea::onthefly::apply_regular_gpu_tmp_data::free() {

  if(h_x != nullptr) cudaFreeHost(h_x);
  for(unsigned int i = 0; i < h_y.size(); i++) cudaFreeHost(h_y[i]);
  for(unsigned int i = 0; i < d_x.size(); i++) cudaFree(d_x[i]);
  for(unsigned int i = 0; i < d_y.size(); i++) cudaFree(d_y[i]);

  h_x = nullptr;
  h_y.clear();
  d_x.clear();
  d_y.clear();
  pitch_x.clear();
  pitch_y.clear();
  ld_x.clear();
  ld_y.clear();
  gpu_i_tst_begins.clear();

}















besthea::onthefly::gpu_uniform_spacetime_tensor_mesh::gpu_uniform_spacetime_tensor_mesh(const besthea::mesh::uniform_spacetime_tensor_mesh & orig_mesh) {

  cudaGetDeviceCount(&n_gpus);
  // TODO: error messages when no cuda-capable gpu found
  
  metadata.timestep = orig_mesh.get_timestep();
  metadata.n_temporal_elements = orig_mesh.get_n_temporal_elements();
  metadata.n_elems = orig_mesh.get_spatial_surface_mesh()->get_n_elements();
  metadata.n_nodes = orig_mesh.get_spatial_surface_mesh()->get_n_nodes();

  per_gpu_data.resize(n_gpus);
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {

    cudaSetDevice(gpu_idx);

    mesh_raw_data &curr_gpu_data = per_gpu_data[gpu_idx];

    cudaMalloc(&curr_gpu_data.d_element_areas,       metadata.n_elems * sizeof(*curr_gpu_data.d_element_areas));
    cudaMalloc(&curr_gpu_data.d_node_coords,     3 * metadata.n_nodes * sizeof(*curr_gpu_data.d_node_coords));
    cudaMalloc(&curr_gpu_data.d_element_nodes,   3 * metadata.n_elems * sizeof(*curr_gpu_data.d_element_nodes));
    cudaMalloc(&curr_gpu_data.d_element_normals, 3 * metadata.n_elems * sizeof(*curr_gpu_data.d_element_normals));

    cudaMemcpy(curr_gpu_data.d_element_areas,   orig_mesh.get_spatial_surface_mesh()->get_all_areas().data(),        metadata.n_elems * sizeof(*curr_gpu_data.d_element_areas), cudaMemcpyHostToDevice);
    cudaMemcpy(curr_gpu_data.d_node_coords,     orig_mesh.get_spatial_surface_mesh()->get_all_nodes().data(),    3 * metadata.n_nodes * sizeof(*curr_gpu_data.d_node_coords),   cudaMemcpyHostToDevice);
    cudaMemcpy(curr_gpu_data.d_element_nodes,   orig_mesh.get_spatial_surface_mesh()->get_all_elements().data(), 3 * metadata.n_elems * sizeof(*curr_gpu_data.d_element_nodes), cudaMemcpyHostToDevice);
    cudaMemcpy(curr_gpu_data.d_element_normals, orig_mesh.get_spatial_surface_mesh()->get_all_normals().data(),  3 * metadata.n_elems * sizeof(*curr_gpu_data.d_element_normals), cudaMemcpyHostToDevice);
  }

}



besthea::onthefly::gpu_uniform_spacetime_tensor_mesh::~gpu_uniform_spacetime_tensor_mesh() {

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {

    cudaSetDevice(gpu_idx);

    mesh_raw_data &curr_gpu_data = per_gpu_data[gpu_idx];

    cudaFree(curr_gpu_data.d_element_areas);
    cudaFree(curr_gpu_data.d_node_coords);
    cudaFree(curr_gpu_data.d_element_nodes);
    cudaFree(curr_gpu_data.d_element_normals);
  }

}

