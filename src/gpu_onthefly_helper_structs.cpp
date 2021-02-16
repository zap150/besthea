
#include "besthea/gpu_onthefly_helper_structs.h"

#include <cuda_runtime.h>


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
