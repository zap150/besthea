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

#include "besthea/uniform_spacetime_tensor_mesh_gpu.h"

#include <exception>
#include <cuda_runtime.h>





besthea::mesh::uniform_spacetime_tensor_mesh_gpu::
  uniform_spacetime_tensor_mesh_gpu(
    const besthea::mesh::uniform_spacetime_tensor_mesh & orig_mesh) {

  n_gpus = 0;
  cudaGetDeviceCount(&n_gpus);
  
  if(n_gpus == 0 || cudaGetLastError() == cudaErrorNoDevice) {
    if(besthea::settings::output_verbosity.warnings >= 1) {
      std::cerr << "BESTHEA Warning: Constructing GPU mesh, "
        "but no cuda-capable GPUs were detected.\n";
    }
    n_gpus = 0;
  }
  
  metadata.timestep = orig_mesh.get_timestep();
  metadata.n_temporal_elements = orig_mesh.get_n_temporal_elements();
  metadata.n_elems = orig_mesh.get_spatial_surface_mesh()->get_n_elements();
  metadata.n_nodes = orig_mesh.get_spatial_surface_mesh()->get_n_nodes();



  // convert aos->soa on CPU, see how long it takes, if makes sense doing it on gpu
  const std::vector<sc> & element_areas = orig_mesh.get_spatial_surface_mesh()->get_all_areas();
  const std::vector<sc> & node_coords_aos = orig_mesh.get_spatial_surface_mesh()->get_all_nodes();
  const std::vector<lo> & element_nodes_aos = orig_mesh.get_spatial_surface_mesh()->get_all_elements();
  const std::vector<sc> & element_normals_aos = orig_mesh.get_spatial_surface_mesh()->get_all_normals();
  std::vector<sc> node_coords_xs(metadata.n_nodes);
  std::vector<sc> node_coords_ys(metadata.n_nodes);
  std::vector<sc> node_coords_zs(metadata.n_nodes);
  std::vector<lo> element_nodes_e1(metadata.n_elems);
  std::vector<lo> element_nodes_e2(metadata.n_elems);
  std::vector<lo> element_nodes_e3(metadata.n_elems);
  std::vector<sc> element_normals_xs(metadata.n_elems);
  std::vector<sc> element_normals_ys(metadata.n_elems);
  std::vector<sc> element_normals_zs(metadata.n_elems);
  for(lo i = 0; i < metadata.n_nodes; i++) {
    node_coords_xs[i] = node_coords_aos[3 * i + 0];
    node_coords_ys[i] = node_coords_aos[3 * i + 1];
    node_coords_zs[i] = node_coords_aos[3 * i + 2];
  }
  for(lo i = 0; i < metadata.n_elems; i++) {
    element_nodes_e1[i] = element_nodes_aos[3 * i + 0];
    element_nodes_e2[i] = element_nodes_aos[3 * i + 1];
    element_nodes_e3[i] = element_nodes_aos[3 * i + 2];
  }
  for(lo i = 0; i < metadata.n_elems; i++) {
    element_normals_xs[i] = element_normals_aos[3 * i + 0];
    element_normals_ys[i] = element_normals_aos[3 * i + 1];
    element_normals_zs[i] = element_normals_aos[3 * i + 2];
  }



  per_gpu_data.resize(n_gpus);
  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {

    cudaSetDevice(gpu_idx);

    mesh_raw_data &curr_gpu_data = per_gpu_data[gpu_idx];

    cudaMalloc(&curr_gpu_data.d_element_areas,      metadata.n_elems * sizeof(*curr_gpu_data.d_element_areas));
    cudaMalloc(&curr_gpu_data.d_node_coords.xs,     metadata.n_nodes * sizeof(*curr_gpu_data.d_node_coords.xs));
    cudaMalloc(&curr_gpu_data.d_node_coords.ys,     metadata.n_nodes * sizeof(*curr_gpu_data.d_node_coords.ys));
    cudaMalloc(&curr_gpu_data.d_node_coords.zs,     metadata.n_nodes * sizeof(*curr_gpu_data.d_node_coords.zs));
    cudaMalloc(&curr_gpu_data.d_element_nodes.e1,   metadata.n_elems * sizeof(*curr_gpu_data.d_element_nodes.e1));
    cudaMalloc(&curr_gpu_data.d_element_nodes.e2,   metadata.n_elems * sizeof(*curr_gpu_data.d_element_nodes.e2));
    cudaMalloc(&curr_gpu_data.d_element_nodes.e3,   metadata.n_elems * sizeof(*curr_gpu_data.d_element_nodes.e3));
    cudaMalloc(&curr_gpu_data.d_element_normals.xs, metadata.n_elems * sizeof(*curr_gpu_data.d_element_normals.xs));
    cudaMalloc(&curr_gpu_data.d_element_normals.ys, metadata.n_elems * sizeof(*curr_gpu_data.d_element_normals.ys));
    cudaMalloc(&curr_gpu_data.d_element_normals.zs, metadata.n_elems * sizeof(*curr_gpu_data.d_element_normals.zs));

    cudaMemcpyAsync(curr_gpu_data.d_element_areas,      element_areas.data(),      metadata.n_elems * sizeof(*curr_gpu_data.d_element_areas),      cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_node_coords.xs,     node_coords_xs.data(),     metadata.n_nodes * sizeof(*curr_gpu_data.d_node_coords.xs),     cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_node_coords.ys,     node_coords_ys.data(),     metadata.n_nodes * sizeof(*curr_gpu_data.d_node_coords.ys),     cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_node_coords.zs,     node_coords_zs.data(),     metadata.n_nodes * sizeof(*curr_gpu_data.d_node_coords.zs),     cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_element_nodes.e1,   element_nodes_e1.data(),   metadata.n_elems * sizeof(*curr_gpu_data.d_element_nodes.e1),   cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_element_nodes.e2,   element_nodes_e2.data(),   metadata.n_elems * sizeof(*curr_gpu_data.d_element_nodes.e2),   cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_element_nodes.e3,   element_nodes_e3.data(),   metadata.n_elems * sizeof(*curr_gpu_data.d_element_nodes.e3),   cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_element_normals.xs, element_normals_xs.data(), metadata.n_elems * sizeof(*curr_gpu_data.d_element_normals.xs), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_element_normals.ys, element_normals_ys.data(), metadata.n_elems * sizeof(*curr_gpu_data.d_element_normals.ys), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(curr_gpu_data.d_element_normals.zs, element_normals_zs.data(), metadata.n_elems * sizeof(*curr_gpu_data.d_element_normals.zs), cudaMemcpyHostToDevice, 0);

  }
  


  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    
    cudaSetDevice(gpu_idx);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
      std::cerr << "BESTHEA Error: gpu mesh init, device "
        << gpu_idx << ", detected cuda error "
        << err << ": " << cudaGetErrorString(err) << ".\n";
      free();
      throw std::runtime_error("BESTHEA Exception: cuda error");
    }

  }

  if(n_gpus > 0) {
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
      std::cerr << "BESTHEA Error: gpu mesh init, detected cuda error "
        << err << ": " << cudaGetErrorString(err) << ".\n";
      free();
      throw std::runtime_error("BESTHEA Exception: cuda error");
    }
  }

}





besthea::mesh::uniform_spacetime_tensor_mesh_gpu::
  ~uniform_spacetime_tensor_mesh_gpu() {

  free();
}





void besthea::mesh::uniform_spacetime_tensor_mesh_gpu::free() {

  for(unsigned int gpu_idx = 0; gpu_idx < per_gpu_data.size(); gpu_idx++) {

    cudaSetDevice(gpu_idx);

    mesh_raw_data &curr_gpu_data = per_gpu_data[gpu_idx];

    cudaFree(curr_gpu_data.d_element_areas);
    cudaFree(curr_gpu_data.d_node_coords.xs);
    cudaFree(curr_gpu_data.d_node_coords.ys);
    cudaFree(curr_gpu_data.d_node_coords.zs);
    cudaFree(curr_gpu_data.d_element_nodes.e1);
    cudaFree(curr_gpu_data.d_element_nodes.e2);
    cudaFree(curr_gpu_data.d_element_nodes.e3);
    cudaFree(curr_gpu_data.d_element_normals.xs);
    cudaFree(curr_gpu_data.d_element_normals.ys);
    cudaFree(curr_gpu_data.d_element_normals.zs);
  }
  per_gpu_data.clear();

  if(n_gpus > 0) {
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
      std::cerr << "BESTHEA Error: gpu mesh free, detected cuda error "
        << err << ": " << cudaGetErrorString(err) << ".\n";
      throw std::runtime_error("BESTHEA Exception: cuda error");
    }
  }

}






