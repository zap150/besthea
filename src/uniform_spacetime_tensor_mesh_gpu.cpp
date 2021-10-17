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

#include "besthea/gpu_onthefly_helpers.h"

#include <cuda_runtime.h>
#include <exception>

besthea::mesh::uniform_spacetime_tensor_mesh_gpu::
  uniform_spacetime_tensor_mesh_gpu(
    const besthea::mesh::uniform_spacetime_tensor_mesh & orig_mesh ) {
  n_gpus = 0;
  cudaError_t errDc = cudaGetDeviceCount( &n_gpus );
  if ( errDc != cudaErrorNoDevice )
    CUDA_CHECK( errDc );

  if ( n_gpus == 0 || errDc == cudaErrorNoDevice ) {
    if ( besthea::settings::output_verbosity.warnings >= 1 ) {
      std::cerr << "BESTHEA Warning: Constructing GPU mesh, "
                   "but no cuda-capable GPUs were detected.\n";
    }
    n_gpus = 0;
  }

  metadata.timestep = orig_mesh.get_timestep( );
  metadata.n_temporal_elements = orig_mesh.get_n_temporal_elements( );
  metadata.n_elems = orig_mesh.get_spatial_surface_mesh( )->get_n_elements( );
  metadata.n_nodes = orig_mesh.get_spatial_surface_mesh( )->get_n_nodes( );

  per_gpu_data.resize( n_gpus );
  for ( int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++ ) {
    CUDA_CHECK( cudaSetDevice( gpu_idx ) );

    mesh_raw_data & curr_gpu_data = per_gpu_data[ gpu_idx ];

    CUDA_CHECK( cudaMalloc( &curr_gpu_data.d_element_areas,
      1 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_areas ) ) );
    CUDA_CHECK( cudaMalloc( &curr_gpu_data.d_node_coords,
      3 * metadata.n_nodes * sizeof( *curr_gpu_data.d_node_coords ) ) );
    CUDA_CHECK( cudaMalloc( &curr_gpu_data.d_element_nodes,
      3 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_nodes ) ) );
    CUDA_CHECK( cudaMalloc( &curr_gpu_data.d_element_normals,
      3 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_normals ) ) );

    CUDA_CHECK( cudaMemcpy( curr_gpu_data.d_element_areas,
      orig_mesh.get_spatial_surface_mesh( )->get_areas( ).data( ),
      1 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_areas ),
      cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( curr_gpu_data.d_node_coords,
      orig_mesh.get_spatial_surface_mesh( )->get_nodes( ).data( ),
      3 * metadata.n_nodes * sizeof( *curr_gpu_data.d_node_coords ),
      cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( curr_gpu_data.d_element_nodes,
      orig_mesh.get_spatial_surface_mesh( )->get_elements( ).data( ),
      3 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_nodes ),
      cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( curr_gpu_data.d_element_normals,
      orig_mesh.get_spatial_surface_mesh( )->get_normals( ).data( ),
      3 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_normals ),
      cudaMemcpyHostToDevice ) );
  }
}

besthea::mesh::uniform_spacetime_tensor_mesh_gpu::
  ~uniform_spacetime_tensor_mesh_gpu( ) {
  free( );
}

void besthea::mesh::uniform_spacetime_tensor_mesh_gpu::free( ) {
  for ( unsigned int gpu_idx = 0; gpu_idx < per_gpu_data.size( ); gpu_idx++ ) {
    CUDA_CHECK( cudaSetDevice( gpu_idx ) );

    mesh_raw_data & curr_gpu_data = per_gpu_data[ gpu_idx ];

    CUDA_CHECK( cudaFree( curr_gpu_data.d_element_areas ) );
    CUDA_CHECK( cudaFree( curr_gpu_data.d_node_coords ) );
    CUDA_CHECK( cudaFree( curr_gpu_data.d_element_nodes ) );
    CUDA_CHECK( cudaFree( curr_gpu_data.d_element_normals ) );
  }
  per_gpu_data.clear( );
}
