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

/** @file uniform_spacetime_tensor_mesh_gpu.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_GPU_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_GPU_H_

#include <vector>

#include "besthea/settings.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"



namespace besthea::mesh {
  class uniform_spacetime_tensor_mesh_gpu;
}



/**
 *  Class representing uniform spacetime mesh resident in the GPU memory.
 */
class besthea::mesh::uniform_spacetime_tensor_mesh_gpu {
public:
  /**
   *  Struct containing GPU-resident space mesh raw data.
   */
  struct mesh_raw_data {
    sc * d_element_areas;
    sc * d_node_coords; // XYZXYZXYZXYZ...
    lo * d_element_nodes; // 123123123123...
    sc * d_element_normals; // XYZXYZXYZXYZ
    mesh_raw_data() : d_element_areas(nullptr), d_node_coords(nullptr), d_element_nodes(nullptr), d_element_normals(nullptr) { }
  };

  /**
   *  Struct containing spacetime mesh metadata.
   */
  struct mesh_raw_metadata {
    sc timestep;
    lo n_temporal_elements;
    lo n_elems;
    lo n_nodes;
  };

public:
  /**
   * Constructor. Creates this instance and copies necessary data tu GPU memory.
   * @param[in] orig_mesh The original mesh.
   */
  uniform_spacetime_tensor_mesh_gpu(const besthea::mesh::uniform_spacetime_tensor_mesh & orig_mesh);
  
  /**
   * Destructor.
   */
  ~uniform_spacetime_tensor_mesh_gpu();
  
  /**
   * Returns metadata structure holding information about this mesh.
   */
  const mesh_raw_metadata & get_metadata() const { return metadata; }
  
  /**
   * Returns vector of structures holding pointers to data on GPUs.
   */
  const std::vector<mesh_raw_data> & get_per_gpu_data() const { return per_gpu_data; }
  
  /**
   * Returns the used number of GPUs.
   */
  int get_n_gpus() const { return n_gpus; }

private:
  /**
   * Frees the allocated GPU memory.
   */
  void free();  
    
private:
  mesh_raw_metadata metadata; //!< Metadata about this mesh.
  std::vector<mesh_raw_data> per_gpu_data; //!< Pointers to GPU-resident data.
  int n_gpus; //!< Number of used GPUs.
};




#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_GPU_H_ */
