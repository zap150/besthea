
/** @file gpu_onthefly_helper_structs.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPER_STRUCTS_H_
#define INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPER_STRUCTS_H_

#include <cmath>

#include "besthea/settings.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"



namespace besthea::onthefly {
  struct quadrature_reference_raw;
  struct quadrature_nodes_raw;
  struct mesh_raw_data;
  struct mesh_raw_metadata;
  struct heat_kernel_parameters;
  struct apply_regular_gpu_tmp_data;
  class gpu_uniform_spacetime_tensor_mesh;
}





struct besthea::onthefly::quadrature_reference_raw {
  sc _x1_ref[64];
  sc _x2_ref[64];
  sc _y1_ref[64];
  sc _y2_ref[64];
  sc _w[64];
  lo _size; // actual size
};

struct besthea::onthefly::quadrature_nodes_raw {
  sc xs[64];
  sc ys[64];
  sc zs[64];
};

struct besthea::onthefly::mesh_raw_data {
  sc * d_element_areas;
  sc * d_node_coords; // XYZXYZXYZXYZ...
  lo * d_element_nodes; // 123123123123...
  sc * d_element_normals; // XYZXYZXYZXYZ
};

struct besthea::onthefly::mesh_raw_metadata {
  sc timestep;
  lo n_temporal_elements;
  lo n_elems;
  lo n_nodes;
};

struct besthea::onthefly::heat_kernel_parameters {
  sc alpha;
  sc sqrt_alpha;
  sc alpha_2;
  sc pi;
  sc sqrt_pi;
  heat_kernel_parameters(sc alpha_) {
    this->alpha = alpha_;
    sqrt_alpha = std::sqrt(alpha_);
    alpha_2 = alpha_ * alpha_;
    pi = M_PI;
    sqrt_pi = std::sqrt(M_PI);
  }
};

struct besthea::onthefly::apply_regular_gpu_tmp_data {
  sc *d_x;
  sc *d_y;
  size_t pitch_x, pitch_y; // pitch in bytes
  lo ld_x, ld_y; // leading dimension in elements
};

class besthea::onthefly::gpu_uniform_spacetime_tensor_mesh {
private:
  mesh_raw_metadata metadata;
  std::vector<mesh_raw_data> per_gpu_data;
  int n_gpus;
public:
  gpu_uniform_spacetime_tensor_mesh(const besthea::mesh::uniform_spacetime_tensor_mesh & orig_mesh);
  ~gpu_uniform_spacetime_tensor_mesh();
  const mesh_raw_metadata & get_metadata() const { return metadata; }
  const std::vector<mesh_raw_data> & get_per_gpu_data() const { return per_gpu_data; }
  int get_n_gpus() const { return n_gpus; }

};



#endif /* INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPER_STRUCTS_H_ */
