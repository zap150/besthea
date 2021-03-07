
/** @file gpu_onthefly_helpers.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_
#define INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_

#include <cmath>

#include "besthea/settings.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"



namespace besthea::onthefly {
  template<int quadr_order>
  struct quadrature_reference_raw;

  template<int quadr_order>
  struct quadrature_nodes_raw;

  struct mesh_raw_data;

  struct mesh_raw_metadata;
  
  struct heat_kernel_parameters;

  struct apply_regular_gpu_tmp_data;

  class gpu_uniform_spacetime_tensor_mesh;
  
#ifdef __NVCC__
  __host__ __device__
#endif
  constexpr int qo2qs(int quadr_order) {
    switch(quadr_order) {
      case 5:
        return 49;
      case 4:
        return 36;
      case 2:
        return 9;
      case 1:
      default:
        return 1;
    }
  }

  extern bool is_gpu_quadr_order5_initialized;
  extern bool is_gpu_quadr_order4_initialized;
  extern bool is_gpu_quadr_order2_initialized;
  extern bool is_gpu_quadr_order1_initialized;

}





template<int quadr_order>
struct besthea::onthefly::quadrature_reference_raw {
  sc _x1_ref[qo2qs(quadr_order)];
  sc _x2_ref[qo2qs(quadr_order)];
  sc _y1_ref[qo2qs(quadr_order)];
  sc _y2_ref[qo2qs(quadr_order)];
  sc _w[qo2qs(quadr_order)];
};





template<int quadr_order>
struct besthea::onthefly::quadrature_nodes_raw {
  sc xs[qo2qs(quadr_order)];
  sc ys[qo2qs(quadr_order)];
  sc zs[qo2qs(quadr_order)];
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
  sc * h_x; // raw data on host
  std::vector<sc*> h_y; // raw data on host
  std::vector<sc*> d_x, d_y; // raw data on device
  std::vector<size_t> pitch_x, pitch_y; // pitch in bytes
  std::vector<lo> ld_x, ld_y; // leading dimension in elements
  std::vector<lo> gpu_i_tst_begins;
  
  apply_regular_gpu_tmp_data();
  apply_regular_gpu_tmp_data(const apply_regular_gpu_tmp_data & that) = delete;
  ~apply_regular_gpu_tmp_data();
  void allocate(int n_gpus, lo x_block_count, lo x_size_of_block, lo y_block_count, lo y_size_of_block, lo n_elems);
  void free();
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



#endif /* INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_ */
