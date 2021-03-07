
/** @file gpu_onthefly_helpers.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_
#define INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_

#include <cmath>
#include <vector>

#include "besthea/settings.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"
#include "besthea/time_measurer.h"
#include "besthea/time_measurer_cuda.h"



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

  class apply_load_distribution;

  struct timer_collection;
  
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
  
  apply_regular_gpu_tmp_data();
  apply_regular_gpu_tmp_data(const apply_regular_gpu_tmp_data & that) = delete;
  ~apply_regular_gpu_tmp_data();
  void allocate(int n_gpus, lo x_block_count, lo x_size_of_block, lo y_block_count, lo y_size_of_block);
  void free();
  void print_times() const;
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





class besthea::onthefly::apply_load_distribution {
private:
  lo cpu_n_tst_elems;
  double cpu_n_tst_elems_target;
  std::vector<lo> gpu_i_tst_begins;
  int n_gpus;
  lo n_elems;
public:
  apply_load_distribution(int n_gpus, lo n_elems, lo init_cpu_n_tst_elems);
  void update_gpu_begins();
  void adapt(double cpu_time_sing_del0, double cpu_time_reg, double gpu_time, lo cpu_chunk_size, double inertia);
  lo get_cpu_begin() const { return 0; }
  lo get_cpu_end() const { return cpu_n_tst_elems; }
  lo get_cpu_count() const { return cpu_n_tst_elems; }
  lo get_gpu_begin(int gpu_idx) const { return gpu_i_tst_begins[gpu_idx]; }
  lo get_gpu_end(int gpu_idx) const { return gpu_i_tst_begins[gpu_idx+1]; }
  lo get_gpu_count(int gpu_idx) const { return get_gpu_end(gpu_idx) - get_gpu_begin(gpu_idx); }
};





struct besthea::onthefly::timer_collection {
  std::vector<besthea::tools::time_measurer_cuda> gpu_all;
  std::vector<besthea::tools::time_measurer_cuda> gpu_copyin;
  std::vector<besthea::tools::time_measurer_cuda> gpu_compute;
  std::vector<besthea::tools::time_measurer_cuda> gpu_copyout;
  besthea::tools::time_measurer cpu_scalein;
  besthea::tools::time_measurer cpu_regular;
  besthea::tools::time_measurer cpu_singular;
  besthea::tools::time_measurer cpu_delta0;
  besthea::tools::time_measurer cpu_all;
  besthea::tools::time_measurer combined;
  timer_collection(int n_gpus);
  void print_all();
  void print_one(std::vector<besthea::tools::time_measurer_cuda> & timers);
  double get_gpu_all_time();
};





#endif /* INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_ */
