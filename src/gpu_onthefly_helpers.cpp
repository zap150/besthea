
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



void besthea::onthefly::apply_regular_gpu_tmp_data::allocate(int n_gpus,
  lo x_block_count, lo x_size_of_block, lo y_block_count, lo y_size_of_block) {

  h_y.resize(n_gpus);
  d_x.resize(n_gpus);
  d_y.resize(n_gpus);
  pitch_x.resize(n_gpus);
  pitch_y.resize(n_gpus);
  ld_x.resize(n_gpus);
  ld_y.resize(n_gpus);

  cudaMallocHost(&h_x, x_block_count * x_size_of_block * sizeof(sc));

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    cudaSetDevice(gpu_idx);

    cudaMallocHost(&h_y[gpu_idx], y_block_count * y_size_of_block * sizeof(sc));

    cudaMallocPitch(&d_x[gpu_idx], &pitch_x[gpu_idx], x_size_of_block * sizeof(sc), x_block_count);
    cudaMallocPitch(&d_y[gpu_idx], &pitch_y[gpu_idx], y_size_of_block * sizeof(sc), y_block_count);
    
    ld_x[gpu_idx] = pitch_x[gpu_idx] / sizeof(sc);
    ld_y[gpu_idx] = pitch_y[gpu_idx] / sizeof(sc);
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















besthea::onthefly::apply_load_distribution::apply_load_distribution(int n_gpus_, lo n_elems_, lo gpu_chunk_size_) {

  this->n_gpus = n_gpus_;
  this->n_elems = n_elems_;
  this->gpu_chunk_size = gpu_chunk_size_;
  this->min_cpu_tst_elems = std::min(n_elems, (lo)omp_get_max_threads());
  
  this->cpu_n_tst_elems = adjust_cpu_count(0.0);
  this->cpu_n_tst_elems_target = (double)this->cpu_n_tst_elems;

  this->gpu_i_tst_begins.resize(n_gpus+1);
  update_gpu_begins();

}



void besthea::onthefly::apply_load_distribution::update_gpu_begins() {

  lo gpus_n_tst_elems = n_elems - cpu_n_tst_elems;

  for(int gpu_idx = 0; gpu_idx <= n_gpus; gpu_idx++) {
    gpu_i_tst_begins[gpu_idx] = (gpus_n_tst_elems * gpu_idx) / n_gpus;
  }

}



void besthea::onthefly::apply_load_distribution::adapt(
  double cpu_time_const, double cpu_time_scaling, double gpu_time_const, double gpu_time_scaling, double inertia) {

  printf("Before adapt: %6ld %12.6f\n", cpu_n_tst_elems, cpu_n_tst_elems_target);

  lo gpu_n_tst_elems = n_elems - cpu_n_tst_elems;
  double cpu_n_elems_ideal;

  if(gpu_n_tst_elems > 0) {
    double time_per_elem_gpu = gpu_time_scaling / gpu_n_tst_elems;
    double time_per_elem_cpu = cpu_time_scaling / cpu_n_tst_elems;

    cpu_n_elems_ideal =
        (n_elems * time_per_elem_gpu - cpu_time_const + gpu_time_const)
        /
        (time_per_elem_cpu + time_per_elem_gpu);
  }
  else {
    cpu_n_elems_ideal = (double)n_elems;
  }
    
  this->cpu_n_tst_elems_target = cpu_n_tst_elems_target * inertia + cpu_n_elems_ideal * (1 - inertia);

  this->cpu_n_tst_elems = adjust_cpu_count(cpu_n_tst_elems_target);

  this->update_gpu_begins();

  printf("After adapt:  %6ld %12.6f\n", cpu_n_tst_elems, cpu_n_tst_elems_target);
  print();

}



lo besthea::onthefly::apply_load_distribution::adjust_cpu_count(double suggested) const {  
  suggested = std::max(suggested, 0.0);

  lo suggested_gpu_elems = (lo)std::ceil(n_elems - suggested);

  lo chunk = gpu_chunk_size * n_gpus;
  lo chunked_gpu_elems = ((suggested_gpu_elems - 1) / chunk + 1) * chunk;
  
  if( chunked_gpu_elems > n_elems - min_cpu_tst_elems)
    chunked_gpu_elems = ((n_elems - min_cpu_tst_elems) / chunk) * chunk;
  
  return n_elems - chunked_gpu_elems;
}



void besthea::onthefly::apply_load_distribution::print() {
  printf("Total %ld CPU %ld GPU %ld:", n_elems, get_cpu_count(), get_gpu_count_total());
  for(unsigned int i = 0; i < gpu_i_tst_begins.size(); i++)
    printf(" %3ld", gpu_i_tst_begins[i]);
  printf("\n");
}
















besthea::onthefly::timer_collection::timer_collection(int n_gpus) {
  gpu_all.resize(n_gpus);
  gpu_copyin.resize(n_gpus);
  gpu_compute.resize(n_gpus);
  gpu_copyout.resize(n_gpus);

  for(int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    cudaSetDevice(gpu_idx);
    gpu_all[gpu_idx].init(gpu_idx, 0);
    gpu_copyin[gpu_idx].init(gpu_idx, 0);
    gpu_compute[gpu_idx].init(gpu_idx, 0);
    gpu_copyout[gpu_idx].init(gpu_idx, 0);
  }
}



void besthea::onthefly::timer_collection::print_all() {
  printf("gpu_copyin:   ");
  print_timers(gpu_copyin);
  printf("gpu_compute:  ");
  print_timers(gpu_compute);
  printf("gpu_copyout:  ");
  print_timers(gpu_copyout);
  printf("gpu_all:      ");
  print_timers(gpu_all);
  printf("cpu_scalein:  %10.6f\n", cpu_scalein.get_time());
  printf("cpu_regular:  %10.6f\n", cpu_regular.get_time());
  printf("cpu_singular: %10.6f\n", cpu_singular.get_time());
  printf("cpu_delta0:   %10.6f\n", cpu_delta0.get_time());
  printf("cpu_all:      %10.6f\n", cpu_all.get_time());
  printf("gpu_max:      %10.6f\n", get_gpu_time_all());
  printf("combined:     %10.6f\n", combined.get_time());
}



void besthea::onthefly::timer_collection::print_timers(std::vector<besthea::tools::time_measurer_cuda> & timers) {
  for(unsigned int i = 0; i < timers.size(); i++) {
    printf("%10.6f  ", timers[i].get_time());
  }
  printf("\n");
}



double besthea::onthefly::timer_collection::get_cpu_time_const() {
  return cpu_singular.get_time() + cpu_delta0.get_time();
}

double besthea::onthefly::timer_collection::get_cpu_time_scaling() {
  return cpu_regular.get_time();
}

double besthea::onthefly::timer_collection::get_gpu_time_const() {
  double max_time_copyin = -1;
  for(unsigned int i = 0; i < gpu_copyin.size(); i++) {
    max_time_copyin = std::max(max_time_copyin, gpu_copyin[i].get_time());
  }
  double max_time_copyout = -1;
  for(unsigned int i = 0; i < gpu_copyout.size(); i++) {
    max_time_copyout = std::max(max_time_copyout, gpu_copyout[i].get_time());
  }
  return max_time_copyin + max_time_copyout;
}

double besthea::onthefly::timer_collection::get_gpu_time_scaling() {
  double max_time_compute = -1;
  for(unsigned int i = 0; i < gpu_compute.size(); i++) {
    max_time_compute = std::max(max_time_compute, gpu_compute[i].get_time());
  }
  return max_time_compute;
}

double besthea::onthefly::timer_collection::get_gpu_time_all() {
  double max_time_gpu_all = -1;
  for(unsigned int i = 0; i < gpu_all.size(); i++) {
    max_time_gpu_all = std::max(max_time_gpu_all, gpu_all[i].get_time());
  }
  return max_time_gpu_all;
}






