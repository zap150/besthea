#include <cstdlib>
#include <filesystem>
#include <cstdio>
#include "besthea/besthea.h"

using namespace besthea;
using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem::onthefly;
using namespace besthea::bem;
using namespace besthea::tools;



int main( int argc, char * argv[] ) {

  if(argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
    printf("Usage: ./matrix_vector_product finess_level gpu_alg_ver repetitions pre_repetitions quadr_order_reg quadr_order_sng\n");
    return 0;
  }
  
  time_measurer tm_init, tm_check;
  time_measurer tm_Vma, tm_Vmm, tm_Vfc, tm_Vfg;
  time_measurer tm_Kma, tm_Kmm, tm_Kfc, tm_Kfg;
  time_measurer tm_Ama, tm_Amm, tm_Afc, tm_Afg;
  time_measurer tm_Dma, tm_Dmm, tm_Dfc, tm_Dfg;
  
  std::string mesh_file_12 = "../../besthea/examples/mesh_files/cube_12.txt";
  std::string mesh_file_24 = "../../besthea/examples/mesh_files/cube_24.txt";
  sc heat_capacity_constant = 1.0;

  bool doMem    = true;
  bool doFlyCpu = true;
  bool doFlyGpu = true;
  bool doV = true;
  bool doK = true;
  bool doA = true;
  bool doD = true;

  bool printCheckErrors = false;
  
  besthea::settings::output_verbosity.timers = 2;
  besthea::settings::output_verbosity.onthefly_loadbalance = 2;
  
  lo finess_level = 3;
  int gpu_alg_ver = 1;
  int repetitions = 1;
  int pre_repetitions = 0;
  int quadr_order_sng = 4;
  int quadr_order_reg = 4;

  tm_init.start();

  if(argc > 1) finess_level = atoi(argv[1]);
  if(argc > 2) gpu_alg_ver = atoi(argv[2]);
  if(argc > 3) repetitions = atoi(argv[3]);
  if(argc > 4) pre_repetitions = atoi(argv[4]);
  if(argc > 5) quadr_order_reg = atoi(argv[5]);
  if(argc > 6) quadr_order_sng = atoi(argv[6]);

  int order_reg_V = quadr_order_reg;
  int order_reg_K = quadr_order_reg;
  int order_reg_A = quadr_order_reg;
  int order_reg_D = quadr_order_reg;
  int order_sng_V = quadr_order_sng;
  int order_sng_K = quadr_order_sng;
  int order_sng_A = quadr_order_sng;
  int order_sng_D = quadr_order_sng;

  //srand(time(nullptr));

  // finess_level   1  2   3   4   5    6    7    8     9
  // n_timesteps    2  4   8  16  32   64  128  256   512
  // n_space_elems 48 96 192 384 768 1536 3072 6144 12288  ...
  // orig_sp_elems 12 24  12  24  12   24   12   24    12
  // space_refine   1  1   2   2   3    3    4    4     5
  lo n_timesteps = std::exp2(finess_level);
  sc end_time = 1.0;
  std::string mesh_file;
  if(finess_level % 2 == 0) {
    mesh_file = mesh_file_24;
  } else {
    mesh_file = mesh_file_12;
  }
  int space_refine = (finess_level + 1) / 2;
  
  // load spatial mesh from file and refine
  triangular_surface_mesh space_mesh;
  space_mesh.load( mesh_file );
  space_mesh.refine( space_refine );

  // create spacetime mesh as a tensor product of spatial and temporal meshes
  uniform_spacetime_tensor_mesh spacetime_mesh( space_mesh, end_time, n_timesteps );
  uniform_spacetime_tensor_mesh_gpu gpu_spacetime_mesh(spacetime_mesh);

  // print some info
  spacetime_mesh.print_info();
  printf("Sqrt of spatial element area (~hx) %f\n", std::sqrt(space_mesh.area(0)));
  printf("Timestep length (ht) %f\n", end_time / n_timesteps);
  printf("hx^2/ht %f\n", space_mesh.area(0) / (end_time / n_timesteps));
  printf("Using quadrature order regular  %d\n", quadr_order_reg);
  printf("Using quadrature order singular %d\n", quadr_order_sng);
  printf("Using GPU algorithm version %d\n", gpu_alg_ver);

  // boundary element spaces
  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );

  // matrix preparation
  spacetime_heat_sl_kernel_antiderivative  kernel_v( heat_capacity_constant );
  spacetime_heat_dl_kernel_antiderivative  kernel_k( heat_capacity_constant );
  spacetime_heat_adl_kernel_antiderivative kernel_a( heat_capacity_constant );
  spacetime_heat_hs_kernel_antiderivative  kernel_d( heat_capacity_constant );
  block_lower_triangular_toeplitz_matrix V_mem; // single layer operator
  block_lower_triangular_toeplitz_matrix K_mem; // double layer operator
  block_lower_triangular_toeplitz_matrix A_mem; // adjoint double layer operator
  block_lower_triangular_toeplitz_matrix D_mem; // hypersingular operator
  uniform_spacetime_be_assembler         assembler_v(kernel_v, space_p0, space_p0,                     order_sng_V, order_reg_V);
  uniform_spacetime_be_matrix_onthefly_cpu V_fly_cpu(kernel_v, space_p0, space_p0,                     order_sng_V, order_reg_V);
  uniform_spacetime_be_matrix_onthefly_gpu V_fly_gpu(kernel_v, space_p0, space_p0, gpu_spacetime_mesh, order_sng_V, order_reg_V, gpu_alg_ver);
  uniform_spacetime_be_assembler         assembler_k(kernel_k, space_p0, space_p1,                     order_sng_K, order_reg_K);
  uniform_spacetime_be_matrix_onthefly_cpu K_fly_cpu(kernel_k, space_p0, space_p1,                     order_sng_K, order_reg_K);
  uniform_spacetime_be_matrix_onthefly_gpu K_fly_gpu(kernel_k, space_p0, space_p1, gpu_spacetime_mesh, order_sng_K, order_reg_K, gpu_alg_ver);
  uniform_spacetime_be_assembler         assembler_a(kernel_a, space_p1, space_p0,                     order_sng_A, order_reg_A);
  uniform_spacetime_be_matrix_onthefly_cpu A_fly_cpu(kernel_a, space_p1, space_p0,                     order_sng_A, order_reg_A);
  uniform_spacetime_be_matrix_onthefly_gpu A_fly_gpu(kernel_a, space_p1, space_p0, gpu_spacetime_mesh, order_sng_A, order_reg_A, gpu_alg_ver);
  uniform_spacetime_be_assembler         assembler_d(kernel_d, space_p1, space_p1,                     order_sng_D, order_reg_D);
  uniform_spacetime_be_matrix_onthefly_cpu D_fly_cpu(kernel_d, space_p1, space_p1,                     order_sng_D, order_reg_D);
  uniform_spacetime_be_matrix_onthefly_gpu D_fly_gpu(kernel_d, space_p1, space_p1, gpu_spacetime_mesh, order_sng_D, order_reg_D, gpu_alg_ver);

  // initialize vectors
  block_vector xV  (n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVm (n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVfc(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVfg(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      xV.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      yVm.set(b, i, 2);
      yVfc.set(b, i, 2);
      yVfg.set(b, i, 2);
    }
  }
  block_vector xK  (n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yKm (n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yKfc(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yKfg(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      xK.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      yKm.set(b, i, 2);
      yKfc.set(b, i, 2);
      yKfg.set(b, i, 2);
    }    
  }
  block_vector xA  (n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yAm (n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yAfc(n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yAfg(n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      xA.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      yAm.set(b, i, 2);
      yAfc.set(b, i, 2);
      yAfg.set(b, i, 2);
    }    
  }
  block_vector xD  (n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yDm (n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yDfc(n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yDfg(n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      xD.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      yDm.set(b, i, 2);
      yDfc.set(b, i, 2);
      yDfg.set(b, i, 2);
    }
  }

  sc alpha = 3;
  sc beta = 5;

  tm_init.stop();



  // all the assembly and multiplications

  if(doV) {
    if(doMem) {
      printf("V mem assembly\n");
      for(int i = 0; i < pre_repetitions; i++) { V_mem.clear(); assembler_v.assemble( V_mem ); }
      tm_Vma.start();
      for(int i = 0; i < repetitions; i++) { V_mem.clear(); assembler_v.assemble( V_mem ); }
      tm_Vma.stop();

      printf("V mem multiply\n");
      for(int i = 0; i < pre_repetitions; i++) V_mem.apply(xV, yVm, false, alpha, beta);
      tm_Vmm.start();
      for(int i = 0; i < repetitions; i++) V_mem.apply(xV, yVm, false, alpha, beta);
      tm_Vmm.stop();
    }
    if(doFlyCpu) {
      printf("V fly cpu\n");
      for(int i = 0; i < pre_repetitions; i++) V_fly_cpu.apply(xV, yVfc, false, alpha, beta);
      tm_Vfc.start();
      for(int i = 0; i < repetitions; i++) V_fly_cpu.apply(xV, yVfc, false, alpha, beta);
      tm_Vfc.stop();
    }
    if(doFlyGpu) {
      printf("V fly gpu\n");
      for(int i = 0; i < pre_repetitions; i++) V_fly_gpu.apply(xV, yVfg, false, alpha, beta);
      tm_Vfg.start();
      for(int i = 0; i < repetitions; i++) V_fly_gpu.apply(xV, yVfg, false, alpha, beta);
      tm_Vfg.stop();
    }
  }



  if(doK) {
    if(doMem) {
      printf("K mem assembly\n");
      for(int i = 0; i < pre_repetitions; i++) { K_mem.clear(); assembler_k.assemble( K_mem ); }
      tm_Kma.start();
      for(int i = 0; i < repetitions; i++) { K_mem.clear(); assembler_k.assemble( K_mem ); }
      tm_Kma.stop();
      
      printf("K mem multiply\n");
      for(int i = 0; i < pre_repetitions; i++) K_mem.apply(xK, yKm, false, alpha, beta);
      tm_Kmm.start();
      for(int i = 0; i < repetitions; i++) K_mem.apply(xK, yKm, false, alpha, beta);
      tm_Kmm.stop();
    }
    if(doFlyCpu) {
      printf("K fly cpu\n");
      for(int i = 0; i < pre_repetitions; i++) K_fly_cpu.apply(xK, yKfc, false, alpha, beta);
      tm_Kfc.start();
      for(int i = 0; i < repetitions; i++) K_fly_cpu.apply(xK, yKfc, false, alpha, beta);
      tm_Kfc.stop();
    }
    if(doFlyGpu) {
      printf("K fly gpu\n");
      for(int i = 0; i < pre_repetitions; i++) K_fly_gpu.apply(xK, yKfg, false, alpha, beta);
      tm_Kfg.start();
      for(int i = 0; i < repetitions; i++) K_fly_gpu.apply(xK, yKfg, false, alpha, beta);
      tm_Kfg.stop();
    }
  }


  // A and K are a little different by quadrature errors on elements with shared edge
  if(doA) {
    if(doMem) {
      printf("A mem assembly\n");
      for(int i = 0; i < pre_repetitions; i++) { A_mem.clear(); assembler_a.assemble( A_mem ); }
      tm_Ama.start();
      for(int i = 0; i < repetitions; i++) { A_mem.clear(); assembler_a.assemble( A_mem ); }
      tm_Ama.stop();
      
      printf("A mem multiply\n");
      for(int i = 0; i < pre_repetitions; i++) A_mem.apply(xA, yAm, false, alpha, beta);
      tm_Amm.start();
      for(int i = 0; i < repetitions; i++) A_mem.apply(xA, yAm, false, alpha, beta);
      tm_Amm.stop();
    }
    if(doFlyCpu) {
      printf("A fly cpu\n");
      for(int i = 0; i < pre_repetitions; i++) A_fly_cpu.apply(xA, yAfc, false, alpha, beta);
      tm_Afc.start();
      for(int i = 0; i < repetitions; i++) A_fly_cpu.apply(xA, yAfc, false, alpha, beta);
      tm_Afc.stop();
    }
    if(doFlyGpu) {
      printf("A fly gpu\n");
      for(int i = 0; i < pre_repetitions; i++) A_fly_gpu.apply(xA, yAfg, false, alpha, beta);
      tm_Afg.start();
      for(int i = 0; i < repetitions; i++) A_fly_gpu.apply(xA, yAfg, false, alpha, beta);
      tm_Afg.stop();
    }
  }



  if(doD) {
    if(doMem) {
      printf("D mem assembly\n");
      for(int i = 0; i < pre_repetitions; i++) { D_mem.clear(); assembler_d.assemble( D_mem ); }
      tm_Dma.start();
      for(int i = 0; i < repetitions; i++) { D_mem.clear(); assembler_d.assemble( D_mem ); }
      tm_Dma.stop();
      
      printf("D mem multiply\n");
      for(int i = 0; i < pre_repetitions; i++) D_mem.apply(xD, yDm, false, alpha, beta);
      tm_Dmm.start();
      for(int i = 0; i < repetitions; i++) D_mem.apply(xD, yDm, false, alpha, beta);
      tm_Dmm.stop();
    }
    if(doFlyCpu) {
      printf("D fly cpu\n");
      for(int i = 0; i < pre_repetitions; i++) D_fly_cpu.apply(xD, yDfc, false, alpha, beta);
      tm_Dfc.start();
      for(int i = 0; i < repetitions; i++) D_fly_cpu.apply(xD, yDfc, false, alpha, beta);
      tm_Dfc.stop();
    }
    if(doFlyGpu) {
      printf("D fly gpu\n");
      for(int i = 0; i < pre_repetitions; i++) D_fly_gpu.apply(xD, yDfg, false, alpha, beta);
      tm_Dfg.start();
      for(int i = 0; i < repetitions; i++) D_fly_gpu.apply(xD, yDfg, false, alpha, beta);
      tm_Dfg.stop();
    }
  }
  


  // checking the results

  tm_check.start();

  bool equalVc = true;
  bool equalVg = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yVm.get(b, i);
      sc vf = yVfc.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        if(printCheckErrors)
          printf("Vectors Vc dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalVc = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yVm.get(b, i);
      sc vf = yVfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        if(printCheckErrors)
          printf("Vectors Vg dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalVg = false;
      }
    }
  }

  bool equalKc = true;
  bool equalKg = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yKm.get(b, i);
      sc vf = yKfc.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        if(printCheckErrors)
          printf("Vectors Kc dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalKc = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yKm.get(b, i);
      sc vf = yKfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        if(printCheckErrors)
          printf("Vectors Kg dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalKg = false;
      }
    }
  }

  bool equalAc = true;
  bool equalAg = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      sc vm = yAm.get(b, i);
      sc vf = yAfc.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        if(printCheckErrors)
          printf("Vectors Ac dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalAc = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      sc vm = yAm.get(b, i);
      sc vf = yAfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        if(printCheckErrors)
          printf("Vectors Ag dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalAg = false;
      }
    }
  }

  bool equalDc = true;
  bool equalDg = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      sc vm = yDm.get(b, i);
      sc vf = yDfc.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        if(printCheckErrors)
          printf("Vectors Dc dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalDc = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      sc vm = yDm.get(b, i);
      sc vf = yDfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        if(printCheckErrors)
          printf("Vectors Dg dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalDg = false;
      }
    }
  }

  tm_check.stop();



  // print results
  printf("Vectors Vc are%s equal!\n", (equalVc ? "" : " NOT"));
  printf("Vectors Vg are%s equal!\n", (equalVg ? "" : " NOT"));
  printf("Vectors Kc are%s equal!\n", (equalKc ? "" : " NOT"));
  printf("Vectors Kg are%s equal!\n", (equalKg ? "" : " NOT"));
  printf("Vectors Ac are%s equal!\n", (equalAc ? "" : " NOT"));
  printf("Vectors Ag are%s equal!\n", (equalAg ? "" : " NOT"));
  printf("Vectors Dc are%s equal!\n", (equalDc ? "" : " NOT"));
  printf("Vectors Dg are%s equal!\n", (equalDg ? "" : " NOT"));
  printf("\n");
  printf("Time init:  %10.6f\n", tm_init.get_time());
  printf("Time check: %10.6f\n", tm_check.get_time());
  printf("\n");
  printf("Time multiply (avg)    V            K            A            D\n");
  printf("mem_assemble        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vma.get_time()/repetitions, tm_Kma.get_time()/repetitions, tm_Ama.get_time()/repetitions, tm_Dma.get_time()/repetitions);
  printf("mem_multiply        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vmm.get_time()/repetitions, tm_Kmm.get_time()/repetitions, tm_Amm.get_time()/repetitions, tm_Dmm.get_time()/repetitions);
  printf("mem_total           %10.6f   %10.6f   %10.6f   %10.6f\n", (tm_Vma.get_time() + tm_Vmm.get_time())/repetitions, (tm_Kma.get_time() + tm_Kmm.get_time())/repetitions, (tm_Ama.get_time() + tm_Amm.get_time())/repetitions, (tm_Dma.get_time() + tm_Dmm.get_time())/repetitions);
  printf("fly_mult_cpu        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vfc.get_time()/repetitions, tm_Kfc.get_time()/repetitions, tm_Afc.get_time()/repetitions, tm_Dfc.get_time()/repetitions);
  printf("fly_mult_gpu        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vfg.get_time()/repetitions, tm_Kfg.get_time()/repetitions, tm_Afg.get_time()/repetitions, tm_Dfg.get_time()/repetitions);

  

  return 0;
}
