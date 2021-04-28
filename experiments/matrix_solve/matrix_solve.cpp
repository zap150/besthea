#include "besthea/besthea.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace besthea;
using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem::onthefly;
using namespace besthea::bem;
using namespace besthea::tools;

struct cauchy_data {
  static sc dirichlet( sc x1, sc x2, sc x3, const coordinates< 3 > & n, sc t ) {
    sc norm2 = ( x1 - _y[ 0 ] ) * ( x1 - _y[ 0 ] )
      + ( x2 - _y[ 1 ] ) * ( x2 - _y[ 1 ] )
      + ( x3 - _y[ 2 ] ) * ( x3 - _y[ 2 ] );
    sc value = std::pow( 4.0 * M_PI * _alpha * ( t + _shift ), -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * ( t + _shift ) ) );

    return value;
  }

  static sc neumann( sc x1, sc x2, sc x3, const coordinates< 3 > & n, sc t ) {
    sc dot = ( x1 - _y[ 0 ] ) * n[ 0 ] + ( x2 - _y[ 1 ] ) * n[ 1 ]
      + ( x3 - _y[ 2 ] ) * n[ 2 ];
    sc value = ( -1.0 / ( 2.0 * ( t + _shift ) ) ) * dot
      * dirichlet( x1, x2, x3, n, t );

    return value;
  }

  static sc initial( sc x1, sc x2, sc x3 ) {
    sc norm2 = ( x1 - _y[ 0 ] ) * ( x1 - _y[ 0 ] )
      + ( x2 - _y[ 1 ] ) * ( x2 - _y[ 1 ] )
      + ( x3 - _y[ 2 ] ) * ( x3 - _y[ 2 ] );
    sc value = std::pow( 4.0 * M_PI * _alpha * _shift, -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * _shift ) );

    return value;
  }

  static constexpr sc _alpha{ 1.0 };
  static constexpr std::array< sc, 3 > _y{ 1.5, 1.5, 1.5 };
  static constexpr sc _shift{ 0.0 };
};






int main( int argc, char * argv[] ) {

  if(argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
    printf("Usage: ./matrix_solve finess_level gpu_alg_ver repetitions pre_repetitions quadr_order_reg quadr_order_sng\n");
    return 0;
  }

  besthea::tools::time_measurer tm_init, tm_init_gpu_mesh;
  besthea::tools::time_measurer tm_init_fly_gpu_v, tm_init_fly_gpu_k, tm_init_fly_gpu_a, tm_init_fly_gpu_d;
  besthea::tools::time_measurer tm_init_fly_cpu_v, tm_init_fly_cpu_k, tm_init_fly_cpu_a, tm_init_fly_cpu_d;
  besthea::tools::time_measurer tm_assemble_v_mem, tm_assemble_k_mem, tm_assemble_d_mem, tm_assemble_m_mem;
  besthea::tools::time_measurer tm_rhs_dir_mem, tm_rhs_dir_fly_cpu, tm_rhs_dir_fly_gpu;
  besthea::tools::time_measurer tm_rhs_neu_mem, tm_rhs_neu_fly_cpu, tm_rhs_neu_fly_gpu;
  besthea::tools::time_measurer tm_solve_dir_mem, tm_solve_dir_fly_cpu, tm_solve_dir_fly_gpu;
  besthea::tools::time_measurer tm_solve_neu_mem, tm_solve_neu_fly_cpu, tm_solve_neu_fly_gpu;

  tm_init.start();

  std::string mesh_file_12 = "../../besthea/examples/mesh_files/cube_12.txt";
  std::string mesh_file_24 = "../../besthea/examples/mesh_files/cube_24.txt";

  bool do_mem    = true;
  bool do_fly_cpu = true;
  bool do_fly_gpu = true;
  bool do_dirichlet = true; // solve with system matrix V
  bool do_neumann = true; // solve with system matrix D
  
  lo finess_level = 3;
  int gpu_alg_ver = 1;
  int repetitions = 1;
  int pre_repetitions = 0;
  int quadr_order_sng = 4;
  int quadr_order_reg = 4;

  if(argc > 1) finess_level = atoi(argv[1]);
  if(argc > 2) gpu_alg_ver = atoi(argv[2]);
  if(argc > 3) repetitions = atoi(argv[3]);
  if(argc > 4) pre_repetitions = atoi(argv[4]);
  if(argc > 5) quadr_order_reg = atoi(argv[5]);
  if(argc > 6) quadr_order_sng = atoi(argv[6]);

  int total_repetitions = repetitions + pre_repetitions;

  int order_reg_V = quadr_order_reg;
  int order_reg_K = quadr_order_reg;
  int order_reg_A = quadr_order_reg;
  int order_reg_D = quadr_order_reg;
  int order_sng_V = quadr_order_sng;
  int order_sng_K = quadr_order_sng;
  int order_sng_A = quadr_order_sng;
  int order_sng_D = quadr_order_sng;

  besthea::settings::output_verbosity.timers = 0;
  besthea::settings::output_verbosity.onthefly_loadbalance = 0;

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
  tm_init_gpu_mesh.start();
  uniform_spacetime_tensor_mesh_gpu gpu_spacetime_mesh(spacetime_mesh);
  tm_init_gpu_mesh.stop();

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
  spacetime_heat_sl_kernel_antiderivative  kernel_v( cauchy_data::_alpha );
  spacetime_heat_dl_kernel_antiderivative  kernel_k( cauchy_data::_alpha );
  spacetime_heat_adl_kernel_antiderivative kernel_a( cauchy_data::_alpha );
  spacetime_heat_hs_kernel_antiderivative  kernel_d( cauchy_data::_alpha );

  // matrix preparation
  block_lower_triangular_toeplitz_matrix V_mem; // single layer operator
  block_lower_triangular_toeplitz_matrix K_mem; // double layer operator
  //block_lower_triangular_toeplitz_matrix A_mem; // adjoint double layer operator
  block_lower_triangular_toeplitz_matrix D_mem; // hypersingular operator

  uniform_spacetime_be_assembler         assembler_v(kernel_v, space_p0, space_p0, order_sng_V, order_reg_V);
  uniform_spacetime_be_assembler         assembler_k(kernel_k, space_p0, space_p1, order_sng_K, order_reg_K);
  //uniform_spacetime_be_assembler         assembler_a(kernel_a, space_p1, space_p0, order_sng_A, order_reg_A);
  uniform_spacetime_be_assembler         assembler_d(kernel_d, space_p1, space_p1, order_sng_D, order_reg_D);

  tm_init_fly_cpu_v.start();
  uniform_spacetime_be_matrix_onthefly_cpu V_fly_cpu(kernel_v, space_p0, space_p0, order_sng_V, order_reg_V);
  tm_init_fly_cpu_v.stop();
  tm_init_fly_cpu_k.start();
  uniform_spacetime_be_matrix_onthefly_cpu K_fly_cpu(kernel_k, space_p0, space_p1, order_sng_K, order_reg_K);
  tm_init_fly_cpu_k.stop();
  tm_init_fly_cpu_a.start();
  uniform_spacetime_be_matrix_onthefly_cpu A_fly_cpu(kernel_a, space_p1, space_p0, order_sng_A, order_reg_A);
  tm_init_fly_cpu_a.stop();
  tm_init_fly_cpu_d.start();
  uniform_spacetime_be_matrix_onthefly_cpu D_fly_cpu(kernel_d, space_p1, space_p1, order_sng_D, order_reg_D);
  tm_init_fly_cpu_d.stop();  

  tm_init_fly_gpu_v.start();
  uniform_spacetime_be_matrix_onthefly_gpu V_fly_gpu(kernel_v, space_p0, space_p0, gpu_spacetime_mesh, order_sng_V, order_reg_V, gpu_alg_ver);
  tm_init_fly_gpu_v.stop();
  tm_init_fly_gpu_k.start();
  uniform_spacetime_be_matrix_onthefly_gpu K_fly_gpu(kernel_k, space_p0, space_p1, gpu_spacetime_mesh, order_sng_K, order_reg_K, gpu_alg_ver);
  tm_init_fly_gpu_k.stop();
  tm_init_fly_gpu_a.start();
  uniform_spacetime_be_matrix_onthefly_gpu A_fly_gpu(kernel_a, space_p1, space_p0, gpu_spacetime_mesh, order_sng_A, order_reg_A, gpu_alg_ver);
  tm_init_fly_gpu_a.stop();
  tm_init_fly_gpu_d.start();
  uniform_spacetime_be_matrix_onthefly_gpu D_fly_gpu(kernel_d, space_p1, space_p1, gpu_spacetime_mesh, order_sng_D, order_reg_D, gpu_alg_ver);
  tm_init_fly_gpu_d.stop();

  uniform_spacetime_be_identity M( space_p0, space_p1, 1 );

  tm_init.stop();

  // assemble matrices
  if(do_dirichlet) {
    printf("assembly V\n");
    tm_assemble_v_mem.start();
    assembler_v.assemble(V_mem);
    tm_assemble_v_mem.stop();
  }
  if(do_neumann) {
    printf("assembly D\n");
    tm_assemble_d_mem.start();
    assembler_d.assemble(D_mem);
    tm_assemble_d_mem.stop();
  }
  printf("assembly K\n");
  tm_assemble_k_mem.start();
  assembler_k.assemble(K_mem);
  tm_assemble_k_mem.stop();
  printf("assembly M\n");
  tm_assemble_m_mem.start();
  M.assemble();
  tm_assemble_m_mem.stop();
  
  sc prec = 1e-8;
  sc iter = 500;

  sc rel_error_dir_mem =     0.0;
  sc rel_error_dir_fly_cpu = 0.0;
  sc rel_error_dir_fly_gpu = 0.0;
  sc rel_error_neu_mem =     0.0;
  sc rel_error_neu_fly_cpu = 0.0;
  sc rel_error_neu_fly_gpu = 0.0;
  
  sc gmres_prec_dir_mem =     prec;
  sc gmres_prec_dir_fly_cpu = prec;
  sc gmres_prec_dir_fly_gpu = prec;
  sc gmres_prec_neu_mem =     prec;
  sc gmres_prec_neu_fly_cpu = prec;
  sc gmres_prec_neu_fly_gpu = prec;

  lo gmres_iter_dir_mem =     iter;
  lo gmres_iter_dir_fly_cpu = iter;
  lo gmres_iter_dir_fly_gpu = iter;
  lo gmres_iter_neu_mem =     iter;
  lo gmres_iter_neu_fly_cpu = iter;
  lo gmres_iter_neu_fly_gpu = iter;

  if(do_dirichlet) {
    printf("dirichlet\n");

    // boundary condition vector
    block_vector bc_dir;
    space_p1.L2_projection( cauchy_data::dirichlet, bc_dir );
    
    // right-hand-side vector
    block_vector rhs_mem    (spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());
    block_vector rhs_fly_cpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());
    block_vector rhs_fly_gpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());

    if(do_mem) {
      printf("  rhs mem\n");
      tm_rhs_dir_mem.start();
      M.apply(bc_dir, rhs_mem, false, 0.5, 0.0);
      K_mem.apply(bc_dir, rhs_mem, false, 1.0, 1.0);
      tm_rhs_dir_mem.stop();
    }
    if(do_fly_cpu) {
      printf("  rhs fly cpu\n");
      tm_rhs_dir_fly_cpu.start();
      M.apply(bc_dir, rhs_fly_cpu, false, 0.5, 0.0);
      K_fly_cpu.apply(bc_dir, rhs_fly_cpu, false, 1.0, 1.0);
      tm_rhs_dir_fly_cpu.stop();
    }
    if(do_fly_gpu) {
      printf("  rhs fly gpu\n");
      tm_rhs_dir_fly_gpu.start();
      M.apply(bc_dir, rhs_fly_gpu, false, 0.5, 0.0);
      K_fly_gpu.apply(bc_dir, rhs_fly_gpu, false, 1.0, 1.0);
      tm_rhs_dir_fly_gpu.stop();
    }

    // solution vectors
    block_vector sol_mem    (spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());
    block_vector sol_fly_cpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());
    block_vector sol_fly_gpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());

    // solve
    if(do_mem) {
      printf("  solve mem\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_mem.copy(rhs_mem);
        gmres_prec_dir_mem = prec;
        gmres_iter_dir_mem = iter;
        if(i >= pre_repetitions) tm_solve_dir_mem.start();
        V_mem.mkl_fgmres_solve    (rhs_mem,     sol_mem,     gmres_prec_dir_mem,     gmres_iter_dir_mem,     gmres_iter_dir_mem);
        if(i >= pre_repetitions) tm_solve_dir_mem.stop();
      }
    }
    if(do_fly_cpu) {
      printf("  solve fly cpu\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_fly_cpu.copy(rhs_fly_cpu);
        gmres_prec_dir_fly_cpu = prec;
        gmres_iter_dir_fly_cpu = iter;
        if(i >= pre_repetitions) tm_solve_dir_fly_cpu.start();
        V_fly_cpu.mkl_fgmres_solve(rhs_fly_cpu, sol_fly_cpu, gmres_prec_dir_fly_cpu, gmres_iter_dir_fly_cpu, gmres_iter_dir_fly_cpu);
        if(i >= pre_repetitions) tm_solve_dir_fly_cpu.stop();
      }
    }
    if(do_fly_gpu) {
      printf("  solve fly gpu\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_fly_gpu.copy(rhs_fly_gpu);
        gmres_prec_dir_fly_gpu = prec;
        gmres_iter_dir_fly_gpu = iter;
        if(i >= pre_repetitions) tm_solve_dir_fly_gpu.start();
        V_fly_gpu.mkl_fgmres_solve(rhs_fly_gpu, sol_fly_gpu, gmres_prec_dir_fly_gpu, gmres_iter_dir_fly_gpu, gmres_iter_dir_fly_gpu);
        if(i >= pre_repetitions) tm_solve_dir_fly_gpu.stop();
      }
    }

    rel_error_dir_mem =     space_p0.L2_relative_error(cauchy_data::neumann, sol_mem);
    rel_error_dir_fly_cpu = space_p0.L2_relative_error(cauchy_data::neumann, sol_fly_cpu);
    rel_error_dir_fly_gpu = space_p0.L2_relative_error(cauchy_data::neumann, sol_fly_gpu);

  }



  if(do_neumann) {
    printf("neumann\n");
    
    // boundary condition vector
    block_vector bc_neu;
    space_p0.L2_projection( cauchy_data::neumann, bc_neu );
    
    // right-hand-side vector
    block_vector rhs_mem    (spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());
    block_vector rhs_fly_cpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());
    block_vector rhs_fly_gpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());

    if(do_mem) {
      printf("  rhs mem\n");
      tm_rhs_neu_mem.start();
      M.apply(bc_neu, rhs_mem, true, 0.5, 0.0);
      K_mem.apply(bc_neu, rhs_mem, true, -1.0, 1.0);
      tm_rhs_neu_mem.stop();
    }
    if(do_fly_cpu) {
      printf("  rhs fly cpu\n");
      tm_rhs_neu_fly_cpu.start();
      M.apply(bc_neu, rhs_fly_cpu, true, 0.5, 0.0);
      A_fly_cpu.apply(bc_neu, rhs_fly_cpu, false, -1.0, 1.0);
      tm_rhs_neu_fly_cpu.stop();
    }
    if(do_fly_gpu) {
      printf("  rhs fly gpu\n");
      tm_rhs_neu_fly_gpu.start();
      M.apply(bc_neu, rhs_fly_gpu, true, 0.5, 0.0);
      A_fly_gpu.apply(bc_neu, rhs_fly_gpu, false, -1.0, 1.0);
      tm_rhs_neu_fly_gpu.stop();
    }

    // solution vectors
    block_vector sol_mem    (spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());
    block_vector sol_fly_cpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());
    block_vector sol_fly_gpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());

    // solve
    if(do_mem) {
      printf("  solve mem\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_mem.copy(rhs_mem);
        gmres_prec_neu_mem = prec;
        gmres_iter_neu_mem = iter;
        if(i >= pre_repetitions) tm_solve_neu_mem.start();
        D_mem.mkl_fgmres_solve    (rhs_mem,     sol_mem,     gmres_prec_neu_mem,     gmres_iter_neu_mem,     gmres_iter_neu_mem);
        if(i >= pre_repetitions) tm_solve_neu_mem.stop();
      }
    }
    if(do_fly_cpu) {
      printf("  solve fly cpu\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_fly_cpu.copy(rhs_fly_cpu);
        gmres_prec_neu_fly_cpu = prec;
        gmres_iter_neu_fly_cpu = iter;
        if(i >= pre_repetitions) tm_solve_neu_fly_cpu.start();
        D_fly_cpu.mkl_fgmres_solve(rhs_fly_cpu, sol_fly_cpu, gmres_prec_neu_fly_cpu, gmres_iter_neu_fly_cpu, gmres_iter_neu_fly_cpu);
        if(i >= pre_repetitions) tm_solve_neu_fly_cpu.stop();
      }
    }
    if(do_fly_gpu) {
      printf("  solve fly gpu\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_fly_gpu.copy(rhs_fly_gpu);
        gmres_prec_neu_fly_gpu = prec;
        gmres_iter_neu_fly_gpu = iter;
        if(i >= pre_repetitions) tm_solve_neu_fly_gpu.start();
        D_fly_gpu.mkl_fgmres_solve(rhs_fly_gpu, sol_fly_gpu, gmres_prec_neu_fly_gpu, gmres_iter_neu_fly_gpu, gmres_iter_neu_fly_gpu);
        if(i >= pre_repetitions) tm_solve_neu_fly_gpu.stop();
      }
    }
    
    rel_error_neu_mem =     space_p1.L2_relative_error(cauchy_data::dirichlet, sol_mem);
    rel_error_neu_fly_cpu = space_p1.L2_relative_error(cauchy_data::dirichlet, sol_fly_cpu);
    rel_error_neu_fly_gpu = space_p1.L2_relative_error(cauchy_data::dirichlet, sol_fly_gpu);

  }




  printf("\n");
  printf("Dirichlet problem:\n");
  printf("                  mem        fly_cpu     fly_gpu\n");
  printf("Dir precision:  %10.3e  %10.3e  %10.3e\n", gmres_prec_dir_mem, gmres_prec_dir_fly_cpu, gmres_prec_dir_fly_gpu);
  printf("Dir iterations: %10ld  %10ld  %10ld\n",    gmres_iter_dir_mem, gmres_iter_dir_fly_cpu, gmres_iter_dir_fly_gpu);
  printf("Dir rel_error:  %10.3e  %10.3e  %10.3e\n", rel_error_dir_mem, rel_error_dir_fly_cpu, rel_error_dir_fly_gpu);
  printf("\n");
  printf("Neumann problem:\n");
  printf("                  mem        fly_cpu     fly_gpu\n");
  printf("Neu precision:  %10.3e  %10.3e  %10.3e\n", gmres_prec_neu_mem, gmres_prec_neu_fly_cpu, gmres_prec_neu_fly_gpu);
  printf("Neu iterations: %10ld  %10ld %10ld\n",     gmres_iter_neu_mem, gmres_iter_neu_fly_cpu, gmres_iter_neu_fly_gpu);
  printf("Neu rel_error:  %10.3e  %10.3e  %10.3e\n", rel_error_neu_mem, rel_error_neu_fly_cpu, rel_error_neu_fly_gpu);
  printf("\n");
  printf("Time init total: %.6f\n", tm_init.get_time());
  printf("Time init gpu mesh: %.6f\n", tm_init_gpu_mesh.get_time());
  printf("Time init fly classes:\n");
  printf("               V           K           A           D\n");
  printf("fly_cpu     %10.6f  %10.6f  %10.6f  %10.6f\n", tm_init_fly_cpu_v.get_time(), tm_init_fly_cpu_k.get_time(), tm_init_fly_cpu_a.get_time(), tm_init_fly_cpu_d.get_time());
  printf("fly_gpu     %10.6f  %10.6f  %10.6f  %10.6f\n", tm_init_fly_gpu_v.get_time(), tm_init_fly_gpu_k.get_time(), tm_init_fly_gpu_a.get_time(), tm_init_fly_gpu_d.get_time());
  printf("\n");
  printf("Assemble:\n");
  printf("Assemble V: %10.6f\n", tm_assemble_v_mem.get_time());
  printf("Assemble K: %10.6f\n", tm_assemble_k_mem.get_time());
  printf("Assemble D: %10.6f\n", tm_assemble_d_mem.get_time());
  printf("Assemble M: %10.6f\n", tm_assemble_m_mem.get_time());  
  printf("\n");
  printf("Right hand side vector:\n");
  printf("                    dirichlet   neumann\n");
  printf("rhs mem           %10.6f  %10.6f\n", tm_rhs_dir_mem.get_time(),     tm_rhs_neu_mem.get_time());
  printf("rhs fly_cpu       %10.6f  %10.6f\n", tm_rhs_dir_fly_cpu.get_time(), tm_rhs_neu_fly_cpu.get_time());
  printf("rhs fly_gpu       %10.6f  %10.6f\n", tm_rhs_dir_fly_gpu.get_time(), tm_rhs_neu_fly_gpu.get_time());
  printf("\n");
  printf("Solving the system:\n");
  printf("                    dirichlet   neumann\n");
  printf("solve mem         %10.6f  %10.6f\n", tm_solve_dir_mem.get_time() / repetitions,     tm_solve_neu_mem.get_time() / repetitions);
  printf("solve fly_cpu     %10.6f  %10.6f\n", tm_solve_dir_fly_cpu.get_time() / repetitions, tm_solve_neu_fly_cpu.get_time() / repetitions);
  printf("solve fly_gpu     %10.6f  %10.6f\n", tm_solve_dir_fly_gpu.get_time() / repetitions, tm_solve_neu_fly_gpu.get_time() / repetitions);
  printf("\n");
  printf("Time per iteration:\n");
  printf("                    dirichlet   neumann\n");
  printf("iter mem          %10.6f  %10.6f\n", tm_solve_dir_mem.get_time() / gmres_iter_dir_mem / repetitions,     tm_solve_neu_mem.get_time() / gmres_iter_neu_mem / repetitions);
  printf("iter fly_cpu      %10.6f  %10.6f\n", tm_solve_dir_fly_cpu.get_time() / gmres_iter_dir_fly_cpu / repetitions, tm_solve_neu_fly_cpu.get_time() / gmres_iter_neu_fly_cpu / repetitions);
  printf("iter fly_gpu      %10.6f  %10.6f\n", tm_solve_dir_fly_gpu.get_time() / gmres_iter_dir_fly_gpu / repetitions, tm_solve_neu_fly_gpu.get_time() / gmres_iter_neu_fly_gpu / repetitions);
  printf("\n");
  printf("Total time:\n");
  printf("                    dirichlet   neumann\n");
  printf("total mem         %10.6f  %10.6f\n", tm_solve_dir_mem.get_time() / repetitions + tm_assemble_v_mem.get_time() + tm_assemble_k_mem.get_time() + tm_assemble_m_mem.get_time() + tm_rhs_dir_mem.get_time(), tm_solve_neu_mem.get_time() / repetitions + tm_assemble_d_mem.get_time() + tm_assemble_k_mem.get_time() + tm_assemble_m_mem.get_time() + tm_rhs_neu_mem.get_time());
  printf("total fly_cpu     %10.6f  %10.6f\n", tm_solve_dir_fly_cpu.get_time() / repetitions + tm_assemble_m_mem.get_time() + tm_rhs_dir_fly_cpu.get_time() + tm_init_fly_cpu_v.get_time() + tm_init_fly_cpu_k.get_time(), tm_solve_neu_fly_cpu.get_time() / repetitions + tm_assemble_m_mem.get_time() + tm_rhs_neu_fly_cpu.get_time() + tm_init_fly_cpu_d.get_time() + tm_init_fly_cpu_a.get_time());
  printf("total fly_gpu     %10.6f  %10.6f\n", tm_solve_dir_fly_gpu.get_time() / repetitions + tm_assemble_m_mem.get_time() + tm_rhs_dir_fly_gpu.get_time() + tm_init_fly_gpu_v.get_time() + tm_init_fly_gpu_k.get_time() + tm_init_gpu_mesh.get_time(), tm_solve_neu_fly_gpu.get_time() / repetitions + tm_assemble_m_mem.get_time() + tm_rhs_neu_fly_gpu.get_time() + tm_init_fly_gpu_d.get_time() + tm_init_fly_gpu_a.get_time() + tm_init_gpu_mesh.get_time());
 


  

  return 0;
}
