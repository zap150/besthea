#include <cstdlib>
#include <iostream>
#include "besthea/besthea_cuda.h"
#include <lyra/lyra.hpp>

using namespace besthea;
using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem::onthefly;
using namespace besthea::bem;
using namespace besthea::tools;



struct config {
  config(int argc, char * argv[]) {
    bool help = false;

    auto cli = lyra::help( help )
      | lyra::opt( mesh_file, "surface mesh" )[ "--mesh" ](
        "Surface mesh of the spatial domain, "
        "refined by the 'refine' parameter" )
          .required( )
      | lyra::opt( space_refine, "spatial surface mesh refinement" )
        [ "--space-refine" ]( "Number of surface mesh refinements" )
      | lyra::opt( n_timeslices, "number of timesteps" )[ "--timesteps" ](
        "Number of timesteps for the given mesh" )
      | lyra::opt( end_time, "end time" )[ "--endtime" ]( "End time" )
      | lyra::opt( fgmres_prec, "FGMRES relative precision" )[ "--fgmres_prec" ](
        "Relative precision of the FGMRES solver (default 1e-8) " )
      | lyra::opt( heat_capacity_constant, "heat capacity constant" )[ "--hc" ](
         "The value of the heat capacity constant alpha" )
      | lyra::opt( do_in_memory )[ "--do-inmemory" ](
        "Make the performance measurement of the original in-memory algorithm" )
      | lyra::opt( do_onthefly_cpu )
        [ "--do-onthefly-cpu" ]
        ( "Make the performance measurement of the CPU on-the-fly algorithm" )
      | lyra::opt( do_onthefly_gpu )
        [ "--do-onthefly-gpu" ]
        ( "Make the performance measurement of the GPU on-the-fly algorithm" )
      | lyra::opt( do_dirichlet )[ "--do-dirichlet" ](
        "Measure the performance of solving the Dirichlet problem" )
      | lyra::opt( do_neumann )[ "--do-neumann" ](
        "Measure the performance of solving the Neumann problem" )
      | lyra::opt( pre_repetitions, "warmup count" )[ "--warmups" ](
        "Number of times the algorithm is run before the measurement phase" )
      | lyra::opt( repetitions, "repetition count" )[ "--repetitions" ](
        "Number of times the algorithm is run and measured" )
      | lyra::opt( qo_sng, "quadrature order singular" )[ "--qo-singular" ](
        "Quadrature order used for singular integrals" )
      | lyra::opt( qo_reg, "quadrature order regular" )[ "--qo-regular" ](
        "Quadrature order used for regular integrals" );

    auto result = cli.parse( { argc, argv } );

    if ( !result ) {
      std::cerr << "Error in command line: " << result.errorMessage( )
                << std::endl;
      std::cout << cli << std::endl;
      exit( 1 );
    }

    if ( help ) {
      std::cout << cli << std::endl;
      exit( 0 );
    }
  }

  void dump() {
    std::cout << "config:" << std::endl;
    std::cout << "  Spatial surface mesh file:                         "
      << mesh_file << std::endl;
    std::cout << "  Number of refinements of the spatial surface mesh: "
      << space_refine << std::endl;
    std::cout << "  Number of timeslices:                              "
      << n_timeslices << std::endl;
    std::cout << "  End time:                                          "
      << end_time << std::endl;
    std::cout << "  FGMRES relative precision:                         "
      << fgmres_prec << std::endl;
    std::cout << "  Heat capacity constant alpha:                      "
      << heat_capacity_constant << std::endl;
    std::cout << "  In memory:                                         "
      << (do_in_memory ? "yes" : "no") << std::endl;
    std::cout << "  Onthefly CPU:                                      "
      << (do_onthefly_cpu ? "yes" : "no") << std::endl;
    std::cout << "  Onthefly GPU:                                      "
      << (do_onthefly_gpu ? "yes" : "no") << std::endl;
    std::cout << "  Dirichlet problem:                                 "
      << (do_dirichlet ? "yes" : "no") << std::endl;
    std::cout << "  Neumann problem:                                   "
      << (do_neumann ? "yes" : "no") << std::endl;
    std::cout << "  Warmup rounds:                                     "
      << pre_repetitions << std::endl;
    std::cout << "  Repetitions:                                       "
      << repetitions << std::endl;
    std::cout << "  Quadrature order for sing. int.:                   "
      << qo_sng << std::endl;
    std::cout << "  Quadrature order for reg. int.:                    "
      << qo_reg << std::endl;
  }

  std::string mesh_file = "";
  int space_refine = 0;
  lo n_timeslices = 8;
  sc end_time = 1;
  sc fgmres_prec = 1e-8;
  sc heat_capacity_constant = 0.5;
  bool do_in_memory = false;
  bool do_onthefly_cpu = false;
  bool do_onthefly_gpu = false;
  bool do_dirichlet = false;
  bool do_neumann = false;
  int pre_repetitions = 2;
  int repetitions = 10;
  int qo_sng = 4;
  int qo_reg = 4;
}; // struct config



struct cauchy_data {
  static sc dirichlet( sc x1, sc x2, sc x3, const coordinates< 3 > &, sc t ) {
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
    int dummy = 0.0;
    sc value = std::pow( 4.0 * M_PI * _alpha * _shift, -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * _shift + dummy ) );

    return value;
  }

  static sc _alpha;
  static constexpr std::array< sc, 3 > _y{ 0.0, 0.0, 1.5 };
  static constexpr sc _shift{ 0.0 };
};
sc cauchy_data::_alpha;





int main( int argc, char * argv[] ) {

  config c( argc, argv );
  cauchy_data::_alpha = c.heat_capacity_constant;

  timer tm_init, tm_init_gpu_mesh;
  timer tm_init_fly_gpu_v, tm_init_fly_gpu_k, tm_init_fly_gpu_a, tm_init_fly_gpu_d;
  timer tm_init_fly_cpu_v, tm_init_fly_cpu_k, tm_init_fly_cpu_a, tm_init_fly_cpu_d;
  timer tm_assemble_v_mem, tm_assemble_k_mem, tm_assemble_d_mem, tm_assemble_m_mem;
  timer tm_rhs_dir_mem, tm_rhs_dir_fly_cpu, tm_rhs_dir_fly_gpu;
  timer tm_rhs_neu_mem, tm_rhs_neu_fly_cpu, tm_rhs_neu_fly_gpu;
  timer tm_solve_dir_mem, tm_solve_dir_fly_cpu, tm_solve_dir_fly_gpu;
  timer tm_solve_neu_mem, tm_solve_neu_fly_cpu, tm_solve_neu_fly_gpu;

  tm_init.start();

  int total_repetitions = c.repetitions + c.pre_repetitions;

  besthea::settings::output_verbosity.timers = 0;
  besthea::settings::output_verbosity.onthefly_loadbalance = 0;
  
  // load spatial mesh from file and refine
  triangular_surface_mesh space_mesh;
  space_mesh.load( c.mesh_file );
  space_mesh.refine( c.space_refine );

  // create spacetime mesh as a tensor product of spatial and temporal meshes
  uniform_spacetime_tensor_mesh spacetime_mesh( space_mesh, c.end_time, c.n_timeslices );
  tm_init_gpu_mesh.start();
  uniform_spacetime_tensor_mesh_gpu gpu_spacetime_mesh(spacetime_mesh);
  tm_init_gpu_mesh.stop();

  // print some info
  c.dump();
  spacetime_mesh.print_info();

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

  uniform_spacetime_be_assembler         assembler_v(kernel_v, space_p0, space_p0, c.qo_sng, c.qo_reg);
  uniform_spacetime_be_assembler         assembler_k(kernel_k, space_p0, space_p1, c.qo_sng, c.qo_reg);
  //uniform_spacetime_be_assembler         assembler_a(kernel_a, space_p1, space_p0, c.qo_sng, c.qo_reg);
  uniform_spacetime_be_assembler         assembler_d(kernel_d, space_p1, space_p1, c.qo_sng, c.qo_reg);

  tm_init_fly_cpu_v.start();
  uniform_spacetime_be_matrix_onthefly_cpu V_fly_cpu(kernel_v, space_p0, space_p0, c.qo_sng, c.qo_reg);
  tm_init_fly_cpu_v.stop();
  tm_init_fly_cpu_k.start();
  uniform_spacetime_be_matrix_onthefly_cpu K_fly_cpu(kernel_k, space_p0, space_p1, c.qo_sng, c.qo_reg);
  tm_init_fly_cpu_k.stop();
  tm_init_fly_cpu_a.start();
  uniform_spacetime_be_matrix_onthefly_cpu A_fly_cpu(kernel_a, space_p1, space_p0, c.qo_sng, c.qo_reg);
  tm_init_fly_cpu_a.stop();
  tm_init_fly_cpu_d.start();
  uniform_spacetime_be_matrix_onthefly_cpu D_fly_cpu(kernel_d, space_p1, space_p1, c.qo_sng, c.qo_reg);
  tm_init_fly_cpu_d.stop();  

  tm_init_fly_gpu_v.start();
  uniform_spacetime_be_matrix_onthefly_gpu V_fly_gpu(kernel_v, space_p0, space_p0, gpu_spacetime_mesh, c.qo_sng, c.qo_reg);
  tm_init_fly_gpu_v.stop();
  tm_init_fly_gpu_k.start();
  uniform_spacetime_be_matrix_onthefly_gpu K_fly_gpu(kernel_k, space_p0, space_p1, gpu_spacetime_mesh, c.qo_sng, c.qo_reg);
  tm_init_fly_gpu_k.stop();
  tm_init_fly_gpu_a.start();
  uniform_spacetime_be_matrix_onthefly_gpu A_fly_gpu(kernel_a, space_p1, space_p0, gpu_spacetime_mesh, c.qo_sng, c.qo_reg);
  tm_init_fly_gpu_a.stop();
  tm_init_fly_gpu_d.start();
  uniform_spacetime_be_matrix_onthefly_gpu D_fly_gpu(kernel_d, space_p1, space_p1, gpu_spacetime_mesh, c.qo_sng, c.qo_reg);
  tm_init_fly_gpu_d.stop();

  uniform_spacetime_be_identity M( space_p0, space_p1, 1 );

  tm_init.stop();

  if(c.do_in_memory) {
    // assemble matrices
    if(c.do_dirichlet) {
      printf("assembly V\n");
      for(int i = 0; i < total_repetitions; i++) {
        V_mem.clear();
        if(i >= c.pre_repetitions) tm_assemble_v_mem.start(true);
        assembler_v.assemble(V_mem);
        if(i >= c.pre_repetitions) tm_assemble_v_mem.stop();
      }
    }
    if(c.do_neumann) {
      printf("assembly D\n");
      for(int i = 0; i < total_repetitions; i++) {
        D_mem.clear();
        if(i >= c.pre_repetitions) tm_assemble_d_mem.start(true);
        assembler_d.assemble(D_mem);
        if(i >= c.pre_repetitions) tm_assemble_d_mem.stop();
      }
    }
    printf("assembly K\n");
    for(int i = 0; i < total_repetitions; i++) {
      K_mem.clear();
      if(i >= c.pre_repetitions) tm_assemble_k_mem.start(true);
      assembler_k.assemble(K_mem);
      if(i >= c.pre_repetitions) tm_assemble_k_mem.stop();
    }
  }
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

  if(c.do_dirichlet) {
    printf("dirichlet\n");

    // boundary condition vector
    block_vector bc_dir;
    space_p1.L2_projection( cauchy_data::dirichlet, bc_dir );
    
    // right-hand-side vector
    block_vector rhs_mem    (spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());
    block_vector rhs_fly_cpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());
    block_vector rhs_fly_gpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_elements());

    if(c.do_in_memory) {
      printf("  rhs mem\n");
      tm_rhs_dir_mem.start();
      M.apply(bc_dir, rhs_mem, false, 0.5, 0.0);
      K_mem.apply(bc_dir, rhs_mem, false, 1.0, 1.0);
      tm_rhs_dir_mem.stop();
    }
    if(c.do_onthefly_cpu) {
      printf("  rhs fly cpu\n");
      tm_rhs_dir_fly_cpu.start();
      M.apply(bc_dir, rhs_fly_cpu, false, 0.5, 0.0);
      K_fly_cpu.apply(bc_dir, rhs_fly_cpu, false, 1.0, 1.0);
      tm_rhs_dir_fly_cpu.stop();
    }
    if(c.do_onthefly_gpu) {
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
    if(c.do_in_memory) {
      printf("  solve mem\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_mem.copy(rhs_mem);
        gmres_prec_dir_mem = prec;
        gmres_iter_dir_mem = iter;
        if(i >= c.pre_repetitions) tm_solve_dir_mem.start(true);
        V_mem.mkl_fgmres_solve    (rhs_mem,     sol_mem,     gmres_prec_dir_mem,     gmres_iter_dir_mem,     gmres_iter_dir_mem);
        if(i >= c.pre_repetitions) tm_solve_dir_mem.stop();
      }
    }
    if(c.do_onthefly_cpu) {
      printf("  solve fly cpu\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_fly_cpu.copy(rhs_fly_cpu);
        gmres_prec_dir_fly_cpu = prec;
        gmres_iter_dir_fly_cpu = iter;
        if(i >= c.pre_repetitions) tm_solve_dir_fly_cpu.start(true);
        V_fly_cpu.mkl_fgmres_solve(rhs_fly_cpu, sol_fly_cpu, gmres_prec_dir_fly_cpu, gmres_iter_dir_fly_cpu, gmres_iter_dir_fly_cpu);
        if(i >= c.pre_repetitions) tm_solve_dir_fly_cpu.stop();
      }
    }
    if(c.do_onthefly_gpu) {
      printf("  solve fly gpu\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_fly_gpu.copy(rhs_fly_gpu);
        gmres_prec_dir_fly_gpu = prec;
        gmres_iter_dir_fly_gpu = iter;
        if(i >= c.pre_repetitions) tm_solve_dir_fly_gpu.start(true);
        V_fly_gpu.mkl_fgmres_solve(rhs_fly_gpu, sol_fly_gpu, gmres_prec_dir_fly_gpu, gmres_iter_dir_fly_gpu, gmres_iter_dir_fly_gpu);
        if(i >= c.pre_repetitions) tm_solve_dir_fly_gpu.stop();
      }
    }

    rel_error_dir_mem =     space_p0.L2_relative_error(cauchy_data::neumann, sol_mem);
    rel_error_dir_fly_cpu = space_p0.L2_relative_error(cauchy_data::neumann, sol_fly_cpu);
    rel_error_dir_fly_gpu = space_p0.L2_relative_error(cauchy_data::neumann, sol_fly_gpu);

  }



  if(c.do_neumann) {
    printf("neumann\n");
    
    // boundary condition vector
    block_vector bc_neu;
    space_p0.L2_projection( cauchy_data::neumann, bc_neu );
    
    // right-hand-side vector
    block_vector rhs_mem    (spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());
    block_vector rhs_fly_cpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());
    block_vector rhs_fly_gpu(spacetime_mesh.get_n_temporal_elements(), spacetime_mesh.get_n_spatial_nodes());

    if(c.do_in_memory) {
      printf("  rhs mem\n");
      tm_rhs_neu_mem.start();
      M.apply(bc_neu, rhs_mem, true, 0.5, 0.0);
      K_mem.apply(bc_neu, rhs_mem, true, -1.0, 1.0);
      tm_rhs_neu_mem.stop();
    }
    if(c.do_onthefly_cpu) {
      printf("  rhs fly cpu\n");
      tm_rhs_neu_fly_cpu.start();
      M.apply(bc_neu, rhs_fly_cpu, true, 0.5, 0.0);
      A_fly_cpu.apply(bc_neu, rhs_fly_cpu, false, -1.0, 1.0);
      tm_rhs_neu_fly_cpu.stop();
    }
    if(c.do_onthefly_gpu) {
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
    if(c.do_in_memory) {
      printf("  solve mem\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_mem.copy(rhs_mem);
        gmres_prec_neu_mem = prec;
        gmres_iter_neu_mem = iter;
        if(i >= c.pre_repetitions) tm_solve_neu_mem.start(true);
        D_mem.mkl_fgmres_solve    (rhs_mem,     sol_mem,     gmres_prec_neu_mem,     gmres_iter_neu_mem,     gmres_iter_neu_mem);
        if(i >= c.pre_repetitions) tm_solve_neu_mem.stop();
      }
    }
    if(c.do_onthefly_cpu) {
      printf("  solve fly cpu\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_fly_cpu.copy(rhs_fly_cpu);
        gmres_prec_neu_fly_cpu = prec;
        gmres_iter_neu_fly_cpu = iter;
        if(i >= c.pre_repetitions) tm_solve_neu_fly_cpu.start(true);
        D_fly_cpu.mkl_fgmres_solve(rhs_fly_cpu, sol_fly_cpu, gmres_prec_neu_fly_cpu, gmres_iter_neu_fly_cpu, gmres_iter_neu_fly_cpu);
        if(i >= c.pre_repetitions) tm_solve_neu_fly_cpu.stop();
      }
    }
    if(c.do_onthefly_gpu) {
      printf("  solve fly gpu\n");
      for(int i = 0; i < total_repetitions; i++) {
        sol_fly_gpu.copy(rhs_fly_gpu);
        gmres_prec_neu_fly_gpu = prec;
        gmres_iter_neu_fly_gpu = iter;
        if(i >= c.pre_repetitions) tm_solve_neu_fly_gpu.start(true);
        D_fly_gpu.mkl_fgmres_solve(rhs_fly_gpu, sol_fly_gpu, gmres_prec_neu_fly_gpu, gmres_iter_neu_fly_gpu, gmres_iter_neu_fly_gpu);
        if(i >= c.pre_repetitions) tm_solve_neu_fly_gpu.stop();
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
  printf("Neu iterations: %10ld  %10ld  %10ld\n",     gmres_iter_neu_mem, gmres_iter_neu_fly_cpu, gmres_iter_neu_fly_gpu);
  printf("Neu rel_error:  %10.3e  %10.3e  %10.3e\n", rel_error_neu_mem, rel_error_neu_fly_cpu, rel_error_neu_fly_gpu);
  printf("\n");
  printf("Time init total: %.6f\n", tm_init.get_elapsed_time_in_seconds());
  printf("Time init gpu mesh: %.6f\n", tm_init_gpu_mesh.get_elapsed_time_in_seconds());
  printf("Time init fly classes:\n");
  printf("               V           K           A           D\n");
  printf("fly_cpu     %10.6f  %10.6f  %10.6f  %10.6f\n", tm_init_fly_cpu_v.get_elapsed_time_in_seconds(), tm_init_fly_cpu_k.get_elapsed_time_in_seconds(), tm_init_fly_cpu_a.get_elapsed_time_in_seconds(), tm_init_fly_cpu_d.get_elapsed_time_in_seconds());
  printf("fly_gpu     %10.6f  %10.6f  %10.6f  %10.6f\n", tm_init_fly_gpu_v.get_elapsed_time_in_seconds(), tm_init_fly_gpu_k.get_elapsed_time_in_seconds(), tm_init_fly_gpu_a.get_elapsed_time_in_seconds(), tm_init_fly_gpu_d.get_elapsed_time_in_seconds());
  printf("\n");
  printf("Assemble:\n");
  printf("Assemble V: %10.6f\n", tm_assemble_v_mem.get_elapsed_time_in_seconds() / c.repetitions);
  printf("Assemble K: %10.6f\n", tm_assemble_k_mem.get_elapsed_time_in_seconds() / c.repetitions);
  printf("Assemble D: %10.6f\n", tm_assemble_d_mem.get_elapsed_time_in_seconds() / c.repetitions);
  printf("Assemble M: %10.6f\n", tm_assemble_m_mem.get_elapsed_time_in_seconds() / c.repetitions);  
  printf("\n");
  printf("Right hand side vector:\n");
  printf("                    dirichlet   neumann\n");
  printf("rhs mem           %10.6f  %10.6f\n", tm_rhs_dir_mem.get_elapsed_time_in_seconds(),     tm_rhs_neu_mem.get_elapsed_time_in_seconds());
  printf("rhs fly_cpu       %10.6f  %10.6f\n", tm_rhs_dir_fly_cpu.get_elapsed_time_in_seconds(), tm_rhs_neu_fly_cpu.get_elapsed_time_in_seconds());
  printf("rhs fly_gpu       %10.6f  %10.6f\n", tm_rhs_dir_fly_gpu.get_elapsed_time_in_seconds(), tm_rhs_neu_fly_gpu.get_elapsed_time_in_seconds());
  printf("\n");
  printf("Solving the system:\n");
  printf("                    dirichlet   neumann\n");
  printf("solve mem         %10.6f  %10.6f\n", tm_solve_dir_mem.get_elapsed_time_in_seconds() / c.repetitions,     tm_solve_neu_mem.get_elapsed_time_in_seconds() / c.repetitions);
  printf("solve fly_cpu     %10.6f  %10.6f\n", tm_solve_dir_fly_cpu.get_elapsed_time_in_seconds() / c.repetitions, tm_solve_neu_fly_cpu.get_elapsed_time_in_seconds() / c.repetitions);
  printf("solve fly_gpu     %10.6f  %10.6f\n", tm_solve_dir_fly_gpu.get_elapsed_time_in_seconds() / c.repetitions, tm_solve_neu_fly_gpu.get_elapsed_time_in_seconds() / c.repetitions);
  printf("\n");
  printf("Time per iteration:\n");
  printf("                    dirichlet   neumann\n");
  printf("iter mem          %10.6f  %10.6f\n", tm_solve_dir_mem.get_elapsed_time_in_seconds() / gmres_iter_dir_mem / c.repetitions,     tm_solve_neu_mem.get_elapsed_time_in_seconds() / gmres_iter_neu_mem / c.repetitions);
  printf("iter fly_cpu      %10.6f  %10.6f\n", tm_solve_dir_fly_cpu.get_elapsed_time_in_seconds() / gmres_iter_dir_fly_cpu / c.repetitions, tm_solve_neu_fly_cpu.get_elapsed_time_in_seconds() / gmres_iter_neu_fly_cpu / c.repetitions);
  printf("iter fly_gpu      %10.6f  %10.6f\n", tm_solve_dir_fly_gpu.get_elapsed_time_in_seconds() / gmres_iter_dir_fly_gpu / c.repetitions, tm_solve_neu_fly_gpu.get_elapsed_time_in_seconds() / gmres_iter_neu_fly_gpu / c.repetitions);
  printf("\n");
  printf("Total time:\n");
  printf("                    dirichlet   neumann\n");
  printf("total mem         %10.6f  %10.6f\n", tm_solve_dir_mem.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_v_mem.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_k_mem.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_m_mem.get_elapsed_time_in_seconds() + tm_rhs_dir_mem.get_elapsed_time_in_seconds(), tm_solve_neu_mem.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_d_mem.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_k_mem.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_m_mem.get_elapsed_time_in_seconds() + tm_rhs_neu_mem.get_elapsed_time_in_seconds());
  printf("total fly_cpu     %10.6f  %10.6f\n", tm_solve_dir_fly_cpu.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_m_mem.get_elapsed_time_in_seconds() + tm_rhs_dir_fly_cpu.get_elapsed_time_in_seconds() + tm_init_fly_cpu_v.get_elapsed_time_in_seconds() + tm_init_fly_cpu_k.get_elapsed_time_in_seconds(), tm_solve_neu_fly_cpu.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_m_mem.get_elapsed_time_in_seconds() + tm_rhs_neu_fly_cpu.get_elapsed_time_in_seconds() + tm_init_fly_cpu_d.get_elapsed_time_in_seconds() + tm_init_fly_cpu_a.get_elapsed_time_in_seconds());
  printf("total fly_gpu     %10.6f  %10.6f\n", tm_solve_dir_fly_gpu.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_m_mem.get_elapsed_time_in_seconds() + tm_rhs_dir_fly_gpu.get_elapsed_time_in_seconds() + tm_init_fly_gpu_v.get_elapsed_time_in_seconds() + tm_init_fly_gpu_k.get_elapsed_time_in_seconds() + tm_init_gpu_mesh.get_elapsed_time_in_seconds(), tm_solve_neu_fly_gpu.get_elapsed_time_in_seconds() / c.repetitions + tm_assemble_m_mem.get_elapsed_time_in_seconds() + tm_rhs_neu_fly_gpu.get_elapsed_time_in_seconds() + tm_init_fly_gpu_d.get_elapsed_time_in_seconds() + tm_init_fly_gpu_a.get_elapsed_time_in_seconds() + tm_init_gpu_mesh.get_elapsed_time_in_seconds());
 


  

  return 0;
}
