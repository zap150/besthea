#include <cstdlib>
#include <cstdio>
#include <string>
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
      | lyra::opt( do_V )[ "--do-V" ](
        "Include the single layer matrix V in the experiment" )
      | lyra::opt( do_K )[ "--do-K" ](
        "Include the double layer matrix K in the experiment" )
      | lyra::opt( do_A )[ "--do-KT" ](
        "Include the adjoint double layer matrix K^T in the experiment" )
      | lyra::opt( do_D )[ "--do-D" ](
        "Include the hypersingular matrix D in the experiment" )
      | lyra::opt( pre_repetitions, "warmup count" )[ "--warmups" ](
        "Number of times the algorithm is run before the measurement phase" )
      | lyra::opt( repetitions, "repetition count" )[ "--repetitions" ](
        "Number of times the algorithm is run and measured" )
      | lyra::opt( qo_sng, "quadrature order singular" )[ "--qo-singular" ](
        "Quadrature order used for singular integrals" )
      | lyra::opt( qo_reg, "quadrature order regular" )[ "--qo-regular" ](
        "Quadrature order used for regular integrals" )
      | lyra::opt( gpu_alg, "gpu algorithm" )[ "--gpu-alg" ](
        "GPU algorithm version used" );

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
    std::cout << "  Heat capacity constant alpha:                      "
      << heat_capacity_constant << std::endl;
    std::cout << "  In memory:                                         "
      << (do_in_memory ? "yes" : "no") << std::endl;
    std::cout << "  Onthefly CPU:                                      "
      << (do_onthefly_cpu ? "yes" : "no") << std::endl;
    std::cout << "  Onthefly GPU:                                      "
      << (do_onthefly_gpu ? "yes" : "no") << std::endl;
    std::cout << "  Matrix V:                                          "
      << (do_V ? "yes" : "no") << std::endl;
    std::cout << "  Matrix K:                                          "
      << (do_K ? "yes" : "no") << std::endl;
    std::cout << "  Matrix KT:                                         "
      << (do_A ? "yes" : "no") << std::endl;
    std::cout << "  Matrix D:                                          "
      << (do_D ? "yes" : "no") << std::endl;
    std::cout << "  Warmup rounds:                                     "
      << pre_repetitions << std::endl;
    std::cout << "  Repetitions:                                       "
      << repetitions << std::endl;
    std::cout << "  Quadrature order for sing. int.:                   "
      << qo_sng << std::endl;
    std::cout << "  Quadrature order for reg. int.:                    "
      << qo_reg << std::endl;
    std::cout << "  GPU algorithm version:                             "
      << gpu_alg << std::endl;
  }

  std::string mesh_file = "";
  int space_refine = 0;
  lo n_timeslices = 8;
  sc end_time = 1;
  sc heat_capacity_constant = 0.5;
  bool do_in_memory = false;
  bool do_onthefly_cpu = false;
  bool do_onthefly_gpu = false;
  bool do_V = false;
  bool do_K = false;
  bool do_A = false;
  bool do_D = false;
  int pre_repetitions = 2;
  int repetitions = 10;
  int qo_sng = 4;
  int qo_reg = 4;
  int gpu_alg = 4;
}; // struct config





int main( int argc, char * argv[] ) {

  config c( argc, argv );
  
  timer tm_init, tm_check;
  timer tm_Vma, tm_Vmm, tm_Vfc, tm_Vfg;
  timer tm_Kma, tm_Kmm, tm_Kfc, tm_Kfg;
  timer tm_Ama, tm_Amm, tm_Afc, tm_Afg;
  timer tm_Dma, tm_Dmm, tm_Dfc, tm_Dfg;

  bool printCheckErrors = true;
  
  besthea::settings::output_verbosity.timers = 2;
  besthea::settings::output_verbosity.onthefly_loadbalance = 1;

  tm_init.start();

  //srand(time(nullptr));

  // load spatial mesh from file and refine
  triangular_surface_mesh space_mesh;
  space_mesh.load( c.mesh_file );
  space_mesh.refine( c.space_refine );

  // create spacetime mesh as a tensor product of spatial and temporal meshes
  uniform_spacetime_tensor_mesh spacetime_mesh( space_mesh, c.end_time, c.n_timeslices );
  uniform_spacetime_tensor_mesh_gpu gpu_spacetime_mesh( spacetime_mesh );

  // print some info
  c.dump();
  spacetime_mesh.print_info();

  // boundary element spaces
  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );

  // matrix preparation
  spacetime_heat_sl_kernel_antiderivative  kernel_v( c.heat_capacity_constant );
  spacetime_heat_dl_kernel_antiderivative  kernel_k( c.heat_capacity_constant );
  spacetime_heat_adl_kernel_antiderivative kernel_a( c.heat_capacity_constant );
  spacetime_heat_hs_kernel_antiderivative  kernel_d( c.heat_capacity_constant );
  block_lower_triangular_toeplitz_matrix V_mem; // single layer operator
  block_lower_triangular_toeplitz_matrix K_mem; // double layer operator
  block_lower_triangular_toeplitz_matrix A_mem; // adjoint double layer operator
  block_lower_triangular_toeplitz_matrix D_mem; // hypersingular operator
  uniform_spacetime_be_assembler         assembler_v(kernel_v, space_p0, space_p0,                     c.qo_sng, c.qo_reg);
  uniform_spacetime_be_matrix_onthefly_cpu V_fly_cpu(kernel_v, space_p0, space_p0,                     c.qo_sng, c.qo_reg);
  uniform_spacetime_be_matrix_onthefly_gpu V_fly_gpu(kernel_v, space_p0, space_p0, gpu_spacetime_mesh, c.qo_sng, c.qo_reg, c.gpu_alg);
  uniform_spacetime_be_assembler         assembler_k(kernel_k, space_p0, space_p1,                     c.qo_sng, c.qo_reg);
  uniform_spacetime_be_matrix_onthefly_cpu K_fly_cpu(kernel_k, space_p0, space_p1,                     c.qo_sng, c.qo_reg);
  uniform_spacetime_be_matrix_onthefly_gpu K_fly_gpu(kernel_k, space_p0, space_p1, gpu_spacetime_mesh, c.qo_sng, c.qo_reg, c.gpu_alg);
  uniform_spacetime_be_assembler         assembler_a(kernel_a, space_p1, space_p0,                     c.qo_sng, c.qo_reg);
  uniform_spacetime_be_matrix_onthefly_cpu A_fly_cpu(kernel_a, space_p1, space_p0,                     c.qo_sng, c.qo_reg);
  uniform_spacetime_be_matrix_onthefly_gpu A_fly_gpu(kernel_a, space_p1, space_p0, gpu_spacetime_mesh, c.qo_sng, c.qo_reg, c.gpu_alg);
  uniform_spacetime_be_assembler         assembler_d(kernel_d, space_p1, space_p1,                     c.qo_sng, c.qo_reg);
  uniform_spacetime_be_matrix_onthefly_cpu D_fly_cpu(kernel_d, space_p1, space_p1,                     c.qo_sng, c.qo_reg);
  uniform_spacetime_be_matrix_onthefly_gpu D_fly_gpu(kernel_d, space_p1, space_p1, gpu_spacetime_mesh, c.qo_sng, c.qo_reg, c.gpu_alg);

  // initialize vectors
  block_vector xV  (c.n_timeslices, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVm (c.n_timeslices, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVfc(c.n_timeslices, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVfg(c.n_timeslices, spacetime_mesh.get_n_spatial_elements(), false);
  for (lo b = 0; b < c.n_timeslices; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      xV.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      yVm.set(b, i, 2);
      yVfc.set(b, i, 2);
      yVfg.set(b, i, 2);
    }
  }
  block_vector xK  (c.n_timeslices, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yKm (c.n_timeslices, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yKfc(c.n_timeslices, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yKfg(c.n_timeslices, spacetime_mesh.get_n_spatial_elements(), false);
  for (lo b = 0; b < c.n_timeslices; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      xK.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      yKm.set(b, i, 2);
      yKfc.set(b, i, 2);
      yKfg.set(b, i, 2);
    }    
  }
  block_vector xA  (c.n_timeslices, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yAm (c.n_timeslices, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yAfc(c.n_timeslices, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yAfg(c.n_timeslices, spacetime_mesh.get_n_spatial_nodes(), false);
  for (lo b = 0; b < c.n_timeslices; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      xA.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      yAm.set(b, i, 2);
      yAfc.set(b, i, 2);
      yAfg.set(b, i, 2);
    }    
  }
  block_vector xD  (c.n_timeslices, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yDm (c.n_timeslices, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yDfc(c.n_timeslices, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yDfg(c.n_timeslices, spacetime_mesh.get_n_spatial_nodes(), false);
  for (lo b = 0; b < c.n_timeslices; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      xD.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      yDm.set(b, i, 2);
      yDfc.set(b, i, 2);
      yDfg.set(b, i, 2);
    }
  }

  // coefficients for GEMV: y = alpha * A * x + beta * y
  sc alpha = 3;
  sc beta = 5;

  tm_init.stop();



  // all the assembly and multiplications

  if(c.do_V) {
    if(c.do_in_memory) {
      printf("V mem assembly\n");
      for(int i = 0; i < c.pre_repetitions; i++) { V_mem.clear(); assembler_v.assemble( V_mem ); }
      tm_Vma.start();
      for(int i = 0; i < c.repetitions; i++) { V_mem.clear(); assembler_v.assemble( V_mem ); }
      tm_Vma.stop();

      printf("V mem multiply\n");
      for(int i = 0; i < c.pre_repetitions; i++) V_mem.apply(xV, yVm, false, alpha, beta);
      tm_Vmm.start();
      for(int i = 0; i < c.repetitions; i++) V_mem.apply(xV, yVm, false, alpha, beta);
      tm_Vmm.stop();

      V_mem.clear();
    }
    if(c.do_onthefly_cpu) {
      printf("V fly cpu\n");
      for(int i = 0; i < c.pre_repetitions; i++) V_fly_cpu.apply(xV, yVfc, false, alpha, beta);
      tm_Vfc.start();
      for(int i = 0; i < c.repetitions; i++) V_fly_cpu.apply(xV, yVfc, false, alpha, beta);
      tm_Vfc.stop();
    }
    if(c.do_onthefly_gpu) {
      printf("V fly gpu\n");
      for(int i = 0; i < c.pre_repetitions; i++) V_fly_gpu.apply(xV, yVfg, false, alpha, beta);
      tm_Vfg.start();
      for(int i = 0; i < c.repetitions; i++) V_fly_gpu.apply(xV, yVfg, false, alpha, beta);
      tm_Vfg.stop();
    }
  }



  if(c.do_K) {
    if(c.do_in_memory) {
      printf("K mem assembly\n");
      for(int i = 0; i < c.pre_repetitions; i++) { K_mem.clear(); assembler_k.assemble( K_mem ); }
      tm_Kma.start();
      for(int i = 0; i < c.repetitions; i++) { K_mem.clear(); assembler_k.assemble( K_mem ); }
      tm_Kma.stop();
      
      printf("K mem multiply\n");
      for(int i = 0; i < c.pre_repetitions; i++) K_mem.apply(xK, yKm, false, alpha, beta);
      tm_Kmm.start();
      for(int i = 0; i < c.repetitions; i++) K_mem.apply(xK, yKm, false, alpha, beta);
      tm_Kmm.stop();

      K_mem.clear();
    }
    if(c.do_onthefly_cpu) {
      printf("K fly cpu\n");
      for(int i = 0; i < c.pre_repetitions; i++) K_fly_cpu.apply(xK, yKfc, false, alpha, beta);
      tm_Kfc.start();
      for(int i = 0; i < c.repetitions; i++) K_fly_cpu.apply(xK, yKfc, false, alpha, beta);
      tm_Kfc.stop();
    }
    if(c.do_onthefly_gpu) {
      printf("K fly gpu\n");
      for(int i = 0; i < c.pre_repetitions; i++) K_fly_gpu.apply(xK, yKfg, false, alpha, beta);
      tm_Kfg.start();
      for(int i = 0; i < c.repetitions; i++) K_fly_gpu.apply(xK, yKfg, false, alpha, beta);
      tm_Kfg.stop();
    }
  }


  // A and K are a little different by quadrature errors on elements with shared edge
  if(c.do_A) {
    if(c.do_in_memory) {
      printf("A mem assembly\n");
      for(int i = 0; i < c.pre_repetitions; i++) { A_mem.clear(); assembler_a.assemble( A_mem ); }
      tm_Ama.start();
      for(int i = 0; i < c.repetitions; i++) { A_mem.clear(); assembler_a.assemble( A_mem ); }
      tm_Ama.stop();
      
      printf("A mem multiply\n");
      for(int i = 0; i < c.pre_repetitions; i++) A_mem.apply(xA, yAm, false, alpha, beta);
      tm_Amm.start();
      for(int i = 0; i < c.repetitions; i++) A_mem.apply(xA, yAm, false, alpha, beta);
      tm_Amm.stop();

      A_mem.clear();
    }
    if(c.do_onthefly_cpu) {
      printf("A fly cpu\n");
      for(int i = 0; i < c.pre_repetitions; i++) A_fly_cpu.apply(xA, yAfc, false, alpha, beta);
      tm_Afc.start();
      for(int i = 0; i < c.repetitions; i++) A_fly_cpu.apply(xA, yAfc, false, alpha, beta);
      tm_Afc.stop();
    }
    if(c.do_onthefly_gpu) {
      printf("A fly gpu\n");
      for(int i = 0; i < c.pre_repetitions; i++) A_fly_gpu.apply(xA, yAfg, false, alpha, beta);
      tm_Afg.start();
      for(int i = 0; i < c.repetitions; i++) A_fly_gpu.apply(xA, yAfg, false, alpha, beta);
      tm_Afg.stop();
    }
  }



  if(c.do_D) {
    if(c.do_in_memory) {
      printf("D mem assembly\n");
      for(int i = 0; i < c.pre_repetitions; i++) { D_mem.clear(); assembler_d.assemble( D_mem ); }
      tm_Dma.start();
      for(int i = 0; i < c.repetitions; i++) { D_mem.clear(); assembler_d.assemble( D_mem ); }
      tm_Dma.stop();
      
      printf("D mem multiply\n");
      for(int i = 0; i < c.pre_repetitions; i++) D_mem.apply(xD, yDm, false, alpha, beta);
      tm_Dmm.start();
      for(int i = 0; i < c.repetitions; i++) D_mem.apply(xD, yDm, false, alpha, beta);
      tm_Dmm.stop();

      D_mem.clear();
    }
    if(c.do_onthefly_cpu) {
      printf("D fly cpu\n");
      for(int i = 0; i < c.pre_repetitions; i++) D_fly_cpu.apply(xD, yDfc, false, alpha, beta);
      tm_Dfc.start();
      for(int i = 0; i < c.repetitions; i++) D_fly_cpu.apply(xD, yDfc, false, alpha, beta);
      tm_Dfc.stop();
    }
    if(c.do_onthefly_gpu) {
      printf("D fly gpu\n");
      for(int i = 0; i < c.pre_repetitions; i++) D_fly_gpu.apply(xD, yDfg, false, alpha, beta);
      tm_Dfg.start();
      for(int i = 0; i < c.repetitions; i++) D_fly_gpu.apply(xD, yDfg, false, alpha, beta);
      tm_Dfg.stop();
    }
  }
  


  // checking the results

  tm_check.start();

  if(c.do_in_memory && c.do_onthefly_cpu && c.do_V)
  {
    bool equalVc = true;
    for (lo b = 0; b < c.n_timeslices; b++) {
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
    printf("Vectors Vc are%s equal!\n", (equalVc ? "" : " NOT"));
  }

  if(c.do_in_memory && c.do_onthefly_gpu && c.do_V)
  {
    bool equalVg = true;
    for (lo b = 0; b < c.n_timeslices; b++) {
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
    printf("Vectors Vg are%s equal!\n", (equalVg ? "" : " NOT"));
  }

  if(c.do_in_memory && c.do_onthefly_cpu && c.do_K)
  {
    bool equalKc = true;
    for (lo b = 0; b < c.n_timeslices; b++) {
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
    printf("Vectors Kc are%s equal!\n", (equalKc ? "" : " NOT"));
  }
  
  if(c.do_in_memory && c.do_onthefly_gpu && c.do_K)
  {
    bool equalKg = true;
    for (lo b = 0; b < c.n_timeslices; b++) {
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
    printf("Vectors Kg are%s equal!\n", (equalKg ? "" : " NOT"));
  }

  if(c.do_in_memory && c.do_onthefly_cpu && c.do_A)
  {
    bool equalAc = true;
    for (lo b = 0; b < c.n_timeslices; b++) {
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
    printf("Vectors Ac are%s equal!\n", (equalAc ? "" : " NOT"));
  }
  
  if(c.do_in_memory && c.do_onthefly_gpu && c.do_A)
  {
    bool equalAg = true;
    for (lo b = 0; b < c.n_timeslices; b++) {
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
    printf("Vectors Ag are%s equal!\n", (equalAg ? "" : " NOT"));
  }

  if(c.do_in_memory && c.do_onthefly_cpu && c.do_D)
  {
    bool equalDc = true;
    for (lo b = 0; b < c.n_timeslices; b++) {
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
    printf("Vectors Dc are%s equal!\n", (equalDc ? "" : " NOT"));
  }
  
  if(c.do_in_memory && c.do_onthefly_gpu && c.do_D)
  {
    bool equalDg = true;
    for (lo b = 0; b < c.n_timeslices; b++) {
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
    printf("Vectors Dg are%s equal!\n", (equalDg ? "" : " NOT"));
  }

  tm_check.stop();



  // print results
  printf("\n");
  printf("Time init:  %10.6f\n", tm_init.get_elapsed_time_in_seconds());
  printf("Time check: %10.6f\n", tm_check.get_elapsed_time_in_seconds());
  printf("\n");
  printf("Time multiply (avg)    V            K            A            D\n");
  printf("mem_assemble        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vma.get_elapsed_time_in_seconds()/c.repetitions, tm_Kma.get_elapsed_time_in_seconds()/c.repetitions, tm_Ama.get_elapsed_time_in_seconds()/c.repetitions, tm_Dma.get_elapsed_time_in_seconds()/c.repetitions);
  printf("mem_multiply        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vmm.get_elapsed_time_in_seconds()/c.repetitions, tm_Kmm.get_elapsed_time_in_seconds()/c.repetitions, tm_Amm.get_elapsed_time_in_seconds()/c.repetitions, tm_Dmm.get_elapsed_time_in_seconds()/c.repetitions);
  printf("mem_total           %10.6f   %10.6f   %10.6f   %10.6f\n", (tm_Vma.get_elapsed_time_in_seconds() + tm_Vmm.get_elapsed_time_in_seconds())/c.repetitions, (tm_Kma.get_elapsed_time_in_seconds() + tm_Kmm.get_elapsed_time_in_seconds())/c.repetitions, (tm_Ama.get_elapsed_time_in_seconds() + tm_Amm.get_elapsed_time_in_seconds())/c.repetitions, (tm_Dma.get_elapsed_time_in_seconds() + tm_Dmm.get_elapsed_time_in_seconds())/c.repetitions);
  printf("fly_mult_cpu        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vfc.get_elapsed_time_in_seconds()/c.repetitions, tm_Kfc.get_elapsed_time_in_seconds()/c.repetitions, tm_Afc.get_elapsed_time_in_seconds()/c.repetitions, tm_Dfc.get_elapsed_time_in_seconds()/c.repetitions);
  printf("fly_mult_gpu        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vfg.get_elapsed_time_in_seconds()/c.repetitions, tm_Kfg.get_elapsed_time_in_seconds()/c.repetitions, tm_Afg.get_elapsed_time_in_seconds()/c.repetitions, tm_Dfg.get_elapsed_time_in_seconds()/c.repetitions);

  

  return 0;
}
