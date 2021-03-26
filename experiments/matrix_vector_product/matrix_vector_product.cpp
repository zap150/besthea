#include "besthea/besthea.h"
#include "besthea/uniform_spacetime_be_matrix_onthefly_cpu.h"
#include "besthea/uniform_spacetime_be_matrix_onthefly_gpu.h"

#include <cstdlib>
#include <filesystem>
#include <cstdio>

using namespace besthea;
using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::linear_algebra::onthefly;
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
  
  time_measurer tm_init, tm_check;
  time_measurer tm_Vma, tm_Vmm, tm_Vfc, tm_Vfg;
  time_measurer tm_Kma, tm_Kmm, tm_Kfc, tm_Kfg;
  time_measurer tm_Ama, tm_Amm, tm_Afc, tm_Afg;
  time_measurer tm_Dma, tm_Dmm, tm_Dfc, tm_Dfg;

  tm_init.start();

  //srand(time(nullptr));
  
  besthea::settings::output_verbosity.timers = 0;
  besthea::settings::output_verbosity.onthefly_loadbalance = 0;

  std::string file;

  // default values
  file = "../../besthea/examples/mesh_files/cube_192.txt";
  int refine = 1;
  lo n_timesteps = 8;
  sc end_time = 1.0;

  // read from commandl line
  if ( argc > 1 ) {
    file.assign( argv[ 1 ] );
  }
  if ( argc > 2 ) {
    n_timesteps = std::atoi( argv[ 2 ] );
  }
  if ( argc > 3 ) {
    end_time = std::atof( argv[ 3 ] );
  }
  if ( argc > 4 ) {
    refine = std::atoi( argv[ 4 ] );
  }

  // load spatial mesh from file and refine it
  triangular_surface_mesh space_mesh;
  space_mesh.load( file );
  space_mesh.refine( refine );

  // refine number of timesteps
  n_timesteps *= std::exp2( 2 * refine );

  // create spacetime mesh as a tensor product of spatial and temporal meshes
  uniform_spacetime_tensor_mesh spacetime_mesh(
    space_mesh, end_time, n_timesteps );
  uniform_spacetime_tensor_mesh_gpu gpu_spacetime_mesh(spacetime_mesh);

  space_mesh.print_info( );
  spacetime_mesh.print_info( );

  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );

  // numerical quadrature orders
  lo order_sng_V = 1;  // for singular integrals (adjacent or identical spatial elements)
  lo order_reg_V = 1;  // disjoint spatial elements
  lo order_sng_K = 1;
  lo order_reg_K = 1;
  lo order_sng_A = 1;
  lo order_reg_A = 1;
  lo order_sng_D = 1;
  lo order_reg_D = 1;

  int gpuAlgVer = 4;

  bool doMem    = true;
  bool doFlyCpu = true;
  bool doFlyGpu = true;

  // create matrix assembler
  spacetime_heat_sl_kernel_antiderivative  kernel_v( cauchy_data::_alpha );
  spacetime_heat_dl_kernel_antiderivative  kernel_k( cauchy_data::_alpha );
  spacetime_heat_adl_kernel_antiderivative kernel_a( cauchy_data::_alpha );
  spacetime_heat_hs_kernel_antiderivative  kernel_d( cauchy_data::_alpha );
  block_lower_triangular_toeplitz_matrix V_mem; // single layer operator
  block_lower_triangular_toeplitz_matrix K_mem; // double layer operator
  block_lower_triangular_toeplitz_matrix A_mem; // adjoint double layer operator
  block_lower_triangular_toeplitz_matrix D_mem; // hypersingular operator
  uniform_spacetime_be_assembler         assembler_v(kernel_v, space_p0, space_p0, order_sng_V, order_reg_V);
  uniform_spacetime_be_matrix_onthefly_cpu V_fly_cpu(kernel_v, space_p0, space_p0, order_sng_V, order_reg_V);
  uniform_spacetime_be_matrix_onthefly_gpu V_fly_gpu(kernel_v, space_p0, space_p0, order_sng_V, order_reg_V, gpu_spacetime_mesh, gpuAlgVer);
  uniform_spacetime_be_assembler         assembler_k(kernel_k, space_p0, space_p1, order_sng_K, order_reg_K);
  uniform_spacetime_be_matrix_onthefly_cpu K_fly_cpu(kernel_k, space_p0, space_p1, order_sng_K, order_reg_K);
  uniform_spacetime_be_matrix_onthefly_gpu K_fly_gpu(kernel_k, space_p0, space_p1, order_sng_K, order_reg_K, gpu_spacetime_mesh, gpuAlgVer);
  uniform_spacetime_be_assembler         assembler_a(kernel_k, space_p0, space_p1, order_sng_A, order_reg_A);
  //uniform_spacetime_be_matrix_onthefly_cpu A_fly_cpu(kernel_a, space_p1, space_p0, order_sng_A, order_reg_A);
  //uniform_spacetime_be_matrix_onthefly_gpu A_fly_gpu(kernel_a, space_p1, space_p0, order_sng_A, order_reg_A, gpu_spacetime_mesh, gpuAlgVer);
  uniform_spacetime_be_assembler         assembler_d(kernel_d, space_p1, space_p1, order_sng_D, order_reg_D);
  uniform_spacetime_be_matrix_onthefly_cpu D_fly_cpu(kernel_d, space_p1, space_p1, order_sng_D, order_reg_D);
  uniform_spacetime_be_matrix_onthefly_gpu D_fly_gpu(kernel_d, space_p1, space_p1, order_sng_D, order_reg_D, gpu_spacetime_mesh, gpuAlgVer);

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

  

  tm_Vma.start(); printf("V mem assembly\n");
  if(doMem) assembler_v.assemble( V_mem );
  tm_Vma.stop();
  tm_Vmm.start(); printf("V mem multiply\n");
  if(doMem) V_mem.apply(xV, yVm, false, alpha, beta);
  tm_Vmm.stop();

  tm_Vfc.start(); printf("V fly cpu\n");
  if(doFlyCpu) V_fly_cpu.apply(xV, yVfc, false, alpha, beta);
  tm_Vfc.stop();
  tm_Vfg.start(); printf("V fly gpu\n");
  //for(int r = 0; r < 10; r++)
  if(doFlyGpu) V_fly_gpu.apply(xV, yVfg, false, alpha, beta);
  tm_Vfg.stop();



  tm_Kma.start(); printf("K mem assembly\n");
  if(doMem) assembler_k.assemble( K_mem );
  tm_Kma.stop();
  tm_Kmm.start(); printf("K mem multiply\n");
  if(doMem) K_mem.apply(xK, yKm, false, alpha, beta);
  tm_Kmm.stop();

  tm_Kfc.start(); printf("K fly cpu\n");
  if(doFlyCpu) K_fly_cpu.apply(xK, yKfc, false, alpha, beta);
  tm_Kfc.stop();
  tm_Kfg.start(); printf("K fly gpu\n");
  if(doFlyGpu) K_fly_gpu.apply(xK, yKfg, false, alpha, beta);
  tm_Kfg.stop();



  tm_Ama.start(); printf("A mem assembly\n");
  if(doMem) assembler_a.assemble( A_mem );
  tm_Ama.stop();
  tm_Amm.start(); printf("A mem multiply\n");
  if(doMem) A_mem.apply(xA, yAm, true, alpha, beta); // A is just K with transposed blocks
  tm_Amm.stop();
  
  tm_Afc.start(); printf("A fly cpu\n");
  //if(doFlyCpu) A_fly_cpu.apply(xA, yAfc, false, alpha, beta);
  tm_Afc.stop();
  tm_Afg.start(); printf("A fly gpu\n");
  //if(doFlyGpu) A_fly_gpu.apply(xA, yAfg, false, alpha, beta);
  tm_Afg.stop();



  tm_Dma.start(); printf("D mem assembly\n");
  if(doMem) assembler_d.assemble( D_mem );
  tm_Dma.stop();
  tm_Dmm.start(); printf("D mem multiply\n");
  if(doMem) D_mem.apply(xD, yDm, false, alpha, beta);
  tm_Dmm.stop();

  tm_Dfc.start(); printf("D fly cpu\n");
  if(doFlyCpu) D_fly_cpu.apply(xD, yDfc, false, alpha, beta);
  tm_Dfc.stop();
  tm_Dfg.start(); printf("D fly gpu\n");
  if(doFlyGpu) D_fly_gpu.apply(xD, yDfg, false, alpha, beta);
  tm_Dfg.stop();
  


  tm_check.start();

  bool equalVc = true;
  bool equalVg = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yVm.get(b, i);
      sc vf = yVfc.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        //printf("Vectors Vc dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalVc = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yVm.get(b, i);
      sc vf = yVfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        //printf("Vectors Vg dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
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
        //printf("Vectors Kc dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalKc = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yKm.get(b, i);
      sc vf = yKfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        //printf("Vectors Kg dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
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
        //printf("Vectors Ac dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalAc = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      sc vm = yAm.get(b, i);
      sc vf = yAfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        //printf("Vectors Ag dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
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
        //printf("Vectors Dc dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalDc = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      sc vm = yDm.get(b, i);
      sc vf = yDfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        //printf("Vectors Dg dont match: B%ld I%ld %f %f\n", b, i, vm, vf);
        equalDg = false;
      }
    }
  }

  tm_check.stop();



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
  printf("Time multiply       V            K            A            D\n");
  printf("mem assemble     %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vma.get_time(), tm_Kma.get_time(), tm_Ama.get_time(), tm_Dma.get_time());
  printf("mem multiply     %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vmm.get_time(), tm_Kmm.get_time(), tm_Amm.get_time(), tm_Dmm.get_time());
  printf("mem total        %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vma.get_time() + tm_Vmm.get_time(), tm_Kma.get_time() + tm_Kmm.get_time(), tm_Ama.get_time() + tm_Amm.get_time(), tm_Dma.get_time() + tm_Dmm.get_time());
  printf("fly mult cpu     %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vfc.get_time(), tm_Kfc.get_time(), tm_Afc.get_time(), tm_Dfc.get_time());
  printf("fly mult gpu     %10.6f   %10.6f   %10.6f   %10.6f\n", tm_Vfg.get_time(), tm_Kfg.get_time(), tm_Afg.get_time(), tm_Dfg.get_time());

  

  return 0;
}
