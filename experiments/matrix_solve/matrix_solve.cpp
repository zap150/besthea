#include "besthea/besthea.h"
#include "besthea/uniform_spacetime_be_onthefly_matrix_cpu.h"
#include "besthea/uniform_spacetime_be_onthefly_matrix_gpu.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace besthea;
using namespace besthea::onthefly;
using namespace besthea::mesh;
using namespace besthea::linear_algebra;
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

  srand(time(nullptr));

  std::string file;

  // default values
  file = "../examples/mesh_files/cube_12.txt";
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

  space_mesh.print_info( );
  spacetime_mesh.print_info( );

  timer t;

  uniform_spacetime_be_space< basis_tri_p0 > space_p0( spacetime_mesh );
  uniform_spacetime_be_space< basis_tri_p1 > space_p1( spacetime_mesh );

  // numerical quadrature orders
  lo order_sing = 4;  // for singular integrals (adjacent or identical spatial elements)
  lo order_reg  = 4;  // disjoint spatial elements

  // create matrix assembler
  spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );
  block_lower_triangular_toeplitz_matrix V_mem;
  uniform_spacetime_be_onthefly_matrix_cpu V_fly_cpu(kernel_v, space_p0, space_p0, order_sing, order_reg);
  uniform_spacetime_be_onthefly_matrix_gpu V_fly_gpu(kernel_v, space_p0, space_p0, order_sing, order_reg);
  uniform_spacetime_be_assembler assembler_v(kernel_v, space_p0, space_p0, order_sing, order_reg );

  block_vector bV  (n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector xVm (n_timesteps, spacetime_mesh.get_n_spatial_elements(), true);
  block_vector xVfc(n_timesteps, spacetime_mesh.get_n_spatial_elements(), true);
  block_vector xVfg(n_timesteps, spacetime_mesh.get_n_spatial_elements(), true);
  block_vector yVm (n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVfc(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVfg(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      bV.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
  }

  sc err = 1e-10;
  sc errVm = err, errVfc = err, errVfg = err;
  lo iters = 10000;
  lo itersVm = iters, itersVfc = iters, itersVfg = iters;

  

  t.reset( "InMemory V  assemble" );
  assembler_v.assemble( V_mem );
  t.measure( );
  t.reset( "InMemory V  solve" );
  V_mem.mkl_fgmres_solve(bV, xVm, errVm, itersVm);
  t.measure( );

  t.reset( "OnTheFly Vc solve" );
  V_fly_cpu.mkl_fgmres_solve(bV, xVfc, errVfc, itersVfc);
  t.measure();
  
  t.reset( "OnTheFly Vg solve" );
  V_fly_gpu.mkl_fgmres_solve(bV, xVfg, errVfg, itersVfg);
  t.measure();

  V_mem.apply(xVm, yVm);
  V_fly_cpu.apply(xVfc, yVfc);
  V_fly_gpu.apply(xVfg, yVfg);
  


  std::cout << "Iters:  mem " << itersVm << "\t\tfly cpu " << itersVfc << "\t\tfly gpu " << itersVfg << "\n";
  std::cout << "Errors: mem " << errVm   << "\tfly cpu " << errVfc   << "\tfly gpu " << errVfg   << "\n";


  bool equal_xVc = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = xVm.get(b, i);
      sc vf = xVfc.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        std::cout << "Solutions Vc dont match: B" << b << " I" << i << " " << vm << " " << vf << "\n";
        equal_xVc = false;
      }
    }
  }
  bool equal_xVg = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = xVm.get(b, i);
      sc vf = xVfg.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        std::cout << "Solutions Vg dont match: B" << b << " I" << i << " " << vm << " " << vf << "\n";
        equal_xVg = false;
      }
    }
  }

  bool equal_Vm = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc v1 = bV.get(b, i);
      sc v2 = yVm.get(b, i);
      if( std::abs((v1 - v2) / v1) > 1e-6 ) {
        std::cout << "Check multiplications mem dont match: B" << b << " I" << i << " " << v1 << " " << v2 << "\n";
        equal_Vm = false;
      }
    }
  }
  bool equal_Vfc = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc v1 = bV.get(b, i);
      sc v2 = yVfc.get(b, i);
      if( std::abs((v1 - v2) / v1) > 1e-6 ) {
        std::cout << "Check multiplications fly cpu dont match: B" << b << " I" << i << " " << v1 << " " << v2 << "\n";
        equal_Vfc = false;
      }
    }
  }
  bool equal_Vfg = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc v1 = bV.get(b, i);
      sc v2 = yVfg.get(b, i);
      if( std::abs((v1 - v2) / v1) > 1e-6 ) {
        std::cout << "Check multiplications fly gpu dont match: B" << b << " I" << i << " " << v1 << " " << v2 << "\n";
        equal_Vfg = false;
      }
    }
  }

  std::cout << "Solutions Vc are" << (equal_xVc ? "" : " NOT") << " equal!\n";
  std::cout << "Solutions Vg are" << (equal_xVg ? "" : " NOT") << " equal!\n";
  std::cout << "Check multiplications mem     are" << (equal_Vm  ? "" : " NOT") << " equal!\n";
  std::cout << "Check multiplications fly cpu are" << (equal_Vfc ? "" : " NOT") << " equal!\n";
  std::cout << "Check multiplications fly gpu are" << (equal_Vfg ? "" : " NOT") << " equal!\n";
  
  

  return 0;
}
