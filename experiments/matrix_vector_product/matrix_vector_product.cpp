#include "besthea/besthea.h"
#include "besthea/uniform_spacetime_be_onthefly_matrix_cpu.h"
#include "besthea/uniform_spacetime_be_onthefly_matrix_gpu.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace besthea;
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
  int refine = 2;
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
  spacetime_heat_dl_kernel_antiderivative kernel_k( cauchy_data::_alpha );
  block_lower_triangular_toeplitz_matrix V_mem;
  block_lower_triangular_toeplitz_matrix K_mem;
  uniform_spacetime_be_onthefly_matrix_cpu V_fly(kernel_v, space_p0, space_p0, order_sing, order_reg);
  uniform_spacetime_be_onthefly_matrix_cpu K_fly(kernel_k, space_p0, space_p1, order_sing, order_reg);
  uniform_spacetime_be_assembler assembler_v(kernel_v, space_p0, space_p0, order_sing, order_reg );
  uniform_spacetime_be_assembler assembler_k(kernel_k, space_p0, space_p1, order_sing, order_reg );

  block_vector xV (n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVm(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yVf(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      xV.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      yVm.set(b, i, 2);
      yVf.set(b, i, 2);
    }    
  }
  block_vector xK (n_timesteps, spacetime_mesh.get_n_spatial_nodes(), false);
  block_vector yKm(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  block_vector yKf(n_timesteps, spacetime_mesh.get_n_spatial_elements(), false);
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_nodes(); i++) {
      xK.set(b, i, (1000.0 * rand()) / RAND_MAX);
    }
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      yKm.set(b, i, 2);
      yKf.set(b, i, 2);
    }    
  }

  sc alpha = 3;
  sc beta = 5;

  


  t.reset( "InMemory V" );
  assembler_v.assemble( V_mem );
  V_mem.apply(xV, yVm, false, alpha, beta);
  t.measure( );

  t.reset( "OnTheFly V" );
  V_fly.apply(xV, yVf, false, alpha, beta);
  t.measure();


  t.reset( "InMemory K" );
  assembler_k.assemble( K_mem );
  K_mem.apply(xK, yKm, false, alpha, beta);
  t.measure( );

  t.reset( "OnTheFly K" );
  K_fly.apply(xK, yKf, false, alpha, beta);
  t.measure();
  



  bool equalV = true;
  bool equalK = true;
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yVm.get(b, i);
      sc vf = yVf.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        std::cout << "Vectors V dont match: B" << b << " I" << i << " " << vm << " " << vf << "\n";
        equalV = false;
      }
    }
  }
  for (lo b = 0; b < n_timesteps; b++) {
    for (lo i = 0; i < spacetime_mesh.get_n_spatial_elements(); i++) {
      sc vm = yKm.get(b, i);
      sc vf = yKf.get(b, i);
      if( std::abs((vm - vf) / vm) > 1e-6 ) {
        std::cout << "Vectors K dont match: B" << b << " I" << i << " " << vm << " " << vf << "\n";
        equalK = false;
      }
    }    
  }

  if(equalV) {
    std::cout << "Vectors V are equal!\n";
  }
  if(equalK) {
    std::cout << "Vectors K are equal!\n";
  }
  
  

  return 0;
}
