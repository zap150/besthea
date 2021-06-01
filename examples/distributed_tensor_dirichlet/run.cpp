/*
Copyright (c) 2020, VSB - Technical University of Ostrava and Graz University of
Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the names of VSB - Technical University of  Ostrava and Graz
  University of Technology nor the names of its contributors may be used to
  endorse or promote products derived from this software without specific prior
  written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL VSB - TECHNICAL UNIVERSITY OF OSTRAVA AND
GRAZ UNIVERSITY OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * Solves the first boundary integral equation (possibly with non-zero initial
 * conditions) for the unknown Neumann datum and evaluates it in a provided set
 * of grid points.
 */

#include "besthea/besthea.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <lyra/lyra.hpp>

using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem;
using namespace besthea::tools;

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
    sc dummy = 0.0;
    sc value = std::pow( 4.0 * M_PI * _alpha * _shift, -1.5 )
      * std::exp( -norm2 / ( 4.0 * _alpha * _shift + dummy ) );

    return value;
  }

  static constexpr sc _alpha{ 1.0 };
  static constexpr std::array< sc, 3 > _y{ 1.5, 1.5, 1.5 };
  static constexpr sc _shift{ 0.0 };
};  // struct cauchy_data

struct config {
  void dump( ) {
    std::cout << "config: " << std::endl;
    std::cout << "  Spatial mesh:                          " << spatial_file
              << std::endl;
    std::cout << "  No. of initial spatial refinements:    "
              << space_init_refine << std::endl;
    std::cout << "  End time:                              " << end_time
              << std::endl;
    std::cout << "  No. of timeslices:                     " << n_timeslices
              << std::endl;
    std::cout << "  No. of space-time refinements:         " << refine
              << std::endl;
    std::cout << "  No. of distribution tree levels:       "
              << distribution_tree_levels << std::endl;
    std::cout << "  Min. no. of elements for refinement:   "
              << n_min_elems_refine << std::endl;
    std::cout << "  Space-time box coupling coefficient:   "
              << st_coupling_coeff << std::endl;
    std::cout << "  Spatial truncation parameter:          " << trunc_space
              << std::endl;
    std::cout << "  Temporal expansion order:              " << temp_order
              << std::endl;
    std::cout << "  Spatial expansion order:               " << spat_order
              << std::endl;
  }

  // spatial mesh data
  std::string spatial_file = "";
  int space_init_refine = 0;
  // temporal mesh data
  sc end_time = 1.0;
  lo n_timeslices = 8;
  // number of refinements for space-time mesh
  int refine = 0;
  // parameters for distributed FMM
  int distribution_tree_levels = -1;
  int n_min_elems_refine = 800;
  sc st_coupling_coeff = 4.5;
  int trunc_space = 2;
  int temp_order = 4;
  int spat_order = 12;
};  // struct config

namespace {
  config configure( int argc, char * argv[] ) {
    config c;
    bool help = false;

    auto cli = lyra::help( help )
      | lyra::opt( c.spatial_file, "surface mesh" )[ "--mesh" ](
        "Surface mesh of the spatial domain, "
        "refined by the 'space_init_refine' parameter" )
          .required( )
      | lyra::opt( c.space_init_refine,
        "initial spatial refinement" )[ "--space_init_refine" ](
        "Number of initial refinements of the spatial mesh" )
          .required( )
      | lyra::opt( c.end_time, "end time" )[ "--endtime" ](
        "End time of the considered time interval. The start time is always "
        "0." )
          .required( )
      | lyra::opt( c.n_timeslices, "number of timeslices" )[ "--timeslices" ](
        "Number of time slices for the given mesh" )
          .required( )
      | lyra::opt( c.refine, "space-time refinement" )[ "--refine" ](
        "Number of refinements of the tensor product space-time mesh "
        "consisting of the spatial surface mesh (see 'spatial_file' and "
        "'space_init_refine') and the time slices (see 'end_time' and "
        "'n_timeslices'). One refinement step includes one uniform refinement "
        "of the spatial mesh and two uniform refinements in time." )
      | lyra::opt( c.distribution_tree_levels,
        "levels of the distribution tree" )[ "--dist_tree_levels" ](
        "Bound for the number of levels of the temporal tree which is used for "
        "the distribution of work among processes. It has to be greater than "
        "ceil( log_2( n_proc ) ) )" )
      | lyra::opt( c.n_min_elems_refine,
        "Minimal number of space-time elements needed for refinement" )
        [ "--n_min_elems_refine" ](
          "In the construction of the space-time cluster tree a cluster is "
          "refined only if it contains more than this number of elements." )
      | lyra::opt( c.st_coupling_coeff,
        "space time box coupling coefficient" )[ "--st_coupling_coeff" ](
        "This constant c determines the relation between the spatial and "
        "temporal half sizes h_x and h_t of the 4D boxes in the space-time "
        "cluster tree according to the criterion ( h_x^2 <= 4 c h_t )." )
      | lyra::opt(
        c.trunc_space, "Spatial truncation parameter" )[ "--trunc_space" ](
        "Determines the number of clusters considered for nearfield "
        "operations. In each linear space direction only 'trunc_space' "
        "clusters are considered, all others are neglected." )
      | lyra::opt( c.temp_order, "Temporal expansion order" )[ "--temp_order" ](
        "Temporal expansion order used for the kernel expansion in the FMM." )
      | lyra::opt( c.spat_order, "Spatial expansion order" )[ "--spat_order" ](
        "Spatial expansion order used for the kernel expansion in the "
        "FMM" );

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

    return c;
  }
}  // namespace

int main( int argc, char * argv[] ) {
  // initialize MPI
  int provided;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  int my_rank, n_processes;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
  MPI_Comm_size( comm, &n_processes );

  config c = configure( argc, argv );
  // set distribution tree levels if it is not initialized by the user
  lo distribution_tree_levels = c.distribution_tree_levels;
  lo min_distribution_tree_levels
    = std::max( 3, (int) std::ceil( std::log2( n_processes ) ) + 1 );
  if ( distribution_tree_levels < min_distribution_tree_levels ) {
    distribution_tree_levels = min_distribution_tree_levels;
  }
  // set some additional parameters:
  // orders of quadrature for computation of nearfield integrals
  lo order_sing = 4;
  lo order_reg = 4;
  // refine time in each refinement step twice:
  lo temp_refine_factor = 2;
  // GMRES parameters: precision and maximal number of iterations
  sc gmres_prec = 1e-8;
  lo gmres_iter = 250;
  // print information about setup
  if ( my_rank == 0 ) {
    c.dump( );
    std::cout << "total refinements in space:              "
              << c.space_init_refine + c.refine << std::endl;
    std::cout << "total refinements in time:               "
              << temp_refine_factor * c.refine << std::endl;
    std::cout << "number of MPI processes:                 " << n_processes
              << std::endl;
    std::cout << "max. number of OpenMP threads:           "
              << omp_get_max_threads( ) << std::endl;
    if ( c.distribution_tree_levels < min_distribution_tree_levels ) {
      std::cout
        << "WARNING: Unsupported choice of no. of distribution tree levels. "
           "Setting it to default value: "
        << distribution_tree_levels << std::endl;
    }
    std::cout << std::endl << "boundary datum:  " << std::endl;
    std::cout << "fundamental solution with alpha = " << cauchy_data::_alpha
              << " and source at tau = 0 and y = (";
    std::cout << cauchy_data::_y[ 0 ] << ", " << cauchy_data::_y[ 1 ] << ", "
              << cauchy_data::_y[ 2 ] << ")" << std::endl
              << std::endl;
  }

  // choose some assignment strategy for assignment of time clusters to
  // processes
  lo process_assignment_strategy = 1;

  // directory to store the files of the distributed space-time mesh which is
  // constructed in the following.
  std::string geometry_dir
    = "./distributed_tensor_dirichlet/temp_geometry_files/";
  std::filesystem::create_directory( geometry_dir );

  // parameters for distributed spacetime mesh
  // refinement of mesh within slices
  lo time_refinement = temp_refine_factor * c.refine;
  lo space_refinement = c.refine;

  MPI_Barrier( comm );
  timer t;

  // generation of distributed mesh: a single process takes care of this and
  // provides several files which are loaded from all processes
  std::string tree_vector_file = geometry_dir + "tree_structure.bin";
  std::string cluster_bounds_file = geometry_dir + "cluster_bounds.bin";
  std::string process_assignment_file = geometry_dir + "process_assignment.bin";

  lo status = 0;
  if ( my_rank == 0 ) {
    t.reset( "mesh generation" );

    // load time mesh defining slices and create temporal tree, whose clusters
    // are distributed among processes
    temporal_mesh time_mesh( 0, c.end_time, c.n_timeslices );
    lo n_min_time_elems = 2;
    time_cluster_tree time_tree(
      time_mesh, distribution_tree_levels, n_min_time_elems );
    // write temporal tree structure, cluster bounds and process assignment to
    // files. These will be used for the assembly of the distributed space-time
    // mesh
    time_tree.print_tree_structure( tree_vector_file );
    time_tree.print_cluster_bounds( cluster_bounds_file );
    time_tree.print_process_assignments( n_processes,
      process_assignment_strategy, process_assignment_file, status );
    if ( status == 0 ) {
      // generate the space-time mesh from the provided spatial mesh and time
      // slices.
      triangular_surface_mesh space_mesh( c.spatial_file );
      if ( c.space_init_refine > 0 ) {
        space_mesh.refine( c.space_init_refine );
      }
      spacetime_mesh_generator generator( space_mesh, c.end_time,
        c.n_timeslices, time_refinement, space_refinement );
      generator.generate( geometry_dir, "test_mesh", "txt" );
    }
    t.measure( );
  }
  // broadcast status to all clusters
  MPI_Bcast( (void *) &status, 1, get_index_type< lo >::MPI_LO( ), 0, comm );
  if ( status > 0 ) {
    if ( my_rank == 0 ) {
      std::cout << "Error in distribution of clusters to processes. Aborting."
                << std::endl;
    }
  } else {
    MPI_Barrier( comm );

    if ( my_rank == 0 ) {
      t.reset( "assembly of distributed mesh and tree" );
    }
    // construct distributed mesh
    distributed_spacetime_tensor_mesh distributed_mesh(
      geometry_dir + "test_mesh_d.txt", tree_vector_file, cluster_bounds_file,
      process_assignment_file, &comm );
    lo n_global_timesteps = distributed_mesh.get_n_temporal_elements( );
    lo n_global_space_elements
      = distributed_mesh.get_local_mesh( )->get_n_spatial_elements( );
    lo n_global_space_nodes
      = distributed_mesh.get_local_mesh( )->get_n_spatial_nodes( );

    // construct the distributed spacetime cluster tree
    lo n_max_levels_spacetime_tree = 20;
    distributed_spacetime_cluster_tree distributed_st_tree( distributed_mesh,
      n_max_levels_spacetime_tree, c.n_min_elems_refine, c.st_coupling_coeff,
      cauchy_data::_alpha, c.trunc_space, &comm, status );

    if ( status > 0 ) {
      if ( my_rank == 0 ) {
        std::cout << "Error in tree construction. Aborting." << std::endl;
      }
    } else {
      // declare boundary element spaces
      distributed_fast_spacetime_be_space< basis_tri_p0 > distributed_space_p0(
        distributed_st_tree );
      distributed_fast_spacetime_be_space< basis_tri_p1 > distributed_space_p1(
        distributed_st_tree );

      MPI_Barrier( comm );

      if ( my_rank == 0 ) {
        t.measure( );
        std::cout << std::endl << "mesh information:" << std::endl;
        std::cout << "number of timesteps = " << n_global_timesteps
                  << std::endl;
        std::cout << "number of spatial triangles = " << n_global_space_elements
                  << std::endl;
        std::cout << "number of spatial vertices = " << n_global_space_nodes
                  << std::endl;
      }

      MPI_Barrier( comm );

      // assemble the double layer matrix K and measure assembly time.
      if ( my_rank == 0 ) {
        t.reset( "assembly of distributed pFMM matrix K" );
      }
      distributed_pFMM_matrix_heat_dl_p0p1 * K
        = new distributed_pFMM_matrix_heat_dl_p0p1;
      spacetime_heat_dl_kernel_antiderivative kernel_k( cauchy_data::_alpha );
      distributed_fast_spacetime_be_assembler distributed_assembler_k( kernel_k,
        distributed_space_p0, distributed_space_p1, &comm, order_sing,
        order_reg, c.temp_order, c.spat_order, cauchy_data::_alpha );
      distributed_assembler_k.assemble( *K );

      MPI_Barrier( comm );
      if ( my_rank == 0 ) {
        t.measure( );
      }

      // compute the L2 projection g of the Dirichlet datum and initialize the
      // rhs of the first boundary integral equation by 1/2 M g (M ... mass
      // matrix)
      if ( my_rank == 0 ) {
        t.reset( "projection of Dirichlet datum + application of mass matrix" );
      }
      distributed_block_vector dirichlet_projection;
      distributed_space_p1.L2_projection(
        cauchy_data::dirichlet, dirichlet_projection );

      besthea::bem::distributed_spacetime_be_identity M(
        distributed_space_p0, distributed_space_p1, 1 );
      M.assemble( );
      // initialize rhs as a distributed vector. Every process just stores the
      // data corresponding to the time steps in its local blocks
      std::vector< lo > local_blocks = distributed_mesh.get_my_timesteps( );
      distributed_block_vector rhs(
        local_blocks, n_global_timesteps, n_global_space_elements );
      M.apply( dirichlet_projection, rhs, false, 0.5, 0.0 );
      if ( my_rank == 0 ) {
        t.measure( );
      }

      MPI_Barrier( comm );
      // add K * g to the rhs.
      if ( my_rank == 0 ) {
        t.reset( "application of K" );
      }
      K->apply( dirichlet_projection, rhs, false, 1.0, 1.0 );
      MPI_Barrier( comm );
      if ( my_rank == 0 ) {
        t.measure( );
      }
      // K is not needed anymore and thus it is deleted to free the memory
      delete K;

      // assemble the single layer matrix V and measure assembly time.
      MPI_Barrier( comm );
      if ( my_rank == 0 ) {
        t.reset( "assembly of distributed pFMM matrix V" );
      }
      distributed_pFMM_matrix_heat_sl_p0p0 * V
        = new distributed_pFMM_matrix_heat_sl_p0p0;
      spacetime_heat_sl_kernel_antiderivative kernel_v( cauchy_data::_alpha );
      distributed_fast_spacetime_be_assembler distributed_assembler_v( kernel_v,
        distributed_space_p0, distributed_space_p0, &comm, order_sing,
        order_reg, c.temp_order, c.spat_order, cauchy_data::_alpha );
      distributed_assembler_v.assemble( *V );
      MPI_Barrier( comm );
      if ( my_rank == 0 ) {
        t.measure( );
      }

      // Use GMRES to solve V q = g for the sought-after Neumann datum q
      if ( my_rank == 0 ) {
        t.reset( "solving for neumann datum" );
      }
      // use the rhs as initial guess of the neumann datum for GMRES
      distributed_block_vector approx_neumann_datum( rhs );
      V->gmres_solve( rhs, approx_neumann_datum, gmres_prec, gmres_iter );
      if ( my_rank == 0 ) {
        t.measure( );
        std::cout << "executed GMRES iterations: " << gmres_iter
                  << ", achieved precision: " << gmres_prec << std::endl;
      }

      MPI_Barrier( comm );

      // compute the relative approximation error of the Neumann datum in the L2
      // norm (approximately)
      sc neumann_l2_relative_error = distributed_space_p0.L2_relative_error(
        cauchy_data::neumann, approx_neumann_datum );

      // compute the L2 projection of the Neumann datum and the corresponding
      // relative error in the L2 norm (best approximation error).
      distributed_block_vector neumann_projection;
      distributed_space_p0.L2_projection(
        cauchy_data::neumann, neumann_projection );
      sc neumann_projection_error = distributed_space_p0.L2_relative_error(
        cauchy_data::neumann, neumann_projection );
      if ( my_rank == 0 ) {
        std::cout << std::endl
                  << "Neumann L2 relative error:  " << neumann_l2_relative_error
                  << std::endl;
        std::cout << "Best approximation error:   " << neumann_projection_error
                  << std::endl;
      }
      delete V;
    }
  }
  MPI_Finalize( );
}
