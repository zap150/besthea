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

#include "besthea/besthea.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <lyra/lyra.hpp>

using namespace besthea::mesh;
using namespace besthea::bem;
using namespace besthea::linear_algebra;
using namespace besthea::tools;

static sc neumann_datum_turn_on(
  sc x1, sc x2, sc x3, const coordinates< 3 > & n, sc t ) {
  const sc x_coeffs[ 3 ] = { 0.2, -0.3, 0.1 };
  const sc turn_on_point = 0.16667;
  sc dot_x = ( x_coeffs[ 0 ] * x1 ) * n[ 0 ] + ( x_coeffs[ 1 ] * x2 ) * n[ 1 ]
    + ( x_coeffs[ 2 ] * x3 ) * n[ 2 ];
  sc time_base = 0.25 * t;
  sc time_turn_on = 0.0;
  if ( t >= turn_on_point ) {
    time_turn_on = std::pow( ( t - turn_on_point ), 0.25 );
  }
  sc value = dot_x * ( time_base + time_turn_on );

  return value;
}

struct config {
  void dump( ) {
    std::cout << "config: " << std::endl;
    std::cout << "  Spatial surface mesh:                  "
              << spatial_surface_file << std::endl;
    if ( temporal_file.empty( ) ) {
      std::cout << "  No. of initial spatial volume refinements: "
                << surf_init_refine << std::endl;
      std::cout << "  End time:                              " << end_time
                << std::endl;
      std::cout << "  No. of timeslices:                     " << n_timeslices
                << std::endl;
      std::cout << "  No. of space-time refinements:         " << refine
                << std::endl;
    } else {
      std::cout << "  No. of spatial refinements:            "
                << surf_init_refine + refine << std::endl;
      std::cout << "  Temporal mesh:                         " << temporal_file
                << std::endl;
      std::cout << "  No. of space-time refinements:         " << refine
                << std::endl;
    }
    std::cout << "  No. of distribution tree levels:       "
              << distribution_tree_levels << std::endl;
    std::cout << "  Min. no. of elements for space-time refinement: "
              << n_min_elems_refine_st << std::endl;
    std::cout << "  Space-time box coupling coefficient:   "
              << st_coupling_coeff << std::endl;
    std::cout << "  Spatial truncation parameter:          " << trunc_space
              << std::endl;
    std::cout << "  Temporal expansion order:              " << temp_order
              << std::endl;
    std::cout << "  Spatial expansion order:               " << spat_order
              << std::endl;
    std::cout << "  regular quadrature order (V,K):        " << order_reg
              << std::endl;
    std::cout << "  singular quadrature order (V,K):       " << order_sing
              << std::endl;
    if ( use_aca_recompression ) {
      std::cout << "  ACA recompression:                     " << std::endl;
      std::cout << "  ACA accuracy:                          " << aca_eps
                << std::endl;
      std::cout << "  ACA max. allowed rank:                 " << aca_max_rank
                << std::endl;
      if ( is_diagonal_svd_recompression_disabled ) {
        std::cout << "  ACA: Standard SVD recompression." << std::endl;
      } else {
        std::cout << "  ACA: SVD Recompression based on largest diagonal SV."
                  << std::endl;
      }
    }
    if ( use_time_adaptive_operations ) {
      std::cout << "  Using M2T and S2L operations           " << std::endl;
    }
    if ( compute_right_hand_side ) {
      std::cout << "  ### Computing the right-hand side ###" << std::endl;
    } else {
      std::cout << "  ### Solving the system ###" << std::endl;
      std::cout << "  GMRES precision:                       " << gmres_prec
                << std::endl;
      std::cout << "  GMRES outer iterations:                "
                << gmres_outer_iter << std::endl;
      std::cout << "  GMRES inner iterations:                "
                << gmres_inner_iter << std::endl;
      if ( diagonal_prec ) {
        std::cout << "  Using a diagonal preconditioner" << std::endl;
      }
    }
  }

  // spatial mesh data
  std::string spatial_surface_file = "cube_12_half_scale.txt";
  std::string temporal_file = "";
  int surf_init_refine = 0;
  // temporal mesh data
  sc end_time = 0.25;
  lo n_timeslices = 16;
  // number of refinements for space-time mesh
  int refine = 0;
  // file containing the data of the rhs
  std::string rhs_data_file = "data_files/fine_rhs_vector.bin";
  // parameters for distributed FMM
  int distribution_tree_levels = -1;
  int n_min_elems_refine_st = 800;
  sc st_coupling_coeff = 4.5;
  int trunc_space = 2;
  int temp_order = 4;
  int spat_order = 12;
  int order_sing = 5;
  int order_reg = 5;
  sc gmres_prec = 1e-8;
  lo gmres_inner_iter = 500;
  lo gmres_outer_iter = 1;
  bool print_ensight = false;
  bool print_times = false;
  bool info_only = false;
  bool diagonal_prec = false;
  bool refine_large_leaves = false;
  bool use_time_adaptive_operations = false;
  bool use_aca_recompression = false;
  bool is_diagonal_svd_recompression_disabled = false;
  sc aca_eps = 1e-5;
  lo aca_max_rank = 500;
  bool compute_right_hand_side = false;
};  // struct config

namespace {
  config configure( int argc, char * argv[] ) {
    config c;
    bool help = false;

    auto cli = lyra::help( help )
      | lyra::opt( c.spatial_surface_file, "surface mesh" )[ "--surface_mesh" ](
        "Surface mesh of the spatial domain" )
      | lyra::opt( c.surf_init_refine,
        "initial surface refinement" )[ "--surf_init_refine" ](
        "Number of initial refinements of the spatial surface mesh. They are "
        "only relevant, if a surface mesh file is provided." )
      | lyra::opt( c.temporal_file, "temporal mesh" )[ "--time_mesh" ](
        "Temporal mesh containing the initial time slices." )
      | lyra::opt( c.end_time, "end time" )[ "--endtime" ](
        "End time of the considered time interval. The start time is always "
        "0. If temporal_file is given this value is ignored." )
      | lyra::opt( c.n_timeslices, "number of timeslices" )[ "--timeslices" ](
        "Number of time slices for the given mesh. If temporal_file is given "
        "this value is ignored." )
      | lyra::opt( c.refine, "space-time refinement" )[ "--refine" ](
        "Number of refinements of the tensor product space-time mesh "
        "consisting of the spatial surface mesh (see 'surface_mesh' "
        "and 'surf_init_refine') and the time slices (see 'end_time' and "
        "'n_timeslices', or 'temporal_file'). One refinement step includes one "
        "uniform refinement of the spatial mesh and two uniform refinements in "
        "time." )
      | lyra::opt( c.rhs_data_file, "rhs data file" )[ "--rhs_data" ](
        "Binary file containing the data of the rhs (on a finer mesh (one "
        "additional 2:1 refinement in space and time))" )
      | lyra::opt( c.distribution_tree_levels,
        "levels of the distribution tree" )[ "--dist_tree_levels" ](
        "Bound for the number of levels of the temporal tree which is used for "
        "the distribution of work among processes. It has to be greater than "
        "ceil( log_2( n_proc ) ) )" )
      | lyra::opt( c.n_min_elems_refine_st,
        "Minimal number of space-time elements needed for refinement in the "
        "space-time tree" )[ "--n_min_elems_refine_st" ](
        "In the construction of the space-time cluster tree a cluster is "
        "refined only if it contains more than this number of elements." )
      | lyra::opt( c.st_coupling_coeff,
        "space time box coupling coefficient" )[ "--st_coupling_coeff" ](
        "This constant c determines the relation between the spatial and "
        "temporal half sizes h_x and h_t of the 4D boxes in the space-time "
        "cluster tree according to the criterion ( h_x^2 <= 4 c h_t )." )
      | lyra::opt(
        c.trunc_space, "Spatial truncation parameter" )[ "--trunc_space" ](
        "Determines the number of clusters considered for spatial operations "
        "of pFMM operators. In each linear space direction only 'trunc_space' "
        "clusters are considered, all others are neglected." )
      | lyra::opt( c.temp_order, "Temporal expansion order" )[ "--temp_order" ](
        "Temporal expansion order used for the kernel expansion in the FMM." )
      | lyra::opt( c.spat_order, "Spatial expansion order" )[ "--spat_order" ](
        "Spatial expansion order used for the kernel expansion in the "
        "FMM" )
      | lyra::opt( c.gmres_prec, "GMRES relative precision" )[ "--gmres_prec" ](
        "Relative precision of the GMRES solver (default 1e-8) " )
      | lyra::opt( c.print_ensight )[ "--print_ensight" ](
        "If this flag is set, the resulting approximation of the Neumann datum "
        "and its projection are printed to files in ensight format for "
        "visualization." )
      | lyra::opt( c.print_times )[ "--print_times" ](
        "If this flag is set, the detailed times of the task execution on each "
        "process are printed to files in the directory ./task_timer/." )
      | lyra::opt(
        c.gmres_inner_iter, "GMRES inner iterations" )[ "--gmres_inner_iter" ](
        "Bound for the number of inner iterations for a GMRES with restarts." )
      | lyra::opt(
        c.gmres_outer_iter, "GMRES outer iterations" )[ "--gmres_outer_iter" ](
        "Bound for the number of outer iterations for a GMRES with restarts " )
      | lyra::opt( c.info_only )[ "--info_only" ](
        "If this flag is set, only some information about the involved trees "
        "and matrices is printed" )
      | lyra::opt( c.diagonal_prec )[ "--diagonal_prec" ](
        "If this flag is set, a diagonal preconditioner is used when solving "
        "the system of equations with GMRES" )
      | lyra::opt(
        c.order_reg, "quadrature order regular integrals" )[ "--order_reg" ](
        "Quadrature order used for the numerical computation of regular "
        "boundary integrals for V and K" )
      | lyra::opt(
        c.order_sing, "quadrature order singular integrals" )[ "--order_sing" ](
        "Quadrature order used for the numerical computation of singular "
        "boundary integrals for V and K" )
      | lyra::opt( c.refine_large_leaves )[ "--refine_large_leaves" ](
        "Determines whether large leaf clusters in the space-time cluster tree "
        "which contain only one time-step are additionally refined in space." )
      | lyra::opt(
        c.use_time_adaptive_operations )[ "--use_time_adaptive_operations" ](
        "Determines whether S2L and M2T operations are used for pFMM " )
      | lyra::opt( c.use_aca_recompression )[ "--use_aca" ](
        "Determines whether nearfield operations for spatially separated "
        "clusters are approximated via ACA" )
      | lyra::opt( c.aca_eps, "ACA relative precision" )[ "--aca_eps" ](
        "Accuracy of aca recompression" )
      | lyra::opt( c.aca_max_rank, "ACA max rank" )[ "--aca_max_rank" ](
        "Max rank for ACA recompression" )
      | lyra::opt( c.is_diagonal_svd_recompression_disabled )
        [ "--aca_disable_diag_svd_recompression" ](
          "Determines if the SVD recompression for the ACA using the largest "
          "singular value of the diagonal block as reference value is "
          "disabled. If true, it is disabled, i.e. no recompression is used." )
      | lyra::opt( c.compute_right_hand_side )[ "--compute_right_hand_side" ](
        "If this option is provided the right-hand side of the linear system "
        "is computed. Otherwise the linear system is solved." );

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
  // initialize MPI related parameters
  int provided;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  int my_rank, n_processes;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( comm, &my_rank );
  MPI_Comm_size( comm, &n_processes );
  // initialize parameters from the console arguments
  config c = configure( argc, argv );

  sc heat_alpha = 1.0;

  // set distribution_tree_levels if it is not initialized properly by the user
  lo distribution_tree_levels = c.distribution_tree_levels;
  lo min_distribution_tree_levels
    = std::max( 3, (int) std::ceil( std::log2( n_processes ) ) + 1 );
  if ( distribution_tree_levels < min_distribution_tree_levels ) {
    distribution_tree_levels = min_distribution_tree_levels;
  }
  // set some additional parameters:
  // orders of quadrature for computation of nearfield integrals of various
  // operators
  lo order_reg_tri = 5;
  lo order_reg_line = 5;
  // refine time in each refinement step twice:
  lo temp_refine_factor = 2;
  // print information about setup
  if ( my_rank == 0 ) {
    c.dump( );
    std::cout << "quadrature orders for initial operator; triangle: "
              << order_reg_tri << ", line (time): " << order_reg_line
              << std::endl;
    std::cout << "total refinements in space:              "
              << c.surf_init_refine + c.refine << std::endl;
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
  }
  // choose some assignment strategy for assignment of time clusters to
  // processes
  lo process_assignment_strategy = 1;

  // directory to store the files of the distributed space-time mesh which is
  // constructed in the following.
  std::string geometry_dir = "./temp_geometry_files/";
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
    temporal_mesh * time_mesh;
    if ( c.temporal_file.empty( ) ) {
      time_mesh = new temporal_mesh( 0, c.end_time, c.n_timeslices );
    } else {
      time_mesh = new temporal_mesh( c.temporal_file );
    }
    lo n_min_time_elems = 2;
    time_cluster_tree time_tree(
      *time_mesh, distribution_tree_levels, n_min_time_elems );
    time_tree.print( );

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
      triangular_surface_mesh space_mesh;
      space_mesh.load( c.spatial_surface_file );
      space_mesh.refine( c.surf_init_refine + space_refinement );

      // the spatial mesh was refined already sufficiently, so it is not further
      // refined in the mesh generation process.
      if ( c.temporal_file.empty( ) ) {
        spacetime_mesh_generator generator(
          space_mesh, c.end_time, c.n_timeslices, time_refinement, 0 );
        generator.generate( geometry_dir, "test_mesh", "txt" );
      } else {
        spacetime_mesh_generator generator(
          space_mesh, *time_mesh, time_refinement, 0 );
        generator.generate( geometry_dir, "test_mesh", "txt" );
      }
    }
    delete time_mesh;
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
      process_assignment_file, c.use_time_adaptive_operations, &comm, status );
    // reset status to 0.
    status = 0;
    lo n_global_timesteps = distributed_mesh.get_n_temporal_elements( );
    lo n_global_space_elements
      = distributed_mesh.get_local_mesh( )->get_n_spatial_elements( );
    lo n_global_space_nodes
      = distributed_mesh.get_local_mesh( )->get_n_spatial_nodes( );

    // construct the distributed spacetime cluster tree
    lo n_max_levels_spacetime_tree = 20;
    bool are_different_levels_in_aca_allowed = true;
    distributed_spacetime_cluster_tree distributed_st_tree( distributed_mesh,
      n_max_levels_spacetime_tree, c.n_min_elems_refine_st, c.st_coupling_coeff,
      heat_alpha, c.trunc_space, c.refine_large_leaves, c.use_aca_recompression,
      are_different_levels_in_aca_allowed, c.use_time_adaptive_operations,
      &comm, status );
    distributed_st_tree.print( );
    std::cout << std::endl;
    distributed_st_tree.get_distribution_tree( )->print( );

    if ( status > 0 ) {
      if ( my_rank == 0 ) {
        std::cout << "Error in tree construction. Aborting." << std::endl;
      }
    } else {
      // declare boundary element spaces
      distributed_fast_spacetime_be_space< basis_tri_p0 > distributed_space_p0(
        distributed_st_tree );

      MPI_Barrier( comm );

      if ( my_rank == 0 ) {
        t.measure( );
        std::cout << std::endl
                  << "space-time surface mesh information:" << std::endl;
        std::cout << "number of timesteps = " << n_global_timesteps
                  << std::endl;
        std::cout << "number of spatial triangles = " << n_global_space_elements
                  << std::endl;
        std::cout << "number of spatial vertices = " << n_global_space_nodes
                  << std::endl;
      }
      distributed_st_tree.print_information( 0 );

      MPI_Barrier( comm );

      if ( c.compute_right_hand_side ) {
        // assemble the single layer matrix V and measure the assembly time.
        if ( my_rank == 0 ) {
          t.reset(
            "prepare distributed pFMM matrix V for on the fly application" );
        }
        distributed_pFMM_matrix_heat_sl_p0p0 * V
          = new distributed_pFMM_matrix_heat_sl_p0p0;
        spacetime_heat_sl_kernel_antiderivative kernel_v( heat_alpha );
        distributed_fast_spacetime_be_assembler distributed_assembler_v(
          kernel_v, distributed_space_p0, distributed_space_p0, &comm,
          c.order_sing, c.order_reg, c.temp_order, c.spat_order, heat_alpha,
          c.aca_eps, c.aca_max_rank );
        distributed_assembler_v.initialize_for_on_the_fly_application( *V );

        MPI_Barrier( comm );
        if ( my_rank == 0 ) {
          t.measure( );
        }

        V->print_information( 0 );

        if ( my_rank == 0 ) {
          t.reset( "Compute L2 projection of Neumann datum" );
        }

        distributed_block_vector neumann_projection;
        distributed_space_p0.L2_projection(
          neumann_datum_turn_on, neumann_projection );
        MPI_Barrier( comm );

        if ( my_rank == 0 ) {
          t.measure( );
          t.reset( "Applying V" );
        }

        distributed_block_vector rhs( neumann_projection );
        V->apply_on_the_fly( distributed_assembler_v, neumann_projection, rhs );

        MPI_Barrier( comm );

        if ( my_rank == 0 ) {
          t.measure( );
        }
        delete V;

        std::string data_dir = "./data_files/";
        const std::string vector_file_name = data_dir + "rhs_fine_mesh.bin";
        std::filesystem::create_directory( data_dir );

        vector rhs_classic_vector;
        rhs.copy_to_vector( rhs_classic_vector );
        if ( my_rank == 0 ) {
          write_raw_data_to_bin_file( rhs_classic_vector.data( ),
            rhs_classic_vector.size( ), vector_file_name );
        }
      } else {  // solve the linear system
        // initialize rhs as a distributed vector. Every process just stores
        // the data corresponding to the time steps in its local blocks
        std::vector< lo > local_blocks = distributed_mesh.get_my_timesteps( );
        distributed_block_vector distributed_rhs;

        // load the fine rhs vector from the given binary file
        std::vector< sc > fine_rhs_vector
          = read_vector_from_bin_file< sc >( c.rhs_data_file );
        if ( fine_rhs_vector.size( ) == 0 ) {
          return EXIT_FAILURE;
        }

        // sum up the entries of the fine vector according to the refinement
        // 1:2 in space and time) storing the result in a separate coarse vector
        std::vector< sc > coarse_rhs_vector_summed_up;
        std::vector< std::vector< lo > > local_time_ref_map;
        lo local_start_idx_time = distributed_mesh.get_local_start_idx( );
        lo n_time_refs = 2;
        distributed_mesh.get_local_mesh( )
          ->get_temporal_mesh( )
          ->compute_element_index_map_for_refinement(
            n_time_refs, local_start_idx_time, local_time_ref_map );
        std::vector< std::vector< lo > > space_ref_map;
        distributed_mesh.get_spatial_surface_mesh( )
          ->compute_element_index_map_for_refinement( space_ref_map );
        sum_up_refined_mesh_vector_entries( fine_rhs_vector, n_global_timesteps,
          local_time_ref_map, local_start_idx_time, space_ref_map,
          coarse_rhs_vector_summed_up );

        // copy the coarse vector to the distributed rhs vector and synchronize
        distributed_rhs.copy_from_raw( local_blocks, n_global_timesteps,
          n_global_space_elements, coarse_rhs_vector_summed_up.data( ) );
        distributed_rhs.synchronize_shared_parts( );

        // scale the rhs vector to obtain the projection coefficients of the rhs
        // vector
        distributed_block_vector rhs_projection( distributed_rhs );
        scale_vector_by_inv_elem_size( distributed_mesh, rhs_projection );

        // assemble the single layer matrix V and measure assembly time.
        MPI_Barrier( comm );
        if ( my_rank == 0 ) {
          t.reset( "assembly of distributed pFMM matrix V" );
        }
        distributed_pFMM_matrix_heat_sl_p0p0 * V
          = new distributed_pFMM_matrix_heat_sl_p0p0;
        spacetime_heat_sl_kernel_antiderivative kernel_v( heat_alpha );
        distributed_fast_spacetime_be_assembler distributed_assembler_v(
          kernel_v, distributed_space_p0, distributed_space_p0, &comm,
          c.order_sing, c.order_reg, c.temp_order, c.spat_order, heat_alpha,
          c.aca_eps, c.aca_max_rank );
        if ( c.is_diagonal_svd_recompression_disabled ) {
          distributed_assembler_v.disable_diagonal_svd_recompression( );
        }
        distributed_assembler_v.assemble( *V, c.info_only );

        V->set_task_timer( c.print_times );
        V->set_verbose( true );

        MPI_Barrier( comm );
        if ( my_rank == 0 ) {
          t.measure( );
        }

        V->print_information( 0 );
        if ( c.info_only ) {
          delete V;
        } else {
          // Use GMRES to solve V q = g for the sought-after Neumann datum q
          if ( my_rank == 0 ) {
            t.reset( "solving for neumann datum" );
          }

          // set some parameters for the GMRES with restarts
          sc achieved_prec = 1.0;
          lo executed_inner_iterations = 0;
          lo executed_outer_iterations = 0;
          distributed_diagonal_matrix * preconditioner = nullptr;
          if ( c.diagonal_prec ) {
            distributed_block_vector inverse_diagonal;
            V->get_inverse_diagonal( inverse_diagonal );
            preconditioner
              = new distributed_diagonal_matrix( inverse_diagonal );
          }
          // use the rhs as initial guess of the neumann datum for GMRES
          distributed_block_vector approx_neumann_datum( distributed_rhs );
          while ( achieved_prec > c.gmres_prec
            && executed_outer_iterations < c.gmres_outer_iter ) {
            achieved_prec = c.gmres_prec;
            executed_inner_iterations = c.gmres_inner_iter;
            if ( c.diagonal_prec ) {
              V->gmres_solve( distributed_rhs, approx_neumann_datum,
                achieved_prec, executed_inner_iterations, *preconditioner,
                false );
            } else {
              V->gmres_solve( distributed_rhs, approx_neumann_datum,
                achieved_prec, executed_inner_iterations );
            }
            ++executed_outer_iterations;
          }

          if ( my_rank == 0 ) {
            std::cout << "GMRES outer iteration " << executed_outer_iterations
                      << ": executed inner iterations: "
                      << executed_inner_iterations
                      << ", achieved precision: " << achieved_prec << std::endl;
            t.measure( );
          }

          delete V;

          delete preconditioner;

          MPI_Barrier( comm );

          // compute the approximation error of the Neumann datum in
          // the L2 norm (relative L2)
          sc neumann_l2_relative_error = distributed_space_p0.L2_relative_error(
            neumann_datum_turn_on, approx_neumann_datum, 10, 4 );

          // in addition, compute the L2 projection of the Neumann datum and the
          // corresponding L2 relative error
          distributed_block_vector neumann_projection;
          distributed_space_p0.L2_projection(
            neumann_datum_turn_on, neumann_projection );
          sc neumann_projection_error_rel_l2
            = distributed_space_p0.L2_relative_error(
              neumann_datum_turn_on, neumann_projection, 10, 4 );

          if ( my_rank == 0 ) {
            std::cout << std::endl;
            std::cout << "BEM approximation L2 relative error: "
                      << neumann_l2_relative_error << std::endl;
            std::cout << "Best approximation error, relative L2 norm:   "
                      << neumann_projection_error_rel_l2 << std::endl;

            std::cout << "###" << std::endl;
          }
          if ( c.print_ensight ) {
            std::string ensight_dir = "./example_adaptive_turn_on/ensight";
            if ( my_rank == 0 ) {
              std::filesystem::create_directory( ensight_dir );
              std::cout << "printing ensight files" << std::endl;
            }
            std::vector< std::string > elem_labels = { "Neumann_projection",
              "Neumann_approximation", "Dirichlet_projection" };
            std::vector< distributed_block_vector * > elem_data
              = { &neumann_projection, &approx_neumann_datum, &rhs_projection };
            MPI_Barrier( comm );
            distributed_mesh.print_ensight_case(
              ensight_dir, nullptr, &elem_labels );
            distributed_mesh.print_ensight_geometry( ensight_dir );
            distributed_mesh.print_ensight_datafiles(
              ensight_dir, nullptr, nullptr, &elem_labels, &elem_data );
          }
        }
      }
    }
  }
  if ( my_rank == 0 ) {
    // erase all auxiliary geometry files created during the execution
    std::filesystem::remove_all( geometry_dir );
  }
  MPI_Finalize( );
  return EXIT_SUCCESS;
}
