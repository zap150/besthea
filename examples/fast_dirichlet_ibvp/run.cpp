/*
Copyright (c) 2022, VSB - Technical University of Ostrava and Graz University of
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

// example with non-vanishing initial datum, but vanishing Dirichlet datum
struct cauchy_data_1 {
  static sc dirichlet( [[maybe_unused]] sc x1, [[maybe_unused]] sc x2,
    [[maybe_unused]] sc x3, [[maybe_unused]] const coordinates< 3 > &,
    [[maybe_unused]] sc t ) {
    return 0.0;
  }

  static sc neumann( sc x1, sc x2, sc x3, const coordinates< 3 > & n, sc t ) {
    sc dot = n[ 0 ] * std::cos( M_PI * ( x1 + 0.5 ) )
        * std::sin( M_PI * ( x2 + 0.5 ) ) * std::sin( M_PI * ( x3 + 0.5 ) )
      + n[ 1 ] * std::sin( M_PI * ( x1 + 0.5 ) )
        * std::cos( M_PI * ( x2 + 0.5 ) ) * std::sin( M_PI * ( x3 + 0.5 ) )
      + n[ 2 ] * std::sin( M_PI * ( x1 + 0.5 ) )
        * std::sin( M_PI * ( x2 + 0.5 ) ) * std::cos( M_PI * ( x3 + 0.5 ) );
    sc value = _alpha * M_PI * std::exp( -3 * M_PI * M_PI * _alpha * t ) * dot;
    return value;
  }

  static sc initial( sc x1, sc x2, sc x3 ) {
    sc value = std::sin( M_PI * ( x1 + 0.5 ) ) * std::sin( M_PI * ( x2 + 0.5 ) )
      * std::sin( M_PI * ( x3 + 0.5 ) );
    return value;
  }

  static constexpr sc _alpha{ 1.0 };
  static constexpr bool _zero_dirichlet{ true };
  static constexpr bool _zero_initial{ false };
};

struct cauchy_data_2 {
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

  static constexpr sc _alpha{ 1.0 };
  static constexpr std::array< sc, 3 > _y{ 1.5, 1.5, 1.5 };
  static constexpr sc _shift{ 0.0 };
  static constexpr bool _zero_dirichlet{ false };
  static constexpr bool _zero_initial{ true };
};  // struct cauchy_data

struct config {
  void dump( ) {
    std::cout << "config: " << std::endl;
    std::cout << "  Chosen Cauchy datum: " << cauchy_datum << std::endl;
    std::cout << "  Spatial volume mesh:                   "
              << spatial_volume_file << std::endl;
    if ( !spatial_surface_file.empty( ) ) {
      std::cout << "  Spatial surface mesh:                  "
                << spatial_surface_file << std::endl;
    } else {
      std::cout << "  Spatial surface mesh constructed from volume mesh"
                << std::endl;
    }
    if ( temporal_file.empty( ) ) {
      std::cout << "  No. of initial spatial volume refinements: "
                << vol_init_refine << std::endl;

      std::cout << "  End time:                              " << end_time
                << std::endl;
      std::cout << "  No. of timeslices:                     " << n_timeslices
                << std::endl;
      std::cout << "  No. of space-time refinements:         " << refine
                << std::endl;
    } else {
      std::cout << "  No. of spatial refinements:            "
                << vol_init_refine + refine << std::endl;
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
    std::cout << "  Min. no. of elements for spatial refinement: "
              << n_min_elems_refine_s << std::endl;
    std::cout << "  Temporal expansion order:              " << temp_order
              << std::endl;
    std::cout << "  Spatial expansion order:               " << spat_order
              << std::endl;
    std::cout << "  Singular integral recursion depth:     "
              << sing_int_rec_depth << std::endl;
    std::cout << "  regular quadrature order (V,K):        " << order_reg
              << std::endl;
    std::cout << "  singular quadrature order (V,K):       " << order_sing
              << std::endl;
    std::cout << "  GMRES precision:                       " << gmres_prec
              << std::endl;
    std::cout << "  GMRES outer iterations:                " << gmres_outer_iter
              << std::endl;
    std::cout << "  GMRES inner iterations:                " << gmres_inner_iter
              << std::endl;
    if ( use_aca_recompression ) {
      std::cout << "  ACA recompression:                     " << std::endl;
      std::cout << "  ACA accuracy:                          " << aca_eps
                << std::endl;
      std::cout << "  ACA max. allowed rank:                 " << aca_max_rank
                << std::endl;
      if ( is_diagonal_svd_recompression_disabled ) {
        std::cout << "  ACA: No SVD recompression." << std::endl;
      } else {
        std::cout << "  ACA: SVD Recompression based on largest diagonal SV."
                  << std::endl;
      }
    }
    if ( diagonal_prec ) {
      std::cout << "  Using a diagonal preconditioner        " << std::endl;
    }
    if ( use_time_adaptive_operations ) {
      std::cout << "  Using M2T and S2L operations           " << std::endl;
    }
  }

  // choice of the cauchy datum
  lo cauchy_datum = 1;
  // spatial mesh data
  std::string spatial_volume_file = "cube_12_vol_half_scale.txt";
  std::string spatial_surface_file = "";
  std::string temporal_file = "";
  int vol_init_refine = 2;
  int surf_init_refine = 0;
  // temporal mesh data
  sc end_time = 0.25;
  lo n_timeslices = 16;
  // number of refinements for space-time mesh
  int refine = 0;
  // parameters for distributed FMM
  int distribution_tree_levels = -1;
  int n_min_elems_refine_st = 800;
  sc st_coupling_coeff = 4.5;
  int trunc_space = 2;
  int n_min_elems_refine_s = 100;
  int temp_order = 4;
  int spat_order = 12;
  int order_sing = 5;
  int order_reg = 5;
  int sing_int_rec_depth = 2;
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
};  // struct config

namespace {
  config configure( int argc, char * argv[] ) {
    config c;
    bool help = false;

    auto cli = lyra::help( help )
      | lyra::opt( c.cauchy_datum, "cauchy datum" )[ "--cauchy_datum" ](
        "Cauchy_datum used to provide initial and boundary data as well as "
        "heat capacity constant. Currently supported choices are: 1,2" )
      | lyra::opt( c.spatial_volume_file, "volume mesh" )[ "--volume_mesh" ](
        "Volume mesh of the spatial domain, refined 'vol_init_refine' times" )
      | lyra::opt( c.spatial_surface_file, "surface mesh" )[ "--surface_mesh" ](
        "Surface mesh of the spatial domain, which is not refined anymore" )
      | lyra::opt(
        c.vol_init_refine, "initial volume refinement" )[ "--vol_init_refine" ](
        "Number of initial refinements of the spatial volume mesh" )
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
        "and 'vol_init_refine') and the time slices (see 'end_time' and "
        "'n_timeslices'). One refinement step includes one uniform refinement "
        "of the spatial mesh and two uniform refinements in time." )
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
      | lyra::opt( c.n_min_elems_refine_s,
        "Minimal number of spatial volume elements needed for refinement in "
        "the spatial tree" )[ "--n_min_elems_refine_st" ](
        "In the construction of the space-time cluster tree a cluster is "
        "refined only if it contains more than this number of elements." )
      | lyra::opt( c.temp_order, "Temporal expansion order" )[ "--temp_order" ](
        "Temporal expansion order used for the kernel expansion in the FMM." )
      | lyra::opt( c.spat_order, "Spatial expansion order" )[ "--spat_order" ](
        "Spatial expansion order used for the kernel expansion in the "
        "FMM" )
      | lyra::opt( c.sing_int_rec_depth,
        "Singular integral recursion depth" )[ "--rec_depth" ](
        "Singular integrals of the initial operator are computed using "
        "recursive refinements. This parameter determines the recursion "
        "depth" )
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
        "which contain only one time-step are additionaly refined in space." )
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
          "disabled. If true, it is disabled, i.e. no recompression is used." );

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

  sc heat_alpha = 0.0;
  bool zero_dirichlet = false;
  bool zero_initial = false;
  sc ( *dirichlet_datum )( sc, sc, sc, const coordinates< 3 > &, sc );
  sc ( *neumann_datum )( sc, sc, sc, const coordinates< 3 > &, sc );
  sc ( *initial_datum )( sc, sc, sc );
  // sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc );
  switch ( c.cauchy_datum ) {
    case 1:
      heat_alpha = cauchy_data_1::_alpha;
      zero_dirichlet = cauchy_data_1::_zero_dirichlet;
      zero_initial = cauchy_data_1::_zero_initial;
      dirichlet_datum = &cauchy_data_1::dirichlet;
      neumann_datum = &cauchy_data_1::neumann;
      initial_datum = &cauchy_data_1::initial;
      break;
    case 2:
      heat_alpha = cauchy_data_2::_alpha;
      zero_dirichlet = cauchy_data_2::_zero_dirichlet;
      zero_initial = cauchy_data_2::_zero_initial;
      dirichlet_datum = &cauchy_data_2::dirichlet;
      neumann_datum = &cauchy_data_2::neumann;
      initial_datum = &cauchy_data_2::initial;
      break;
    default:
      std::cout << "Cauchy datum " << c.cauchy_datum << " not supported"
                << std::endl;
      return EXIT_FAILURE;
  }
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
  lo order_reg_tetra = 5;
  lo order_reg_line = 5;
  // refine time in each refinement step twice:
  lo temp_refine_factor = 2;
  // print information about setup
  if ( my_rank == 0 ) {
    c.dump( );
    std::cout << "quadrature orders for initial operator; triangle: "
              << order_reg_tri << ", tetrahedron: " << order_reg_tetra
              << ", line (time): " << order_reg_line << std::endl;
    std::cout << "total refinements in space:              "
              << c.vol_init_refine + c.refine << std::endl;
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

  // load the volume mesh and refine it
  tetrahedral_volume_mesh volume_mesh;
  volume_mesh.load( c.spatial_volume_file );
  volume_mesh.refine( c.vol_init_refine + space_refinement );

  // generation of distributed mesh: a single process takes care of this and
  // provides several files which are loaded from all processes
  // (NOTE: all processes need to be able to access the geometry_dir!)
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

    // time_tree.print( );
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
      if ( c.spatial_surface_file.empty( ) ) {
        space_mesh.from_tetrahedral( volume_mesh );
      } else {
        space_mesh.load( c.spatial_surface_file );
        space_mesh.refine( c.surf_init_refine + space_refinement );
      }
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
        std::cout << std::endl
                  << "space-time surface mesh information:" << std::endl;
        std::cout << "number of timesteps = " << n_global_timesteps
                  << std::endl;
        std::cout << "number of spatial triangles = " << n_global_space_elements
                  << std::endl;
        std::cout << "number of spatial vertices = " << n_global_space_nodes
                  << std::endl;
        // std::cout << "printing local part of distributed tree: " <<
        // std::endl;
        // distributed_st_tree.get_distribution_tree( )->print( );
        // distributed_st_tree.print( 5 );
      }
      distributed_st_tree.print_information( 0 );

      MPI_Barrier( comm );

      // construction of the spatial cluster tree for the volume mesh
      bool print_warnings_tree_construction = false;
      lo n_space_levels = distributed_st_tree.get_global_n_space_levels( );
      // lo n_space_levels = 1;
      if ( my_rank == 0 ) {
        print_warnings_tree_construction = true;
        t.reset( "assembly of volume space cluster tree" );
      }
      volume_space_cluster_tree volume_space_tree( volume_mesh, n_space_levels,
        c.n_min_elems_refine_s, print_warnings_tree_construction );
      // declare the finite element space related to the volume mesh and the
      // initial datum
      fe_space< basis_tetra_p1 > space_p1_tetra( volume_mesh );

      if ( my_rank == 0 ) {
        t.measure( );
        volume_mesh.print_info( );
      }

      MPI_Barrier( comm );

      // initialize rhs as a distributed vector. Every process just stores
      // the data corresponding to the time steps in its local blocks
      std::vector< lo > local_blocks = distributed_mesh.get_my_timesteps( );
      distributed_block_vector rhs(
        local_blocks, n_global_timesteps, n_global_space_elements, true, comm );

      distributed_block_vector dirichlet_projection;
      if ( !zero_dirichlet ) {
        // compute the L2 projection g of the Dirichlet datum and initialize
        // the rhs of the first boundary integral equation by 1/2 M g
        // (M ... mass matrix)
        if ( !c.info_only ) {
          if ( my_rank == 0 ) {
            t.reset(
              "projection of Dirichlet datum + application of mass matrix" );
          }
          distributed_space_p1.L2_projection(
            dirichlet_datum, dirichlet_projection );
          besthea::bem::distributed_spacetime_be_identity M(
            distributed_space_p0, distributed_space_p1, 1 );
          M.assemble( );

          M.apply( dirichlet_projection, rhs, false, 0.5, 0.0 );
          if ( my_rank == 0 ) {
            t.measure( );
          }
        }

        MPI_Barrier( comm );

        // assemble the double layer matrix K and measure assembly time.
        if ( my_rank == 0 ) {
          t.reset( "Initializing double layer operator K" );
        }
        distributed_pFMM_matrix_heat_dl_p0p1 * K
          = new distributed_pFMM_matrix_heat_dl_p0p1;
        spacetime_heat_dl_kernel_antiderivative kernel_k( heat_alpha );
        distributed_fast_spacetime_be_assembler distributed_assembler_k(
          kernel_k, distributed_space_p0, distributed_space_p1, &comm,
          c.order_sing, c.order_reg, c.temp_order, c.spat_order, heat_alpha,
          -1.0 );
        distributed_assembler_k.initialize_for_on_the_fly_application( *K );
        MPI_Barrier( comm );
        if ( my_rank == 0 ) {
          t.measure( );
        }

        if ( c.info_only ) {
          K->print_information( 0 );
        } else {
          // add K * g to the rhs.
          if ( my_rank == 0 ) {
            t.reset( "On the fly application of distributed pFMM matrix K" );
          }
          K->apply_on_the_fly(
            distributed_assembler_k, dirichlet_projection, rhs, 1.0, 1.0 );
          MPI_Barrier( comm );
          if ( my_rank == 0 ) {
            t.measure( );
          }
        }
        // K is not needed anymore; delete it to free the memory
        delete K;
      } else {
        if ( my_rank == 0 ) {
          std::cout << "Provided Dirichlet datum is zero" << std::endl;
        }
      }

      MPI_Barrier( comm );

      if ( !zero_initial ) {
        // compute the projection of the initial datum to the p1 fe space
        vector initial_projection;
        space_p1_tetra.L2_projection( initial_datum, initial_projection );
        sc initial_l2_error = space_p1_tetra.L2_relative_error(
          initial_datum, initial_projection );
        if ( my_rank == 0 ) {
          std::cout << "Initial projection L2 relative error: "
                    << initial_l2_error << std::endl;
        }
        // assemble the initial operator M0
        spacetime_heat_initial_m0_kernel_antiderivative kernel_m0( heat_alpha );
        if ( my_rank == 0 ) {
          t.reset( "Initializing initial potential operator M0" );
        }
        distributed_initial_pFMM_matrix_heat_m0_p0p1 * M0_dist
          = new distributed_initial_pFMM_matrix_heat_m0_p0p1;
        distributed_fast_spacetime_initial_be_assembler
          distributed_assembler_m0( kernel_m0, distributed_space_p0,
            space_p1_tetra, &comm, volume_space_tree, order_reg_tri,
            order_reg_tetra, order_reg_line, c.temp_order, c.spat_order,
            c.sing_int_rec_depth, heat_alpha );
        distributed_assembler_m0.initialize_for_on_the_fly_application(
          *M0_dist );
        MPI_Barrier( comm );
        if ( my_rank == 0 ) {
          t.measure( );
        }

        // apply M0 to the projected initial datum. add the result to the rhs.
        if ( c.info_only ) {
          M0_dist->print_information( 0 );
        } else {
          MPI_Barrier( comm );
          if ( my_rank == 0 ) {
            t.reset( "On the fly application of distributed pFMM matrix M0" );
          }
          M0_dist->apply_on_the_fly(
            distributed_assembler_m0, initial_projection, rhs, -1.0, 1.0 );
          MPI_Barrier( comm );
          if ( my_rank == 0 ) {
            t.measure( );
          }
          M0_dist->print_information( 0 );
        }
        delete M0_dist;
      } else {
        if ( my_rank == 0 ) {
          std::cout << "Provided initial datum is zero" << std::endl;
        }
      }

      // assemble the single layer matrix V and measure assembly time.
      MPI_Barrier( comm );
      distributed_pFMM_matrix_heat_sl_p0p0 * V
        = new distributed_pFMM_matrix_heat_sl_p0p0;
      spacetime_heat_sl_kernel_antiderivative kernel_v( heat_alpha );
      distributed_fast_spacetime_be_assembler distributed_assembler_v( kernel_v,
        distributed_space_p0, distributed_space_p0, &comm, c.order_sing,
        c.order_reg, c.temp_order, c.spat_order, heat_alpha, c.aca_eps,
        c.aca_max_rank );
      if ( c.is_diagonal_svd_recompression_disabled ) {
        distributed_assembler_v.disable_diagonal_svd_recompression( );
      }

      if ( c.info_only ) {
        V->print_information( 0 );
        delete V;
      } else {
        if ( my_rank == 0 ) {
          t.reset( "assembly of distributed pFMM matrix V" );
        }
        distributed_assembler_v.assemble( *V, c.info_only );
        MPI_Barrier( comm );
        if ( my_rank == 0 ) {
          t.measure( );
        }
        V->set_task_timer( c.print_times );
        V->set_verbose( true );

        V->print_information( 0 );
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
          preconditioner = new distributed_diagonal_matrix( inverse_diagonal );
        }
        // use the rhs as initial guess of the neumann datum for GMRES
        distributed_block_vector approx_neumann_datum( rhs );
        while ( achieved_prec > c.gmres_prec
          && executed_outer_iterations < c.gmres_outer_iter ) {
          achieved_prec = c.gmres_prec;
          executed_inner_iterations = c.gmres_inner_iter;
          if ( c.diagonal_prec ) {
            V->gmres_solve( rhs, approx_neumann_datum, achieved_prec,
              executed_inner_iterations, *preconditioner, false );
          } else {
            V->gmres_solve( rhs, approx_neumann_datum, achieved_prec,
              executed_inner_iterations );
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
        // the L2 norm (relative L2, and absolute weighted L2)
        // in addition, compute the L2 projection of the Neumann datum and the
        // corresponding L2 relative error and absolute weighted L2 error
        distributed_block_vector neumann_projection;
        sc neumann_l2_relative_error = distributed_space_p0.L2_relative_error(
          neumann_datum, approx_neumann_datum, 10, 4 );
        distributed_space_p0.L2_projection( neumann_datum, neumann_projection );
        sc neumann_projection_error_rel_l2
          = distributed_space_p0.L2_relative_error(
            neumann_datum, neumann_projection, 10, 4 );

        if ( my_rank == 0 ) {
          std::cout << std::endl;
          std::cout << "BEM approximation L2 relative error: "
                    << neumann_l2_relative_error << std::endl;
          std::cout << "Best approximation error, relative L2 norm:   "
                    << neumann_projection_error_rel_l2 << std::endl;
        }
        if ( c.print_ensight ) {
          std::cout << "printing ensight files" << std::endl;
          std::vector< std::string > node_labels{ "Dirichlet_projection" };
          std::vector< std::string > elem_labels{ "Neumann_projection",
            "Neumann_approximation" };
          std::vector< distributed_block_vector * > elem_data{
            &neumann_projection, &approx_neumann_datum
          };
          std::vector< distributed_block_vector * > node_data{
            &dirichlet_projection
          };
          std::string ensight_dir = "./fast_initial_dirichlet_bvp/ensight";
          if ( my_rank == 0 ) {
            std::filesystem::create_directory( ensight_dir );
          }
          MPI_Barrier( comm );
          distributed_mesh.print_ensight_case(
            ensight_dir, &node_labels, &elem_labels );
          distributed_mesh.print_ensight_geometry( ensight_dir );
          distributed_mesh.print_ensight_datafiles(
            ensight_dir, &node_labels, &node_data, &elem_labels, &elem_data );
        }
      }
    }
  }
  if ( my_rank == 0 ) {
    // erase all auxiliary geometry files created during the execution
    std::filesystem::remove_all( geometry_dir );
  }
  MPI_Finalize( );
}
