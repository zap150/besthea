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

#include "besthea/distributed_fast_spacetime_be_assembler.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/distributed_fast_spacetime_be_space.h"
#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/low_rank_kernel.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/timer.h"

#include <set>
#include <vector>

using besthea::mesh::distributed_spacetime_tensor_mesh;
using besthea::mesh::general_spacetime_cluster;
using besthea::mesh::spacetime_tensor_mesh;
using besthea::mesh::tree_structure;

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::distributed_fast_spacetime_be_assembler( kernel_type &
                                                                 kernel,
  test_space_type & test_space, trial_space_type & trial_space, MPI_Comm * comm,
  int order_singular, int order_regular, int temp_order, int spat_order,
  sc alpha, sc aca_eps, lo aca_max_rank )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ),
    _temp_order( temp_order ),
    _spat_order( spat_order ),
    _m2l_integration_order( _spat_order ),
    _alpha( alpha ),
    _aca_eps( aca_eps ),
    _aca_max_rank( aca_max_rank ),
    _comm( comm ) {
  MPI_Comm_rank( *_comm, &_my_rank );
}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::~distributed_fast_spacetime_be_assembler( ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::assemble( pfmm_matrix_type &
                                                   global_matrix,
  bool info_mode ) const {
  global_matrix.set_MPI_communicator( _comm );
  global_matrix.set_trees( _test_space->get_tree( ) );
  global_matrix.set_orders( _spat_order, _temp_order, _order_regular );
  global_matrix.set_alpha( _alpha );
  global_matrix.set_m2l_integration_order( _m2l_integration_order );
  global_matrix.set_aca_parameters( _aca_eps, _aca_max_rank );

  // number of timesteps have to be the same for test and trial meshes
  lo n_timesteps = _test_space->get_mesh( ).get_n_temporal_elements( );
  // // size of individual blocks
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );
  global_matrix.resize( n_timesteps, n_columns, n_rows );
  // ###########################################################################

  initialize_moment_and_local_contributions( );

  global_matrix.initialize_spatial_m2m_coeffs( );

  global_matrix.initialize_nearfield_containers( );

  // fill the m-list, m2l-list, l-list, m2t-list, s2l-list and n-list of the
  // distributed pFMM matrix and determine the receive information data.
  global_matrix.prepare_fmm( );

  // precompute Chebyshev nodes and values
  global_matrix.compute_chebyshev( );

  // assemble the nearfield matrices of the pFMM matrix
  if ( !info_mode ) {
    assemble_nearfield( global_matrix );
    // sort pointers in _n_list of the matrix for matrix-vector multiplication
    global_matrix.sort_clusters_in_n_list( );
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::assemble_nearfield( pfmm_matrix_type &
    global_matrix ) const {
  const std::vector< general_spacetime_cluster * > *
    clusters_with_nearfield_operations
    = global_matrix.get_pointer_to_clusters_with_nearfield_operations( );

  // FIXME: disabled sorting temporarily

  // for the assembly, sort the clusters with nearfield operations by the size
  // of matrices in the nearfield
  // std::vector< lo > total_sizes(
  //   clusters_with_nearfield_operations->size( ), 0 );
  // for ( std::vector< general_spacetime_cluster * >::size_type cluster_index =
  // 0;
  //       cluster_index < clusters_with_nearfield_operations->size( );
  //       ++cluster_index ) {
  //   general_spacetime_cluster * current_cluster
  //     = ( *clusters_with_nearfield_operations )[ cluster_index ];
  //   std::vector< general_spacetime_cluster * > * nearfield_list
  //     = current_cluster->get_nearfield_list( );
  //   lo n_dofs_target = current_cluster->get_n_dofs< test_space_type >( );
  //   for ( std::vector< general_spacetime_cluster * >::size_type src_index =
  //   0;
  //         src_index < nearfield_list->size( ); ++src_index ) {
  //     general_spacetime_cluster * nearfield_cluster
  //       = ( *nearfield_list )[ src_index ];
  //     lo n_dofs_source = nearfield_cluster->get_n_dofs< trial_space_type >(
  //     ); total_sizes[ cluster_index ] += n_dofs_source * n_dofs_target;
  //   }
  // }
  std::vector< lo > permutation_index(
    clusters_with_nearfield_operations->size( ), 0 );
  for ( lo i = 0; i != lo( permutation_index.size( ) ); i++ ) {
    permutation_index[ i ] = i;
  }
  // sort( permutation_index.begin( ), permutation_index.end( ),
  //   [ & ]( const int & a, const int & b ) {
  //     return ( total_sizes[ a ] > total_sizes[ b ] );
  //   } );

  std::unordered_map< general_spacetime_cluster *, sc >
    largest_sing_val_diag_blocks;
// assemble the nearfield matrices in parallel
#pragma omp parallel for schedule( dynamic, 1 )
  for ( std::vector< general_spacetime_cluster * >::size_type cluster_index = 0;
        cluster_index < clusters_with_nearfield_operations->size( );
        ++cluster_index ) {
    mesh::general_spacetime_cluster * current_cluster
      = ( *clusters_with_nearfield_operations )
        [ permutation_index[ cluster_index ] ];
    if ( current_cluster->get_n_children( ) == 0 ) {
      std::vector< general_spacetime_cluster * > * nearfield_list
        = current_cluster->get_nearfield_list( );
      for ( std::vector< general_spacetime_cluster * >::size_type src_index = 0;
            src_index < nearfield_list->size( ); ++src_index ) {
        mesh::general_spacetime_cluster * nearfield_cluster
          = ( *nearfield_list )[ src_index ];

        full_matrix_type * block = global_matrix.create_nearfield_matrix(
          permutation_index[ cluster_index ], src_index );
        assemble_nearfield_matrix( current_cluster, nearfield_cluster, *block );
        global_matrix.add_to_local_approximated_size(
          block->get_n_rows( ) * block->get_n_columns( ) );
        global_matrix.add_to_local_full_size(
          block->get_n_rows( ) * block->get_n_columns( ) );
        // if an aca recompression of spatially admissible nearfield blocks is
        // active, estimate the largest singular value of the diagonal block.
        if ( _aca_eps > 0.0 && current_cluster == nearfield_cluster ) {
          sc current_max_svd = block->estimate_max_singular_value( );
          largest_sing_val_diag_blocks.insert(
            { current_cluster, current_max_svd } );
        }
      }
    }
  }

  // in case of aca recompression, update the map of largest singular values for
  // non-leaf clusters (where no diagonal nearfield matrix was computed)
  if ( _aca_eps > 0.0 ) {
    for ( std::vector< general_spacetime_cluster * >::size_type cluster_index
          = 0;
          cluster_index < clusters_with_nearfield_operations->size( );
          ++cluster_index ) {
      mesh::general_spacetime_cluster * current_cluster
        = ( *clusters_with_nearfield_operations )
          [ permutation_index[ cluster_index ] ];
      if ( current_cluster->get_n_children( ) > 0 ) {
        std::vector< general_spacetime_cluster * > current_leaf_descendants;
        mesh::distributed_spacetime_cluster_tree * test_distributed_st_tree
          = _test_space->get_tree( );
        test_distributed_st_tree->collect_local_leaves(
          *current_cluster, current_leaf_descendants );
        // set the largest singular value for the current cluster to the maximum
        // of all its descendants.
        sc max_sing_val = 0.0;
        for ( auto leaf_descendant : current_leaf_descendants ) {
          max_sing_val = std::max(
            max_sing_val, largest_sing_val_diag_blocks[ leaf_descendant ] );
        }
        largest_sing_val_diag_blocks.insert(
          { current_cluster, max_sing_val } );
      }
    }
  }

  // auxiliary vectors to count the number of spatially admissible matrices and
  // those of them which could not be compressed (in case of aca recompression)
  std::vector< lo > n_tot_spat_adm_matrices_per_thread(
    omp_get_max_threads( ), 0 );
  std::vector< lo > n_failed_compression_spat_adm_per_thread(
    omp_get_max_threads( ), 0 );
  // next, assemble the spatially admissible nearfield matrices
#pragma omp parallel for schedule( dynamic, 1 )
  for ( std::vector< general_spacetime_cluster * >::size_type cluster_index = 0;
        cluster_index < clusters_with_nearfield_operations->size( );
        ++cluster_index ) {
    mesh::general_spacetime_cluster * current_cluster
      = ( *clusters_with_nearfield_operations )
        [ permutation_index[ cluster_index ] ];
    std::vector< general_spacetime_cluster * > * spat_adm_nearfield_list
      = current_cluster->get_spatially_admissible_nearfield_list( );
    if ( spat_adm_nearfield_list != nullptr ) {
      sc max_singular_value = 0.0;
      if ( _aca_eps > 0.0 ) {
        max_singular_value = largest_sing_val_diag_blocks[ current_cluster ];
      }
      for ( std::vector< general_spacetime_cluster * >::size_type src_index = 0;
            src_index < spat_adm_nearfield_list->size( ); ++src_index ) {
        n_tot_spat_adm_matrices_per_thread[ omp_get_thread_num( ) ]++;
        general_spacetime_cluster * nearfield_cluster
          = ( *spat_adm_nearfield_list )[ src_index ];
        lo n_cols = nearfield_cluster->get_n_dofs< trial_space_type >( );
        lo n_rows = current_cluster->get_n_dofs< test_space_type >( );
        // currently, the matrix is assembled fully, then approximated
        // TODO: to speed-up the process, assemble only the necessary rows and
        // columns
        full_matrix_type * full_block = new full_matrix_type( n_rows, n_cols );
        assemble_nearfield_matrix(
          current_cluster, nearfield_cluster, *full_block );
        bool successful_compression = false;
        if ( _aca_eps > 0.0 ) {
          // try to compress the full_block via aca and an additional svd
          low_rank_matrix_type * lr_block = new low_rank_matrix_type( );
          sc est_compression_error;
          successful_compression = compute_low_rank_approximation( *full_block,
            *lr_block, est_compression_error, true, max_singular_value );
          if ( successful_compression ) {
            delete full_block;
            lo rank = lr_block->get_rank( );
            if ( rank > 0 ) {
              global_matrix.insert_spatially_admissible_nearfield_matrix(
                permutation_index[ cluster_index ], src_index, lr_block );
            } else {
              delete lr_block;
            }
            // update the auxiliary variables measuring the nearfield
            // compression
            global_matrix.add_to_local_approximated_size(
              rank * ( n_rows + n_cols ) );
            global_matrix.add_to_local_full_size( n_rows * n_cols );
          } else {
            delete lr_block;
          }
        }
        if ( !successful_compression ) {
          n_failed_compression_spat_adm_per_thread[ omp_get_thread_num( ) ]++;
          global_matrix.insert_spatially_admissible_nearfield_matrix(
            permutation_index[ cluster_index ], src_index, full_block );
          // update the auxiliary variables measuring the nearfield
          global_matrix.add_to_local_approximated_size( n_rows * n_cols );
          global_matrix.add_to_local_full_size( n_rows * n_cols );
        }
      }
    }
  }
  lo n_failed_compression_spat_adm( 0 ), n_tot_spat_adm_matrices( 0 );
  for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
    n_failed_compression_spat_adm
      += n_failed_compression_spat_adm_per_thread[ i ];
    n_tot_spat_adm_matrices += n_tot_spat_adm_matrices_per_thread[ i ];
  }
  std::cout << "total number of spatially admissible nearfield matrices: "
            << n_tot_spat_adm_matrices << std::endl;
  std::cout << "failed compression in " << n_failed_compression_spat_adm
            << " cases." << std::endl;
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::assemble_nearfield_matrix( general_spacetime_cluster *
                                                   target_cluster,
  general_spacetime_cluster * source_cluster,
  full_matrix_type & nearfield_matrix ) const {
  nearfield_matrix.fill( 0.0 );

  // basis is either basis_tri_p0 or basis_tri_p1.
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );

  const distributed_spacetime_tensor_mesh & distributed_test_mesh
    = _test_space->get_mesh( );
  const distributed_spacetime_tensor_mesh & distributed_trial_mesh
    = _trial_space->get_mesh( );
  const spacetime_tensor_mesh * test_mesh;
  const spacetime_tensor_mesh * trial_mesh;
  lo test_mesh_start_idx, trial_mesh_start_idx;
  if ( target_cluster->get_process_id( ) == _my_rank ) {
    test_mesh = distributed_test_mesh.get_local_mesh( );
    test_mesh_start_idx = distributed_test_mesh.get_local_start_idx( );
  } else {
    test_mesh = distributed_test_mesh.get_nearfield_mesh( );
    test_mesh_start_idx = distributed_test_mesh.get_nearfield_start_idx( );
  }
  if ( source_cluster->get_process_id( ) == _my_rank ) {
    trial_mesh = distributed_trial_mesh.get_local_mesh( );
    trial_mesh_start_idx = distributed_trial_mesh.get_local_start_idx( );
  } else {
    trial_mesh = distributed_trial_mesh.get_nearfield_mesh( );
    trial_mesh_start_idx = distributed_trial_mesh.get_nearfield_start_idx( );
  }

  lo n_test_time_elem = target_cluster->get_n_time_elements( );
  lo n_test_space_dofs = target_cluster->get_n_space_dofs< test_space_type >( );
  lo n_test_space_elem = target_cluster->get_n_space_elements( );
  std::vector< lo > test_elems = target_cluster->get_all_elements( );

  lo n_trial_time_elem = source_cluster->get_n_time_elements( );
  lo n_trial_space_dofs
    = source_cluster->get_n_space_dofs< trial_space_type >( );
  lo n_trial_space_elem = source_cluster->get_n_space_elements( );
  std::vector< lo > trial_elems = source_cluster->get_all_elements( );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  char time_configuration = 0;
  if ( source_cluster->get_level( ) >= target_cluster->get_level( ) ) {
    if ( std::abs( target_cluster->get_time_center( )
           - source_cluster->get_time_center( ) )
      < target_cluster->get_time_half_size( ) ) {
      // source cluster's temporal component is contained in target cluster's
      time_configuration = 1;
    }
  } else {
    if ( std::abs( source_cluster->get_time_center( )
           - target_cluster->get_time_center( ) )
      < source_cluster->get_time_half_size( ) ) {
      // target cluster's temporal component is contained in source cluster's
      time_configuration = 2;
    }
  }

  std::vector< lo > test_loc_access( n_loc_rows );
  std::vector< lo > trial_loc_access( n_loc_columns );

  sc test, trial, value, test_area, trial_area;
  lo size;
  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  sc t0, t1, tau0, tau1;
  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;
  linear_algebra::coordinates< 3 > nx, ny;

  sc * nx_data = nx.data( );
  sc * ny_data = ny.data( );

  quadrature_wrapper my_quadrature;
  init_quadrature( my_quadrature );
  sc * x1_ref = nullptr;
  sc * x2_ref = nullptr;
  sc * y1_ref = nullptr;
  sc * y2_ref = nullptr;
  sc * w = nullptr;
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );
  sc * kernel_data = my_quadrature._kernel_values.data( );

  lo test_elem_time, gl_test_elem_time, gl_test_elem_space;
  lo trial_elem_time, gl_trial_elem_time, gl_trial_elem_space;

  lo test_element_spacetime, trial_element_spacetime;

  // determine the bounds for the loops over the temporal elements
  lo i_test_min = 0;
  lo trial_offset = 0;
  if ( time_configuration == 1 ) {
    // source cluster' s temporal component is contained in target cluster' s
    lo glob_i_trial_min = trial_mesh->get_time_element( trial_elems[ 0 ] );
    lo glob_i_test_min = test_mesh->get_time_element( test_elems[ 0 ] );
    i_test_min = glob_i_trial_min - glob_i_test_min;
  } else if ( time_configuration == 2 ) {
    // target cluster' s temporal component is contained in source cluster' s
    lo glob_i_trial_min = trial_mesh->get_time_element( trial_elems[ 0 ] );
    lo glob_i_test_min = test_mesh->get_time_element( test_elems[ 0 ] );
    trial_offset = glob_i_test_min - glob_i_trial_min;
  }

  for ( lo i_test_time = i_test_min; i_test_time < n_test_time_elem;
        ++i_test_time ) {
    lo i_trial_max = n_trial_time_elem - 1;
    if ( time_configuration == 1 ) {
      i_trial_max = std::min( i_test_time, i_trial_max );
    } else if ( time_configuration == 2 ) {
      i_trial_max = trial_offset + i_test_time;
    }
    for ( lo i_trial_time = 0; i_trial_time <= i_trial_max; ++i_trial_time ) {
      for ( lo i_test_space = 0; i_test_space < n_test_space_elem;
            ++i_test_space ) {
        // get the index of the current spacetime test element and transform
        // it to the local indices in the appropriate mesh (nearfield or
        // local)
        test_element_spacetime
          = distributed_test_mesh.global_2_local( test_mesh_start_idx,
            test_elems[ i_test_time * n_test_space_elem + i_test_space ] );
        // get the indices of the time element and space element of which
        // the spacetime element consists and get some data.
        test_elem_time = test_mesh->get_time_element( test_element_spacetime );
        test_mesh->get_temporal_nodes( test_elem_time, &t0, &t1 );
        gl_test_elem_space
          = test_mesh->get_space_element_index( test_element_spacetime );
        test_mesh->get_spatial_nodes_using_spatial_element_index(
          gl_test_elem_space, x1, x2, x3 );
        test_mesh->get_spatial_normal_using_spatial_element_index(
          gl_test_elem_space, nx );
        test_area = test_mesh->get_spatial_area_using_spatial_index(
          gl_test_elem_space );
        for ( lo i_trial_space = 0; i_trial_space < n_trial_space_elem;
              ++i_trial_space ) {
          // get the appropriate indices of the spacetime trial element, its
          // spatial and temporal parts and some necessary data
          trial_element_spacetime = distributed_trial_mesh.global_2_local(
            trial_mesh_start_idx,
            trial_elems[ i_trial_time * n_trial_space_elem + i_trial_space ] );
          trial_elem_time
            = trial_mesh->get_time_element( trial_element_spacetime );
          trial_mesh->get_temporal_nodes( trial_elem_time, &tau0, &tau1 );
          gl_trial_elem_space
            = trial_mesh->get_space_element_index( trial_element_spacetime );
          // determine the configuration in time and space.
          gl_trial_elem_time = trial_mesh_start_idx + trial_elem_time;
          gl_test_elem_time = test_mesh_start_idx + test_elem_time;
          bool shared_t_element = ( gl_trial_elem_time == gl_test_elem_time );
          bool shared_t_vertex
            = ( gl_trial_elem_time == gl_test_elem_time - 1 );
          if ( shared_t_element || shared_t_vertex ) {
            get_type( gl_test_elem_space, gl_trial_elem_space,
              n_shared_vertices, rot_test, rot_trial );
          } else {
            n_shared_vertices = 0;
            rot_test = 0;
            rot_trial = 0;
          }
          trial_mesh->get_spatial_nodes_using_spatial_element_index(
            gl_trial_elem_space, y1, y2, y3 );
          trial_mesh->get_spatial_normal_using_spatial_element_index(
            gl_trial_elem_space, ny );
          trial_area = trial_mesh->get_spatial_area_using_spatial_index(
            gl_trial_elem_space );

          target_cluster->local_elem_to_local_space_dofs< test_space_type >(
            i_test_space, n_shared_vertices, rot_test, false, test_loc_access );
          source_cluster->local_elem_to_local_space_dofs< trial_space_type >(
            i_trial_space, n_shared_vertices, rot_trial, true,
            trial_loc_access );

          triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices,
            rot_test, rot_trial, my_quadrature );
          x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
          x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
          y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
          y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
          w = my_quadrature._w[ n_shared_vertices ].data( );

          size = my_quadrature._w[ n_shared_vertices ].size( );

          if ( shared_t_element ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel_data[ i_quad ]
                = _kernel->definite_integral_over_same_interval(
                    x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                    x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                    x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, ny_data,
                    t0, t1 )
                * w[ i_quad ];
            }
          } else {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel_data[ i_quad ]
                = _kernel->definite_integral_over_different_intervals(
                    x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                    x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                    x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, ny_data,
                    t0, t1, tau0, tau1 )
                * w[ i_quad ];
            }
          }

          for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
            for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                  ++i_loc_trial ) {
              value = 0.0;
#pragma omp simd \
      aligned( x1_ref, x2_ref, y1_ref, y2_ref, kernel_data : DATA_ALIGN ) \
      private( test, trial ) reduction( + : value ) simdlen( BESTHEA_SIMD_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                // for p0 and p1 basis functions only the values of
                // i_loc_test, x1_ref and x2_ref are used, the other variables
                // are ignored. -> execution should be fine
                test = test_basis.evaluate( gl_test_elem_space, i_loc_test,
                  x1_ref[ i_quad ], x2_ref[ i_quad ], nx_data,
                  n_shared_vertices, rot_test, false );
                trial = trial_basis.evaluate( gl_trial_elem_space, i_loc_trial,
                  y1_ref[ i_quad ], y2_ref[ i_quad ], ny_data,
                  n_shared_vertices, rot_trial, true );
                value += kernel_data[ i_quad ] * test * trial;
              }

              value *= test_area * trial_area;
              nearfield_matrix.add(
                i_test_time * n_test_space_dofs + test_loc_access[ i_loc_test ],
                i_trial_time * n_trial_space_dofs
                  + trial_loc_access[ i_loc_trial ],
                value );
            }
          }
        }
      }
    }
  }
}

/** specialization of
 * @ref distributed_fast_spacetime_be_assembler::assemble_nearfield_matrix for
 * p1p1 hypersingular operator */
template<>
void besthea::bem::distributed_fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  assemble_nearfield_matrix(
    [[maybe_unused]] general_spacetime_cluster * target_cluster,
    [[maybe_unused]] general_spacetime_cluster * source_cluster,
    [[maybe_unused]] full_matrix_type & nearfield_matrix ) const {
  // basis is either basis_tri_p0 or basis_tri_p1.
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );

  const distributed_spacetime_tensor_mesh & distributed_test_mesh
    = _test_space->get_mesh( );
  const distributed_spacetime_tensor_mesh & distributed_trial_mesh
    = _trial_space->get_mesh( );
  const spacetime_tensor_mesh * test_mesh;
  const spacetime_tensor_mesh * trial_mesh;
  lo test_mesh_start_idx, trial_mesh_start_idx;
  if ( target_cluster->get_process_id( ) == _my_rank ) {
    test_mesh = distributed_test_mesh.get_local_mesh( );
    test_mesh_start_idx = distributed_test_mesh.get_local_start_idx( );
  } else {
    test_mesh = distributed_test_mesh.get_nearfield_mesh( );
    test_mesh_start_idx = distributed_test_mesh.get_nearfield_start_idx( );
  }
  if ( source_cluster->get_process_id( ) == _my_rank ) {
    trial_mesh = distributed_trial_mesh.get_local_mesh( );
    trial_mesh_start_idx = distributed_trial_mesh.get_local_start_idx( );
  } else {
    trial_mesh = distributed_trial_mesh.get_nearfield_mesh( );
    trial_mesh_start_idx = distributed_trial_mesh.get_nearfield_start_idx( );
  }

  lo n_test_time_elem = target_cluster->get_n_time_elements( );
  lo n_test_space_dofs = target_cluster->get_n_space_dofs< besthea::bem::
      distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( );
  lo n_test_space_elem = target_cluster->get_n_space_elements( );
  std::vector< lo > test_elems = target_cluster->get_all_elements( );

  lo n_trial_time_elem = source_cluster->get_n_time_elements( );
  lo n_trial_space_dofs = source_cluster->get_n_space_dofs< besthea::bem::
      distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( );
  lo n_trial_space_elem = source_cluster->get_n_space_elements( );
  std::vector< lo > trial_elems = source_cluster->get_all_elements( );

  char time_configuration = 0;
  if ( source_cluster->get_level( ) >= target_cluster->get_level( ) ) {
    if ( std::abs( target_cluster->get_time_center( )
           - source_cluster->get_time_center( ) )
      < target_cluster->get_time_half_size( ) ) {
      // source cluster's temporal component is contained in target cluster's
      time_configuration = 1;
    }
  } else {
    if ( std::abs( source_cluster->get_time_center( )
           - target_cluster->get_time_center( ) )
      < source_cluster->get_time_half_size( ) ) {
      // target cluster's temporal component is contained in source cluster's
      time_configuration = 2;
    }
  }
  std::vector< lo > test_loc_access( 3 );
  std::vector< lo > trial_loc_access( 3 );

  sc areas, test_area;
  sc value11, value12, value13;
  sc value21, value22, value23;
  sc value31, value32, value33;
  lo size;
  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  sc t0, t1, tau0, tau1;
  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;
  linear_algebra::coordinates< 3 > nx, ny;

  sc * nx_data = nx.data( );
  sc * ny_data = ny.data( );

  sc test_curls[ 9 ], trial_curls[ 9 ];
  sc phi1x, phi1y;
  sc kernel1, kernel2;
  lo test_curl_offset, trial_curl_offset;
  sc curl_dot[ 9 ];

  quadrature_wrapper my_quadrature;
  init_quadrature( my_quadrature );
  sc * x1_ref = nullptr;
  sc * x2_ref = nullptr;
  sc * y1_ref = nullptr;
  sc * y2_ref = nullptr;
  sc * w = nullptr;
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo test_elem_time, gl_test_elem_time, gl_test_elem_space;
  lo trial_elem_time, gl_trial_elem_time, gl_trial_elem_space;

  lo test_element_spacetime, trial_element_spacetime;

  // determine the bounds for the loops over the temporal elements
  lo i_test_min = 0;
  lo trial_offset = 0;
  if ( time_configuration == 1 ) {
    // source cluster' s temporal component is contained in target cluster' s
    lo glob_i_trial_min = trial_mesh->get_time_element( trial_elems[ 0 ] );
    lo glob_i_test_min = test_mesh->get_time_element( test_elems[ 0 ] );
    i_test_min = glob_i_trial_min - glob_i_test_min;
  } else if ( time_configuration == 2 ) {
    // target cluster' s temporal component is contained in source cluster' s
    lo glob_i_trial_min = trial_mesh->get_time_element( trial_elems[ 0 ] );
    lo glob_i_test_min = test_mesh->get_time_element( test_elems[ 0 ] );
    trial_offset = glob_i_test_min - glob_i_trial_min;
  }

  for ( lo i_test_time = i_test_min; i_test_time < n_test_time_elem;
        ++i_test_time ) {
    lo i_trial_max = n_trial_time_elem - 1;
    if ( time_configuration == 1 ) {
      i_trial_max = std::min( i_test_time, i_trial_max );
    } else if ( time_configuration == 2 ) {
      i_trial_max = trial_offset + i_test_time;
    }
    for ( lo i_trial_time = 0; i_trial_time <= i_trial_max; ++i_trial_time ) {
      for ( lo i_test_space = 0; i_test_space < n_test_space_elem;
            ++i_test_space ) {
        // get the index of the current spacetime test element and transform
        // it to the local indices in the appropriate mesh (nearfield or
        // local)
        test_element_spacetime
          = distributed_test_mesh.global_2_local( test_mesh_start_idx,
            test_elems[ i_test_time * n_test_space_elem + i_test_space ] );
        // get the indices of the time element and space element of which
        // the spacetime element consists and get some data.
        test_elem_time = test_mesh->get_time_element( test_element_spacetime );
        test_mesh->get_temporal_nodes( test_elem_time, &t0, &t1 );
        gl_test_elem_space
          = test_mesh->get_space_element_index( test_element_spacetime );
        test_mesh->get_spatial_nodes_using_spatial_element_index(
          gl_test_elem_space, x1, x2, x3 );
        test_mesh->get_spatial_normal_using_spatial_element_index(
          gl_test_elem_space, nx );
        test_area = test_mesh->get_spatial_area_using_spatial_index(
          gl_test_elem_space );
        for ( lo i_trial_space = 0; i_trial_space < n_trial_space_elem;
              ++i_trial_space ) {
          // get the appropriate indices of the spacetime trial element, its
          // spatial and temporal parts and some necessary data
          trial_element_spacetime = distributed_trial_mesh.global_2_local(
            trial_mesh_start_idx,
            trial_elems[ i_trial_time * n_trial_space_elem + i_trial_space ] );
          trial_elem_time
            = trial_mesh->get_time_element( trial_element_spacetime );
          trial_mesh->get_temporal_nodes( trial_elem_time, &tau0, &tau1 );
          gl_trial_elem_space
            = trial_mesh->get_space_element_index( trial_element_spacetime );
          // determine the configuration in time and space.
          gl_trial_elem_time = trial_mesh_start_idx + trial_elem_time;
          gl_test_elem_time = test_mesh_start_idx + test_elem_time;
          bool shared_t_element = ( gl_trial_elem_time == gl_test_elem_time );
          bool shared_t_vertex
            = ( gl_trial_elem_time == gl_test_elem_time - 1 );
          if ( shared_t_element || shared_t_vertex ) {
            get_type( gl_test_elem_space, gl_trial_elem_space,
              n_shared_vertices, rot_test, rot_trial );
          } else {
            n_shared_vertices = 0;
            rot_test = 0;
            rot_trial = 0;
          }
          trial_mesh->get_spatial_nodes_using_spatial_element_index(
            gl_trial_elem_space, y1, y2, y3 );
          trial_mesh->get_spatial_normal_using_spatial_element_index(
            gl_trial_elem_space, ny );
          areas = test_area
            * trial_mesh->get_spatial_area_using_spatial_index(
              gl_trial_elem_space );

          target_cluster->local_elem_to_local_space_dofs<
            besthea::bem::distributed_fast_spacetime_be_space<
              besthea::bem::basis_tri_p1 > >(
            i_test_space, n_shared_vertices, rot_test, false, test_loc_access );
          source_cluster->local_elem_to_local_space_dofs<
            besthea::bem::distributed_fast_spacetime_be_space<
              besthea::bem::basis_tri_p1 > >( i_trial_space, n_shared_vertices,
            rot_trial, true, trial_loc_access );

          triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices,
            rot_test, rot_trial, my_quadrature );
          x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
          x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
          y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
          y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
          w = my_quadrature._w[ n_shared_vertices ].data( );

          size = my_quadrature._w[ n_shared_vertices ].size( );

          test_basis.evaluate_curl( gl_test_elem_space, nx, n_shared_vertices,
            rot_test, false, test_curls );
          trial_basis.evaluate_curl( gl_trial_elem_space, ny, n_shared_vertices,
            rot_trial, true, trial_curls );

          for ( lo i_loc_test = 0; i_loc_test < 3; ++i_loc_test ) {
            for ( lo i_loc_trial = 0; i_loc_trial < 3; ++i_loc_trial ) {
              test_curl_offset = 3 * i_loc_test;
              trial_curl_offset = 3 * i_loc_trial;
              curl_dot[ i_loc_trial * 3 + i_loc_test ]
                = test_curls[ test_curl_offset ]
                  * trial_curls[ trial_curl_offset ]
                + test_curls[ test_curl_offset + 1 ]
                  * trial_curls[ trial_curl_offset + 1 ]
                + test_curls[ test_curl_offset + 2 ]
                  * trial_curls[ trial_curl_offset + 2 ];
            }
          }

          value11 = value12 = value13 = 0.0;
          value21 = value22 = value23 = 0.0;
          value31 = value32 = value33 = 0.0;

          if ( shared_t_element ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref \
                          : DATA_ALIGN )                                  \
  aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w: DATA_ALIGN ) \
       private( kernel1, kernel2, phi1x, phi1y ) \
       reduction( + : value11, value12, value13 ) \
       reduction( + : value21, value22, value23 ) \
       reduction( + : value31, value32, value33 ) \
       simdlen( BESTHEA_SIMD_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              _kernel->definite_integral_over_same_interval(
                x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, ny_data, t0,
                t1, &kernel1, &kernel2 );

              phi1x = (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ];
              phi1y = (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ];
              // phi2* = *1_ref[ i_quad ] and phi3* = *2_ref[ i_quad ];

              value11 += ( kernel1 * curl_dot[ 0 ] + kernel2 * phi1x * phi1y )
                * w[ i_quad ];
              value21 += ( kernel1 * curl_dot[ 1 ]
                           + kernel2 * x1_ref[ i_quad ] * phi1y )
                * w[ i_quad ];
              value31 += ( kernel1 * curl_dot[ 2 ]
                           + kernel2 * x2_ref[ i_quad ] * phi1y )
                * w[ i_quad ];
              value12 += ( kernel1 * curl_dot[ 3 ]
                           + kernel2 * phi1x * y1_ref[ i_quad ] )
                * w[ i_quad ];
              value22 += ( kernel1 * curl_dot[ 4 ]
                           + kernel2 * x1_ref[ i_quad ] * y1_ref[ i_quad ] )
                * w[ i_quad ];
              value32 += ( kernel1 * curl_dot[ 5 ]
                           + kernel2 * x2_ref[ i_quad ] * y1_ref[ i_quad ] )
                * w[ i_quad ];
              value13 += ( kernel1 * curl_dot[ 6 ]
                           + kernel2 * phi1x * y2_ref[ i_quad ] )
                * w[ i_quad ];
              value23 += ( kernel1 * curl_dot[ 7 ]
                           + kernel2 * x1_ref[ i_quad ] * y2_ref[ i_quad ] )
                * w[ i_quad ];
              value33 += ( kernel1 * curl_dot[ 8 ]
                           + kernel2 * x2_ref[ i_quad ] * y2_ref[ i_quad ] )
                * w[ i_quad ];
            }
          } else {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref \
                          : DATA_ALIGN )                                  \
  aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w: DATA_ALIGN ) \
       private( kernel1, kernel2, phi1x, phi1y ) \
       reduction( + : value11, value12, value13 ) \
       reduction( + : value21, value22, value23 ) \
       reduction( + : value31, value32, value33 ) \
       simdlen( BESTHEA_SIMD_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              _kernel->definite_integral_over_different_intervals(
                x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, ny_data, t0,
                t1, tau0, tau1, &kernel1, &kernel2 );

              phi1x = (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ];
              phi1y = (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ];
              // phi2* = *1_ref[ i_quad ] and phi3* = *2_ref[ i_quad ];

              value11 += ( kernel1 * curl_dot[ 0 ] + kernel2 * phi1x * phi1y )
                * w[ i_quad ];
              value21 += ( kernel1 * curl_dot[ 1 ]
                           + kernel2 * x1_ref[ i_quad ] * phi1y )
                * w[ i_quad ];
              value31 += ( kernel1 * curl_dot[ 2 ]
                           + kernel2 * x2_ref[ i_quad ] * phi1y )
                * w[ i_quad ];
              value12 += ( kernel1 * curl_dot[ 3 ]
                           + kernel2 * phi1x * y1_ref[ i_quad ] )
                * w[ i_quad ];
              value22 += ( kernel1 * curl_dot[ 4 ]
                           + kernel2 * x1_ref[ i_quad ] * y1_ref[ i_quad ] )
                * w[ i_quad ];
              value32 += ( kernel1 * curl_dot[ 5 ]
                           + kernel2 * x2_ref[ i_quad ] * y1_ref[ i_quad ] )
                * w[ i_quad ];
              value13 += ( kernel1 * curl_dot[ 6 ]
                           + kernel2 * phi1x * y2_ref[ i_quad ] )
                * w[ i_quad ];
              value23 += ( kernel1 * curl_dot[ 7 ]
                           + kernel2 * x1_ref[ i_quad ] * y2_ref[ i_quad ] )
                * w[ i_quad ];
              value33 += ( kernel1 * curl_dot[ 8 ]
                           + kernel2 * x2_ref[ i_quad ] * y2_ref[ i_quad ] )
                * w[ i_quad ];
            }
          }

          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 0 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 0 ],
            value11 * areas );
          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 0 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 1 ],
            value12 * areas );
          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 0 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 2 ],
            value13 * areas );
          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 1 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 0 ],
            value21 * areas );
          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 1 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 1 ],
            value22 * areas );
          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 1 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 2 ],
            value23 * areas );
          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 2 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 0 ],
            value31 * areas );
          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 2 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 1 ],
            value32 * areas );
          nearfield_matrix.add_atomic(
            i_test_time * n_test_space_dofs + test_loc_access[ 2 ],
            i_trial_time * n_trial_space_dofs + trial_loc_access[ 2 ],
            value33 * areas );
        }
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
bool besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::compute_low_rank_approximation( const full_matrix_type &
                                                        full_nf_matrix,
  low_rank_matrix_type & lr_nf_matrix, sc & estimated_eps,
  bool enable_svd_recompression, sc svd_recompression_reference_value ) const {
  lo i, j, ell;             // indices
  lo ik, jk;                // Pivot-Indices
  lo k;                     // current rank
  sc old_eps;               // old accuracy
  sc error2 = 0.0;          // square of the Frobenius-norm of the update
  sc frobnr2 = 0.0;         // square of the Frobenius-norm of the block
  sc crit2;                 // auxiliary variable
  sc tmp, max, pmax = 1.0;  // auxiliary variables
  sc errU, errV;            // variables used for error computation
  sc scale;                 // auxiliary variable
  sc *p, *q;                // pointer to current rank-1-matrix
  lo row_dim = full_nf_matrix.get_n_rows( );
  lo col_dim = full_nf_matrix.get_n_columns( );
  bool successful_compression = false;

  besthea::bem::kernel_from_matrix nf_matrix_kernel( &full_nf_matrix );

  // construct low rank components u and v_aux (large enough)
  full_matrix_type u( row_dim, col_dim, true );
  full_matrix_type v( col_dim, row_dim, true );

  sc * u_data = u.data( );
  sc * v_data = v.data( );

  old_eps = _aca_eps;

  // containers for row and column indices
  std::vector< lo > Zi( row_dim, 1 );
  std::vector< lo > Zj( col_dim, 1 );

  // initialisation of indices
  k = 0;
  ik = 0;
  jk = 0;

  while ( 1 ) {
    p = u_data + k * row_dim;  // memory address p = u_k
    // compute u_k
    {
      // generate a new column
      //      #pragma omp parallel for
      for ( i = 0; i < row_dim; i++ )
        if ( Zi[ i ] )
          p[ i ] = nf_matrix_kernel.evaluate( i, jk );  // F(i, jk);

      // compute the residuum
      for ( ell = 0; ell < k; ell++ ) {
        scale = v_data[ ell * col_dim + jk ];
        q = u_data + ell * row_dim;
        for ( i = 0; i < row_dim; i++ )
          if ( Zi[ i ] )
            p[ i ] -= scale * q[ i ];
      }

      // compute the maximum (pivot)
      {
        max = 0.0;
        for ( i = 0; i < row_dim; i++ ) {
          if ( Zi[ i ] ) {
            tmp = std::abs( p[ i ] );

            if ( tmp > max ) {
              max = tmp;
              ik = i;
            }
          }
        }
        pmax = std::max( pmax, max );
      }
    }

    // check for zero column (TODO: currently hard coded)
    if ( max < 1.0e-30 * pmax ) {
      Zj[ jk ] = 0;
      for ( jk = 0; jk < col_dim && Zj[ jk ] == 0; jk++ ) {
      };

      if ( jk == col_dim )  // all columns considered
      {
        if ( enable_svd_recompression && k > 0 ) {
          low_rank_recompression( u, v, svd_recompression_reference_value, k );
        }
        error2 = frobnr2 * old_eps * old_eps;

        if ( k * ( col_dim + row_dim ) < col_dim * row_dim ) {
          successful_compression = true;
        }

        break;
      }
    } else {
      // scale u_k
      scale = 1.0 / p[ ik ];
      for ( i = 0; i < row_dim; i++ )
        if ( Zi[ i ] )
          p[ i ] *= scale;

      q = v_data + k * col_dim;  // memory address q = v_k

      // compute v_k
      {
        // generate new row
        //        #pragma omp parallel for
        for ( j = 0; j < col_dim; j++ )
          if ( Zj[ j ] )
            q[ j ] = nf_matrix_kernel.evaluate( ik, j );  // F(ik, j);

        // compute the residuum
        for ( ell = 0; ell < k; ell++ ) {
          scale = u_data[ ell * row_dim + ik ];
          p = v_data + ell * col_dim;
          for ( j = 0; j < col_dim; j++ )
            if ( Zj[ j ] )
              q[ j ] -= p[ j ] * scale;
        }

        // exclude rows and columns in the future iterations.
        Zi[ ik ] = 0;
        Zj[ jk ] = 0;

        // compute the maximum (pivot)
        max = 0.0;

        for ( j = 0; j < col_dim; j++ ) {
          if ( Zj[ j ] ) {
            tmp = std::abs( q[ j ] );
            if ( tmp > max ) {
              max = tmp;
              jk = j;
            }
          }
        }
        pmax = std::max( pmax, max );
      }

      // compute the stopping criterion
      {
        for ( ell = 0; ell < k; ell++ ) {
          errU = cblas_ddot(
            row_dim, &u_data[ ell * row_dim ], 1, &u_data[ k * row_dim ], 1 );
          errV = cblas_ddot(
            col_dim, &v_data[ k * col_dim ], 1, &v_data[ ell * col_dim ], 1 );
          frobnr2 += 2.0 * errU * errV;
        }

        errU = cblas_ddot(
          row_dim, &u_data[ k * row_dim ], 1, &u_data[ k * row_dim ], 1 );
        errV = cblas_ddot(
          col_dim, &v_data[ k * col_dim ], 1, &v_data[ k * col_dim ], 1 );
        error2 = errU * errV;
        frobnr2 += error2;

        crit2 = ( _aca_eps ) * (_aca_eps) *frobnr2;
      }

      // increase the rank
      k++;

      // check the stopping criterion
      if ( error2 < crit2 ) {
        if ( enable_svd_recompression && k > 0 ) {
          low_rank_recompression( u, v, svd_recompression_reference_value, k );
        }

        if ( k * ( col_dim + row_dim ) < col_dim * row_dim )
          successful_compression = true;

        break;
      } else {
        if ( k < _aca_max_rank && k >= std::min( col_dim, row_dim ) ) {
          if ( enable_svd_recompression ) {
            low_rank_recompression(
              u, v, svd_recompression_reference_value, k );
          }
          error2 = frobnr2 * old_eps * old_eps;
          if ( k * ( col_dim + row_dim ) < col_dim * row_dim )
            successful_compression = true;

          break;
        }
      }
    }
  }

  // resize u, v to achieved rank
  u.resize( row_dim, k, false );
  v.resize( col_dim, k, false );

  lr_nf_matrix
    = std::move( low_rank_matrix_type( std::move( u ), std::move( v ) ) );

  estimated_eps = sqrt( error2 / frobnr2 );

  return successful_compression;
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::low_rank_recompression( full_matrix_type & u,
  full_matrix_type & v, sc svd_recompression_reference_value,
  lo & rank ) const {
  lo kmax;               // old rank
  long i, j, k;          // indices
  long lwork;            // size of workspace for LAPACK
  long info;             // returninfo of LAPACK-routines
  int minsize, maxsize;  // block sizes
  sc * r_work;           // workspace for LAPACK
  sc * sigma;            // singular values
  sc *atmp, *btmp;       // auxiliary matrices
  sc * rarb;             // R-matrix
  sc *tau1, *tau2;       // auxiliary factor for QR-decomposition
  sc *u_aux, *v_aux;     // auxiliary matrices for SVD
  sc * csigma;           // singular values
  sc * qr_work;          // Workspace for LAPACK

  int rc_svd_flops = 0;  // Estimate of the required floating-point operations
                         // for recompression

  sc * u_data = u.data( );
  sc * v_data = v.data( );
  lo row_dim = u.get_n_rows( );
  lo col_dim = v.get_n_rows( );

  minsize = std::min( row_dim, col_dim );
  maxsize = std::max( row_dim, col_dim );

  kmax = rank;

  atmp = new sc[ ( row_dim + col_dim ) * kmax ];
  btmp = &atmp[ row_dim * kmax ];

  // copy U to auxiliary container
  for ( k = 0; k < kmax; k++ )
    for ( i = 0; i < row_dim; i++ )
      atmp[ i + k * row_dim ] = u_data[ i + k * row_dim ];

#ifdef DEBUG_COMPRESS
  printBlock< T >( "U", atmp, row_dim, kmax );
#endif

  // copy V to auxiliary container
  for ( k = 0; k < kmax; k++ )
    for ( j = 0; j < col_dim; j++ )
      btmp[ j + k * col_dim ] = v_data[ j + k * col_dim ];

#ifdef DEBUG_COMPRESS
  printBlock< T >( "V", btmp, col_dim, kmax );
#endif

  // allocate storage for auxiliary variables and workspace
  lwork = 8 * minsize + maxsize;
  qr_work = new sc[ lwork + row_dim + col_dim + minsize + 3 * kmax * kmax ];
  r_work = new sc[ minsize + 5 * kmax * kmax ];
  tau1 = &qr_work[ lwork ];
  tau2 = &qr_work[ lwork + row_dim ];
  csigma = &qr_work[ lwork + row_dim + col_dim ];
  rarb = &qr_work[ lwork + row_dim + col_dim + minsize ];
  u_aux = &qr_work[ lwork + row_dim + col_dim + minsize + kmax * kmax ];
  v_aux = &qr_work[ lwork + row_dim + col_dim + minsize + kmax * kmax
    + kmax * kmax ];
  sigma = &r_work[ 5 * kmax * kmax ];

  // QR-decomposition of U
  dgeqrf( &row_dim, &kmax, atmp, &row_dim, tau1, qr_work, &lwork, &info );
  // update the number of flops by an appropriate estimate for that operation.
  rc_svd_flops += 2 * kmax * kmax * row_dim;

#ifdef DEBUG_COMPRESS
  if ( info != 0 ) {
    std::cout << "fatal error in recompression: RU" << std::endl;
    exit( 1 );
  }
  printBlockFill( "RU", atmp, row_dim, kmax, row_dim );
#endif

  // QR-decomposition of V
  dgeqrf( &col_dim, &kmax, btmp, &col_dim, tau2, qr_work, &lwork, &info );
  // update the number of flops by an appropriate estimate for this operation.
  rc_svd_flops += 2 * kmax * kmax * col_dim;

#ifdef DEBUG_COMPRESS
  if ( info != 0 ) {
    std::cout << "fatal error in recompression: RV" << std::endl;
    exit( 1 );
  }
  printBlockFill( "RV", btmp, col_dim, kmax, col_dim );
#endif

  // determine RU*RV'  (RU is the R matrix in QR decomposition of U, RV similar)
  for ( i = 0; i < kmax * kmax; i++ ) rarb[ i ] = 0.0;

  for ( i = 0; i < kmax; i++ )
    for ( j = 0; j < kmax; j++ )
      for ( k = std::max( i, j ); k < kmax; k++ )
        // replaces for (k=0; k<kmax; k++) if ((k>=i) && (k>=j))
        rarb[ i + kmax * j ]
          += atmp[ i + k * row_dim ] * btmp[ j + k * col_dim ];

#ifdef DEBUG_COMPRESS
  printBlock< sc >( "R", rarb, kmax, kmax );
#endif

  // determine the matrix QU, i.e. Q in the QR decomposition of U
  dorgqr(
    &row_dim, &kmax, &kmax, atmp, &row_dim, tau1, qr_work, &lwork, &info );

#ifdef DEBUG_COMPRESS
  if ( info != 0 ) {
    std::cout << "fatal error in recompression: QU" << std::endl;
    exit( 1 );
  }
  printBlock< T >( "QU", atmp, row_dim, kmax );
#endif

  // determine the matrix QV, i.e. Q in the QR decomposition of V
  dorgqr(
    &col_dim, &kmax, &kmax, btmp, &col_dim, tau2, qr_work, &lwork, &info );

#ifdef DEBUG_COMPRESS
  if ( info != 0 ) {
    std::cout << "fatal error in recompression: QV" << std::endl;
    exit( 1 );
  }
  printBlock< sc >( "QV", btmp, col_dim, kmax );
#endif

  // sc diff = 0.0;  // auxiliary variable
  // for ( i = 0; i < kmax * kmax; i++ ) {
  //   if ( std::abs( rarb[ i ] ) < ( *eps ) * ( *eps ) )
  //     rarb[ i ] = 0.0;
  //   diff += std::abs( rarb[ i ] );
  // }

  // if ( diff < ( *eps ) * ( *eps ) ) {
  //   std::cout << "RECOMPRESSION: setting new rank to 0" << std::endl;
  //   *rank = 0;
  // } else
  {
    // determine the SVD of RU*RV'
    dgesvd( "A", "A", &kmax, &kmax, rarb, &kmax, sigma, u_aux, &kmax, v_aux,
      &kmax, qr_work, &lwork, &info );
    // update the number of flops by an appropriate estimate for this operation.
    rc_svd_flops += 60 * kmax * kmax * kmax;

    for ( i = 0; i < kmax; i++ ) csigma[ i ] = (sc) sigma[ i ];

#ifdef DEBUG_COMPRESS
    if ( info != 0 ) {
      std::cout << "fatal error in recompression: SVN of R" << std::endl;
      exit( 1 );
    }
    printBlock< sc >( "UR", u_aux, kmax, kmax );
    printBlock< sc >( "SR", sigma, 1, kmax );
    printBlockT< sc >( "VR", v_aux, kmax, kmax );
#endif

    for ( i = 0; i < kmax; i++ ) {
      // scale UR with singular values
      cblas_dscal(
        kmax, csigma[ i ], &u_aux[ i * kmax ], 1 );  // for MKL and cblas
      // dscal_ (&kmax, &sigma[i], &u_aux[i*kmax], eins_);
      // copy VR
      cblas_dcopy(
        kmax, &v_aux[ i ], kmax, &rarb[ i * kmax ], 1 );  // for MKL and cblas
      // dcopy_ (&kmax, &v_aux[i], &kmax, &rarb[i*kmax], eins_);
    }

#ifdef DEBUG_COMPRESS
    printBlock< sc >( "URS", u_aux, kmax, kmax );
#endif

    sc actual_reference_value = sigma[ 0 ];
    if ( svd_recompression_reference_value > 0.0 ) {
      actual_reference_value = svd_recompression_reference_value;
    }

    for ( k = 0; ( k < kmax ) && ( k < row_dim ) && ( k < col_dim )
          && ( sigma[ k ] > _aca_eps * actual_reference_value );
          k++ ) {
    };
    // std::cout << "RECOMPRESSION: first singular value is " << sigma[ 0 ]
    //           << ", original rank " << kmax << ", new rank " << k;
    // if ( k < kmax ) {
    //   std::cout << ", first truncated svd: " << sigma[ k ];
    // }
    // std::cout << std::endl;

    rank = k;
    if ( k > 0 ) {
      // determine U = QU*UR
      cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, row_dim, k, kmax,
        1.0, atmp, row_dim, u_aux, kmax, 0.0, u_data, row_dim );

      // determine V = QV*VR
      cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, col_dim, k, kmax,
        1.0, btmp, col_dim, rarb, kmax, 0.0, v_data, col_dim );
      // update the number of flops by an appropriate estimate for this
      // operation.
      rc_svd_flops += 2 * ( row_dim + col_dim ) * k * k;
    }

#ifdef DEBUG_COMPRESS
    printBlock< T >( "UC", u_data, row_dim, *rank );
    printBlock< T >( "VC", v_data, col_dim, *rank );
#endif
  }

  delete[] qr_work;
  delete[] r_work;
  delete[] atmp;

  // std::cout << "required flops is " << rc_svd_flops << std::endl;

  return;
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::get_type( lo i_test_space,
  lo i_trial_space, int & n_shared_vertices, int & rot_test,
  int & rot_trial ) const {
  // check for identical
  if ( i_test_space == i_trial_space ) {
    n_shared_vertices = 3;
    rot_test = 0;
    rot_trial = 0;
    return;
  }

  linear_algebra::indices< 3 > test_elem;
  linear_algebra::indices< 3 > trial_elem;

  // assuming that the spatial meshes of the nearfield mesh and the local
  // mesh coincide, we can simply consider the local meshes.
  _test_space->get_mesh( )
    .get_local_mesh( )
    ->get_spatial_element_using_spatial_index( i_test_space, test_elem );
  _trial_space->get_mesh( )
    .get_local_mesh( )
    ->get_spatial_element_using_spatial_index( i_trial_space, trial_elem );

  // check for shared edge
  for ( int i_rot_test = 0; i_rot_test < 3; ++i_rot_test ) {
    for ( int i_rot_trial = 0; i_rot_trial < 3; ++i_rot_trial ) {
      if ( ( trial_elem[ i_rot_trial ]
             == test_elem[ map[ ( i_rot_test + 1 ) ] ] )
        && ( trial_elem[ map[ ( i_rot_trial + 1 ) ] ]
          == test_elem[ i_rot_test ] ) ) {
        n_shared_vertices = 2;
        rot_test = i_rot_test;
        rot_trial = i_rot_trial;
        return;
      }
    }
  }

  // check for shared vertex
  for ( int i_rot_test = 0; i_rot_test < 3; ++i_rot_test ) {
    for ( int i_rot_trial = 0; i_rot_trial < 3; ++i_rot_trial ) {
      if ( test_elem[ i_rot_test ] == trial_elem[ i_rot_trial ] ) {
        n_shared_vertices = 1;
        rot_test = i_rot_test;
        rot_trial = i_rot_trial;
        return;
      }
    }
  }

  // disjoint
  n_shared_vertices = 0;
  rot_test = 0;
  rot_trial = 0;
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::init_quadrature( quadrature_wrapper &
    my_quadrature ) const {
  // Use triangle rules for disjoint elements
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x1
    = quadrature::triangle_x1( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x2
    = quadrature::triangle_x2( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_w
    = quadrature::triangle_w( _order_regular );
  lo tri_size = tri_w.size( );
  lo tri_size2 = tri_size * tri_size;

  int n_shared_vertices = 0;
  my_quadrature._x1_ref[ n_shared_vertices ].resize( tri_size2 );
  my_quadrature._x2_ref[ n_shared_vertices ].resize( tri_size2 );
  my_quadrature._y1_ref[ n_shared_vertices ].resize( tri_size2 );
  my_quadrature._y2_ref[ n_shared_vertices ].resize( tri_size2 );
  my_quadrature._w[ n_shared_vertices ].resize( tri_size2 );

  lo counter = 0;
  for ( lo i_x = 0; i_x < tri_size; ++i_x ) {
    for ( lo i_y = 0; i_y < tri_size; ++i_y ) {
      my_quadrature._x1_ref[ n_shared_vertices ][ counter ] = tri_x1[ i_x ];
      my_quadrature._x2_ref[ n_shared_vertices ][ counter ] = tri_x2[ i_x ];
      my_quadrature._y1_ref[ n_shared_vertices ][ counter ] = tri_x1[ i_y ];
      my_quadrature._y2_ref[ n_shared_vertices ][ counter ] = tri_x2[ i_y ];
      my_quadrature._w[ n_shared_vertices ][ counter ]
        = tri_w[ i_x ] * tri_w[ i_y ];
      ++counter;
    }
  }

  // Use tensor Gauss rules for singular configurations
  const std::vector< sc, besthea::allocator_type< sc > > & line_x
    = quadrature::line_x( _order_singular );
  const std::vector< sc, besthea::allocator_type< sc > > & line_w
    = quadrature::line_w( _order_singular );
  lo line_size = line_x.size( );
  lo line_size4 = line_size * line_size * line_size * line_size;
  sc jacobian = 0.0;

  for ( n_shared_vertices = 1; n_shared_vertices <= 3; ++n_shared_vertices ) {
    my_quadrature._x1_ref[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );
    my_quadrature._x2_ref[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );
    my_quadrature._y1_ref[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );
    my_quadrature._y2_ref[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );
    my_quadrature._w[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );

    counter = 0;
    for ( int i_simplex = 0; i_simplex < n_simplices[ n_shared_vertices ];
          ++i_simplex ) {
      for ( lo i_ksi = 0; i_ksi < line_size; ++i_ksi ) {
        for ( lo i_eta1 = 0; i_eta1 < line_size; ++i_eta1 ) {
          for ( lo i_eta2 = 0; i_eta2 < line_size; ++i_eta2 ) {
            for ( lo i_eta3 = 0; i_eta3 < line_size; ++i_eta3 ) {
              hypercube_to_triangles( line_x[ i_ksi ], line_x[ i_eta1 ],
                line_x[ i_eta2 ], line_x[ i_eta3 ], n_shared_vertices,
                i_simplex,
                my_quadrature._x1_ref[ n_shared_vertices ][ counter ],
                my_quadrature._x2_ref[ n_shared_vertices ][ counter ],
                my_quadrature._y1_ref[ n_shared_vertices ][ counter ],
                my_quadrature._y2_ref[ n_shared_vertices ][ counter ],
                jacobian );
              my_quadrature._w[ n_shared_vertices ][ counter ] = 4.0 * jacobian
                * line_w[ i_ksi ] * line_w[ i_eta1 ] * line_w[ i_eta2 ]
                * line_w[ i_eta3 ];
              ++counter;
            }
          }
        }
      }
    }
  }

  lo size = std::max( tri_size2,
    *std::max_element( n_simplices.begin( ), n_simplices.end( ) )
      * line_size4 );
  my_quadrature._x1.resize( size );
  my_quadrature._x2.resize( size );
  my_quadrature._x3.resize( size );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );
  my_quadrature._kernel_values.resize( size );
  my_quadrature._kernel_values_2.resize( size );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::init_quadrature_polynomials( quadrature_wrapper &
    my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._y1_ref_cheb = quadrature::triangle_x1( _order_regular );
  my_quadrature._y2_ref_cheb = quadrature::triangle_x2( _order_regular );
  my_quadrature._wy_cheb = quadrature::triangle_w( _order_regular );

  lo size = my_quadrature._wy_cheb.size( );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );

  my_quadrature._y1_polynomial.resize( size );
  my_quadrature._y2_polynomial.resize( size );
  my_quadrature._y3_polynomial.resize( size );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::hypercube_to_triangles_vertex( sc ksi,
  sc eta1, sc eta2, sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref,
  sc & y2_ref, sc & jacobian ) const {
  jacobian = ksi * ksi * ksi * eta2;

  switch ( simplex ) {
    case 0:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1;
      y1_ref = ksi * eta2 * ( 1 - eta3 );
      y2_ref = ksi * eta2 * eta3;
      break;
    case 1:
      x1_ref = ksi * eta2 * ( 1 - eta3 );
      x2_ref = ksi * eta2 * eta3;
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1;
      break;
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::
  triangle_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    quadrature_wrapper & my_quadrature ) const {
  const sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  const sc * y2_ref = my_quadrature._y2_ref_cheb.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo size = my_quadrature._wy_cheb.size( );

  // x1, x2, x3 are vectors in R^3,
  // y%_mapped are the %th components of the vectors to which y%_ref is mapped
#pragma omp simd aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * y1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * y2_ref[ i ];
    y2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * y1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * y2_ref[ i ];
    y3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * y1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * y2_ref[ i ];
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::hypercube_to_triangles_edge( sc ksi,
  sc eta1, sc eta2, sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref,
  sc & y2_ref, sc & jacobian ) const {
  jacobian = ksi * ksi * ksi * eta1 * eta1;

  switch ( simplex ) {
    case 0:
      x1_ref = ksi * ( 1 - eta1 * eta3 );
      x2_ref = ksi * ( eta1 * eta3 );
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1 * ( 1 - eta2 );
      break;
    case 1:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1;
      y1_ref = ksi * ( 1 - eta1 * eta2 );
      y2_ref = ksi * eta1 * eta2 * ( 1 - eta3 );
      jacobian *= eta2;
      break;
    case 2:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1 * ( 1 - eta2 );
      y1_ref = ksi * ( 1 - eta1 * eta2 * eta3 );
      y2_ref = ksi * eta1 * eta2 * eta3;
      jacobian *= eta2;
      break;
    case 3:
      x1_ref = ksi * ( 1 - eta1 * eta2 );
      x2_ref = ksi * eta1 * eta2 * ( 1 - eta3 );
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1;
      jacobian *= eta2;
      break;
    case 4:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      y1_ref = ksi * ( 1 - eta1 * eta2 );
      y2_ref = ksi * eta1 * eta2;
      jacobian *= eta2;
      break;
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::
  triangles_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3, int n_shared_vertices,
    int rot_test, int rot_trial, quadrature_wrapper & my_quadrature ) const {
  const sc * x1rot = nullptr;
  const sc * x2rot = nullptr;
  const sc * x3rot = nullptr;
  const sc * y1rot = nullptr;
  const sc * y2rot = nullptr;
  const sc * y3rot = nullptr;

  switch ( rot_test ) {
    case 0:
      x1rot = x1.data( );
      x2rot = x2.data( );
      x3rot = x3.data( );
      break;
    case 1:
      x1rot = x2.data( );
      x2rot = x3.data( );
      x3rot = x1.data( );
      break;
    case 2:
      x1rot = x3.data( );
      x2rot = x1.data( );
      x3rot = x2.data( );
      break;
  }

  switch ( rot_trial ) {
    case 0:
      if ( n_shared_vertices == 2 ) {
        y1rot = y2.data( );
        y2rot = y1.data( );
        y3rot = y3.data( );
      } else {
        y1rot = y1.data( );
        y2rot = y2.data( );
        y3rot = y3.data( );
      }
      break;
    case 1:
      if ( n_shared_vertices == 2 ) {
        y1rot = y3.data( );
        y2rot = y2.data( );
        y3rot = y1.data( );
      } else {
        y1rot = y2.data( );
        y2rot = y3.data( );
        y3rot = y1.data( );
      }
      break;
    case 2:
      if ( n_shared_vertices == 2 ) {
        y1rot = y1.data( );
        y2rot = y3.data( );
        y3rot = y2.data( );
      } else {
        y1rot = y3.data( );
        y2rot = y1.data( );
        y3rot = y2.data( );
      }
      break;
  }

  const sc * x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
  const sc * x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
  const sc * y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );

  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo size = my_quadrature._w[ n_shared_vertices ].size( );

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped        \
                          : DATA_ALIGN ) aligned( x1_ref, x2_ref \
                                                  : DATA_ALIGN ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1rot[ 0 ] + ( x2rot[ 0 ] - x1rot[ 0 ] ) * x1_ref[ i ]
      + ( x3rot[ 0 ] - x1rot[ 0 ] ) * x2_ref[ i ];
    x2_mapped[ i ] = x1rot[ 1 ] + ( x2rot[ 1 ] - x1rot[ 1 ] ) * x1_ref[ i ]
      + ( x3rot[ 1 ] - x1rot[ 1 ] ) * x2_ref[ i ];
    x3_mapped[ i ] = x1rot[ 2 ] + ( x2rot[ 2 ] - x1rot[ 2 ] ) * x1_ref[ i ]
      + ( x3rot[ 2 ] - x1rot[ 2 ] ) * x2_ref[ i ];
  }

#pragma omp simd aligned( y1_mapped, y2_mapped, y3_mapped        \
                          : DATA_ALIGN ) aligned( y1_ref, y2_ref \
                                                  : DATA_ALIGN ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = y1rot[ 0 ] + ( y2rot[ 0 ] - y1rot[ 0 ] ) * y1_ref[ i ]
      + ( y3rot[ 0 ] - y1rot[ 0 ] ) * y2_ref[ i ];
    y2_mapped[ i ] = y1rot[ 1 ] + ( y2rot[ 1 ] - y1rot[ 1 ] ) * y1_ref[ i ]
      + ( y3rot[ 1 ] - y1rot[ 1 ] ) * y2_ref[ i ];
    y3_mapped[ i ] = y1rot[ 2 ] + ( y2rot[ 2 ] - y1rot[ 2 ] ) * y1_ref[ i ]
      + ( y3rot[ 2 ] - y1rot[ 2 ] ) * y2_ref[ i ];
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::hypercube_to_triangles_identical( sc ksi,
  sc eta1, sc eta2, sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref,
  sc & y2_ref, sc & jacobian ) const {
  jacobian = ksi * ksi * ksi * eta1 * eta1 * eta2;

  switch ( simplex ) {
    case 0:
      x1_ref = ksi * eta1 * ( 1 - eta2 );
      x2_ref = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
      y1_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      y2_ref = ksi * ( 1 - eta1 );
      break;
    case 1:
      x1_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      x2_ref = ksi * ( 1 - eta1 );
      y1_ref = ksi * eta1 * ( 1 - eta2 );
      y2_ref = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
      break;
    case 2:
      x1_ref = ksi * ( 1 - eta1 * ( 1 - eta2 * ( 1 - eta3 ) ) );
      x2_ref = ksi * eta1 * ( 1 - eta2 * ( 1 - eta3 ) );
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1 * ( 1 - eta2 );
      break;
    case 3:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1 * ( 1 - eta2 );
      y1_ref = ksi * ( 1 - eta1 * ( 1 - eta2 * ( 1 - eta3 ) ) );
      y2_ref = ksi * eta1 * ( 1 - eta2 * ( 1 - eta3 ) );
      break;
    case 4:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      y1_ref = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
      y2_ref = ksi * eta1 * ( 1 - eta2 );
      break;
    case 5:
      x1_ref = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
      x2_ref = ksi * eta1 * ( 1 - eta2 );
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      break;
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::cluster_to_polynomials( quadrature_wrapper &
                                                my_quadrature,
  sc start_0, sc end_0, sc start_1, sc end_1, sc start_2, sc end_2 ) const {
  for ( lo i = 0; i < my_quadrature._y1_polynomial.size( ); ++i ) {
    my_quadrature._y1_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y1[ i ] - start_0 ) / ( end_0 - start_0 );
    my_quadrature._y2_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y2[ i ] - start_1 ) / ( end_1 - start_1 );
    my_quadrature._y3_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y3[ i ] - start_2 ) / ( end_2 - start_2 );
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::initialize_moment_and_local_contributions( ) const {
  lou spat_contribution_size
    = ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
  lou contribution_size = ( _temp_order + 1 ) * spat_contribution_size;
  mesh::tree_structure * trial_distribution_tree
    = _trial_space->get_tree( )->get_distribution_tree( );
  // initialize moments and spatial moments for all relevant clusters in the
  // space-time cluster tree (of the trial space).
  trial_distribution_tree->allocate_moments_in_tree(
    *trial_distribution_tree->get_root( ), contribution_size );
  trial_distribution_tree->allocate_spatial_moments_in_tree(
    *trial_distribution_tree->get_root( ), spat_contribution_size,
    _trial_space->get_tree( )->get_start_space_refinement( ) );

  mesh::tree_structure * test_distribution_tree
    = _test_space->get_tree( )->get_distribution_tree( );
  // initialize space-time and spatial local contributions for all relevant
  // clusters in the space-time cluster tree (of the test space).
  test_distribution_tree->allocate_local_contributions_in_tree(
    *test_distribution_tree->get_root( ), contribution_size );
  test_distribution_tree->allocate_spatial_local_contributions_in_tree(
    *test_distribution_tree->get_root( ), spat_contribution_size,
    _test_space->get_tree( )->get_start_space_refinement( ) );
}

template class besthea::bem::distributed_fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >;
// template class besthea::bem::distributed_fast_spacetime_be_assembler<
//   besthea::bem::spacetime_heat_sl_kernel_antiderivative,
//   besthea::bem::distributed_fast_spacetime_be_space<
//     besthea::bem::basis_tri_p1 >,
//   besthea::bem::distributed_fast_spacetime_be_space<
//     besthea::bem::basis_tri_p1 > >;
template class besthea::bem::distributed_fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;
template class besthea::bem::distributed_fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >;
template class besthea::bem::distributed_fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;
