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
  sc alpha )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ),
    _temp_order( temp_order ),
    _spat_order( spat_order ),
    _m2l_integration_order( _spat_order ),
    _alpha( alpha ),
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
  global_matrix.set_order( _spat_order, _temp_order, _order_regular );
  global_matrix.set_alpha( _alpha );
  global_matrix.set_m2l_integration_order( _m2l_integration_order );

  // number of timesteps have to be the same for test and trial meshes
  lo n_timesteps = _test_space->get_mesh( ).get_n_temporal_elements( );
  // // size of individual blocks
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );
  global_matrix.resize( n_timesteps, n_rows, n_columns );
  // ###########################################################################

  global_matrix.compute_spatial_m2m_coeffs( );

  initialize_moment_and_local_contributions( );

  // fill the m-list, m2l-list, n-list and l2l-list of the distributed pFMM
  // matrix and determine the receive information data.
  global_matrix.prepare_fmm( );

  // precompute Chebyshev nodes and values
  global_matrix.compute_chebyshev( );

  // assemble the nearfield matrices of the pFMM matrix
  if ( !info_mode ) {
    assemble_nearfield( global_matrix );
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::assemble_nearfield( pfmm_matrix_type &
    global_matrix ) const {
  const std::vector< general_spacetime_cluster * > & local_leaves
    = _test_space->get_tree( )->get_local_leaves( );

#pragma omp parallel for
  for ( std::vector< general_spacetime_cluster * >::size_type leaf_index = 0;
        leaf_index < local_leaves.size( ); ++leaf_index ) {
    general_spacetime_cluster * current_cluster = local_leaves[ leaf_index ];
    std::vector< general_spacetime_cluster * > * nearfield_list
      = current_cluster->get_nearfield_list( );
    for ( std::vector< general_spacetime_cluster * >::size_type src_index = 0;
          src_index < nearfield_list->size( ); ++src_index ) {
      general_spacetime_cluster * nearfield_cluster
        = ( *nearfield_list )[ src_index ];

      full_matrix_type * block
        = global_matrix.create_nearfield_matrix( leaf_index, src_index );
      assemble_nearfield_matrix( current_cluster, nearfield_cluster, *block );
    }
  }
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
  //#pragma omp parallel
  {
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
        //#pragma omp for schedule( dynamic )
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
          test_elem_time
            = test_mesh->get_time_element( test_element_spacetime );
          test_mesh->get_temporal_nodes( test_elem_time, &t0, &t1 );
          gl_test_elem_space
            = test_mesh->get_space_element( test_element_spacetime );
          test_mesh->get_spatial_nodes( gl_test_elem_space, x1, x2, x3 );
          test_mesh->get_spatial_normal( gl_test_elem_space, nx );
          test_area = test_mesh->spatial_area( gl_test_elem_space );
          for ( lo i_trial_space = 0; i_trial_space < n_trial_space_elem;
                ++i_trial_space ) {
            // get the appropriate indices of the spacetime trial element, its
            // spatial and temporal parts and some necessary data
            trial_element_spacetime
              = distributed_trial_mesh.global_2_local( trial_mesh_start_idx,
                trial_elems[ i_trial_time * n_trial_space_elem
                  + i_trial_space ] );
            trial_elem_time
              = trial_mesh->get_time_element( trial_element_spacetime );
            trial_mesh->get_temporal_nodes( trial_elem_time, &tau0, &tau1 );
            gl_trial_elem_space
              = trial_mesh->get_space_element( trial_element_spacetime );
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
            trial_mesh->get_spatial_nodes( gl_trial_elem_space, y1, y2, y3 );
            trial_mesh->get_spatial_normal( gl_trial_elem_space, ny );
            trial_area = trial_mesh->spatial_area( gl_trial_elem_space );

            target_cluster->local_elem_to_local_space_dofs< test_space_type >(
              i_test_space, n_shared_vertices, rot_test, false,
              test_loc_access );
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
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                kernel_data[ i_quad ]
                  = _kernel->definite_integral_over_same_interval(
                      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                      x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data,
                      ny_data, t0, t1 )
                  * w[ i_quad ];
              }
            } else {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                kernel_data[ i_quad ]
                  = _kernel->definite_integral_over_different_intervals(
                      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                      x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data,
                      ny_data, t0, t1, tau0, tau1 )
                  * w[ i_quad ];
              }
            }

            for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
              for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                    ++i_loc_trial ) {
                value = 0.0;
#pragma omp simd \
      aligned( x1_ref, x2_ref, y1_ref, y2_ref, kernel_data : DATA_ALIGN ) \
      private( test, trial ) reduction( + : value ) simdlen( DATA_WIDTH )
                for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                  // for p0 and p1 basis functions only the values of
                  // i_loc_test, x1_ref and x2_ref are used, the other variables
                  // are ignored. -> execution should be fine
                  test = test_basis.evaluate( gl_test_elem_space, i_loc_test,
                    x1_ref[ i_quad ], x2_ref[ i_quad ], nx_data,
                    n_shared_vertices, rot_test, false );
                  trial = trial_basis.evaluate( gl_trial_elem_space,
                    i_loc_trial, y1_ref[ i_quad ], y2_ref[ i_quad ], ny_data,
                    n_shared_vertices, rot_trial, true );
                  value += kernel_data[ i_quad ] * test * trial;
                }

                value *= test_area * trial_area;
                nearfield_matrix.add( i_test_time * n_test_space_dofs
                    + test_loc_access[ i_loc_test ],
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
}

/** specialization for p1p1 hypersingular operator */
template<>
void besthea::bem::distributed_fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  assemble_nearfield_matrix(
    [[maybe_unused]] general_spacetime_cluster * target_cluster,
    [[maybe_unused]] general_spacetime_cluster * source_cluster,
    [[maybe_unused]] full_matrix_type & nearfield_matrix ) const {
  /*
auto & test_basis = _test_space->get_basis( );
auto & trial_basis = _trial_space->get_basis( );
auto test_mesh = _test_space->get_mesh( );
auto trial_mesh = _trial_space->get_mesh( );

time_cluster_type & test_time_cluster = target_cluster->get_time_cluster(
); lo n_test_time_elem = test_time_cluster.get_n_elements( );
space_cluster_type & test_space_cluster
 = target_cluster->get_space_cluster( );
lo n_test_space_dofs = test_space_cluster.get_n_dofs<
 besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >(
 );
lo n_test_space_elem = test_space_cluster.get_n_elements( );
std::vector< lo > test_space_elems = test_space_cluster.get_all_elements(
);

time_cluster_type & trial_time_cluster =
source_cluster->get_time_cluster( ); lo n_trial_time_elem =
trial_time_cluster.get_n_elements( ); space_cluster_type &
trial_space_cluster
 = source_cluster->get_space_cluster( );
lo n_trial_space_dofs = trial_space_cluster.get_n_dofs<
 besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >(
 );
lo n_trial_space_elem = trial_space_cluster.get_n_elements( );
std::vector< lo > trial_space_elems =
trial_space_cluster.get_all_elements( );

bool same_time_cluster
 = ( test_time_cluster.get_center( ) == trial_time_cluster.get_center( )
 );
#pragma omp parallel
{
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

 lo gl_test_elem_time, gl_test_elem_space;
 lo gl_trial_elem_time, gl_trial_elem_space;

 for ( lo i_test_time = 0; i_test_time < n_test_time_elem; ++i_test_time
 ) {
   gl_test_elem_time = test_time_cluster.get_element( i_test_time );
   test_mesh->get_temporal_nodes( gl_test_elem_time, &t0, &t1 );
   lo i_trial_max = n_trial_time_elem - 1;
   if ( same_time_cluster ) {
     i_trial_max = i_test_time;
   }
   for ( lo i_trial_time = 0; i_trial_time <= i_trial_max;
   ++i_trial_time ) {
     gl_trial_elem_time = trial_time_cluster.get_element( i_trial_time
     ); trial_mesh->get_temporal_nodes( gl_trial_elem_time, &tau0, &tau1
     );

     bool shared_t_element = t0 == tau0;
     bool shared_t_vertex = t0 == tau1;
#pragma omp for schedule( dynamic, 1 )
     for ( lo i_test_space = 0; i_test_space < n_test_space_elem;
           ++i_test_space ) {
       gl_test_elem_space = test_space_elems[ i_test_space ];

       test_mesh->get_spatial_nodes( gl_test_elem_space, x1, x2, x3 );
       test_mesh->get_spatial_normal( gl_test_elem_space, nx );
       test_area = test_mesh->spatial_area( gl_test_elem_space );

       for ( lo i_trial_space = 0; i_trial_space < n_trial_space_elem;
             ++i_trial_space ) {
         gl_trial_elem_space = trial_space_elems[ i_trial_space ];

         if ( shared_t_element || shared_t_vertex ) {
           get_type( gl_test_elem_space, gl_trial_elem_space,
             n_shared_vertices, rot_test, rot_trial );
         } else {
           n_shared_vertices = 0;
           rot_test = 0;
           rot_trial = 0;
         }
         trial_mesh->get_spatial_nodes( gl_trial_elem_space, y1, y2, y3
         ); trial_mesh->get_spatial_normal( gl_trial_elem_space, ny );
         areas = test_area * trial_mesh->spatial_area(
         gl_trial_elem_space );

         test_space_cluster
           .local_elem_to_local_dofs<
           besthea::bem::fast_spacetime_be_space<
             besthea::bem::basis_tri_p1 > >( i_test_space,
             n_shared_vertices, rot_test, false, test_loc_access );
         trial_space_cluster
           .local_elem_to_local_dofs<
           besthea::bem::fast_spacetime_be_space<
             besthea::bem::basis_tri_p1 > >( i_trial_space,
             n_shared_vertices, rot_trial, true, trial_loc_access );

         triangles_to_geometry( x1, x2, x3, y1, y2, y3,
         n_shared_vertices,
           rot_test, rot_trial, my_quadrature );
         x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
         x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
         y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
         y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
         w = my_quadrature._w[ n_shared_vertices ].data( );

         size = my_quadrature._w[ n_shared_vertices ].size( );

         test_basis.evaluate_curl( gl_test_elem_space, nx,
         n_shared_vertices,
           rot_test, false, test_curls );
         trial_basis.evaluate_curl( gl_trial_elem_space, ny,
           n_shared_vertices, rot_trial, true, trial_curls );

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
#pragma omp simd \
       aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref :
         DATA_ALIGN ) \
       aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w :
         DATA_ALIGN ) \
       private( kernel1, kernel2, phi1x, phi1y ) \
       reduction( + : value11, value12, value13 ) \
       reduction( + : value21, value22, value23 ) \
       reduction( + : value31, value32, value33 ) \
       simdlen( DATA_WIDTH )
           for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
             _kernel->definite_integral_over_same_interval(
               x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
               x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
               x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data,
               ny_data, t0, t1, &kernel1, &kernel2 );

             phi1x = (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ];
             phi1y = (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ];
             // phi2* = *1_ref[ i_quad ] and phi3* = *2_ref[ i_quad ];

             value11 += ( kernel1 * curl_dot[ 0 ] + kernel2 * phi1x *
             phi1y )
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
                          + kernel2 * x1_ref[ i_quad ] * y1_ref[ i_quad
                          ] )
               * w[ i_quad ];
             value32 += ( kernel1 * curl_dot[ 5 ]
                          + kernel2 * x2_ref[ i_quad ] * y1_ref[ i_quad
                          ] )
               * w[ i_quad ];
             value13 += ( kernel1 * curl_dot[ 6 ]
                          + kernel2 * phi1x * y2_ref[ i_quad ] )
               * w[ i_quad ];
             value23 += ( kernel1 * curl_dot[ 7 ]
                          + kernel2 * x1_ref[ i_quad ] * y2_ref[ i_quad
                          ] )
               * w[ i_quad ];
             value33 += ( kernel1 * curl_dot[ 8 ]
                          + kernel2 * x2_ref[ i_quad ] * y2_ref[ i_quad
                          ] )
               * w[ i_quad ];
           }
         } else {
#pragma omp simd \
       aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref :
         DATA_ALIGN ) \
       aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w :
         DATA_ALIGN ) \
       private( kernel1, kernel2, phi1x, phi1y ) \
       reduction( + : value11, value12, value13 ) \
       reduction( + : value21, value22, value23 ) \
       reduction( + : value31, value32, value33 ) \
       simdlen( DATA_WIDTH )
           for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
             _kernel->definite_integral_over_different_intervals(
               x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
               x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
               x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data,
               ny_data, t0, t1, tau0, tau1, &kernel1, &kernel2 );

             phi1x = (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ];
             phi1y = (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ];
             // phi2* = *1_ref[ i_quad ] and phi3* = *2_ref[ i_quad ];

             value11 += ( kernel1 * curl_dot[ 0 ] + kernel2 * phi1x *
             phi1y )
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
                          + kernel2 * x1_ref[ i_quad ] * y1_ref[ i_quad
                          ] )
               * w[ i_quad ];
             value32 += ( kernel1 * curl_dot[ 5 ]
                          + kernel2 * x2_ref[ i_quad ] * y1_ref[ i_quad
                          ] )
               * w[ i_quad ];
             value13 += ( kernel1 * curl_dot[ 6 ]
                          + kernel2 * phi1x * y2_ref[ i_quad ] )
               * w[ i_quad ];
             value23 += ( kernel1 * curl_dot[ 7 ]
                          + kernel2 * x1_ref[ i_quad ] * y2_ref[ i_quad
                          ] )
               * w[ i_quad ];
             value33 += ( kernel1 * curl_dot[ 8 ]
                          + kernel2 * x2_ref[ i_quad ] * y2_ref[ i_quad
                          ] )
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
*/
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

  // assuming that the spatial meshes of the nearfield mesh and the local mesh
  // coincide, we can simply consider the local meshes.
  _test_space->get_mesh( ).get_local_mesh( )->get_spatial_element(
    i_test_space, test_elem );
  _trial_space->get_mesh( ).get_local_mesh( )->get_spatial_element(
    i_trial_space, trial_elem );

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
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
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
  simdlen( DATA_WIDTH )
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
  simdlen( DATA_WIDTH )
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
  lou contribution_size = ( _temp_order + 1 )
    * ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
  tree_structure * trial_distribution_tree
    = _trial_space->get_tree( )->get_distribution_tree( );
  trial_distribution_tree->initialize_moment_contributions(
    *trial_distribution_tree->get_root( ), contribution_size );
  tree_structure * test_distribution_tree
    = _test_space->get_tree( )->get_distribution_tree( );
  test_distribution_tree->initialize_local_contributions(
    *test_distribution_tree->get_root( ), contribution_size );
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
// template class besthea::bem::distributed_fast_spacetime_be_assembler<
//   besthea::bem::spacetime_heat_hs_kernel_antiderivative,
//   besthea::bem::distributed_fast_spacetime_be_space<
//     besthea::bem::basis_tri_p1 >,
//   besthea::bem::distributed_fast_spacetime_be_space<
//     besthea::bem::basis_tri_p1 > >;
