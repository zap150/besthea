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

#include "besthea/fast_spacetime_be_assembler.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/timer.h"

#include <set>
#include <vector>

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::fast_spacetime_be_assembler( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular, int temp_order, int spat_order,
  sc alpha, sc cutoff_param, bool uniform )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ),
    _cutoff_param( cutoff_param ),
    _uniform( uniform ),
    _temp_order( temp_order ),
    _spat_order( spat_order ),
    _m2l_integration_order( _spat_order ),
    _alpha( alpha ),
    _lagrange( temp_order ),
    _chebyshev( spat_order ) {
  lo levels = _test_space->get_tree( )->get_space_tree( )->get_levels( );
  const std::vector< sc > & bb_size
    = _test_space->get_tree( )->get_space_tree( )->get_bounding_box( );
  sc size_x = bb_size[ 0 ] / std::pow( 2.0, levels - 1 );
  sc size_y = bb_size[ 1 ] / std::pow( 2.0, levels - 1 );
  sc size_z = bb_size[ 2 ] / std::pow( 2.0, levels - 1 );

  _space_cluster_size
    = std::sqrt( size_x * size_x + size_y * size_y + size_z * size_z );
  //   precompute_nonzeros( );
}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::~fast_spacetime_be_assembler( ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
bool besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::is_spatial_nearfield( [[maybe_unused]] lo test_idx,
  [[maybe_unused]] lo trial_idx ) const {
  //   auto test_mesh = _test_space->get_spatial_mesh( );
  //   auto trial_mesh = _trial_space->get_spatial_mesh( );
  //
  //   linear_algebra::coordinates< 3 > test_c;
  //   linear_algebra::coordinates< 3 > trial_c;
  //
  //   test_mesh->get_centroid( test_idx, test_c );
  //   trial_mesh->get_centroid( trial_idx, trial_c );
  //   sc dist = ( test_c[ 0 ] - trial_c[ 0 ] ) * ( test_c[ 0 ] - trial_c[ 0 ] )
  //     + ( test_c[ 1 ] - trial_c[ 1 ] ) * ( test_c[ 1 ] - trial_c[ 1 ] )
  //     + ( test_c[ 2 ] - trial_c[ 2 ] ) * ( test_c[ 2 ] - trial_c[ 2 ] );
  // TODO: criterion turned off for debugging reasons
  //   return dist <= _cutoff_param * _space_cluster_size * _cutoff_param
  //     * _space_cluster_size;
  return true;
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::precompute_nonzeros( ) {
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_spatial_elements( );

  _nonzeros.reserve( n_test_elements * 10 );
  for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
    for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
      if ( is_spatial_nearfield( i_test, i_trial ) ) {
        _nonzeros.push_back( std::make_pair( i_test, i_trial ) );
      }
    }
  }
  _nonzeros.shrink_to_fit( );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble( pfmm_matrix_type & global_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );

  global_matrix.set_tree( _test_space->get_tree( ) );
  global_matrix.set_order( _spat_order, _temp_order );
  global_matrix.set_alpha( _alpha );
  global_matrix.set_m2l_integration_order( _m2l_integration_order );
  global_matrix.set_uniform(
    false );  // for now, ignore the possible uniformity

  // number of timesteps should be the same for test and
  // trial meshes
  lo n_timesteps = test_mesh->get_n_temporal_elements( );

  // size of individual blocks
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );

  global_matrix.resize( n_timesteps, n_rows, n_columns );

  std::vector< spacetime_cluster_type * > spacetime_leaves
    = _test_space->get_tree( )->get_leaves( );
  // find all space cluster and time clusters which are part of a spacetime
  // leaf cluster and store them ( each only once ) in sets
  std::set< time_cluster_type * > time_clusters_spacetime_leaves;
  std::set< space_cluster_type * > space_clusters_spacetime_leaves;
  for ( auto it = spacetime_leaves.begin( ); it != spacetime_leaves.end( );
        ++it ) {
    time_clusters_spacetime_leaves.insert( &( ( *it )->get_time_cluster( ) ) );
    space_clusters_spacetime_leaves.insert(
      &( ( *it )->get_space_cluster( ) ) );
  }

  compute_required_data(
    time_clusters_spacetime_leaves, space_clusters_spacetime_leaves );

  global_matrix.compute_spatial_m2m_coeffs( );
  global_matrix.compute_temporal_m2m_matrices( );

  initialize_moment_and_local_contributions( );

#ifdef NEARFIELD_CLUSTERWISE
  assemble_clusterwise_nearfield( global_matrix );
#else
  if ( !_uniform ) {
    assemble_nearfield( global_matrix );
    // assemble_farfield_nonapproximated( global_matrix );
  } else {
    assemble_nonapproximated_uniform( global_matrix );
  }
#endif
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble_clusterwise_nearfield( pfmm_matrix_type &
    global_matrix ) const {
  std::vector< spacetime_cluster_type * > & leaves
    = _test_space->get_tree( )->get_leaves( );

  spacetime_cluster_type * current_cluster;
  spacetime_cluster_type * nearfield_cluster;

  full_matrix_type * block;

  for ( std::vector< spacetime_cluster_type * >::size_type leaf_index = 0;
        leaf_index < leaves.size( ); ++leaf_index ) {
    current_cluster = leaves[ leaf_index ];
    std::vector< spacetime_cluster_type * > & nearfield_list
      = *( current_cluster->get_nearfield_list( ) );
    for ( std::vector< spacetime_cluster_type * >::size_type source_index = 0;
          source_index < nearfield_list.size( ); ++source_index ) {
      nearfield_cluster = nearfield_list[ source_index ];

      block = global_matrix.create_clusterwise_nearfield_matrix(
        leaf_index, source_index );

      assemble_clusterwise_nearfield_matrix(
        current_cluster, nearfield_cluster, *block );
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble_nearfield( pfmm_matrix_type & global_matrix )
  const {
  std::vector< mesh::time_cluster * > & leaves
    = _test_space->get_tree( )->get_time_tree( )->get_leaves( );

  mesh::time_cluster * current_cluster;
  mesh::time_cluster * neighbor_cluster;

  lo global_elem_i, global_elem_j;
  sc t0, t1, tau0, tau1;
  full_matrix_type * block;

  for ( auto it = leaves.begin( ); it != leaves.end( ); ++it ) {
    current_cluster = *it;
    neighbor_cluster = current_cluster->get_left_neighbour( );

    // go over every element in the current time cluster
    for ( lo i = 0; i < current_cluster->get_n_elements( ); ++i ) {
      global_elem_i = current_cluster->get_element( i );
      // std::cout << "TEST: " << global_elem_i << std::endl << "   ";

      _test_space->get_tree( )->get_time_tree( )->get_mesh( ).get_nodes(
        global_elem_i, &t0, &t1 );

      // first, assemble diagonal block
      block
        = global_matrix.create_nearfield_matrix( global_elem_i, global_elem_i );
      assemble_nearfield_matrix( t0, t1, t0, t1, 0, *block );
      //      if ( i == 0 ) {
      //        block->print( );
      //        std::cout << "end" << std::endl;
      //        for ( lo q = 0; q < 10; ++q ) std::cout << std::endl;
      //      }

      // next, compute interaction of the cluster with itself
      // (this requires the temporal elements within the cluster to be
      // sorted)
      for ( lo j = 0; j < i; ++j ) {
        global_elem_j = current_cluster->get_element( j );
        // std::cout << global_elem_j << " ";
        _trial_space->get_tree( )->get_time_tree( )->get_mesh( ).get_nodes(
          global_elem_j, &tau0, &tau1 );
        block = global_matrix.create_nearfield_matrix(
          global_elem_i, global_elem_j );
        // in case that the index global_elem_j is less than global_elem_i - 1
        // the time intervals are separated, otherwise they share a vertex.
        char time_configuration
          = ( global_elem_j < global_elem_i - 1 ? (char) 2 : (char) 1 );
        assemble_nearfield_matrix(
          t0, t1, tau0, tau1, time_configuration, *block );
      }
      // std::cout << std::endl << "   ";
      // next interact with the previous cluster
      if ( neighbor_cluster != nullptr ) {
        for ( lo j = 0; j < neighbor_cluster->get_n_elements( ); ++j ) {
          global_elem_j = neighbor_cluster->get_element( j );
          // std::cout << global_elem_j << " ";
          _trial_space->get_tree( )->get_time_tree( )->get_mesh( ).get_nodes(
            global_elem_j, &tau0, &tau1 );
          block = global_matrix.create_nearfield_matrix(
            global_elem_i, global_elem_j );
          // in case that the index global_elem_j is less than global_elem_i - 1
          // the time intervals are separated, otherwise they share a vertex.
          char time_configuration
            = ( global_elem_j < global_elem_i - 1 ? (char) 2 : (char) 1 );
          assemble_nearfield_matrix(
            t0, t1, tau0, tau1, time_configuration, *block );
        }
      }
      // std::cout << std::endl;
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble_farfield_nonapproximated( pfmm_matrix_type &
    global_matrix ) const {
  std::vector< mesh::time_cluster * > & leaves
    = _test_space->get_tree( )->get_time_tree( )->get_leaves( );

  mesh::time_cluster * current_cluster;
  mesh::time_cluster * farfield_cluster = nullptr;

  lo global_elem_i, global_elem_j;
  sc t0, t1, tau0, tau1;
  full_matrix_type * block;

  for ( auto it = leaves.begin( ); it != leaves.end( ); ++it ) {
    current_cluster = *it;

    // go over every element in the current time cluster
    for ( lo i = 0; i < current_cluster->get_n_elements( ); ++i ) {
      if ( current_cluster->get_left_neighbour( ) != nullptr ) {
        farfield_cluster
          = current_cluster->get_left_neighbour( )->get_left_neighbour( );
      }
      global_elem_i = current_cluster->get_element( i );
      // std::cout << "TEST: " << global_elem_i << std::endl << "   ";

      _test_space->get_tree( )->get_time_tree( )->get_mesh( ).get_nodes(
        global_elem_i, &t0, &t1 );

      // next interact with the previous cluster
      while ( farfield_cluster != nullptr ) {
        for ( lo j = 0; j < farfield_cluster->get_n_elements( ); ++j ) {
          global_elem_j = farfield_cluster->get_element( j );
          // std::cout << global_elem_j << " ";
          _trial_space->get_tree( )->get_time_tree( )->get_mesh( ).get_nodes(
            global_elem_j, &tau0, &tau1 );
          block = global_matrix.create_farfield_matrix(
            global_elem_i, global_elem_j );
          // time configuration is always 2 in the farfield (i.e. time intervals
          // are disjoint)
          char time_configuration = 2;
          assemble_nearfield_matrix(
            t0, t1, tau0, tau1, time_configuration, *block );
        }
        // std::cout << std::endl;
        farfield_cluster = farfield_cluster->get_left_neighbour( );
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble_nonapproximated_uniform( pfmm_matrix_type &
    global_matrix ) const {
  std::vector< mesh::time_cluster * > & leaves
    = _test_space->get_tree( )->get_time_tree( )->get_leaves( );

  mesh::time_cluster * current_cluster;

  lo global_elem_i, global_elem_j;
  sc t0, t1, tau0, tau1;
  full_matrix_type * block;

  for ( auto it = leaves.begin( ); it != leaves.end( ); ++it ) {
    current_cluster = *it;

    char time_configuration = 0;  // start with same time element.

    // go over every element in the current time cluster
    for ( lo i = 0; i < current_cluster->get_n_elements( ); ++i ) {
      global_elem_i = current_cluster->get_element( i );
      // std::cout << global_elem_i << std::endl;

      _test_space->get_tree( )->get_time_tree( )->get_mesh( ).get_nodes(
        global_elem_i, &t0, &t1 );

      // interact with the first cluster
      global_elem_j = 0;
      _trial_space->get_tree( )->get_time_tree( )->get_mesh( ).get_nodes(
        global_elem_j, &tau0, &tau1 );

      block
        = global_matrix.create_nearfield_matrix( global_elem_i, global_elem_j,
          leaves.size( ) * current_cluster->get_n_elements( ) - global_elem_i );

      assemble_nearfield_matrix(
        t0, t1, tau0, tau1, time_configuration, *block );

      if ( i < 2 ) {
        // update time configuration for next step in loop.
        ++time_configuration;
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::
  assemble_clusterwise_nearfield_matrix(
    spacetime_cluster_type * target_cluster,
    spacetime_cluster_type * source_cluster,
    full_matrix_type & nearfield_matrix ) const {
  nearfield_matrix.fill( 0.0 );

  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  time_cluster_type & test_time_cluster = target_cluster->get_time_cluster( );
  lo n_test_time_elem = test_time_cluster.get_n_elements( );
  space_cluster_type & test_space_cluster
    = target_cluster->get_space_cluster( );
  lo n_test_space_dofs = test_space_cluster.get_n_dofs< test_space_type >( );
  lo n_test_space_elem = test_space_cluster.get_n_elements( );
  std::vector< lo > test_space_elems = test_space_cluster.get_all_elements( );

  time_cluster_type & trial_time_cluster = source_cluster->get_time_cluster( );
  lo n_trial_time_elem = trial_time_cluster.get_n_elements( );
  space_cluster_type & trial_space_cluster
    = source_cluster->get_space_cluster( );
  lo n_trial_space_dofs = trial_space_cluster.get_n_dofs< trial_space_type >( );
  lo n_trial_space_elem = trial_space_cluster.get_n_elements( );
  std::vector< lo > trial_space_elems = trial_space_cluster.get_all_elements( );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  // bool same_time_cluster
  //   = ( test_time_cluster.get_center( ) == trial_time_cluster.get_center( )
  //   );
  bool same_time_cluster = ( &test_time_cluster == &trial_time_cluster );
#pragma omp parallel
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

    lo gl_test_elem_time, gl_test_elem_space;
    lo gl_trial_elem_time, gl_trial_elem_space;

    for ( lo i_test_time = 0; i_test_time < n_test_time_elem; ++i_test_time ) {
      gl_test_elem_time = test_time_cluster.get_element( i_test_time );
      test_mesh->get_temporal_nodes( gl_test_elem_time, &t0, &t1 );
      lo i_trial_max = n_trial_time_elem - 1;
      if ( same_time_cluster ) {
        i_trial_max = i_test_time;
      }
      for ( lo i_trial_time = 0; i_trial_time <= i_trial_max; ++i_trial_time ) {
        gl_trial_elem_time = trial_time_cluster.get_element( i_trial_time );
        trial_mesh->get_temporal_nodes( gl_trial_elem_time, &tau0, &tau1 );

        bool shared_t_element = ( gl_trial_elem_time == gl_test_elem_time );
        bool shared_t_vertex = ( gl_trial_elem_time == gl_test_elem_time - 1 );
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

            trial_mesh->get_spatial_nodes( gl_trial_elem_space, y1, y2, y3 );
            trial_mesh->get_spatial_normal( gl_trial_elem_space, ny );
            trial_area = trial_mesh->spatial_area( gl_trial_elem_space );

            test_space_cluster.local_elem_to_local_dofs< test_space_type >(
              i_test_space, n_shared_vertices, rot_test, false,
              test_loc_access );
            trial_space_cluster.local_elem_to_local_dofs< trial_space_type >(
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
                  test = test_basis.evaluate( gl_test_elem_space, i_loc_test,
                    x1_ref[ i_quad ], x2_ref[ i_quad ], nx_data,
                    n_shared_vertices, rot_test, false );
                  trial = trial_basis.evaluate( gl_trial_elem_space,
                    i_loc_trial, y1_ref[ i_quad ], y2_ref[ i_quad ], ny_data,
                    n_shared_vertices, rot_trial, true );
                  value += kernel_data[ i_quad ] * test * trial;
                }

                value *= test_area * trial_area;
                nearfield_matrix.add_atomic( i_test_time * n_test_space_dofs
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
void besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  assemble_clusterwise_nearfield_matrix(
    spacetime_cluster_type * target_cluster,
    spacetime_cluster_type * source_cluster,
    full_matrix_type & nearfield_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  time_cluster_type & test_time_cluster = target_cluster->get_time_cluster( );
  lo n_test_time_elem = test_time_cluster.get_n_elements( );
  space_cluster_type & test_space_cluster
    = target_cluster->get_space_cluster( );
  lo n_test_space_dofs = test_space_cluster.get_n_dofs<
    besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( );
  lo n_test_space_elem = test_space_cluster.get_n_elements( );
  std::vector< lo > test_space_elems = test_space_cluster.get_all_elements( );

  time_cluster_type & trial_time_cluster = source_cluster->get_time_cluster( );
  lo n_trial_time_elem = trial_time_cluster.get_n_elements( );
  space_cluster_type & trial_space_cluster
    = source_cluster->get_space_cluster( );
  lo n_trial_space_dofs = trial_space_cluster.get_n_dofs<
    besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( );
  lo n_trial_space_elem = trial_space_cluster.get_n_elements( );
  std::vector< lo > trial_space_elems = trial_space_cluster.get_all_elements( );

  // bool same_time_cluster
  //   = ( test_time_cluster.get_center( ) == trial_time_cluster.get_center( )
  //   );
  bool same_time_cluster = ( &test_time_cluster == &trial_time_cluster );
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

    for ( lo i_test_time = 0; i_test_time < n_test_time_elem; ++i_test_time ) {
      gl_test_elem_time = test_time_cluster.get_element( i_test_time );
      test_mesh->get_temporal_nodes( gl_test_elem_time, &t0, &t1 );
      lo i_trial_max = n_trial_time_elem - 1;
      if ( same_time_cluster ) {
        i_trial_max = i_test_time;
      }
      for ( lo i_trial_time = 0; i_trial_time <= i_trial_max; ++i_trial_time ) {
        gl_trial_elem_time = trial_time_cluster.get_element( i_trial_time );
        trial_mesh->get_temporal_nodes( gl_trial_elem_time, &tau0, &tau1 );

        bool shared_t_element = ( gl_trial_elem_time == gl_test_elem_time );
        bool shared_t_vertex = ( gl_trial_elem_time == gl_test_elem_time - 1 );
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
            trial_mesh->get_spatial_nodes( gl_trial_elem_space, y1, y2, y3 );
            trial_mesh->get_spatial_normal( gl_trial_elem_space, ny );
            areas = test_area * trial_mesh->spatial_area( gl_trial_elem_space );

            test_space_cluster
              .local_elem_to_local_dofs< besthea::bem::fast_spacetime_be_space<
                besthea::bem::basis_tri_p1 > >( i_test_space, n_shared_vertices,
                rot_test, false, test_loc_access );
            trial_space_cluster
              .local_elem_to_local_dofs< besthea::bem::fast_spacetime_be_space<
                besthea::bem::basis_tri_p1 > >( i_trial_space,
                n_shared_vertices, rot_trial, true, trial_loc_access );

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
            aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref : DATA_ALIGN ) \
            aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w : DATA_ALIGN ) \
            private( kernel1, kernel2, phi1x, phi1y ) \
            reduction( + : value11, value12, value13 ) \
            reduction( + : value21, value22, value23 ) \
            reduction( + : value31, value32, value33 ) \
            simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                _kernel->definite_integral_over_same_interval(
                  x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                  x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                  x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, ny_data,
                  t0, t1, &kernel1, &kernel2 );

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
#pragma omp simd \
            aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref : DATA_ALIGN ) \
            aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w : DATA_ALIGN ) \
            private( kernel1, kernel2, phi1x, phi1y ) \
            reduction( + : value11, value12, value13 ) \
            reduction( + : value21, value22, value23 ) \
            reduction( + : value31, value32, value33 ) \
            simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                _kernel->definite_integral_over_different_intervals(
                  x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                  x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                  x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, ny_data,
                  t0, t1, tau0, tau1, &kernel1, &kernel2 );

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
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble_nearfield_matrix( sc t0, sc t1, sc tau0, sc tau1,
  char temporal_configuration, full_matrix_type & nearfield_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_spatial_elements( );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

#pragma omp parallel
  {
    std::vector< lo > test_l2g( n_loc_rows );
    std::vector< lo > trial_l2g( n_loc_columns );

    sc test, trial, value, test_area, trial_area;
    lo size;
    int n_shared_vertices = 0;
    int rot_test = 0;
    int rot_trial = 0;

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

#pragma omp for schedule( dynamic, 1 )
    for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
      for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
        test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
        test_mesh->get_spatial_normal( i_test, nx );
        test_area = test_mesh->spatial_area( i_test );
        // special quadrature in case of same element ( configuration 0 ) or
        // in case of shared vertex ( configuration 1 ).
        if ( temporal_configuration == 0 || temporal_configuration == 1 ) {
          get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );
        } else {
          n_shared_vertices = 0;
          rot_test = 0;
          rot_trial = 0;
        }
        trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
        trial_mesh->get_spatial_normal( i_trial, ny );
        trial_area = trial_mesh->spatial_area( i_trial );

        test_basis.local_to_global(
          i_test, n_shared_vertices, rot_test, false, test_l2g );
        trial_basis.local_to_global(
          i_trial, n_shared_vertices, rot_trial, true, trial_l2g );

        triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices,
          rot_test, rot_trial, my_quadrature );
        x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
        x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
        y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
        y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
        w = my_quadrature._w[ n_shared_vertices ].data( );

        size = my_quadrature._w[ n_shared_vertices ].size( );

        if ( temporal_configuration == 0 ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
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
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
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
  private( test, trial ) reduction( + : value ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              test = test_basis.evaluate( i_test, i_loc_test, x1_ref[ i_quad ],
                x2_ref[ i_quad ], nx_data, n_shared_vertices, rot_test, false );
              trial = trial_basis.evaluate( i_trial, i_loc_trial,
                y1_ref[ i_quad ], y2_ref[ i_quad ], ny_data, n_shared_vertices,
                rot_trial, true );

              value += kernel_data[ i_quad ] * test * trial;
            }
            value *= test_area * trial_area;

            nearfield_matrix.add_atomic(
              test_l2g[ i_loc_test ], trial_l2g[ i_loc_trial ], value );
          }
        }
      }
    }
  }
}

//! template specialization for hypersingular p1p1 matrix
template<>
void besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  assemble_nearfield_matrix( sc t0, sc t1, sc tau0, sc tau1,
    char temporal_configuration, full_matrix_type & nearfield_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  // size of individual blocks
  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_spatial_elements( );

#pragma omp parallel
  {
    std::vector< lo > test_l2g( 3 );
    std::vector< lo > trial_l2g( 3 );

    sc areas, test_area;
    sc value11, value12, value13;
    sc value21, value22, value23;
    sc value31, value32, value33;
    lo size;
    int n_shared_vertices = 0;
    int rot_test = 0;
    int rot_trial = 0;

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

#pragma omp for schedule( dynamic, 1 )
    for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
      for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
        test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
        test_mesh->get_spatial_normal( i_test, nx );
        test_area = test_mesh->spatial_area( i_test );

        // special quadrature in case of same element ( configuration 0 ) or
        // in case of shared vertex ( configuration 1 ).
        if ( temporal_configuration == 0 || temporal_configuration == 1 ) {
          get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );
        } else {
          n_shared_vertices = 0;
          rot_test = 0;
          rot_trial = 0;
        }
        trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
        trial_mesh->get_spatial_normal( i_trial, ny );
        areas = test_area * trial_mesh->spatial_area( i_trial );

        test_basis.local_to_global(
          i_test, n_shared_vertices, rot_test, false, test_l2g );
        trial_basis.local_to_global(
          i_trial, n_shared_vertices, rot_trial, true, trial_l2g );

        triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices,
          rot_test, rot_trial, my_quadrature );
        x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
        x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
        y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
        y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
        w = my_quadrature._w[ n_shared_vertices ].data( );

        size = my_quadrature._w[ n_shared_vertices ].size( );

        test_basis.evaluate_curl(
          i_test, nx, n_shared_vertices, rot_test, false, test_curls );
        trial_basis.evaluate_curl(
          i_trial, ny, n_shared_vertices, rot_trial, true, trial_curls );

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

        if ( temporal_configuration == 0 ) {
#pragma omp simd \
        aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref : DATA_ALIGN ) \
        aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w : DATA_ALIGN ) \
        private( kernel1, kernel2, phi1x, phi1y ) \
        reduction( + : value11, value12, value13 ) \
        reduction( + : value21, value22, value23 ) \
        reduction( + : value31, value32, value33 ) \
        simdlen( DATA_WIDTH )
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
#pragma omp simd \
        aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref : DATA_ALIGN ) \
        aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w : DATA_ALIGN ) \
        private( kernel1, kernel2, phi1x, phi1y ) \
        reduction( + : value11, value12, value13 ) \
        reduction( + : value21, value22, value23 ) \
        reduction( + : value31, value32, value33 ) \
        simdlen( DATA_WIDTH )
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
          test_l2g[ 0 ], trial_l2g[ 0 ], value11 * areas );
        nearfield_matrix.add_atomic(
          test_l2g[ 0 ], trial_l2g[ 1 ], value12 * areas );
        nearfield_matrix.add_atomic(
          test_l2g[ 0 ], trial_l2g[ 2 ], value13 * areas );
        nearfield_matrix.add_atomic(
          test_l2g[ 1 ], trial_l2g[ 0 ], value21 * areas );
        nearfield_matrix.add_atomic(
          test_l2g[ 1 ], trial_l2g[ 1 ], value22 * areas );
        nearfield_matrix.add_atomic(
          test_l2g[ 1 ], trial_l2g[ 2 ], value23 * areas );
        nearfield_matrix.add_atomic(
          test_l2g[ 2 ], trial_l2g[ 0 ], value31 * areas );
        nearfield_matrix.add_atomic(
          test_l2g[ 2 ], trial_l2g[ 1 ], value32 * areas );
        nearfield_matrix.add_atomic(
          test_l2g[ 2 ], trial_l2g[ 2 ], value33 * areas );
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::get_type( lo i_test, lo i_trial, int & n_shared_vertices,
  int & rot_test, int & rot_trial ) const {
  // check for identical
  if ( i_test == i_trial ) {
    n_shared_vertices = 3;
    rot_test = 0;
    rot_trial = 0;
    return;
  }

  linear_algebra::indices< 3 > test_elem;
  linear_algebra::indices< 3 > trial_elem;

  _test_space->get_mesh( )->get_spatial_element( i_test, test_elem );
  _trial_space->get_mesh( )->get_spatial_element( i_trial, trial_elem );

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
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::init_quadrature( quadrature_wrapper & my_quadrature )
  const {
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
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
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
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::hypercube_to_triangles_vertex( sc ksi, sc eta1, sc eta2,
  sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
  sc & jacobian ) const {
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
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::
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
  // y%_mapped are the %th components of the vectors to which y#_ref is mapped
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
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::hypercube_to_triangles_edge( sc ksi, sc eta1, sc eta2,
  sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
  sc & jacobian ) const {
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
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::
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
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::hypercube_to_triangles_identical( sc ksi, sc eta1,
  sc eta2, sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref,
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
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::compute_lagrange_quadrature( time_cluster_type *
    time_cluster ) const {
  lo n_temp_elems = time_cluster->get_n_elements( );
  full_matrix_type & L = time_cluster->get_lagrange_quad( );
  L.resize( _temp_order + 1, n_temp_elems );
  L.fill( 0.0 );

  const std::vector< sc, besthea::allocator_type< sc > > & line_t
    = quadrature::line_x( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & line_w
    = quadrature::line_w( _order_regular );

  vector_type eval_points( line_t.size( ) );
  vector_type evaluation( line_t.size( ) );
  sc padding = _test_space->get_tree( )
                 ->get_time_tree( )
                 ->get_paddings( )[ time_cluster->get_level( ) ];
  sc cluster_t_start
    = time_cluster->get_center( ) - time_cluster->get_half_size( ) - padding;
  sc cluster_t_end
    = time_cluster->get_center( ) + time_cluster->get_half_size( ) + padding;

  linear_algebra::coordinates< 1 > elem_t_start;
  linear_algebra::coordinates< 1 > elem_t_end;

  sc cluster_size = cluster_t_end - cluster_t_start;
  sc elem_size;

  for ( lo i = 0; i < n_temp_elems; ++i ) {
    lo elem = time_cluster->get_element( i );
    time_cluster->get_mesh( ).get_nodes( elem, elem_t_start, elem_t_end );
    elem_size = elem_t_end[ 0 ] - elem_t_start[ 0 ];
    // compute the quadrature points in the current element in relative
    // coordinates with respect to the time cluster and transform them to [-1,1]
    for ( std::vector< sc, besthea::allocator_type< sc > >::size_type j = 0;
          j < line_t.size( ); ++j ) {
      eval_points[ j ] = -1.0
        + 2.0
          * ( elem_t_start[ 0 ] + elem_size * line_t[ j ] - cluster_t_start )
          / cluster_size;
    }

    for ( lo j = 0; j <= _temp_order; ++j ) {
      _lagrange.evaluate( j, eval_points, evaluation );
      for ( lo k = 0; k < eval_points.size( ); ++k ) {
        L.add( j, i, evaluation[ k ] * line_w[ k ] );
      }
      L.set( j, i, L.get( j, i ) * elem_size );
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::compute_lagrange_drv_integrals( time_cluster_type *
    time_cluster ) const {
  lo n_temp_elems = time_cluster->get_n_elements( );
  full_matrix_type & L_drv = time_cluster->get_lagrange_drv_int( );
  L_drv.resize( _temp_order + 1, n_temp_elems );

  // Lagrange polynomials are evaluated at the borders of the elements
  vector_type eval_points( 2 );
  vector_type evaluation( 2 );
  sc padding = _test_space->get_tree( )
                 ->get_time_tree( )
                 ->get_paddings( )[ time_cluster->get_level( ) ];
  sc cluster_t_start
    = time_cluster->get_center( ) - time_cluster->get_half_size( ) - padding;
  sc cluster_t_end
    = time_cluster->get_center( ) + time_cluster->get_half_size( ) + padding;

  linear_algebra::coordinates< 1 > elem_t_start;
  linear_algebra::coordinates< 1 > elem_t_end;

  sc cluster_size = cluster_t_end - cluster_t_start;

  for ( lo i = 0; i < n_temp_elems; ++i ) {
    lo elem = time_cluster->get_element( i );
    time_cluster->get_mesh( ).get_nodes( elem, elem_t_start, elem_t_end );
    // compute the quadrature points in the current element in relative
    // coordinates with respect to the time cluster and transform them to [-1,1]
    eval_points[ 0 ]
      = -1.0 + 2.0 * ( elem_t_start[ 0 ] - cluster_t_start ) / cluster_size;
    eval_points[ 1 ]
      = -1.0 + 2.0 * ( elem_t_end[ 0 ] - cluster_t_start ) / cluster_size;
    for ( lo j = 0; j <= _temp_order; ++j ) {
      _lagrange.evaluate( j, eval_points, evaluation );
      L_drv.set( j, i, evaluation[ 1 ] - evaluation[ 0 ] );
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
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

//! template specialization for hypersingular p1p1 matrix
template<>
void besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  compute_surface_curls( space_cluster_type * cluster ) const {
  std::vector< sc > & surf_curls_dim0 = cluster->get_surf_curls( 0 );
  std::vector< sc > & surf_curls_dim1 = cluster->get_surf_curls( 1 );
  std::vector< sc > & surf_curls_dim2 = cluster->get_surf_curls( 2 );

  const std::vector< lo > & space_elements = cluster->get_all_elements( );
  // check first whether the surface curls have already been computed
  // ATTENTION: the surface curls are only computed for the trial space (
  // assuming that test and trial spaces coincide)
  if ( ( surf_curls_dim0.size( ) == 0 ) && ( surf_curls_dim1.size( ) == 0 )
    && ( surf_curls_dim2.size( ) == 0 ) ) {
    lo n_space_elems = cluster->get_n_elements( );
    surf_curls_dim0.resize( 3 * n_space_elems );
    surf_curls_dim1.resize( 3 * n_space_elems );
    surf_curls_dim2.resize( 3 * n_space_elems );
    sc local_curls[ 9 ];
    linear_algebra::coordinates< 3 > normal;
    for ( lo j = 0; j < n_space_elems; ++j ) {
      _trial_space->get_mesh( )->get_spatial_normal(
        space_elements[ j ], normal );
      _trial_space->get_basis( ).evaluate_curl(
        space_elements[ j ], normal, 0, 0, 0, local_curls );
      surf_curls_dim0[ 3 * j ] = local_curls[ 0 ];
      surf_curls_dim0[ 3 * j + 1 ] = local_curls[ 3 ];
      surf_curls_dim0[ 3 * j + 2 ] = local_curls[ 6 ];
      surf_curls_dim1[ 3 * j ] = local_curls[ 1 ];
      surf_curls_dim1[ 3 * j + 1 ] = local_curls[ 4 ];
      surf_curls_dim1[ 3 * j + 2 ] = local_curls[ 7 ];
      surf_curls_dim2[ 3 * j ] = local_curls[ 2 ];
      surf_curls_dim2[ 3 * j + 1 ] = local_curls[ 5 ];
      surf_curls_dim2[ 3 * j + 2 ] = local_curls[ 8 ];
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::
  compute_surface_curls( [[maybe_unused]] space_cluster_type * cluster ) const {
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::compute_required_data( std::set< time_cluster_type * > &
  /*time_clusters_spacetime_leaves*/,
  std::set< space_cluster_type * > & /*space_clusters_spacetime_leaves*/ )
  const {
  std::cout << "compute_required_data NOT IMPLEMENTED" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
void besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  compute_required_data(
    std::set< time_cluster_type * > & time_clusters_spacetime_leaves,
    std::set< space_cluster_type * > & space_clusters_spacetime_leaves ) const {
  // Compute only Chebyshev quadratures (not their derivatives) only if they
  // have not been computed yet.
  // WARNING: Only the number of columns of the Chebyshev quadrature of the
  //          first spatial cluster is checked.
  //   if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) &&
  //     ( ( *space_clusters_spacetime_leaves.begin( ) )->
  //               get_chebyshev_quad_p0( ).get_n_columns( ) !=
  //     ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6
  //     ) )
  // ATTENTION: in the current version it is not checked if quadratures etc.
  //            have been computed!!! still, each computation is done in a
  //            separate step in case that the checks are added
  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      compute_chebyshev_quadrature_p0( *it );
    }
  }

  if ( time_clusters_spacetime_leaves.size( ) > 0 ) {
    for ( auto it = time_clusters_spacetime_leaves.begin( );
          it != time_clusters_spacetime_leaves.end( ); ++it ) {
      compute_lagrange_quadrature( *it );
    }
  }
}

//! template specialization for single layer p1p1 matrix
template<>
void besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  compute_required_data(
    std::set< time_cluster_type * > & time_clusters_spacetime_leaves,
    std::set< space_cluster_type * > & space_clusters_spacetime_leaves ) const {
  // ATTENTION: in the current version it is not checked if quadratures etc.
  //            have been computed!!! still, each computation is done in a
  //            separate step in case that the checks are added
  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      ( *it )->compute_node_mapping( );
      compute_chebyshev_quadrature_p1( *it );
    }
  }

  if ( time_clusters_spacetime_leaves.size( ) > 0 ) {
    for ( auto it = time_clusters_spacetime_leaves.begin( );
          it != time_clusters_spacetime_leaves.end( ); ++it ) {
      compute_lagrange_quadrature( *it );
    }
  }
}

//! template specialization for double layer p0p1 matrix
template<>
void besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  compute_required_data(
    std::set< time_cluster_type * > & time_clusters_spacetime_leaves,
    std::set< space_cluster_type * > & space_clusters_spacetime_leaves ) const {
  // Compute quadratures of the Chebyshev polynomials and their derivatives in
  // case that they do not have been computed yet.
  // WARNING: Only the number of columns of the quadratures of the
  //          first spatial cluster is checked.
  //  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) &&
  //    ( ( *space_clusters_spacetime_leaves.begin( ) )->
  //              get_chebyshev_quad_p0( ).get_n_columns( ) !=
  //    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6
  //    ) )
  // ATTENTION: in the current version it is not checked if quadratures etc.
  //            have been computed!!! still, each computation is done in a
  //            separate step in case that the checks are added
  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      compute_chebyshev_quadrature_p0( *it );
    }
  }
  //  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) &&
  //    ( ( *space_clusters_spacetime_leaves.begin( ) )->
  //          get_normal_drv_chebyshev_quad( ).get_n_columns( ) !=
  //    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6
  //    ) )
  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      // update the node mappings (needed for non-leaf space clusters)
      //       TODO: should this be done here, or somewhere else in the code?
      ( *it )->compute_node_mapping( );
      compute_normal_drv_chebyshev_quadrature( *it );
    }
  }

  // Check if lagrange quadratures have already been set and set them if not.
  if ( time_clusters_spacetime_leaves.size( ) > 0 ) {
    for ( auto it = time_clusters_spacetime_leaves.begin( );
          it != time_clusters_spacetime_leaves.end( ); ++it ) {
      compute_lagrange_quadrature( *it );
    }
  }
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
void besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  compute_required_data(
    std::set< time_cluster_type * > & time_clusters_spacetime_leaves,
    std::set< space_cluster_type * > & space_clusters_spacetime_leaves ) const {
  // Compute quadratures of the Chebyshev polynomials and their derivatives in
  // case that they do not have been computed yet.
  // WARNING: Only the number of columns of the quadratures of the
  //          first spatial cluster is checked.
  //  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) &&
  //    ( ( *space_clusters_spacetime_leaves.begin( ) )->
  //              get_chebyshev_quad_p0( ).get_n_columns( ) !=
  //    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6
  //    ) )
  // ATTENTION: in the current version it is not checked if quadratures etc.
  //            have been computed!!! still, each computation is done in a
  //            separate step in case that the checks are added
  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      compute_chebyshev_quadrature_p0( *it );
    }
  }
  //  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) &&
  //    ( ( *space_clusters_spacetime_leaves.begin( ) )->
  //          get_normal_drv_chebyshev_quad( ).get_n_columns( ) !=
  //    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6
  //    ) )
  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      // update the node mappings (needed for non-leaf space clusters)
      //       TODO: should this be done here, or somewhere else in the code?
      ( *it )->compute_node_mapping( );
      compute_normal_drv_chebyshev_quadrature( *it );
    }
  }

  if ( time_clusters_spacetime_leaves.size( ) > 0 ) {
    for ( auto it = time_clusters_spacetime_leaves.begin( );
          it != time_clusters_spacetime_leaves.end( ); ++it ) {
      compute_lagrange_quadrature( *it );
    }
  }
}

//! template specialization for hypersingular p1p1 matrix
template<>
void besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  compute_required_data(
    std::set< time_cluster_type * > & time_clusters_spacetime_leaves,
    std::set< space_cluster_type * > & space_clusters_spacetime_leaves ) const {
  // Compute quadratures of the Chebyshev polynomials and quadratures of
  // Chebyshev polynomials times p1 basis functions times normal vectors in
  // case that they do not have been computed yet.
  // ATTENTION: in the current version it is not checked if quadratures etc.
  //            have been computed!!! still, each computation is done in a
  //            separate step in case that the checks are added
  // WARNING: Only the number of columns of the quadratures of the
  //          first spatial cluster is checked.
  //  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) &&
  //    ( ( *space_clusters_spacetime_leaves.begin( ) )->
  //              get_chebyshev_quad_p0( ).get_n_columns( ) !=
  //    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6
  //    ) )
  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      compute_chebyshev_quadrature_p0( *it );
    }
  }

  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      // update the node mappings (needed for non-leaf space clusters)
      //       TODO: should this be done here, or somewhere else in the code?
      ( *it )->compute_node_mapping( );
      compute_chebyshev_times_normal_quadrature( *it );
    }
  }

  if ( ( space_clusters_spacetime_leaves.size( ) > 0 ) ) {
    for ( auto it = space_clusters_spacetime_leaves.begin( );
          it != space_clusters_spacetime_leaves.end( ); ++it ) {
      compute_surface_curls( *it );
    }
  }

  if ( time_clusters_spacetime_leaves.size( ) > 0 ) {
    for ( auto it = time_clusters_spacetime_leaves.begin( );
          it != time_clusters_spacetime_leaves.end( ); ++it ) {
      compute_lagrange_quadrature( *it );
    }
  }

  if ( time_clusters_spacetime_leaves.size( ) > 0 ) {
    for ( auto it = time_clusters_spacetime_leaves.begin( );
          it != time_clusters_spacetime_leaves.end( ); ++it ) {
      compute_lagrange_drv_integrals( *it );
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::compute_chebyshev_quadrature_p0( space_cluster_type *
    space_cluster ) const {
  full_matrix_type & T = space_cluster->get_chebyshev_quad_p0( );

  lo n_space_elems = space_cluster->get_n_elements( );
  T.resize( n_space_elems,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );

  // get some info on the current cluster
  vector_type cluster_center( 3 );
  vector_type cluster_half( 3 );
  space_cluster->get_center( cluster_center );
  space_cluster->get_half_size( cluster_half );
  sc padding = _test_space->get_tree( )
                 ->get_space_tree( )
                 ->get_paddings( )[ space_cluster->get_level( ) ];
  sc start_0 = cluster_center[ 0 ] - cluster_half[ 0 ] - padding;
  sc end_0 = cluster_center[ 0 ] + cluster_half[ 0 ] + padding;
  sc start_1 = cluster_center[ 1 ] - cluster_half[ 1 ] - padding;
  sc end_1 = cluster_center[ 1 ] + cluster_half[ 1 ] + padding;
  sc start_2 = cluster_center[ 2 ] - cluster_half[ 2 ] - padding;
  sc end_2 = cluster_center[ 2 ] + cluster_half[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );

  sc elem_area;
  lo elem;

  for ( lo i = 0; i < n_space_elems; ++i ) {
    elem = space_cluster->get_element( i );
    space_cluster->get_mesh( ).get_nodes( elem, y1, y2, y3 );
    elem_area = space_cluster->get_mesh( ).area( elem );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          sc quad = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            quad += cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ] * wy[ j ];
          }
          quad *= elem_area;
          T.set( i, current_index, quad );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::compute_chebyshev_quadrature_p1( space_cluster_type *
    space_cluster ) const {
  full_matrix_type & T = space_cluster->get_chebyshev_quad_p1( );
  lo n_space_nodes = space_cluster->get_n_nodes( );
  lo n_space_elems = space_cluster->get_n_elements( );
  T.resize( n_space_nodes,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  T.fill( 0.0 );

  // get some info on the current cluster
  vector_type cluster_center( 3 );
  vector_type cluster_half( 3 );
  space_cluster->get_center( cluster_center );
  space_cluster->get_half_size( cluster_half );
  sc padding = _test_space->get_tree( )
                 ->get_space_tree( )
                 ->get_paddings( )[ space_cluster->get_level( ) ];
  sc start_0 = cluster_center[ 0 ] - cluster_half[ 0 ] - padding;
  sc end_0 = cluster_center[ 0 ] + cluster_half[ 0 ] + padding;
  sc start_1 = cluster_center[ 1 ] - cluster_half[ 1 ] - padding;
  sc end_1 = cluster_center[ 1 ] + cluster_half[ 1 ] + padding;
  sc start_2 = cluster_center[ 2 ] - cluster_half[ 2 ] - padding;
  sc end_2 = cluster_center[ 2 ] + cluster_half[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );

  sc elem_area;
  lo elem;

  sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  sc * y2_ref = my_quadrature._y2_ref_cheb.data( );

  sc value1, value2, value3;
  const std::vector< lo > & elems_2_local_nodes
    = space_cluster->get_elems_2_local_nodes( );

  for ( lo i = 0; i < n_space_elems; ++i ) {
    elem = space_cluster->get_element( i );
    space_cluster->get_mesh( ).get_nodes( elem, y1, y2, y3 );
    elem_area = space_cluster->get_mesh( ).area( elem );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          value1 = 0.0;
          value2 = 0.0;
          value3 = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            sc weighted_cheb_prod = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ] * wy[ j ] * elem_area;
            value1
              += weighted_cheb_prod * ( (sc) 1.0 - y1_ref[ j ] - y2_ref[ j ] );
            value2 += weighted_cheb_prod * y1_ref[ j ];
            value3 += weighted_cheb_prod * y2_ref[ j ];
          }

          // if ( elems_2_local_nodes[ 3 * i ] == 9 ) {
          //   std::cout << "element is " << i << std::endl;

          // }

          T.add_atomic( elems_2_local_nodes[ 3 * i ], current_index, value1 );
          T.add_atomic(
            elems_2_local_nodes[ 3 * i + 1 ], current_index, value2 );
          T.add_atomic(
            elems_2_local_nodes[ 3 * i + 2 ], current_index, value3 );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::
  compute_normal_drv_chebyshev_quadrature(
    space_cluster_type * space_cluster ) const {
  full_matrix_type & T_drv = space_cluster->get_normal_drv_chebyshev_quad( );
  lo n_space_nodes = space_cluster->get_n_nodes( );
  lo n_space_elems = space_cluster->get_n_elements( );
  T_drv.resize( n_space_nodes,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  T_drv.fill( 0.0 );

  // get some info on the current cluster
  vector_type cluster_center( 3 );
  vector_type cluster_half( 3 );
  space_cluster->get_center( cluster_center );
  space_cluster->get_half_size( cluster_half );
  sc padding = _test_space->get_tree( )
                 ->get_space_tree( )
                 ->get_paddings( )[ space_cluster->get_level( ) ];
  sc start_0 = cluster_center[ 0 ] - cluster_half[ 0 ] - padding;
  sc end_0 = cluster_center[ 0 ] + cluster_half[ 0 ] + padding;
  sc start_1 = cluster_center[ 1 ] - cluster_half[ 1 ] - padding;
  sc end_1 = cluster_center[ 1 ] + cluster_half[ 1 ] + padding;
  sc start_2 = cluster_center[ 2 ] - cluster_half[ 2 ] - padding;
  sc end_2 = cluster_center[ 2 ] + cluster_half[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );
  // same for evaluations of scaled derivatives of Chebyshev polynomials
  vector_type cheb_drv_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_drv_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_drv_dim_2( ( _spat_order + 1 ) * size_quad );

  sc elem_area;
  lo elem;

  sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  sc * y2_ref = my_quadrature._y2_ref_cheb.data( );

  sc value1, value2, value3;
  linear_algebra::coordinates< 3 > grad;
  const std::vector< lo > & elems_2_local_nodes
    = space_cluster->get_elems_2_local_nodes( );

  linear_algebra::coordinates< 3 > normal;
  for ( lo i = 0; i < n_space_elems; ++i ) {
    elem = space_cluster->get_element( i );
    space_cluster->get_mesh( ).get_normal( elem, normal );
    space_cluster->get_mesh( ).get_nodes( elem, y1, y2, y3 );
    elem_area = space_cluster->get_mesh( ).area( elem );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y1_polynomial, cheb_drv_dim_0 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y2_polynomial, cheb_drv_dim_1 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y3_polynomial, cheb_drv_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          value1 = 0.0;
          value2 = 0.0;
          value3 = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            grad[ 0 ] = cheb_drv_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ]
              / ( cluster_half[ 0 ] + padding );
            grad[ 1 ] = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_drv_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ]
              / ( cluster_half[ 1 ] + padding );
            grad[ 2 ] = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_drv_dim_2[ beta2 * size_quad + j ]
              / ( cluster_half[ 2 ] + padding );
            sc weighted_normal_derivative
              = wy[ j ] * elem_area * normal.dot( grad );
            value1 += weighted_normal_derivative
              * ( (sc) 1.0 - y1_ref[ j ] - y2_ref[ j ] );
            value2 += weighted_normal_derivative * y1_ref[ j ];
            value3 += weighted_normal_derivative * y2_ref[ j ];
          }

          T_drv.add_atomic( elems_2_local_nodes[ 3 * i ], current_index,
            _kernel->get_alpha( ) * value1 );
          T_drv.add_atomic( elems_2_local_nodes[ 3 * i + 1 ], current_index,
            _kernel->get_alpha( ) * value2 );
          T_drv.add_atomic( elems_2_local_nodes[ 3 * i + 2 ], current_index,
            _kernel->get_alpha( ) * value3 );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::
  compute_chebyshev_times_normal_quadrature(
    space_cluster_type * space_cluster ) const {
  full_matrix_type & T_dim0 = space_cluster->get_cheb_times_normal_quad( 0 );
  full_matrix_type & T_dim1 = space_cluster->get_cheb_times_normal_quad( 1 );
  full_matrix_type & T_dim2 = space_cluster->get_cheb_times_normal_quad( 2 );
  lo n_space_nodes = space_cluster->get_n_nodes( );
  lo n_space_elems = space_cluster->get_n_elements( );
  T_dim0.resize( n_space_nodes,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  T_dim1.resize( n_space_nodes,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  T_dim2.resize( n_space_nodes,
    ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6 );
  T_dim0.fill( 0.0 );
  T_dim1.fill( 0.0 );
  T_dim2.fill( 0.0 );

  // get some info on the current cluster
  vector_type cluster_center( 3 );
  vector_type cluster_half( 3 );
  space_cluster->get_center( cluster_center );
  space_cluster->get_half_size( cluster_half );
  sc padding = _test_space->get_tree( )
                 ->get_space_tree( )
                 ->get_paddings( )[ space_cluster->get_level( ) ];
  sc start_0 = cluster_center[ 0 ] - cluster_half[ 0 ] - padding;
  sc end_0 = cluster_center[ 0 ] + cluster_half[ 0 ] + padding;
  sc start_1 = cluster_center[ 1 ] - cluster_half[ 1 ] - padding;
  sc end_1 = cluster_center[ 1 ] + cluster_half[ 1 ] + padding;
  sc start_2 = cluster_center[ 2 ] - cluster_half[ 2 ] - padding;
  sc end_2 = cluster_center[ 2 ] + cluster_half[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );

  sc elem_area;
  lo elem;

  sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  sc * y2_ref = my_quadrature._y2_ref_cheb.data( );

  sc value1, value2, value3;
  const std::vector< lo > & elems_2_local_nodes
    = space_cluster->get_elems_2_local_nodes( );

  linear_algebra::coordinates< 3 > normal;
  for ( lo i = 0; i < n_space_elems; ++i ) {
    elem = space_cluster->get_element( i );
    space_cluster->get_mesh( ).get_normal( elem, normal );
    space_cluster->get_mesh( ).get_nodes( elem, y1, y2, y3 );
    elem_area = space_cluster->get_mesh( ).area( elem );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          value1 = 0.0;
          value2 = 0.0;
          value3 = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            sc weigth_poly = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ] * wy[ j ] * elem_area;
            value1 += weigth_poly * ( (sc) 1.0 - y1_ref[ j ] - y2_ref[ j ] );
            value2 += weigth_poly * y1_ref[ j ];
            value3 += weigth_poly * y2_ref[ j ];
          }
          T_dim0.add_atomic(
            elems_2_local_nodes[ 3 * i ], current_index, value1 * normal[ 0 ] );
          T_dim0.add_atomic( elems_2_local_nodes[ 3 * i + 1 ], current_index,
            value2 * normal[ 0 ] );
          T_dim0.add_atomic( elems_2_local_nodes[ 3 * i + 2 ], current_index,
            value3 * normal[ 0 ] );

          T_dim1.add_atomic(
            elems_2_local_nodes[ 3 * i ], current_index, value1 * normal[ 1 ] );
          T_dim1.add_atomic( elems_2_local_nodes[ 3 * i + 1 ], current_index,
            value2 * normal[ 1 ] );
          T_dim1.add_atomic( elems_2_local_nodes[ 3 * i + 2 ], current_index,
            value3 * normal[ 1 ] );

          T_dim2.add_atomic(
            elems_2_local_nodes[ 3 * i ], current_index, value1 * normal[ 2 ] );
          T_dim2.add_atomic( elems_2_local_nodes[ 3 * i + 1 ], current_index,
            value2 * normal[ 2 ] );
          T_dim2.add_atomic( elems_2_local_nodes[ 3 * i + 2 ], current_index,
            value3 * normal[ 2 ] );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::initialize_moment_and_local_contributions( ) const {
  lo n_rows_contribution = _temp_order + 1;
  lo n_columns_contribution
    = ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;

  _trial_space->get_tree( )->initialize_moment_contributions(
    _trial_space->get_tree( )->get_root( ), n_rows_contribution,
    n_columns_contribution );
  _test_space->get_tree( )->initialize_local_contributions(
    _test_space->get_tree( )->get_root( ), n_rows_contribution,
    n_columns_contribution );
}

template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

// template class besthea::bem::fast_spacetime_be_assembler<
//   besthea::bem::spacetime_heat_dl_kernel_antiderivative,
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
// template class besthea::bem::fast_spacetime_be_assembler<
//   besthea::bem::spacetime_heat_dl_kernel_antiderivative,
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

// template class besthea::bem::fast_spacetime_be_assembler<
//   besthea::bem::spacetime_heat_adl_kernel_antiderivative,
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
// template class besthea::bem::fast_spacetime_be_assembler<
//   besthea::bem::spacetime_heat_adl_kernel_antiderivative,
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
//   besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
