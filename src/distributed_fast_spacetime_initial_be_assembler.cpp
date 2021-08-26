/*
Copyright (c) 2021, VSB - Technical University of Ostrava and Graz University of
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

#include "besthea/distributed_fast_spacetime_initial_be_assembler.h"

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
using besthea::mesh::volume_space_cluster;
using besthea::mesh::volume_space_cluster_tree;

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type, trial_space_type >::
  distributed_fast_spacetime_initial_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    MPI_Comm * comm, mesh::volume_space_cluster_tree & space_source_tree,
    slou spatial_nearfield_limit, int order_regular_tri,
    int order_regular_tetra, int order_regular_line, int temp_order,
    int spat_order, sc alpha )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _space_source_tree( &space_source_tree ),
    _spatial_nearfield_limit( spatial_nearfield_limit ),
    _order_regular_tri( order_regular_tri ),
    _order_regular_tetra( order_regular_tetra ),
    _order_regular_line( order_regular_line ),
    _temp_order( temp_order ),
    _spat_order( spat_order ),
    _m2l_integration_order( _spat_order ),
    _alpha( alpha ),
    _comm( comm ) {
  MPI_Comm_rank( *_comm, &_my_rank );
}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::~distributed_fast_spacetime_initial_be_assembler( ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type, trial_space_type >::assemble( pfmm_matrix_type &
                                                   global_matrix,
  bool info_mode ) const {
  global_matrix.set_MPI_communicator( _comm );

  // determine and set the dimensions of the matrix
  lo n_timesteps = _test_space->get_mesh( ).get_n_temporal_elements( );
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  lo n_loc_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );
  global_matrix.resize( n_columns, n_loc_rows * n_timesteps );

  // initialize additional data needed for the computation
  global_matrix.set_orders( _spat_order, _temp_order, _order_regular_tri,
    _order_regular_tetra, _order_regular_line );
  global_matrix.set_alpha( _alpha );
  global_matrix.set_m2l_integration_order( _m2l_integration_order );
  // ###########################################################################
  global_matrix.initialize_fmm_data(
    _test_space->get_tree( ), _space_source_tree, _spatial_nearfield_limit );

  global_matrix.initialize_spatial_m2m_coeffs( );

  initialize_moment_and_local_contributions( );

  // precompute Chebyshev nodes and values
  global_matrix.compute_chebyshev( );

  // assemble the nearfield matrices of the pFMM matrix
  if ( !info_mode ) {
    assemble_nearfield( global_matrix );
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type, trial_space_type >::assemble_nearfield( pfmm_matrix_type &
    global_matrix ) const {
  const std::vector< std::pair< mesh::general_spacetime_cluster *,
    std::vector< mesh::volume_space_cluster * > > > & nearfield_list_vector
    = global_matrix.get_nearfield_list_vector( );

  // todo: enable sorting for better performance
  //   // first, sort by size of matrices in the nearfield
  //   std::vector< lo > total_sizes( local_leaves.size( ), 0 );
  //   for ( std::vector< general_spacetime_cluster * >::size_type leaf_index =
  //   0;
  //         leaf_index < local_leaves.size( ); ++leaf_index ) {
  //     general_spacetime_cluster * current_cluster = local_leaves[ leaf_index
  //     ]; std::vector< general_spacetime_cluster * > * nearfield_list
  //       = current_cluster->get_nearfield_list( );
  //     lo n_dofs_target = current_cluster->get_n_dofs< test_space_type >( );
  //     for ( std::vector< general_spacetime_cluster * >::size_type src_index =
  //     0;
  //           src_index < nearfield_list->size( ); ++src_index ) {
  //       general_spacetime_cluster * nearfield_cluster
  //         = ( *nearfield_list )[ src_index ];
  //       lo n_dofs_source = nearfield_cluster->get_n_dofs< trial_space_type >(
  //       ); total_sizes[ leaf_index ] += n_dofs_source * n_dofs_target;
  //     }
  //   }
  //   std::vector< lo > permutation_index( total_sizes.size( ), 0 );
  //   for ( lo i = 0; i != permutation_index.size( ); i++ ) {
  //     permutation_index[ i ] = i;
  //   }
  //   sort( permutation_index.begin( ), permutation_index.end( ),
  //     [ & ]( const int & a, const int & b ) {
  //       return ( total_sizes[ a ] > total_sizes[ b ] );
  //     } );

#pragma omp parallel for schedule( dynamic, 1 )
  for ( lo leaf_index = nearfield_list_vector.size( ) - 1; leaf_index > -1;
        --leaf_index ) {
    mesh::general_spacetime_cluster * current_cluster
      = nearfield_list_vector[ leaf_index ].first;
    //  nearfield_list_vector[ permutation_index[ leaf_index ] ].first;
    const std::vector< mesh::volume_space_cluster * > & nearfield_list
      = nearfield_list_vector[ leaf_index ].second;
    //  nearfield_list_vector[ permutation_index[ leaf_index ] ].second;
    for ( lou src_index = 0; src_index < nearfield_list.size( ); ++src_index ) {
      const volume_space_cluster * nearfield_cluster
        = nearfield_list[ src_index ];
      full_matrix_type * block
        = global_matrix.create_nearfield_matrix( leaf_index, src_index );
      // permutation_index[ leaf_index ], src_index );
      assemble_nearfield_matrix( current_cluster, nearfield_cluster, *block );
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type, trial_space_type >::
  assemble_nearfield_matrix( const general_spacetime_cluster * target_cluster,
    const volume_space_cluster * source_cluster,
    full_matrix_type & nearfield_matrix ) const {
  auto & trial_basis = _trial_space->get_basis( );
  const auto & trial_mesh = _trial_space->get_mesh( );

  auto & test_basis = _test_space->get_basis( );
  const distributed_spacetime_tensor_mesh & distributed_test_mesh
    = _test_space->get_mesh( );
  const spacetime_tensor_mesh * test_mesh;
  lo test_mesh_start_idx;
  if ( target_cluster->get_process_id( ) == _my_rank ) {
    test_mesh = distributed_test_mesh.get_local_mesh( );
    test_mesh_start_idx = distributed_test_mesh.get_local_start_idx( );
  } else {
    test_mesh = distributed_test_mesh.get_nearfield_mesh( );
    test_mesh_start_idx = distributed_test_mesh.get_nearfield_start_idx( );
  }

  lo n_test_time_elem = target_cluster->get_n_time_elements( );
  lo n_test_space_dofs = target_cluster->get_n_space_dofs< test_space_type >( );
  lo n_test_space_elem = target_cluster->get_n_space_elements( );
  std::vector< lo > test_elems = target_cluster->get_all_elements( );

  lo n_trial_elem = source_cluster->get_n_elements( );
  std::vector< lo > trial_elems = source_cluster->get_all_elements( );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  /////////////////////////////////

  std::vector< lo > test_local_access( n_loc_rows );
  std::vector< lo > trial_local_access( n_loc_columns );

  sc test, trial, value, test_area, trial_area;

  sc t0, t1;
  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3, y4;
  linear_algebra::coordinates< 3 > nx;

  lo test_elem_spacetime, test_elem_time, gl_test_elem_space;
  lo gl_trial_elem;

  sc * nx_data = nx.data( );

  quadrature_wrapper my_quadrature;
  init_quadrature( my_quadrature );
  sc * x1_ref = my_quadrature._x1_ref.data( );
  ;
  sc * x2_ref = my_quadrature._x2_ref.data( );
  ;
  sc * y1_ref = my_quadrature._y1_ref.data( );
  sc * y2_ref = my_quadrature._y2_ref.data( );
  sc * y3_ref = my_quadrature._y3_ref.data( );
  sc * w = my_quadrature._w.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );
  sc * kernel_data = my_quadrature._kernel_values.data( );
  lo size = my_quadrature._w.size( );

  for ( lo i_test_time = 0; i_test_time < n_test_time_elem; ++i_test_time ) {
    for ( lo i_test_space = 0; i_test_space < n_test_space_elem;
          ++i_test_space ) {
      // get the index of the current spacetime test element and transform
      // it to the local indices in the appropriate mesh (nearfield or
      // local)
      test_elem_spacetime
        = distributed_test_mesh.global_2_local( test_mesh_start_idx,
          test_elems[ i_test_time * n_test_space_elem + i_test_space ] );
      // get the indices of the time element and space element of which
      // the spacetime element consists and get some data.
      test_elem_time = test_mesh->get_time_element( test_elem_spacetime );
      test_mesh->get_temporal_nodes( test_elem_time, &t0, &t1 );
      gl_test_elem_space = test_mesh->get_space_element( test_elem_spacetime );
      test_mesh->get_spatial_nodes( gl_test_elem_space, x1, x2, x3 );
      test_mesh->get_spatial_normal( gl_test_elem_space, nx );
      test_area = test_mesh->spatial_area( gl_test_elem_space );

      for ( lo i_trial = 0; i_trial < n_trial_elem; ++i_trial ) {
        gl_trial_elem = trial_elems[ i_trial ];
        trial_mesh.get_nodes( gl_trial_elem, y1, y2, y3, y4 );
        trial_area = trial_mesh.area( gl_trial_elem );
        // when determining the local space dofs, the relative position of the
        // spatial surface test element and volume trial element is not regarded
        // (i.e. routines are called with n_shared_vertices = 0, rotation = 0,
        // swap = false)
        target_cluster->local_elem_to_local_space_dofs< test_space_type >(
          i_test_space, 0, 0, false, test_local_access );
        source_cluster->local_elem_to_local_dofs< trial_space_type >(
          i_trial, trial_local_access );

        triangle_and_tetrahedron_to_geometry(
          x1, x2, x3, y1, y2, y3, y4, my_quadrature );

        // treat the first time-step separately
        if ( test_mesh_start_idx == 0 && test_elem_time == 0 ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
          for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
            kernel_data[ i_quad ]
              = _kernel->definite_integral_over_first_interval(
                  x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                  x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                  x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, t1 )
              * w[ i_quad ];
          }
        } else {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
          for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
            kernel_data[ i_quad ]
              = _kernel->definite_integral_over_regular_interval(
                  x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                  x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                  x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, t0, t1 )
              * w[ i_quad ];
          }
        }

        for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
          for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                ++i_loc_trial ) {
            value = 0.0;
#pragma omp simd \
    	aligned( x1_ref, x2_ref, y1_ref, y2_ref, y3_ref, kernel_data : DATA_ALIGN ) \
    	private( test, trial ) reduction( + : value ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              test = test_basis.evaluate( gl_test_elem_space, i_loc_test,
                x1_ref[ i_quad ], x2_ref[ i_quad ], nx_data );
              trial = trial_basis.evaluate( i_trial, i_loc_trial,
                y1_ref[ i_quad ], y2_ref[ i_quad ], y3_ref[ i_quad ] );

              value += kernel_data[ i_quad ] * test * trial;
            }
            value *= test_area * trial_area;

            nearfield_matrix.add(
              i_test_time * n_test_space_dofs + test_local_access[ i_loc_test ],
              trial_local_access[ i_loc_trial ], value );
          }
        }
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type, trial_space_type >::init_quadrature( quadrature_wrapper &
    my_quadrature ) const {
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x1
    = quadrature::triangle_x1( _order_regular_tri );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x2
    = quadrature::triangle_x2( _order_regular_tri );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_w
    = quadrature::triangle_w( _order_regular_tri );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_y1
    = quadrature::tetrahedron_x1( _order_regular_tetra );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_y2
    = quadrature::tetrahedron_x2( _order_regular_tetra );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_y3
    = quadrature::tetrahedron_x3( _order_regular_tetra );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_w
    = quadrature::tetrahedron_w( _order_regular_tetra );

  lo tri_size = tri_w.size( );
  lo tetra_size = tetra_w.size( );
  lo size = tri_size * tetra_size;

  my_quadrature._x1_ref.resize( size );
  my_quadrature._x2_ref.resize( size );
  my_quadrature._y1_ref.resize( size );
  my_quadrature._y2_ref.resize( size );
  my_quadrature._y3_ref.resize( size );
  my_quadrature._w.resize( size );

  lo counter = 0;
  for ( lo i_x = 0; i_x < tri_size; ++i_x ) {
    for ( lo i_y = 0; i_y < tetra_size; ++i_y ) {
      my_quadrature._x1_ref[ counter ] = tri_x1[ i_x ];
      my_quadrature._x2_ref[ counter ] = tri_x2[ i_x ];
      my_quadrature._y1_ref[ counter ] = tetra_y1[ i_y ];
      my_quadrature._y2_ref[ counter ] = tetra_y2[ i_y ];
      my_quadrature._y3_ref[ counter ] = tetra_y3[ i_y ];
      my_quadrature._w[ counter ] = tri_w[ i_x ] * tetra_w[ i_y ];
      ++counter;
    }
  }

  my_quadrature._x1.resize( size );
  my_quadrature._x2.resize( size );
  my_quadrature._x3.resize( size );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );
  my_quadrature._kernel_values.resize( size );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type, trial_space_type >::
  triangle_and_tetrahedron_to_geometry(
    const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3,
    const linear_algebra::coordinates< 3 > & y4,
    quadrature_wrapper & my_quadrature ) const {
  const sc * y1_ref = my_quadrature._y1_ref.data( );
  const sc * y2_ref = my_quadrature._y2_ref.data( );
  const sc * y3_ref = my_quadrature._y3_ref.data( );
  const sc * x1_ref = my_quadrature._x1_ref.data( );
  const sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );

  lo size = my_quadrature._w.size( );

#pragma omp simd aligned(                                 \
  y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, y3_ref \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = y1[ 0 ] + ( y2[ 0 ] - y1[ 0 ] ) * y1_ref[ i ]
      + ( y3[ 0 ] - y1[ 0 ] ) * y2_ref[ i ]
      + ( y4[ 0 ] - y1[ 0 ] ) * y3_ref[ i ];
    y2_mapped[ i ] = y1[ 1 ] + ( y2[ 1 ] - y1[ 1 ] ) * y1_ref[ i ]
      + ( y3[ 1 ] - y1[ 1 ] ) * y2_ref[ i ]
      + ( y4[ 1 ] - y1[ 1 ] ) * y3_ref[ i ];
    y3_mapped[ i ] = y1[ 2 ] + ( y2[ 2 ] - y1[ 2 ] ) * y1_ref[ i ]
      + ( y3[ 2 ] - y1[ 2 ] ) * y2_ref[ i ]
      + ( y4[ 2 ] - y1[ 2 ] ) * y3_ref[ i ];
  }

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ];
    x2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ];
    x3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ];
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
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
void besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::initialize_moment_and_local_contributions( ) const {
  lou spat_contribution_size
    = ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
  // allocate memory to store moments for all volume clusters
  _space_source_tree->initialize_moment_contributions(
    *_space_source_tree->get_root( ), spat_contribution_size );

  lou contribution_size = spat_contribution_size * ( _temp_order + 1 );
  tree_structure * test_distribution_tree
    = _test_space->get_tree( )->get_distribution_tree( );
  test_distribution_tree->initialize_local_contributions_initial_op(
    *test_distribution_tree->get_root( ), contribution_size );
}

template class besthea::bem::distributed_fast_spacetime_initial_be_assembler<
  besthea::bem::spacetime_heat_initial_m0_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >;

template class besthea::bem::distributed_fast_spacetime_initial_be_assembler<
  besthea::bem::spacetime_heat_initial_m1_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >;
