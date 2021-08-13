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
    int order_regular_tri, int order_regular_tetra, int order_regular_line,
    int temp_order, int spat_order, sc alpha )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _space_source_tree( &space_source_tree ),
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
    _test_space->get_tree( ), _space_source_tree );

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
  //   const std::vector< general_spacetime_cluster * > & local_leaves
  //     = _test_space->get_tree( )->get_local_leaves( );

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

  // #pragma omp parallel for schedule( dynamic, 1 )
  //   for ( std::vector< general_spacetime_cluster * >::size_type leaf_index =
  //   0;
  //         leaf_index < local_leaves.size( ); ++leaf_index ) {
  //     general_spacetime_cluster * current_cluster
  //       = local_leaves[ permutation_index[ leaf_index ] ];
  //     std::vector< general_spacetime_cluster * > * nearfield_list
  //       = current_cluster->get_nearfield_list( );
  //     for ( std::vector< general_spacetime_cluster * >::size_type src_index =
  //     0;
  //           src_index < nearfield_list->size( ); ++src_index ) {
  //       general_spacetime_cluster * nearfield_cluster
  //         = ( *nearfield_list )[ src_index ];

  //       full_matrix_type * block = global_matrix.create_nearfield_matrix(
  //         permutation_index[ leaf_index ], src_index );
  //       assemble_nearfield_matrix( current_cluster, nearfield_cluster, *block
  //       );
  //     }
  //   }

  //   // sort pointers in _n_list of the matrix for matrix-vector
  //   multiplication global_matrix.sort_clusters_in_nearfield( );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::distributed_fast_spacetime_initial_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::assemble_nearfield_matrix( general_spacetime_cluster *
                                                   target_cluster,
  volume_space_cluster * source_cluster,
  full_matrix_type & nearfield_matrix ) const {
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
