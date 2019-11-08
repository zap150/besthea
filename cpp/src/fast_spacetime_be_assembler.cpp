/*
 * Copyright 2019, VSB - Technical University of Ostrava and Graz University of
 * Technology All rights reserved. Redistribution and use in source and binary
 * forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided
 * with the distribution. Neither the name of VSB - Technical University of
 * Ostrava and Graz University of Technology nor the names of its contributors
 * may be used to endorse or promote products  derived from this software
 * without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL VSB - TECHNICAL UNIVERSITY OF OSTRAVA AND
 * GRAZ UNIVERSITY OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "besthea/fast_spacetime_be_assembler.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::fast_spacetime_be_assembler( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::~fast_spacetime_be_assembler( ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble( besthea::linear_algebra::pFMM_matrix &
    global_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  // number of temporal elements and timestep should be the same for test and
  // trial meshes
  lo n_timesteps = test_mesh->get_n_temporal_elements( );

  // size of individual blocks
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );

  global_matrix.resize( n_timesteps * n_rows, n_timesteps * n_columns );

  assemble_nearfield( global_matrix );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::fast_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble_nearfield( besthea::linear_algebra::pFMM_matrix &
    global_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  // number of temporal elements and timestep should be the same for test and
  // trial meshes
  lo n_timesteps = test_mesh->get_n_temporal_elements( );

  // size of individual blocks
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );

  std::vector< mesh::time_cluster * > & leaves
    = _test_space->get_tree( )->get_time_tree( )->get_leaves( );

  mesh::time_cluster * current_cluster;
  mesh::time_cluster * neighbor_cluster;

  for ( auto it = leaves.begin( ); it != leaves.end( ); ++it ) {
    current_cluster = *it;
    neighbor_cluster = current_cluster->get_left_neighbour( );

    // go over every element in the current time cluster
    for ( lo i = 0; i < current_cluster->get_n_elements( ); ++i ) {
      // first, compute interaction of the cluster with itself
      // (this requires the temporal elements within the cluster to be sorted)
      for ( lo j = 0; j <= i; ++j ) {
        global_matrix.create_nearfield_matrix( i, j );
      }

      // next interact with the previous cluster
      for ( lo j = 0; j < neighbor_cluster->get_n_elements( ); ++j ) {
        global_matrix.create_nearfield_matrix( i, j );
      }
    }
  }
}

template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class besthea::bem::fast_spacetime_be_assembler<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
