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

#include "besthea/uniform_spacetime_be_identity.h"

#include "besthea/quadrature.h"

template class besthea::bem::uniform_spacetime_be_identity<
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::uniform_spacetime_be_identity<
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
template class besthea::bem::uniform_spacetime_be_identity<
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template< class test_space_type, class trial_space_type >
besthea::bem::uniform_spacetime_be_identity< test_space_type,
  trial_space_type >::uniform_spacetime_be_identity( test_space_type &
                                                       test_space,
  trial_space_type & trial_space, int order_regular )
  : _data( ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_regular( order_regular ) {
}

template< class test_space_type, class trial_space_type >
besthea::bem::uniform_spacetime_be_identity< test_space_type,
  trial_space_type >::~uniform_spacetime_be_identity( ) {
}

template< class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_identity< test_space_type,
  trial_space_type >::assemble( ) {
  std::vector< los > ii;
  std::vector< los > jj;
  std::vector< sc > vv;

  assemble_triplets( ii, jj, vv );

  lo n_rows = _test_space->get_basis( ).dimension_global( );
  lo n_columns = _trial_space->get_basis( ).dimension_global( );
  _data.set_from_triplets( n_rows, n_columns, ii, jj, vv );
}

template< class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_identity< test_space_type,
  trial_space_type >::assemble( matrix_type & global_matrix ) {
  std::vector< los > ii;
  std::vector< los > jj;
  std::vector< sc > vv;

  assemble_triplets( ii, jj, vv );

  lo n_rows = _test_space->get_basis( ).dimension_global( );
  lo n_columns = _trial_space->get_basis( ).dimension_global( );
  global_matrix.set_from_triplets( n_rows, n_columns, ii, jj, vv );
}

template< class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_identity< test_space_type,
  trial_space_type >::assemble_triplets( std::vector< los > & ii,
  std::vector< los > & jj, std::vector< sc > & vv ) {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto mesh = _test_space->get_mesh( );
  sc timestep = mesh->get_timestep( );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  lo n_elements = mesh->get_n_spatial_elements( );
  std::vector< lo > test_l2g( n_loc_rows );
  std::vector< lo > trial_l2g( n_loc_columns );
  ii.reserve( n_elements * n_loc_rows * n_loc_columns );
  jj.reserve( n_elements * n_loc_rows * n_loc_columns );
  vv.reserve( n_elements * n_loc_rows * n_loc_columns );

  const std::vector< sc > & x1_ref = quadrature::triangle_x1( _order_regular );
  const std::vector< sc > & x2_ref = quadrature::triangle_x2( _order_regular );
  const std::vector< sc > & w = quadrature::triangle_w( _order_regular );
  lo size = w.size( );

  sc value, test, trial, area;
  sc n[ 3 ];
  for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
    mesh->get_spatial_normal( i_elem, n );
    area = mesh->spatial_area( i_elem );

    test_basis.local_to_global( i_elem, test_l2g );
    trial_basis.local_to_global( i_elem, trial_l2g );
    for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
      for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns; ++i_loc_trial ) {
        value = 0.0;
        for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
          test = test_basis.evaluate(
            i_elem, i_loc_test, x1_ref[ i_quad ], x2_ref[ i_quad ], n );
          trial = trial_basis.evaluate(
            i_elem, i_loc_trial, x1_ref[ i_quad ], x2_ref[ i_quad ], n );

          value += w[ i_quad ] * test * trial;
        }
        ii.push_back( test_l2g[ i_loc_test ] );
        jj.push_back( trial_l2g[ i_loc_trial ] );
        vv.push_back( value * timestep * area );
      }
    }
  }
}

template< class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_identity< test_space_type,
  trial_space_type >::apply( const block_vector_type & x, block_vector_type & y,
  bool trans, sc alpha, sc beta ) const {
}
