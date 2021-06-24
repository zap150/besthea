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

#include "besthea/distributed_spacetime_be_identity.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/block_vector.h"
#include "besthea/distributed_block_vector.h"
#include "besthea/distributed_fast_spacetime_be_space.h"

template< class test_space_type, class trial_space_type >
besthea::bem::distributed_spacetime_be_identity< test_space_type,
  trial_space_type >::distributed_spacetime_be_identity( test_space_type &
                                                           test_space,
  trial_space_type & trial_space, int order_regular )
  : spacetime_be_identity< test_space_type, trial_space_type >(
    test_space, trial_space, order_regular ) {
}

template< class test_space_type, class trial_space_type >
besthea::bem::distributed_spacetime_be_identity< test_space_type,
  trial_space_type >::~distributed_spacetime_be_identity( ) {
}

template< class test_space_type, class trial_space_type >
void besthea::bem::distributed_spacetime_be_identity< test_space_type,
  trial_space_type >::assemble_timesteps( ) {
  const auto local_st_mesh = this->_test_space->get_mesh( ).get_local_mesh( );
  lo n_local_timesteps = local_st_mesh->get_n_temporal_elements( );

  this->_timesteps.resize( n_local_timesteps );
  this->_timesteps.shrink_to_fit( );
  for ( lo d = 0; d < n_local_timesteps; ++d ) {
    this->_timesteps[ d ] = local_st_mesh->temporal_length( d );
  }
}

template< class test_space_type, class trial_space_type >
void besthea::bem::
  distributed_spacetime_be_identity< test_space_type, trial_space_type >::apply(
    [[maybe_unused]] const linear_algebra::block_vector & x,
    [[maybe_unused]] linear_algebra::block_vector & y,
    [[maybe_unused]] bool trans, [[maybe_unused]] sc alpha,
    [[maybe_unused]] sc beta ) const {
  std::cout << "ERROR: Apply is not implemented for block vectors (only for "
               "distributed block vectors)"
            << std::endl;
}

template< class test_space_type, class trial_space_type >
void besthea::bem::distributed_spacetime_be_identity< test_space_type,
  trial_space_type >::apply( const linear_algebra::distributed_block_vector & x,
  linear_algebra::distributed_block_vector & y, bool trans, sc alpha,
  sc beta ) const {
  if ( this->_data.get_n_rows( ) > 0 ) {
    lo local_start_idx = this->_test_space->get_mesh( ).get_local_start_idx( );
    for ( lou local_time_index = 0; local_time_index < this->_timesteps.size( );
          ++local_time_index ) {
      this->_data.apply( x.get_block( local_start_idx + local_time_index ),
        y.get_block( local_start_idx + local_time_index ), trans,
        alpha * this->_timesteps[ local_time_index ], beta );
    }
    y.synchronize_shared_parts( );
  } else {
    std::cout
      << "ERROR: Matrix not assembled correctly. Please use assemble routine"
      << std::endl;
  }
}

template class besthea::bem::distributed_spacetime_be_identity<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >;
template class besthea::bem::distributed_spacetime_be_identity<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;
template class besthea::bem::distributed_spacetime_be_identity<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;

// Needed for L2 projection which is const
template class besthea::bem::distributed_spacetime_be_identity<
  const besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  const besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >;
template class besthea::bem::distributed_spacetime_be_identity<
  const besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  const besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;
template class besthea::bem::distributed_spacetime_be_identity<
  const besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  const besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;
