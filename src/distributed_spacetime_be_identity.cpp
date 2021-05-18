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
  const auto local_st_mesh = _test_space->get_mesh->get_local_mesh( );
  lo n_local_timesteps = local_st_mesh->get_n_temporal_elements( );

  _timesteps.resize( n_local_timesteps );
  _timesteps.shrink_to_fit( );
  for ( lo d = 0; d < n_local_timesteps; ++d ) {
    _timesteps[ d ] = local_mesh->temporal_length( d );
  }
}

template< class test_space_type, class trial_space_type >
void besthea::bem::distributed_spacetime_be_identity< test_space_type,
  trial_space_type >::apply( const block_vector_type & x, block_vector_type & y,
  bool trans, sc alpha, sc beta ) const {
  std::cout << "ERROR: Apply is not implemented for block vectors (only for "
               "distributed block vectors)"
            << std::endl;
}

template< class test_space_type, class trial_space_type >
void besthea::bem::distributed_spacetime_be_identity< test_space_type,
  trial_space_type >::apply( const distributed_block_vector_type & x,
  distributed_block_vector_type & y, bool trans, sc alpha, sc beta ) const {
  if ( _data.get_n_rows( ) > 0 ) {
    lo local_start_idx = _test_space->get_mesh( )->get_local_start_idx( );
    for ( lo local_time_index = 0; local_time_index < _timesteps.size( );
          ++local_time_index ) {
      _data.apply( x.get_block( local_start_idx + local_time_index ),
        y.get_block( local_start_idx + local_time_index ), trans,
        alpha * _timesteps[ local_time_index ], beta );
    }
  } else {
    std::cout
      << "ERROR: Matrix not assembled correctly. Please use assemble routine"
      << std::endl;
  }
}
