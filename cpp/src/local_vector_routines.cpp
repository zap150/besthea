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
#include "besthea/local_vector_routines.h"
#include <vector>

template<>
void get_local_part_of_block_vector< 
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >( 
  besthea::mesh::spacetime_cluster* cluster,
  const besthea::linear_algebra::block_vector & block_vector,
  besthea::linear_algebra::vector & local_vector ) {
  lo n_time_elements = cluster->get_time_cluster( ).get_n_elements( );
  const std::vector< lo > & time_elements = 
    cluster->get_time_cluster( ).get_all_elements( );
  lo n_space_elements = cluster->get_space_cluster( ).get_n_elements( );
  const std::vector< lo > & space_elements =
    cluster->get_space_cluster( ).get_all_elements( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      local_vector[ i_time * n_space_elements + i_space ]
        = block_vector.get( time_elements[ i_time ], 
                            space_elements[ i_space ] );
    }
  }
}

template<>
void get_local_part_of_block_vector< 
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( 
  besthea::mesh::spacetime_cluster* cluster,
  const besthea::linear_algebra::block_vector & block_vector,
  besthea::linear_algebra::vector & local_vector ) {
  lo n_time_elements = cluster->get_time_cluster( ).get_n_elements( );
  const std::vector< lo > & time_elements = 
    cluster->get_time_cluster( ).get_all_elements( );
  lo n_space_nodes = cluster->get_space_cluster( ).get_n_nodes( );
  const std::vector< lo > & local_2_global_nodes
    = cluster->get_space_cluster( ).get_local_2_global_nodes( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      local_vector[ i_time * n_space_nodes + i_space ]
        = block_vector.get( time_elements[ i_time ], 
                            local_2_global_nodes[ i_space ] );
    }
  }
}

template<>
void add_local_part_to_block_vector< 
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >( 
  besthea::mesh::spacetime_cluster* cluster,
  const besthea::linear_algebra::vector & local_vector, 
  besthea::linear_algebra::block_vector & block_vector) {
  lo n_time_elements = cluster->get_time_cluster( ).get_n_elements( );
  const std::vector< lo > & time_elements = 
    cluster->get_time_cluster( ).get_all_elements( );
  lo n_space_elements = cluster->get_space_cluster( ).get_n_elements( );
  const std::vector< lo > & space_elements =
    cluster->get_space_cluster( ).get_all_elements( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      block_vector.add( time_elements[ i_time ], space_elements[ i_space ],
        local_vector[ i_time * n_space_elements + i_space ] );
    }
  }
}

template<>
void add_local_part_to_block_vector< 
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >( 
  besthea::mesh::spacetime_cluster* cluster,
  const besthea::linear_algebra::vector & local_vector, 
  besthea::linear_algebra::block_vector & block_vector) {
  
  lo n_time_elements = cluster->get_time_cluster( ).get_n_elements( );
  const std::vector< lo > & time_elements = 
    cluster->get_time_cluster( ).get_all_elements( );
  lo n_space_nodes = cluster->get_space_cluster( ).get_n_nodes( );
  const std::vector< lo > & local_2_global_nodes
    = cluster->get_space_cluster( ).get_local_2_global_nodes( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      block_vector.add( time_elements[ i_time ], 
        local_2_global_nodes[ i_space ], 
        local_vector[ i_time * n_space_nodes + i_space ] );
    }
  }
} 