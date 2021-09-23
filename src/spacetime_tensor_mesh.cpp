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

#include "besthea/spacetime_tensor_mesh.h"

#include <iostream>

besthea::mesh::spacetime_tensor_mesh::spacetime_tensor_mesh(
  triangular_surface_mesh & space_mesh, temporal_mesh & time_mesh ) {
  _space_mesh = &space_mesh;
  _time_mesh = &time_mesh;
}

besthea::mesh::spacetime_tensor_mesh::~spacetime_tensor_mesh( ) {
}

void besthea::mesh::spacetime_tensor_mesh::print_info( ) const {
  std::cout << "besthea::mesh::spacetime_tensor_mesh" << std::endl;
  std::cout << "  elements: " << get_n_elements( )
            << ", nodes: " << get_n_nodes( ) << std::endl;
}

void besthea::mesh::spacetime_tensor_mesh::refine(
  int level, int temporal_order ) {
  refine_space( level );
  refine_time( temporal_order * level );
}

void besthea::mesh::spacetime_tensor_mesh::map_to_unit_sphere( ) {
  _space_mesh->map_to_unit_sphere( );
}

void besthea::mesh::spacetime_tensor_mesh::refine_space( int level ) {
  _space_mesh->refine( level );
}

void besthea::mesh::spacetime_tensor_mesh::refine_time( int level ) {
  _time_mesh->refine( level );
}

bool besthea::mesh::spacetime_tensor_mesh::print_vtu(
  const std::string & directory, const std::vector< std::string > * node_labels,
  const std::vector< linear_algebra::block_vector * > * node_data,
  const std::vector< std::string > * element_labels,
  const std::vector< linear_algebra::block_vector * > * element_data,
  lo time_stride ) const {
  lo n_nodal = node_data ? node_data->size( ) : 0;
  lo n_elem = element_data ? element_data->size( ) : 0;
  std::vector< linear_algebra::vector * > node_data_for_one_ts( n_nodal );
  std::vector< linear_algebra::vector * > elem_data_for_one_ts( n_elem );

  lo counter = 0;
  for ( lo ts = 0; ts < _time_mesh->get_n_elements( ); ts += time_stride ) {
    for ( lo i = 0; i < n_nodal; ++i ) {
      node_data_for_one_ts[ i ] = &( *node_data )[ i ]->get_block( ts );
    }

    for ( lo i = 0; i < n_elem; ++i ) {
      elem_data_for_one_ts[ i ] = &( *element_data )[ i ]->get_block( ts );
    }

    if ( !_space_mesh->print_vtu( directory, node_labels, &node_data_for_one_ts,
           element_labels, &elem_data_for_one_ts, counter++ ) ) {
      return false;
    }
  }

  return true;
}
