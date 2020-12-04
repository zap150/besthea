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

#include "besthea/block_vector.h"

#include "besthea/general_spacetime_cluster.h"
#include "besthea/spacetime_cluster.h"

besthea::linear_algebra::block_vector::block_vector( )
  : _block_size( 0 ), _size( 0 ), _data( ) {
}

besthea::linear_algebra::block_vector::block_vector(
  lo block_size, std::initializer_list< sc > list )
  : _block_size( block_size ),
    _size( list.size( ) ),
    //    _data( block_size, list ) { // Why does this work??
    _data( block_size, vector_type( list ) ) {
}

besthea::linear_algebra::block_vector::block_vector(
  lo block_size, lo size, bool zero )
  : _block_size( block_size ),
    _size( size ),
    _data( block_size, vector_type( size, zero ) ) {
}

besthea::linear_algebra::block_vector::block_vector( const block_vector & that )
  : _block_size( that._block_size ), _size( that._size ), _data( that._data ) {
}

besthea::linear_algebra::block_vector::~block_vector( ) {
}

void besthea::linear_algebra::block_vector::print(
  std::ostream & stream ) const {
  for ( const vector_type & v : _data ) {
    v.print( stream );
  }
}

void besthea::linear_algebra::block_vector::copy_from_raw(
  lo block_size, lo size, const sc * data ) {
  if ( block_size != _block_size ) {
    resize( block_size );
  }
  if ( size != _size ) {
    resize_blocks( size, false );
  }
  for ( lo i = 0; i < block_size; ++i ) {
    _data[ i ].copy_from_raw( size, data + i * size );
  }
}

void besthea::linear_algebra::block_vector::copy_to_raw( sc * data ) const {
  for ( lo i = 0; i < _block_size; ++i ) {
    _data[ i ].copy_to_raw( data + i * _size );
  }
}

void besthea::linear_algebra::block_vector::copy_from_vector(
  lo block_size, lo size, const vector_type & data ) {
  if ( block_size != _block_size ) {
    resize( block_size );
  }
  if ( size != _size ) {
    resize_blocks( size, false );
  }
  for ( lo i = 0; i < block_size; ++i ) {
    _data[ i ].copy_from_raw( size, data.data( ) + i * size );
  }
}

void besthea::linear_algebra::block_vector::copy_to_vector(
  vector_type & data ) const {
  if ( data.size( ) != _block_size * _size ) {
    data.resize( _block_size * _size, false );
  }
  for ( lo i = 0; i < _block_size; ++i ) {
    _data[ i ].copy_to_raw( data.data( ) + i * _size );
  }
}

template<>
void besthea::linear_algebra::block_vector::get_local_part<
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >(
  besthea::mesh::spacetime_cluster * cluster,
  besthea::linear_algebra::vector & local_vector ) const {
  lo n_time_elements = cluster->get_time_cluster( ).get_n_elements( );
  const std::vector< lo > & time_elements
    = cluster->get_time_cluster( ).get_all_elements( );
  lo n_space_elements = cluster->get_space_cluster( ).get_n_elements( );
  const std::vector< lo > & space_elements
    = cluster->get_space_cluster( ).get_all_elements( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      local_vector[ i_time * n_space_elements + i_space ]
        = this->get( time_elements[ i_time ], space_elements[ i_space ] );
    }
  }
}

template<>
void besthea::linear_algebra::block_vector::get_local_part<
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >(
  besthea::mesh::spacetime_cluster * cluster,
  besthea::linear_algebra::vector & local_vector ) const {
  lo n_time_elements = cluster->get_time_cluster( ).get_n_elements( );
  const std::vector< lo > & time_elements
    = cluster->get_time_cluster( ).get_all_elements( );
  lo n_space_nodes = cluster->get_space_cluster( ).get_n_nodes( );
  const std::vector< lo > & local_2_global_nodes
    = cluster->get_space_cluster( ).get_local_2_global_nodes( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      local_vector[ i_time * n_space_nodes + i_space ]
        = this->get( time_elements[ i_time ], local_2_global_nodes[ i_space ] );
    }
  }
}

template<>
void besthea::linear_algebra::block_vector::get_local_part< besthea::bem::
    distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >(
  besthea::mesh::general_spacetime_cluster * cluster,
  besthea::linear_algebra::vector & local_vector ) const {
  lo n_time_elements = cluster->get_n_time_elements( );
  lo n_space_elements = cluster->get_n_space_elements( );

  const std::vector< lo > & spacetime_elements = cluster->get_all_elements( );

  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * cluster_mesh;
  lo mesh_start_idx;
  if ( cluster->get_elements_are_local( ) ) {
    cluster_mesh = distributed_mesh->get_local_mesh( );
    mesh_start_idx = distributed_mesh->get_local_start_idx( );
  } else {
    cluster_mesh = distributed_mesh->get_nearfield_mesh( );
    mesh_start_idx = distributed_mesh->get_nearfield_start_idx( );
  }
  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component to
    // determine the local time index only once
    lo local_time_index
      = cluster_mesh->get_time_element( distributed_mesh->global_2_local(
        mesh_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    lo global_time_index = distributed_mesh->local_2_global_time(
      mesh_start_idx, local_time_index );
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      lo global_space_index = cluster_mesh->get_space_element(
        distributed_mesh->global_2_local( mesh_start_idx,
          spacetime_elements[ i_time * n_space_elements + i_space ] ) );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      local_vector[ i_time * n_space_elements + i_space ]
        = get( global_time_index, global_space_index );
    }
  }
}

template<>
void besthea::linear_algebra::block_vector::get_local_part< besthea::bem::
    distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >(
  besthea::mesh::general_spacetime_cluster * cluster,
  besthea::linear_algebra::vector & local_vector ) const {
  lo n_time_elements = cluster->get_n_time_elements( );
  lo n_space_elements = cluster->get_n_space_elements( );
  lo n_space_nodes = cluster->get_n_space_nodes( );

  const std::vector< lo > & spacetime_elements = cluster->get_all_elements( );
  const std::vector< lo > & local_2_global_nodes
    = cluster->get_local_2_global_nodes( );

  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * cluster_mesh;
  lo mesh_start_idx;
  if ( cluster->get_elements_are_local( ) ) {
    cluster_mesh = distributed_mesh->get_local_mesh( );
    mesh_start_idx = distributed_mesh->get_local_start_idx( );
  } else {
    cluster_mesh = distributed_mesh->get_nearfield_mesh( );
    mesh_start_idx = distributed_mesh->get_nearfield_start_idx( );
  }

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component to
    // determine the local time index only once
    lo local_time_index
      = cluster_mesh->get_time_element( distributed_mesh->global_2_local(
        mesh_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    lo global_time_index = distributed_mesh->local_2_global_time(
      mesh_start_idx, local_time_index );
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      // local_2_global_nodes gives the indices of the spacetime nodes. take
      // the rest from division by the number of global spatial nodes to get the
      // spatial node index
      lo global_space_index = local_2_global_nodes[ i_space ]
        % cluster_mesh->get_n_spatial_nodes( );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      local_vector[ i_time * n_space_nodes + i_space ]
        = get( global_time_index, global_space_index );
    }
  }
}

template<>
void besthea::linear_algebra::block_vector::add_local_part<
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >(
  besthea::mesh::spacetime_cluster * cluster,
  const besthea::linear_algebra::vector & local_vector ) {
  lo n_time_elements = cluster->get_time_cluster( ).get_n_elements( );
  const std::vector< lo > & time_elements
    = cluster->get_time_cluster( ).get_all_elements( );
  lo n_space_elements = cluster->get_space_cluster( ).get_n_elements( );
  const std::vector< lo > & space_elements
    = cluster->get_space_cluster( ).get_all_elements( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      this->add_atomic( time_elements[ i_time ], space_elements[ i_space ],
        local_vector[ i_time * n_space_elements + i_space ] );
    }
  }
}

template<>
void besthea::linear_algebra::block_vector::add_local_part<
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >(
  besthea::mesh::spacetime_cluster * cluster,
  const besthea::linear_algebra::vector & local_vector ) {
  lo n_time_elements = cluster->get_time_cluster( ).get_n_elements( );
  const std::vector< lo > & time_elements
    = cluster->get_time_cluster( ).get_all_elements( );
  lo n_space_nodes = cluster->get_space_cluster( ).get_n_nodes( );
  const std::vector< lo > & local_2_global_nodes
    = cluster->get_space_cluster( ).get_local_2_global_nodes( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      this->add_atomic( time_elements[ i_time ],
        local_2_global_nodes[ i_space ],
        local_vector[ i_time * n_space_nodes + i_space ] );
    }
  }
}

template<>
void besthea::linear_algebra::block_vector::add_local_part< besthea::bem::
    distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >(
  besthea::mesh::general_spacetime_cluster * cluster,
  const besthea::linear_algebra::vector & local_vector ) {
  lo n_time_elements = cluster->get_n_time_elements( );
  lo n_space_elements = cluster->get_n_space_elements( );
  const std::vector< lo > & spacetime_elements = cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );
  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component to
    // determine the local time index only once
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    lo global_time_index = distributed_mesh->local_2_global_time(
      local_start_idx, local_time_index );
    for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
      lo global_space_index = local_mesh->get_space_element(
        distributed_mesh->global_2_local( local_start_idx,
          spacetime_elements[ i_time * n_space_elements + i_space ] ) );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      add_atomic( global_time_index, global_space_index,
        local_vector[ i_time * n_space_elements + i_space ] );
    }
  }
}

template<>
void besthea::linear_algebra::block_vector::add_local_part< besthea::bem::
    distributed_fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >(
  besthea::mesh::general_spacetime_cluster * cluster,
  const besthea::linear_algebra::vector & local_vector ) {
  lo n_time_elements = cluster->get_n_time_elements( );
  lo n_space_elements = cluster->get_n_space_elements( );
  lo n_space_nodes = cluster->get_n_space_nodes( );
  const std::vector< lo > & spacetime_elements = cluster->get_all_elements( );
  const mesh::distributed_spacetime_tensor_mesh * distributed_mesh
    = cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh->get_local_mesh( );
  lo local_start_idx = distributed_mesh->get_local_start_idx( );

  const std::vector< lo > & local_2_global_nodes
    = cluster->get_local_2_global_nodes( );

  for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component to
    // determine the local time index only once
    lo local_time_index
      = local_mesh->get_time_element( distributed_mesh->global_2_local(
        local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
    lo global_time_index = distributed_mesh->local_2_global_time(
      local_start_idx, local_time_index );
    for ( lo i_space = 0; i_space < n_space_nodes; ++i_space ) {
      // local_2_global_nodes gives the indices of the spacetime nodes. take
      // the rest from division by the number of global spatial nodes to get the
      // spatial node index
      lo global_space_index
        = local_2_global_nodes[ i_space ] % local_mesh->get_n_spatial_nodes( );
      // for the spatial mesh no transformation from local 2 global is
      // necessary since there is just one global space mesh at the moment.
      add_atomic( global_time_index, global_space_index,
        local_vector[ i_time * n_space_nodes + i_space ] );
    }
  }
}
