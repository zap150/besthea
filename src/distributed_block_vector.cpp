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

#include "besthea/distributed_block_vector.h"

#include "besthea/general_spacetime_cluster.h"
#include "besthea/spacetime_cluster.h"

besthea::linear_algebra::distributed_block_vector::distributed_block_vector( )
  : _block_size( 0 ), _size( 0 ), _data( ), _owners( ), _duplicated( true ) {
  _comm = MPI_COMM_WORLD;
  MPI_Comm_rank( _comm, &_rank );
}

besthea::linear_algebra::distributed_block_vector::distributed_block_vector(
  lo block_size, std::initializer_list< sc > list, MPI_Comm comm )
  : _block_size( block_size ),
    _size( list.size( ) ),
    _data( block_size, vector_type( list ) ),
    _owners( block_size ),
    _comm( comm ),
    _duplicated( true ) {
  int comm_size;
  MPI_Comm_rank( _comm, &_rank );
  MPI_Comm_size( _comm, &comm_size );

  // vector is duplicated on all MPI ranks:
  std::fill( _owners.begin( ), _owners.end( ), std::vector< int >{ _rank } );

  _my_blocks.resize( _block_size );
  for ( lo i = 0; i < _block_size; ++i ) {
    _my_blocks.push_back( i );
  }

  for ( auto & it : _owners ) {
    for ( int i = 0; i < comm_size; ++i ) {
      if ( i != _rank ) {
        it.push_back( i );
      }
    }
  }
}

besthea::linear_algebra::distributed_block_vector::distributed_block_vector(
  lo block_size, lo size, bool zero, MPI_Comm comm )
  : _block_size( block_size ),
    _size( size ),
    _data( block_size, vector_type( size, zero ) ),
    _owners( block_size ),
    _my_blocks( std::vector< lo >{ } ),
    _comm( comm ),
    _duplicated( true ) {
  int comm_size;
  MPI_Comm_rank( _comm, &_rank );
  MPI_Comm_size( _comm, &comm_size );

  // vector is duplicated on all MPI ranks:
  std::fill( _owners.begin( ), _owners.end( ), std::vector< int >{ _rank } );

  _my_blocks.resize( _block_size );
  for ( lo i = 0; i < _block_size; ++i ) {
    _my_blocks[ i ] = i;
  }

  for ( auto & it : _owners ) {
    for ( int i = 0; i < comm_size; ++i ) {
      if ( i != _rank ) {
        it.push_back( i );
      }
    }
  }
}

besthea::linear_algebra::distributed_block_vector::distributed_block_vector(
  std::vector< lo > & my_blocks, lo block_size, lo size, bool zero,
  MPI_Comm comm )
  : _block_size( block_size ),
    _size( size ),
    _data( block_size ),
    _owners( block_size ),
    _my_blocks( my_blocks ),
    _comm( comm ),
    _duplicated( false ) {
  MPI_Comm_rank( _comm, &_rank );
  for ( auto it : my_blocks ) {
    _owners.at( it ).push_back( _rank );
  }

  communicate_owners( my_blocks );

  lo i = 0;
  for ( auto & it : _data ) {
    if ( _owners.at( i ).at( 0 ) == _rank ) {
      it.resize( size, zero );
    }
    i++;
  }
}

besthea::linear_algebra::distributed_block_vector::distributed_block_vector(
  const distributed_block_vector & that )
  : _block_size( that._block_size ),
    _size( that._size ),
    _data( that._data ),
    _owners( that._owners ),
    _my_blocks( that._my_blocks ),
    _comm( that._comm ),
    _rank( that._rank ),
    _duplicated( that._duplicated ) {
}

besthea::linear_algebra::distributed_block_vector::
  ~distributed_block_vector( ) {
}

void besthea::linear_algebra::distributed_block_vector::resize(
  lo block_size ) {
  int comm_size;

  MPI_Comm_size( _comm, &comm_size );
  _data.resize( block_size );
  _block_size = block_size;
  _owners.resize( block_size );

  // vector is duplicated on all MPI ranks:
  std::fill( _owners.begin( ), _owners.end( ), std::vector< int >{ _rank } );
  _duplicated = true;

  for ( auto & it : _owners ) {
    for ( int i = 0; i < comm_size; ++i ) {
      if ( i != _rank ) {
        it.push_back( i );
      }
    }
  }
}

void besthea::linear_algebra::distributed_block_vector::resize(
  std::vector< lo > & my_blocks, lo block_size ) {
  int comm_size;

  MPI_Comm_size( _comm, &comm_size );
  _data.resize( block_size );
  _block_size = block_size;
  _owners.resize( block_size );

  _my_blocks = my_blocks;

  for ( auto it : my_blocks ) {
    _owners.at( it ).push_back( _rank );
  }

  communicate_owners( my_blocks );
  _duplicated = false;
}

void besthea::linear_algebra::distributed_block_vector::communicate_owners(
  std::vector< lo > & my_blocks ) {
  int comm_size;

  MPI_Comm_size( _comm, &comm_size );
  int * n_blocks_per_rank = new int[ comm_size ];
  std::fill_n( n_blocks_per_rank, comm_size, 0 );
  n_blocks_per_rank[ _rank ] = my_blocks.size( );

  MPI_Allgather(
    MPI_IN_PLACE, 1, MPI_INT, n_blocks_per_rank, 1, MPI_INT, _comm );

  lo length = 0;
  for ( lo i = 0; i < comm_size; ++i ) {
    length += n_blocks_per_rank[ i ];
  }
  lo * blocks_per_rank = new lo[ length ];

  int * offsets = new int[ comm_size ];
  offsets[ 0 ] = 0;
  for ( lo i = 0; i < comm_size; ++i ) {
    offsets[ i ] = offsets[ i - 1 ] + n_blocks_per_rank[ i - 1 ];
  }

  MPI_Allgatherv( my_blocks.data( ), my_blocks.size( ),
    get_index_type< lo >::MPI_LO( ), blocks_per_rank, n_blocks_per_rank,
    offsets, get_index_type< lo >::MPI_LO( ), _comm );

  for ( lo i = 0; i < comm_size; ++i ) {
    for ( lo j = offsets[ i ]; j < offsets[ i ] + n_blocks_per_rank[ i ];
          ++j ) {
      if ( i != _rank ) {
        _owners[ blocks_per_rank[ j ] ].push_back( i );
      }
    }
  }

  delete[] blocks_per_rank;
  delete[] n_blocks_per_rank;
  delete[] offsets;
}

void besthea::linear_algebra::distributed_block_vector::copy(
  distributed_block_vector const & that ) {
  _block_size = that._block_size;
  _size = that._size;
  _data = that._data;
  _owners = that._owners;
  _my_blocks = that._my_blocks;
  _comm = that._comm;
  _rank = that._rank;
  _duplicated = that._duplicated;
}

void besthea::linear_algebra::distributed_block_vector::copy_from_raw(
  lo block_size, lo size, const sc * data ) {
  if ( block_size != _block_size ) {
    int comm_size;

    MPI_Comm_size( _comm, &comm_size );
    resize( block_size );
    _owners.resize( block_size );

    // vector is duplicated on all MPI ranks:
    std::fill( _owners.begin( ), _owners.end( ), std::vector< int >{ _rank } );

    for ( auto & it : _owners ) {
      for ( int i = 0; i < comm_size; ++i ) {
        if ( i != _rank ) {
          it.push_back( i );
        }
      }
    }
  }
  if ( size != _size ) {
    resize_blocks( size, false );
  }
  for ( lo i = 0; i < block_size; ++i ) {
    _data[ i ].copy_from_raw( size, data + i * size );
  }
}

void besthea::linear_algebra::distributed_block_vector::copy_from_raw(
  std::vector< lo > & my_blocks, lo block_size, lo size, const sc * data ) {
  if ( block_size != _block_size ) {
    int comm_size;

    MPI_Comm_size( _comm, &comm_size );
    resize( block_size );
    _owners.resize( block_size, std::vector< int >{ } );

    for ( auto it : my_blocks ) {
      _owners.at( it ).push_back( _rank );
    }

    communicate_owners( my_blocks );
  }
  if ( size != _size ) {
    resize_blocks( size, false );
  }
  for ( lo i = 0; i < block_size; ++i ) {
    if ( _owners[ i ][ 0 ] == _rank ) {
      _data[ i ].copy_from_raw( size, data + i * size );
    }
  }
}

void besthea::linear_algebra::distributed_block_vector::copy_to_raw(
  sc * data ) const {
  for ( lo i = 0; i < _block_size; ++i ) {
    // owner (the lowest rank owning the block) broadcasts the block
    int root;
    if ( _owners[ i ].size( ) == 1 ) {
      root = _owners[ i ][ 0 ];
    } else {
      root = _owners[ i ][ 0 ] < _owners[ i ][ 1 ] ? _owners[ i ][ 0 ]
                                                   : _owners[ i ][ 1 ];
    }

    if ( root == _rank ) {
      MPI_Bcast( (void *) _data[ i ].data( ), _size,
        get_scalar_type< sc >::MPI_SC( ), root, _comm );
      _data[ i ].copy_to_raw( data + i * _size );
    } else {
      MPI_Bcast( data + i * _size, _size, get_scalar_type< sc >::MPI_SC( ),
        root, _comm );
    }
  }
}

void besthea::linear_algebra::distributed_block_vector::copy_from_vector(
  lo block_size, lo size, const vector_type & data ) {
  if ( block_size != _block_size ) {
    int comm_size;

    MPI_Comm_size( _comm, &comm_size );
    resize( block_size );
    _owners.resize( block_size );

    // vector is duplicated on all MPI ranks:
    std::fill( _owners.begin( ), _owners.end( ), std::vector< int >{ _rank } );

    for ( auto & it : _owners ) {
      for ( int i = 0; i < comm_size; ++i ) {
        if ( i != _rank ) {
          it.push_back( i );
        }
      }
    }
  }
  if ( size != _size ) {
    resize_blocks( size, false );
  }
  for ( lo i = 0; i < block_size; ++i ) {
    _data[ i ].copy_from_raw( size, data.data( ) + i * size );
  }
}

void besthea::linear_algebra::distributed_block_vector::copy_from_vector(
  std::vector< lo > & my_blocks, lo block_size, lo size,
  const vector_type & data ) {
  if ( block_size != _block_size ) {
    int comm_size;

    MPI_Comm_size( _comm, &comm_size );
    resize( block_size );
    _owners.resize( block_size, std::vector< int >{ } );

    for ( auto it : my_blocks ) {
      _owners.at( it ).push_back( _rank );
    }

    communicate_owners( my_blocks );
  }
  if ( size != _size ) {
    resize_blocks( size, false );
  }
  for ( lo i = 0; i < block_size; ++i ) {
    if ( _owners[ i ][ 0 ] == _rank ) {
      _data[ i ].copy_from_raw( size, data.data( ) + i * size );
    }
  }
}

void besthea::linear_algebra::distributed_block_vector::copy_to_vector(
  vector_type & data ) const {
  for ( lo i = 0; i < _block_size; ++i ) {
    // owner (the lowest rank owning the block) broadcasts the block
    int root;
    if ( _owners[ i ].size( ) == 1 ) {
      root = _owners[ i ][ 0 ];
    } else {
      root = _owners[ i ][ 0 ] < _owners[ i ][ 1 ] ? _owners[ i ][ 0 ]
                                                   : _owners[ i ][ 1 ];
    }

    if ( root == _rank ) {
      MPI_Bcast( (void *) _data[ i ].data( ), _size,
        get_scalar_type< sc >::MPI_SC( ), root, _comm );
      _data[ i ].copy_to_raw( data.data( ) + i * _size );
    } else {
      MPI_Bcast( data.data( ) + i * _size, _size,
        get_scalar_type< sc >::MPI_SC( ), root, _comm );
    }
  }
}

void besthea::linear_algebra::distributed_block_vector::send_block(
  lo block_idx, int rank, vector_type & data ) const {
  int primary_owner = get_primary_owner( block_idx );
  if ( primary_owner == _rank ) {
    MPI_Send( _data[ block_idx ].data( ), _size,
      get_scalar_type< sc >::MPI_SC( ), rank, 0, _comm );
  }
  if ( rank == _rank ) {
    if ( data.size( ) != _size ) {
      data.resize( _size, false );
    }
    MPI_Status status;
    MPI_Recv( data.data( ), _size, get_scalar_type< sc >::MPI_SC( ),
      primary_owner, 0, _comm, &status );
  }
}

void besthea::linear_algebra::distributed_block_vector::add(
  distributed_block_vector const & v, sc alpha ) {
  for ( lo i = 0; i < _block_size; ++i ) {
    if ( am_i_owner( i ) ) {
      if ( v.am_i_owner( i ) ) {
        _data[ i ].add( v._data[ i ], alpha );
      }
    }
  }
}

sc besthea::linear_algebra::distributed_block_vector::dot(
  distributed_block_vector const & v ) const {
  sc val = 0.0;
  if ( _duplicated ) {
    for ( lo i = 0; i < _block_size; ++i ) {
      val += _data[ i ].dot( v.get_block( i ) );
    }
  } else {
    for ( lo i = 0; i < _block_size; ++i ) {
      if ( _rank == get_primary_owner( i ) ) {
        val += _data[ i ].dot( v.get_block( i ) );
      }
    }
    MPI_Allreduce(
      MPI_IN_PLACE, &val, 1, get_scalar_type< sc >::MPI_SC( ), MPI_SUM, _comm );
  }
  return val;
}

template<>
void besthea::linear_algebra::distributed_block_vector::get_local_part<
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
void besthea::linear_algebra::distributed_block_vector::get_local_part<
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
void besthea::linear_algebra::distributed_block_vector::get_local_part<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >(
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
void besthea::linear_algebra::distributed_block_vector::get_local_part<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >(
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
void besthea::linear_algebra::distributed_block_vector::add_local_part<
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
void besthea::linear_algebra::distributed_block_vector::add_local_part<
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
void besthea::linear_algebra::distributed_block_vector::add_local_part<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >(
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
void besthea::linear_algebra::distributed_block_vector::add_local_part<
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >(
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

void besthea::linear_algebra::distributed_block_vector::print(
  std::ostream & stream ) const {
  for ( const vector_type & v : _data ) {
    v.print( stream );
  }
}

void besthea::linear_algebra::distributed_block_vector::
  synchronize_shared_parts( ) {
  for ( auto it : _my_blocks ) {
    if ( am_i_primary_owner( it ) ) {
      for ( auto i = 1; i < (lo) _owners[ it ].size( ); ++i ) {
        MPI_Send( _data.at( it ).data( ), _size,
          get_scalar_type< sc >::MPI_SC( ), _owners[ it ].at( i ), 0, _comm );
      }
    } else {
      MPI_Recv( _data.at( it ).data( ), _size, get_scalar_type< sc >::MPI_SC( ),
        get_primary_owner( it ), 0, _comm, MPI_STATUS_IGNORE );
    }
  }
}
