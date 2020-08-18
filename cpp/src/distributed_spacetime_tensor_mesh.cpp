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

#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/scheduling_time_cluster.h"
#include "besthea/tree_structure.h"

#include <cmath>
#include <iostream>
#include <set>

besthea::mesh::distributed_spacetime_tensor_mesh::
  distributed_spacetime_tensor_mesh( const std::string & decomposition_file,
    const std::string & tree_file, const std::string & cluster_bounds_file,
    const std::string & distribution_file, MPI_Comm * comm )
  : _my_mesh( nullptr ),
    _space_mesh( nullptr ),
    _time_mesh( nullptr ),
    _dist_tree( nullptr ),
    _comm( comm ) {
  MPI_Comm_rank( *_comm, &_my_rank );
  MPI_Comm_size( *_comm, &_n_processes );

  load( decomposition_file, tree_file, cluster_bounds_file, distribution_file );

  // get global data
  MPI_Barrier( *_comm );
  lo n_elems_array[ 2 ];
  n_elems_array[ 0 ] = _my_mesh->get_n_elements( );
  n_elems_array[ 1 ] = _my_mesh->get_n_temporal_elements( );
  MPI_Allreduce( MPI_IN_PLACE , n_elems_array, 2,
    get_index_type< lo >::MPI_LO( ), MPI_SUM, *_comm );
  _n_global_elements = n_elems_array[ 0 ];
  _n_global_time_elements = n_elems_array[ 1 ];
}

besthea::mesh::distributed_spacetime_tensor_mesh::
  ~distributed_spacetime_tensor_mesh( ) {
  if ( _my_mesh != nullptr ) {
    delete _my_mesh;
  }
  if ( _space_mesh != nullptr ) {
    delete _space_mesh;
  }
  if ( _time_mesh != nullptr ) {
    delete _time_mesh;
  }
  if ( _dist_tree != nullptr ) {
    delete _dist_tree;
  }
}

besthea::mesh::tree_structure const *
  besthea::mesh::distributed_spacetime_tensor_mesh::
  get_distribution_tree( ) const {
  return _dist_tree;
}

besthea::mesh::tree_structure*
  besthea::mesh::distributed_spacetime_tensor_mesh::get_distribution_tree( ) {
  return _dist_tree;
}

void besthea::mesh::distributed_spacetime_tensor_mesh::find_my_slices(
  scheduling_time_cluster * root, std::vector< lo > & slice_indices,
  lo start, lo end ) {
  if ( root->get_n_children( ) > 0 ) {
    std::vector< scheduling_time_cluster * > * children = root->get_children( );
    lo split_index = 0;
    sc center = root->get_center( );

    for ( lo i = start; i < end; ++i ) {
      if ( ( _slices[ i + 1 ] + _slices[ i ] ) / 2.0 <= center ) {
        split_index = i + 1;
      }
    }
    find_my_slices( children->at( 0 ), slice_indices, start, split_index );
    find_my_slices( children->at( 1 ), slice_indices, split_index, end );
  } else {
    lo cluster_owner = root->get_process_id( );
    if ( cluster_owner == _my_rank ) {
      for ( lo i = 0; i < end - start; ++i ) {
        slice_indices.push_back( start + i );
      }
    }
  }
}

void besthea::mesh::distributed_spacetime_tensor_mesh::find_slices_to_load(
  std::set< lo > & nearfield_slice_indices,
  std::set< lo > & local_slice_indices ) const {
  for ( auto leaf_cluster : _dist_tree->get_leaves( ) ) {
    lo cluster_owner = leaf_cluster->get_process_id( );
    if ( cluster_owner == _my_rank ) {
      // add all the indices of slices in the local cluster to the corresponding
      // set
      const std::vector< lo > * slices = leaf_cluster->get_time_slices( );
      if ( slices != nullptr ) {
        for ( auto idx : *slices ) {
          local_slice_indices.insert( idx );
        }
      }
      // remember that this cluster's mesh is available
      leaf_cluster->set_mesh_availability( true );

      // if a cluster in the nearfield is not local, add its slices to the
      // corresponding set
      std::vector< scheduling_time_cluster* >* nearfield
        = leaf_cluster->get_nearfield_list( );
      for ( auto nearfield_cluster : *nearfield ) {
        lo nearfield_cluster_owner = nearfield_cluster->get_process_id( );
        if ( nearfield_cluster_owner != _my_rank ) {
          const std::vector< lo > * nearfield_slices
            = nearfield_cluster->get_time_slices( );
          for ( auto idx : *nearfield_slices ) {
            nearfield_slice_indices.insert( idx );
          }
          // remember that this cluster's mesh is available
          nearfield_cluster->set_mesh_availability( true );
        }
      }
    }
  }
}

bool besthea::mesh::distributed_spacetime_tensor_mesh::load(
  const std::string & decomposition_file, const std::string & tree_file,
  const std::string & cluster_bounds_file,
  const std::string & distribution_file ) {
  // load the file with basic description of the decomposed mesh
  std::ifstream filestream( decomposition_file.c_str( ) );

  if ( !filestream.is_open( ) ) {
    std::cout << "File could not be opened!" << std::endl;
    _n_meshes = 0;
    return false;
  }

  // load the boundary of the time interval

  filestream >> _t_start;
  filestream >> _t_end;

  filestream >> _n_meshes;  // read total number of time slices

  // read the slices definition
  _slices.resize( _n_meshes + 1 );
  sc node;
  for ( lo i = 0; i < _n_meshes + 1; ++i ) {
    filestream >> node;
    _slices[ i ] = node;
  }

  // read and reconstruct temporal tree and distribution of clusters
  _dist_tree = new tree_structure( tree_file, cluster_bounds_file, _my_rank );
  _dist_tree->load_process_assignments( distribution_file );
  _dist_tree->assign_slices_to_clusters( _slices );

  std::set< lo > local_slice_indices;
  std::set< lo > nearfield_slice_indices;
  find_slices_to_load( nearfield_slice_indices, local_slice_indices );

  _n_meshes_per_rank = local_slice_indices.size( );

  lo local_start_mesh = *local_slice_indices.begin( );
  lo nearfield_start_mesh = *nearfield_slice_indices.begin( );
  std::vector< sc > my_time_nodes;
  std::vector< sc > my_nearfield_time_nodes;

  lo current_idx;
  std::string t_file_path;
  std::string s_file_path;

  std::set< lo >::iterator next_slice_to_load
    = nearfield_slice_indices.begin( );
  bool is_local_mesh = false;
  // check whether there are nearfield slices to load
  if ( next_slice_to_load == nearfield_slice_indices.end( ) ) {
    _my_nearfield_start_idx = 0;
    next_slice_to_load = local_slice_indices.begin( );
    is_local_mesh = true;
  }

  lo mesh_idx = 0;
  while ( ( mesh_idx < _n_meshes )
          && ( next_slice_to_load != local_slice_indices.end( ) ) ) {
    filestream >> current_idx;
    filestream >> t_file_path;
    filestream >> s_file_path;

    if ( mesh_idx == *next_slice_to_load ) {
      // check whether the mesh is a nearfield mesh or a local one
      if ( !is_local_mesh ) { // nearfield mesh
        if ( mesh_idx == nearfield_start_mesh ) {
          _my_nearfield_start_idx = current_idx;
        }
        std::ifstream temp_file( t_file_path.c_str( ) );
        if ( !temp_file.is_open( ) ) {
          std::cout << "File " << t_file_path << " could not be opened!"
                    << std::endl;
          return false;
        }
        lo dummy;

        temp_file >> dummy;  // dimension (1)
        temp_file >> dummy;  // nodes per element (2)

        lo n_nodes;
        sc node;
        temp_file >> n_nodes;
        for ( lo i_node = 0; i_node < n_nodes; ++i_node ) {
          temp_file >> node;
          // avoid double entries in nearfield time nodes vector by adding the
          // first node of a mesh only if the mesh is the first mesh.
          if ( i_node != 0 || mesh_idx == nearfield_start_mesh ) {
            my_nearfield_time_nodes.push_back( node );
          }
        }
        temp_file.close( );
      }
      else { // local mesh
        if ( mesh_idx == local_start_mesh ) {
          _local_start_idx = current_idx;
        }
        std::ifstream temp_file( t_file_path.c_str( ) );
        if ( !temp_file.is_open( ) ) {
          std::cout << "File " << t_file_path << " could not be opened!"
                    << std::endl;
          return false;
        }
        lo dummy;

        temp_file >> dummy;  // dimension (1)
        temp_file >> dummy;  // nodes per element (2)

        lo n_nodes;
        sc node;
        temp_file >> n_nodes;
        for ( lo i_node = 0; i_node < n_nodes; ++i_node ) {
          temp_file >> node;
          // again, avoid double entries
          if ( i_node != 0 || mesh_idx == local_start_mesh ) {
            my_time_nodes.push_back( node );
          }
        }

        temp_file.close( );
      }
      ++next_slice_to_load;
      if ( next_slice_to_load == nearfield_slice_indices.end( ) ) {
        next_slice_to_load = local_slice_indices.begin( );
        is_local_mesh = true;
      }
    }
    ++mesh_idx;
  }

  _time_mesh = new temporal_mesh( my_time_nodes );
  _space_mesh = new triangular_surface_mesh( s_file_path );
  _my_mesh = new spacetime_tensor_mesh( *_space_mesh, *_time_mesh );

  if ( my_nearfield_time_nodes.size( ) > 0 ) {
    _nearfield_time_mesh = new temporal_mesh( my_nearfield_time_nodes );
    _nearfield_mesh
      = new spacetime_tensor_mesh( *_space_mesh, *_nearfield_time_mesh );
  } else {
    _nearfield_time_mesh = nullptr;
    _nearfield_mesh = nullptr;
  }

  filestream.close( );

  _dist_tree->reduce_2_essential( );

  return true;
}
