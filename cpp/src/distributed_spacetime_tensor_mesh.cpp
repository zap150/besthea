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

#include <cmath>
#include <iostream>

besthea::mesh::distributed_spacetime_tensor_mesh::
  distributed_spacetime_tensor_mesh( const std::string & decomposition_file,
    const std::string & tree_file, const std::string & cluster_bounds_file,
    const std::string & distribution_file, MPI_Comm * comm )
  : _comm( comm ),
    _my_mesh( nullptr ),
    _space_mesh( nullptr ),
    _time_mesh( nullptr ),
    _dist_tree( nullptr ) {
  MPI_Comm_rank( *_comm, &_my_rank );
  MPI_Comm_size( *_comm, &_n_processes );

  load( decomposition_file, tree_file, cluster_bounds_file, distribution_file );

  // get global data
  MPI_Barrier( *_comm );
  lo local_n_elems = _my_mesh->get_n_elements( );

  MPI_Allreduce( &local_n_elems, &_n_global_elements, 1,
    get_index_type< lo >::MPI_LO( ), MPI_SUM, *_comm );
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
  _dist_tree = new tree_structure( tree_file, cluster_bounds_file );
  _dist_tree->load_process_assignments( distribution_file );
  // std::vector< scheduling_time_cluster * > leaves = _dist_tree->get_leaves(
  // );

  std::vector< lo > slice_indices;
  find_my_slices( _dist_tree->get_root( ), slice_indices, 0, _n_meshes );

  _n_meshes_per_rank = slice_indices.size( );

  lo my_start_mesh = slice_indices.front( );
  //  if ( slice_indices.front( ) > 0 ) {
  //    // let's load also the temporal nearfield slice
  //    my_start_mesh -= 1;
  //  }
  lo my_end_mesh = slice_indices.back( );

  std::vector< sc > my_time_nodes;
  std::vector< sc > my_nearfield_time_nodes;

  lo my_start_idx;
  std::string t_file_path;
  std::string s_file_path;

  for ( lo i = 0; i < _n_meshes; ++i ) {
    filestream >> my_start_idx;
    filestream >> t_file_path;
    filestream >> s_file_path;

    if ( ( i == my_start_mesh - 1 ) ) {
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
        my_nearfield_time_nodes.push_back( node );
      }

      temp_file.close( );
    }

    if ( i >= my_start_mesh && i <= my_end_mesh ) {
      if ( i == my_start_mesh ) {
        _my_start_idx = my_start_idx;
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
        if ( i_node != 0 || i == my_start_mesh ) {
          my_time_nodes.push_back( node );
        }
      }

      temp_file.close( );
    }
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

  _dist_tree->reduce_2_essential( _my_rank );

  return true;
}
