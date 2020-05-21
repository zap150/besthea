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

#include <iostream>

besthea::mesh::distributed_spacetime_tensor_mesh::
  distributed_spacetime_tensor_mesh(
    const std::string & decomposition_file, MPI_Comm * comm )
  : _comm( comm ),
    _my_mesh( nullptr ),
    _space_mesh( nullptr ),
    _time_mesh( nullptr ) {
  MPI_Comm_rank( *_comm, &_my_rank );
  MPI_Comm_size( *_comm, &_n_processes );

  load( decomposition_file );
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
}

bool besthea::mesh::distributed_spacetime_tensor_mesh::load(
  const std::string & decomposition_file ) {
  std::ifstream filestream( decomposition_file.c_str( ) );

  if ( !filestream.is_open( ) ) {
    std::cout << "File could not be opened!" << std::endl;
    _n_meshes = 0;
    return false;
  }

  filestream >> _n_meshes;
  _n_meshes_per_rank = _n_meshes / _n_processes;
  lo rem = _n_meshes % _n_processes;
  if ( _my_rank < rem ) {
    // if number of meshes is not divisible by number of processes, first ranks
    // will own additional mesh
    _n_meshes_per_rank++;
  }

  lo my_start_mesh = 0;
  lo my_end_mesh;

  my_start_mesh = _my_rank * ( _n_meshes / _n_processes );
  my_start_mesh += ( rem < _my_rank ? rem : _my_rank );
  my_end_mesh = my_start_mesh + _n_meshes_per_rank;

  std::vector< spacetime_tensor_mesh * > my_meshes;
  my_meshes.resize( _n_meshes_per_rank );

  std::vector< sc > my_time_nodes;

  lo my_start_idx;
  std::string t_file_path;
  std::string s_file_path;
  for ( lo i = 0; i < _n_meshes; ++i ) {
    filestream >> my_start_idx;
    filestream >> t_file_path;
    filestream >> s_file_path;
    if ( i >= my_start_mesh && i < my_end_mesh ) {
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

  filestream.close( );
  return true;
}
