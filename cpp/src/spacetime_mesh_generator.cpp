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

#include "besthea/spacetime_mesh_generator.h"

besthea::mesh::spacetime_mesh_generator::spacetime_mesh_generator(
  triangular_surface_mesh & space_mesh, temporal_mesh & time_mesh ) {
  _space_mesh = &space_mesh;
  _time_mesh = &time_mesh;
  _delete_time_mesh = false;
  _delete_space_mesh = false;
}

besthea::mesh::spacetime_mesh_generator::spacetime_mesh_generator(
  triangular_surface_mesh & space_mesh, sc end_time, lo n_timesteps ) {
  _space_mesh = &space_mesh;

  _time_mesh = new temporal_mesh( 0.0, end_time, n_timesteps );
  _delete_time_mesh = true;
  _delete_space_mesh = false;
}

besthea::mesh::spacetime_mesh_generator::spacetime_mesh_generator(
  const std::string & file_space, const std::string & file_time ) {
  _time_mesh = new temporal_mesh( file_time );
  _space_mesh = new triangular_surface_mesh( file_space );
  _delete_space_mesh = true;
  _delete_time_mesh = true;
}

besthea::mesh::spacetime_mesh_generator::~spacetime_mesh_generator( ) {
  if ( _delete_time_mesh ) {
    delete _time_mesh;
  }
  if ( _delete_space_mesh ) {
    delete _space_mesh;
  }
}

bool besthea::mesh::spacetime_mesh_generator::generate( lo n_meshes,
  const std::string & directory, const std::string & file_name,
  const std::string & suffix ) {
  if ( n_meshes > _time_mesh->get_n_elements( ) ) {
    std::cout << "Error: Number of output meshes must not exceed number of "
                 "temporal elements in the input mesh."
              << std::endl;
    return false;
  }

  lo n_timesteps_per_mesh = _time_mesh->get_n_elements( ) / n_meshes;
  lo n_remaining_timesteps = _time_mesh->get_n_elements( ) % n_meshes;

  lo curr_start_timestep = 0;

  // decomposition file
  std::string decomposition_file = directory + file_name + "_d" + "." + suffix;
  std::ofstream d_file( decomposition_file.c_str( ) );
  if ( !d_file.is_open( ) ) {
    std::cout << "File '" << decomposition_file << "' could not be opened!"
              << std::endl;
    return false;
  }
  d_file << n_meshes << "\n\n";

  for ( lo i = 0; i < n_meshes; ++i ) {
    lo add_one = i < n_remaining_timesteps ? 1 : 0;
    lo n_local_timesteps = n_timesteps_per_mesh + add_one;
    lo n_local_nodes = n_local_timesteps + 1;

    linear_algebra::indices< 2 > start_element, end_element, element;
    _time_mesh->get_element( curr_start_timestep, start_element );
    _time_mesh->get_element(
      curr_start_timestep + n_local_timesteps, end_element );

    std::string file_path
      = directory + file_name + "_t_" + std::to_string( i ) + "." + suffix;

    std::ofstream file( file_path.c_str( ) );

    if ( !file.is_open( ) ) {
      std::cout << "File '" << file_path << "' could not be opened!"
                << std::endl;
      return false;
    }

    file << "1\n"
         << "2\n\n"
         << n_local_nodes << "\n";

    for ( lo i = 0; i < n_local_timesteps; ++i ) {
      _time_mesh->get_element( curr_start_timestep + i, element );
      file << _time_mesh->get_node( element[ 0 ] ) << "\n";
    }
    file << _time_mesh->get_node( element[ 1 ] ) << "\n\n"
         << n_local_timesteps << "\n";

    for ( lo i = 0; i < n_local_timesteps; ++i ) {
      _time_mesh->get_element( curr_start_timestep + i, element );
      file << element[ 0 ] - curr_start_timestep << " "
           << element[ 1 ] - curr_start_timestep << "\n";
    }

    d_file << "\n" << curr_start_timestep << "\n";
    curr_start_timestep += n_local_timesteps;
    file.close( );

    // for completeness, save also spatial mesh (although it is the same for all
    // temporal meshes)
    std::string space_file = file_name + "_s_" + std::to_string( i );
    _space_mesh->save( directory, space_file, suffix );

    d_file << file_path << "\n";
    d_file << space_file + "." + suffix << "\n";
  }

  d_file.close( );

  return true;
}
