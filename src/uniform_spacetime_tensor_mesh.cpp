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

#include "besthea/uniform_spacetime_tensor_mesh.h"

#include <iomanip>
#include <iostream>

besthea::mesh::uniform_spacetime_tensor_mesh::uniform_spacetime_tensor_mesh(
  triangular_surface_mesh & space_mesh, sc end_time, lo n_timesteps ) {
  this->_space_mesh = &space_mesh;
  _end_time = end_time;
  _n_timesteps = n_timesteps;
  _timestep = end_time / n_timesteps;
  _time_mesh = new temporal_mesh( 0.0, end_time, n_timesteps );
}

besthea::mesh::uniform_spacetime_tensor_mesh::
  ~uniform_spacetime_tensor_mesh( ) {
  delete _time_mesh;
}

void besthea::mesh::uniform_spacetime_tensor_mesh::refine(
  int level, int temporal_order ) {
  refine_space( level );
  refine_time( temporal_order * level );
  _n_timesteps = _time_mesh->get_n_elements( );
  _timestep = _end_time / _n_timesteps;
}

// void besthea::mesh::uniform_spacetime_tensor_mesh::refine_space( int level )
// {
//  _space_mesh->refine( level );
//}
//
// void besthea::mesh::uniform_spacetime_tensor_mesh::refine_time( int level ) {
//  _n_timesteps *= 1 << level;
//  _timestep = _end_time / _n_timesteps;
//}

void besthea::mesh::uniform_spacetime_tensor_mesh::print_info( ) const {
  std::cout << "besthea::mesh::uniform_spacetime_tensor_mesh" << std::endl;
  std::cout << "  spatial elements: " << get_n_spatial_elements( )
            << ", spatial nodes: " << get_n_spatial_nodes( )
            << ", spatial edges: " << get_n_spatial_edges( ) << std::endl;
  std::cout << "  timesteps: " << get_n_temporal_elements( ) << std::endl;
}
