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

#include "besthea/temporal_mesh.h"

#include <iostream>
#include <sstream>

besthea::mesh::temporal_mesh::temporal_mesh(
  sc start_time, sc end_time, lo n_timesteps ) {
  _start_time = start_time;
  _end_time = end_time;
  _n_timesteps = n_timesteps;
  _n_temporal_nodes = n_timesteps + 1;

  _elements.resize( 2 * n_timesteps );
  _nodes.resize( n_timesteps + 1 );

  _nodes[ 0 ] = start_time;
  _nodes[ _n_temporal_nodes - 1 ] = end_time;

  sc timestep = ( end_time - start_time ) / n_timesteps;

  for ( lo i_node = 1; i_node < _n_temporal_nodes - 1; ++i_node ) {
    _nodes[ i_node ] = _nodes[ i_node - 1 ] + timestep;
  }

  for ( lo i_elem = 0; i_elem < _n_timesteps; ++i_elem ) {
    _elements[ 2 * i_elem ] = 2 * i_elem;
    _elements[ 2 * i_elem + 1 ] = 2 * i_elem + 1;
  }

  init_lenghts();
}

besthea::mesh::temporal_mesh::~temporal_mesh( ) {
}

void besthea::mesh::temporal_mesh::print_info( ) const {
  std::cout << "besthea::mesh::temporal_mesh" << std::endl;
  std::cout << " elements: " << _n_timesteps << ", nodes: " << _n_temporal_nodes
            << std::endl;
}

void besthea::mesh::temporal_mesh( const std::string & file ) {
  load( file );
}

bool besthea::mesh::temporal_mesh::load( const std::string & file ) {
  std::ifstream filestream( file.c_str( ) );

  if ( !filestream.is_open( ) ) {
    std::cerr << "File could not be opened!" std::endl;
    return false;
  }
  lo dummy;

  filestream >> dummy;  // dimension (1)
  filestream >> dummy;  // nodes per element (2)
  filestream >> this->_n_temporal_nodes;

  _nodes.resize( _n_temporal_nodes );

  for ( lo i_node = 0; i_node < _n_temporal_nodes; ++i_node ) {
    filestream >> _nodes[ i_node ];
  }

  std::string line;
  std::stringstream linestream;

  filestream >> _n_timesteps;
  filestream.ignore( std::numeric_limits< std::streamsize >::max( ), '\n' );
  _elements.resize( _n_timesteps * 2 );

  for ( lo i = 0; i < _n_timesteps; ++i ) {
    std::getline( filestream, line );
    linestream.clear( );
    linestream.str( line );
    linestream >> _elements[ 2 * i ];
    linestream >> _elements[ 2 * i + 1 ];
  }

  init_lengths( );
}

void besthea::mesh::temporal_mesh::init_lengths( ) {
  for ( lo i_elem = 0; i_elem < _n_timesteps; ++i_elem ) {
    _lengths[ i_elem ] = _nodes[ _elements[ 2 * i_elem + 1 ] ]
      - _nodes[ _elements[ 2 * i_elem ] ];
  }
}

void besthea::mesh::temporal_mesh::refine( int level ) {
  lo new_n_nodes, new_n_timesteps;

  for ( lo l = 0; l < level; ++l ) {
    new_n_nodes = 2 * _n_temporal_nodes - 1;
    new_n_timesteps = 2 * _n_timesteps;

    _nodes.resize( new_n_nodes );
    for ( lo i_node = 0; i_node < _n_temporal_nodes - 1; ++i_node ) {
      _nodes[ new_n_nodes - 2 * i_node - 1 ]
        = _nodes[ _n_temporal_nodes - i_node - 1 ];
      _nodes[ new_n_nodes - 2 * i_node - 2 ]
        = ( _nodes[ _n_temporal_nodes - i_node - 1 ]
            + _nodes[ _n_temporal_nodes - i_node - 1 ] )
        / 2;
    }

    _elements.resize( new_n_timesteps );
    for ( lo i_elem = 0; i_elem < new_n_timesteps; ++i_elem ) {
      _elements[ 2 * i_elem ] = 2 * i_elem;
      _elements[ 2 * i_elem + 1 ] = 2 * i_elem + 1;
    }

    _n_temporal_nodes = new_n_nodes;
    _n_timesteps = new_n_timesteps;
  }

  init_lengths( );
}
