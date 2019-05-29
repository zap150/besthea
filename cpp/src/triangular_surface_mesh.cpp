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

#include "besthea/triangular_surface_mesh.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

besthea::mesh::triangular_surface_mesh::~triangular_surface_mesh( ) {
}

besthea::mesh::triangular_surface_mesh::triangular_surface_mesh( )
  : _n_nodes( 0 ), _n_elements( 0 ), _n_edges( 0 ) {
}

void besthea::mesh::triangular_surface_mesh::print_info( ) {
  std::cout << "besthea::mesh::triangular_surface_mesh" << std::endl;
  std::cout << "  elements: " << _n_elements << ", nodes: " << _n_nodes
            << std::endl;
}

besthea::mesh::triangular_surface_mesh::triangular_surface_mesh(
  const std::string & file ) {
  load( file );
}

bool besthea::mesh::triangular_surface_mesh::load( const std::string & file ) {
  std::ifstream filestream( file.c_str( ) );

  if ( !filestream.is_open( ) ) {
    std::cerr << "File could not be opened!" << std::endl;
    return false;
  }

  lo dummy;

  filestream >> dummy;  // dimension (3)
  filestream >> dummy;  // nodes per element (3)
  filestream >> this->_n_nodes;

  _nodes.resize( 3 * _n_nodes );

  for ( lo i = 0; i < _n_nodes; ++i ) {
    filestream >> _nodes[ 3 * i ];
    filestream >> _nodes[ 3 * i + 1 ];
    filestream >> _nodes[ 3 * i + 2 ];
  }

  std::string line;
  std::stringstream linestream;
  lo n1, n2;

  filestream >> _n_elements;
  filestream.ignore( std::numeric_limits< std::streamsize >::max( ), '\n' );
  _elements.resize( _n_elements * 3 );
  _orientation.first.resize( _n_elements );
  _orientation.second.resize( _n_elements );

  for ( lo i = 0; i < _n_elements; ++i ) {
    std::getline( filestream, line );
    linestream.clear( );
    linestream.str( line );
    linestream >> _elements[ 3 * i ];
    linestream >> _elements[ 3 * i + 1 ];
    linestream >> _elements[ 3 * i + 2 ];

    if ( linestream >> n1 >> n2 ) {
      _orientation.first[ i ] = n1;
      _orientation.second[ i ] = n2;
    } else {
      _orientation.first[ i ] = 0;
      _orientation.second[ i ] = 1;
    }
  }

  filestream.close( );

  init_normals_and_areas( );
  init_normals( );
  init_edges( );

  return true;
}

void besthea::mesh::triangular_surface_mesh::init_normals_and_areas( ) {
  _areas.resize( _n_elements );

  sc x21[ 3 ];
  sc x31[ 3 ];
  sc cross[ 3 ];
  sc norm;

  for ( lo i = 0; i < _n_elements; ++i ) {
    x21[ 0 ] = _nodes[ 3 * _elements[ 3 * i_elem + 1 ] ]
      - _nodes[ 3 * _elements[ 3 * i_elem ] ];
    x21[ 1 ] = _nodes[ 3 * _elements[ 3 * i_elem + 1 ] + 1 ]
      - _nodes[ 3 * _elements[ 3 * i_elem ] + 1 ];
    x21[ 2 ] = _nodes[ 3 * _elements[ 3 * i_elem + 1 ] + 2 ]
      - _nodes[ 3 * _elements[ 3 * i_elem ] + 2 ];

    x31[ 0 ] = _nodes[ 3 * _elements[ 3 * i_elem + 2 ] ]
      - _nodes[ 3 * _elements[ 3 * i_elem ] ];
    x31[ 1 ] = _nodes[ 3 * _elements[ 3 * i_elem + 2 ] + 1 ]
      - _nodes[ 3 * _elements[ 3 * i_elem ] + 1 ];
    x31[ 2 ] = _nodes[ 3 * _elements[ 3 * i_elem + 2 ] + 2 ]
      - _nodes[ 3 * _elements[ 3 * i_elem ] + 2 ];

    cross[ 0 ] = x21[ 1 ] * x31[ 2 ] - x21[ 2 ] * x31[ 1 ];
    cross[ 1 ] = x21[ 2 ] * x31[ 0 ] - x21[ 0 ] * x31[ 2 ];
    cross[ 2 ] = x21[ 0 ] * x31[ 1 ] - x21[ 1 ] * x31[ 0 ];

    norm = std::sqrt( cross[ 0 ] * cross[ 0 ] + cross[ 1 ] * cross[ 1 ]
      + cross[ 2 ] * cross[ 2 ] );

    _areas[ i ] = 0.5 * norm;
    _normals[ 3 * i ] = cross[ 0 ] / norm;
    _normals[ 3 * i + 1 ] = cross[ 1 ] / norm;
    _normals[ 3 * i + 2 ] = cross[ 2 ] / norm;
  }
}

sc besthea::mesh::triangular_surface_mesh::area( lo i_elem ) {
  return _areas[ i_elem ];
}

void besthea::mesh::triangular_surface_mesh::init_edges( ) {
}
