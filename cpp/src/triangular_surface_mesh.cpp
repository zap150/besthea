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
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

besthea::mesh::triangular_surface_mesh::~triangular_surface_mesh( ) {
}

besthea::mesh::triangular_surface_mesh::triangular_surface_mesh( ) {
}

void besthea::mesh::triangular_surface_mesh::print_info( ) {
  std::cout << "besthea::mesh::triangular_surface_mesh" << std::endl;
  std::cout << "  elements: " << this->_n_elements
            << ", nodes: " << this->_n_nodes << std::endl;
}

besthea::mesh::triangular_surface_mesh::triangular_surface_mesh(
  const std::string & file ) {
  this->load( file );
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

  this->_nodes1.resize( this->_n_nodes );
  this->_nodes2.resize( this->_n_nodes );
  this->_nodes3.resize( this->_n_nodes );

  for ( lo i = 0; i < this->_n_nodes; ++i ) {
    filestream >> this->_nodes1[ i ];
    filestream >> this->_nodes2[ i ];
    filestream >> this->_nodes3[ i ];
  }

  std::string line;
  std::stringstream linestream;
  lo n1, n2;

  filestream >> this->_n_elements;
  filestream.ignore( std::numeric_limits< std::streamsize >::max( ), '\n' );
  this->_elements.resize( this->_n_elements * 3 );
  this->_orientation.first.resize( this->_n_elements );
  this->_orientation.second.resize( this->_n_elements );

  for ( lo i = 0; i < this->_n_elements; ++i ) {
    std::getline( filestream, line );
    linestream.clear( );
    linestream.str( line );
    linestream >> this->_elements[ 3 * i ];
    linestream >> this->_elements[ 3 * i + 1 ];
    linestream >> this->_elements[ 3 * i + 2 ];

    if ( linestream >> n1 >> n2 ) {
      this->_orientation.first[ i ] = n1;
      this->_orientation.second[ i ] = n2;
    } else {
      this->_orientation.first[ i ] = 0;
      this->_orientation.second[ i ] = 1;
    }
  }

  filestream.close( );

  this->init_area( );
  this->init_normals( );
  this->init_edges( );

  return true;
}

void besthea::mesh::triangular_surface_mesh::init_area( ) {
}

void besthea::mesh::triangular_surface_mesh::init_normals( ) {
}

void besthea::mesh::triangular_surface_mesh::init_edges( ) {
}
