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
#include <vector>

besthea::mesh::triangular_surface_mesh::~triangular_surface_mesh( ) {
}

void besthea::mesh::triangular_surface_mesh::print_info( ) const {
  std::cout << "besthea::mesh::triangular_surface_mesh" << std::endl;
  std::cout << "  elements: " << _n_elements << ", nodes: " << _n_nodes
            << ", edges: " << _n_edges << std::endl;
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
  init_edges( );

  return true;
}

void besthea::mesh::triangular_surface_mesh::init_normals_and_areas( ) {
  _areas.resize( _n_elements );
  _normals.resize( 3 * _n_elements );

  sc x21[ 3 ];
  sc x31[ 3 ];
  sc cross[ 3 ];
  sc norm;

  for ( lo i_elem = 0; i_elem < _n_elements; ++i_elem ) {
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

    _areas[ i_elem ] = 0.5 * norm;
    _normals[ 3 * i_elem ] = cross[ 0 ] / norm;
    _normals[ 3 * i_elem + 1 ] = cross[ 1 ] / norm;
    _normals[ 3 * i_elem + 2 ] = cross[ 2 ] / norm;
  }
}

void besthea::mesh::triangular_surface_mesh::refine( int level ) {
  lo new_n_nodes, new_n_elements, new_n_edges;
  sc x1[ 3 ], x2[ 3 ];
  lo edge[ 2 ], edges[ 3 ], element[ 3 ];
  lo node1, node2, node3, node4, node5, node6;

  for ( int l = 0; l < level; ++l ) {
    // allocate new arrays
    new_n_nodes = _n_nodes + _n_edges;
    new_n_elements = 4 * _n_elements;
    std::vector< sc > new_nodes( _nodes );
    new_nodes.resize( 3 * new_n_nodes );
    std::vector< lo > new_elements;
    new_elements.resize( 3 * new_n_elements );

    new_n_edges = 2 * _n_edges + 3 * _n_elements;
    std::vector< lo > new_edges;
    new_edges.resize( 2 * new_n_edges );

    std::vector< lo > new_element_to_edges;
    new_element_to_edges.resize( 3 * new_n_elements );

    std::vector< lo > new_orientation_first;
    std::vector< lo > new_orientation_second;
    new_orientation_first.resize( new_n_elements );
    new_orientation_second.resize( new_n_elements );

    // loop through edges to insert new nodes
#pragma omp parallel for private( x1, x2, edge )
    for ( lo i = 0; i < _n_edges; ++i ) {
      get_edge( i, edge );
      get_node( edge[ 0 ], x1 );
      get_node( edge[ 1 ], x2 );
      new_nodes[ 3 * ( _n_nodes + i ) ] = ( x1[ 0 ] + x2[ 0 ] ) / 2.0;
      new_nodes[ 3 * ( _n_nodes + i ) + 1 ] = ( x1[ 1 ] + x2[ 1 ] ) / 2.0;
      new_nodes[ 3 * ( _n_nodes + i ) + 2 ] = ( x1[ 2 ] + x2[ 2 ] ) / 2.0;
      new_edges[ 4 * i ] = edge[ 0 ];
      new_edges[ 4 * i + 1 ] = _n_nodes + i;
      new_edges[ 4 * i + 2 ] = edge[ 1 ];
      new_edges[ 4 * i + 3 ] = _n_nodes + i;
    }

    /* loop through old elements to define new elements
                      node1
                       /\
                      /  \
                     / 1  \
               node4 ______ node6
                   / \ 4  / \
                  / 2 \  / 3 \
                 /_____\/_____\
               node2  node5  node3
     */
#pragma omp parallel for private( \
  element, edges, node1, node2, node3, node4, node5, node6 )
    for ( lo i = 0; i < _n_elements; ++i ) {
      get_element( i, element );
      get_edges( i, edges );

      for ( lo j = 0; j < 4; ++j ) {
        new_orientation_first[ 4 * i + j ] = _orientation.first.at( i );
        new_orientation_second[ 4 * i + j ] = _orientation.second.at( i );
      }

      node1 = element[ 0 ];
      node2 = element[ 1 ];
      node3 = element[ 2 ];
      node4 = _n_nodes + edges[ 0 ];
      node5 = _n_nodes + edges[ 1 ];
      node6 = _n_nodes + edges[ 2 ];

      // first element
      new_elements[ 12 * i ] = node1;
      new_elements[ 12 * i + 1 ] = node4;
      new_elements[ 12 * i + 2 ] = node6;
      // second element
      new_elements[ 12 * i + 3 ] = node4;
      new_elements[ 12 * i + 4 ] = node2;
      new_elements[ 12 * i + 5 ] = node5;
      // third element
      new_elements[ 12 * i + 6 ] = node5;
      new_elements[ 12 * i + 7 ] = node3;
      new_elements[ 12 * i + 8 ] = node6;
      // fourth element
      new_elements[ 12 * i + 9 ] = node4;
      new_elements[ 12 * i + 10 ] = node5;
      new_elements[ 12 * i + 11 ] = node6;

      if ( node4 < node5 ) {
        new_edges[ 4 * _n_edges + 6 * i ] = node4;
        new_edges[ 4 * _n_edges + 6 * i + 1 ] = node5;
      } else {
        new_edges[ 4 * _n_edges + 6 * i ] = node5;
        new_edges[ 4 * _n_edges + 6 * i + 1 ] = node4;
      }

      if ( node5 < node6 ) {
        new_edges[ 4 * _n_edges + 6 * i + 2 ] = node5;
        new_edges[ 4 * _n_edges + 6 * i + 3 ] = node6;
      } else {
        new_edges[ 4 * _n_edges + 6 * i + 2 ] = node6;
        new_edges[ 4 * _n_edges + 6 * i + 3 ] = node5;
      }

      if ( node4 < node6 ) {
        new_edges[ 4 * _n_edges + 6 * i + 4 ] = node4;
        new_edges[ 4 * _n_edges + 6 * i + 5 ] = node6;
      } else {
        new_edges[ 4 * _n_edges + 6 * i + 4 ] = node6;
        new_edges[ 4 * _n_edges + 6 * i + 5 ] = node4;
      }

      new_element_to_edges[ 12 * i ] = 2 * edges[ 0 ];
      new_element_to_edges[ 12 * i + 1 ] = 2 * _n_edges + 3 * i + 2;
      new_element_to_edges[ 12 * i + 2 ] = 2 * edges[ 2 ];

      new_element_to_edges[ 12 * i + 3 ] = 2 * edges[ 0 ];
      new_element_to_edges[ 12 * i + 4 ] = 2 * edges[ 1 ];
      new_element_to_edges[ 12 * i + 5 ] = 2 * _n_edges + 3 * i;

      new_element_to_edges[ 12 * i + 6 ] = 2 * edges[ 1 ];
      new_element_to_edges[ 12 * i + 7 ] = 2 * edges[ 2 ];
      new_element_to_edges[ 12 * i + 8 ] = 2 * _n_edges + 3 * i + 1;

      if ( node1 > node2 ) {
        ++new_element_to_edges[ 12 * i ];
      } else {
        ++new_element_to_edges[ 12 * i + 3 ];
      }

      if ( node1 > node3 ) {
        ++new_element_to_edges[ 12 * i + 2 ];
      } else {
        ++new_element_to_edges[ 12 * i + 7 ];
      }

      if ( node2 > node3 ) {
        ++new_element_to_edges[ 12 * i + 4 ];
      } else {
        ++new_element_to_edges[ 12 * i + 6 ];
      }

      new_element_to_edges[ 12 * i + 9 ] = 2 * _n_edges + 3 * i;
      new_element_to_edges[ 12 * i + 10 ] = 2 * _n_edges + 3 * i + 1;
      new_element_to_edges[ 12 * i + 11 ] = 2 * _n_edges + 3 * i + 2;
    }

    // update the mesh
    _n_nodes = new_n_nodes;
    _n_elements = new_n_elements;
    _nodes.swap( new_nodes );
    _elements.swap( new_elements );

    _n_edges = new_n_edges;
    _edges.swap( new_edges );
    _element_to_edges.swap( new_element_to_edges );

    _orientation.first.swap( new_orientation_first );
    _orientation.second.swap( new_orientation_second );
  }

  init_normals_and_areas( );
}

void besthea::mesh::triangular_surface_mesh::init_edges( ) {
  // allocate class variables
  _element_to_edges.clear( );
  _element_to_edges.resize( 3 * _n_elements );

  // allocate aux. variables
  lo element[ 3 ];

  std::vector< std::vector< lo > > local_edges;
  local_edges.resize( _n_nodes );
  std::vector< std::pair< lo, lo > > element_to_edges_tmp;
  element_to_edges_tmp.resize( 3 * _n_elements );

  // iterate through elements and insert edges with index of starting point
  // lower than ending point
  for ( lo i = 0; i < _n_elements; ++i ) {
    get_element( i, element );

    if ( element[ 0 ] < element[ 1 ] ) {
      local_edges[ element[ 0 ] ].push_back( element[ 1 ] );
    }
    if ( element[ 1 ] < element[ 2 ] ) {
      local_edges[ element[ 1 ] ].push_back( element[ 2 ] );
    }
    if ( element[ 2 ] < element[ 0 ] ) {
      local_edges[ element[ 2 ] ].push_back( element[ 0 ] );
    }
  }

  typedef typename std::vector< lo >::iterator it;
  it edge_position;
  lo f, s;

  // iterate through elements, find its edges in vector edges and
  // fill the mapping vector from element to edges
  for ( lo i = 0; i < _n_elements; ++i ) {
    get_element( i, element );

    if ( element[ 0 ] < element[ 1 ] ) {
      f = element[ 0 ];
      s = element[ 1 ];
    } else {
      f = element[ 1 ];
      s = element[ 0 ];
    }

    edge_position = std::find< it, lo >(
      local_edges[ f ].begin( ), local_edges[ f ].end( ), s );
    if ( edge_position != local_edges[ f ].end( ) && *edge_position == s ) {
      element_to_edges_tmp[ 3 * i ]
        = std::pair< lo, lo >( f, edge_position - local_edges[ f ].begin( ) );
    } else {  // for mesh not being a boundary of a domain (square, etc.)
      local_edges[ f ].push_back( s );
      element_to_edges_tmp[ 3 * i ]
        = std::pair< lo, lo >( f, local_edges[ f ].size( ) - 1 );
    }

    if ( element[ 1 ] < element[ 2 ] ) {
      f = element[ 1 ];
      s = element[ 2 ];
    } else {
      f = element[ 2 ];
      s = element[ 1 ];
    }

    edge_position = std::find< it, lo >(
      local_edges[ f ].begin( ), local_edges[ f ].end( ), s );
    if ( edge_position != local_edges[ f ].end( ) && *edge_position == s ) {
      element_to_edges_tmp[ 3 * i + 1 ]
        = std::pair< lo, lo >( f, edge_position - local_edges[ f ].begin( ) );
    } else {
      local_edges[ f ].push_back( s );
      element_to_edges_tmp[ 3 * i + 1 ]
        = std::pair< lo, lo >( f, local_edges[ f ].size( ) - 1 );
    }

    if ( element[ 2 ] < element[ 0 ] ) {
      f = element[ 2 ];
      s = element[ 0 ];
    } else {
      f = element[ 0 ];
      s = element[ 2 ];
    }

    edge_position = std::find< it, lo >(
      local_edges[ f ].begin( ), local_edges[ f ].end( ), s );
    if ( edge_position != local_edges[ f ].end( ) && *edge_position == s ) {
      element_to_edges_tmp[ 3 * i + 2 ]
        = std::pair< lo, lo >( f, edge_position - local_edges[ f ].begin( ) );
    } else {
      local_edges[ f ].push_back( s );
      element_to_edges_tmp[ 3 * i + 2 ]
        = std::pair< lo, lo >( f, local_edges[ f ].size( ) - 1 );
    }
  }

  std::vector< lo > offsets( _n_nodes );
  lo offset = 0;
  for ( lo i = 0; i < _n_nodes; ++i ) {
    offsets[ i ] = offset;
    offset += local_edges[ i ].size( );
  }
  _n_edges = offset;

  for ( lo i = 0; i < _n_elements; ++i ) {
    _element_to_edges[ 3 * i ] = element_to_edges_tmp[ 3 * i ].second
      + offsets[ element_to_edges_tmp[ 3 * i ].first ];
    _element_to_edges[ 3 * i + 1 ] = element_to_edges_tmp[ 3 * i + 1 ].second
      + offsets[ element_to_edges_tmp[ 3 * i + 1 ].first ];
    _element_to_edges[ 3 * i + 2 ] = element_to_edges_tmp[ 3 * i + 2 ].second
      + offsets[ element_to_edges_tmp[ 3 * i + 2 ].first ];
  }

  // finally convert edges from vector of std::pairs to 1D vector of lo
  _edges.clear( );
  _edges.resize( 2 * _n_edges );

  for ( lo i = 0; i < _n_nodes; ++i ) {
    for ( std::vector< lo >::size_type j = 0; j < local_edges[ i ].size( );
          ++j ) {
      _edges[ 2 * ( offsets[ i ] + j ) ] = i;
      _edges[ 2 * ( offsets[ i ] + j ) + 1 ] = local_edges[ i ][ j ];
    }
  }
}

bool besthea::mesh::triangular_surface_mesh::print_vtu(
  const std::string & file, const std::vector< std::string > * node_labels,
  const std::vector< sc * > * node_data,
  const std::vector< std::string > * element_labels,
  const std::vector< sc * > * element_data ) const {
  std::cout << "Printing '" << file << "' ... ";
  std::cout.flush( );
  std::ofstream file_vtu( file.c_str( ) );

  file_vtu.setf( std::ios::showpoint | std::ios::scientific );
  file_vtu.precision( 6 );

  if ( !file_vtu.is_open( ) ) {
    std::cout << "File could not be opened!" << std::endl;
    return false;
  }

  file_vtu << "<?xml version=\"1.0\"?>" << std::endl;
  file_vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\">"
           << std::endl;
  file_vtu << "  <UnstructuredGrid>" << std::endl;

  file_vtu << "    <Piece NumberOfPoints=\"" << _n_nodes
           << "\" NumberOfCells=\"" << _n_elements << "\">" << std::endl;

  file_vtu << "      <Points>" << std::endl;
  file_vtu << "        <DataArray type=\"Float32\" Name=\"points\""
              " NumberOfComponents=\"3\" format=\"ascii\">"
           << std::endl;

  for ( lo i = 0; i < _n_nodes; ++i ) {
    file_vtu << "          " << _nodes[ 3 * i ] << " " << _nodes[ 3 * i + 1 ]
             << " " << _nodes[ 3 * i + 2 ] << std::endl;
  }

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu << "      </Points>" << std::endl;
  file_vtu << "      <Cells>" << std::endl;
  file_vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\""
              " format=\"ascii\">"
           << std::endl;

  for ( lo i = 0; i < _n_elements; ++i ) {
    file_vtu << "          " << _elements[ 3 * i ] << " "
             << _elements[ 3 * i + 1 ] << " " << _elements[ 3 * i + 2 ]
             << std::endl;
  }

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu
    << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"
    << std::endl;

  for ( lo offset = 1; offset <= _n_elements; offset++ ) {
    file_vtu << "          " << offset * 3 << std::endl;
  }

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu
    << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    << std::endl;
  for ( lo i = 1; i <= _n_elements; i++ ) {
    file_vtu << "          5" << std::endl;
  }

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu << "      </Cells>" << std::endl;

  int n_nodal = 0;
  if ( node_data )
    n_nodal = node_data->size( );

  std::string header, vheader;
  if ( n_nodal > 0 ) {
    file_vtu << "      <PointData ";

    if ( n_nodal > 0 ) {
      header = ( *node_labels )[ 0 ];
      for ( int j = 1; j < n_nodal; j++ ) {
        header += "," + ( *node_labels )[ j ];
      }
      file_vtu << "Scalars=\"" + header + "\"";
    }

    file_vtu << ">" << std::endl;
    for ( int j = 0; j < n_nodal; ++j ) {
      file_vtu << "        <DataArray type=\"Float32\" Name=\""
          + ( *node_labels )[ j ] + "\" format=\"ascii\">"
               << std::endl;
      for ( lo i = 0; i < _n_nodes; i++ ) {
        file_vtu << "          " << ( *node_data )[ j ][ i ] << std::endl;
      }
      file_vtu << "        </DataArray>" << std::endl;
    }

    file_vtu << "      </PointData>" << std::endl;
  }

  int n_element = 0;
  if ( element_data )
    n_element = element_data->size( );

  if ( n_element > 0 ) {
    file_vtu << "      <CellData ";

    if ( n_element > 0 ) {
      header = ( *element_labels )[ 0 ];
      for ( int j = 1; j < n_element; ++j ) {
        header += "," + ( *element_labels )[ j ];
      }
      file_vtu << "Scalars=\"" + header + "\"";
    }

    file_vtu << ">" << std::endl;
    for ( int j = 0; j < n_element; ++j ) {
      file_vtu << "        <DataArray type=\"Float32\" Name=\""
          + ( *element_labels )[ j ] + "\" format=\"ascii\">"
               << std::endl;
      for ( lo i = 0; i < _n_elements; i++ ) {
        file_vtu << "          " << ( *element_data )[ j ][ i ] << std::endl;
      }
      file_vtu << "        </DataArray>" << std::endl;
    }

    file_vtu << "      </CellData>" << std::endl;
  }

  file_vtu << "    </Piece>" << std::endl;
  file_vtu << "  </UnstructuredGrid>" << std::endl;
  file_vtu << "</VTKFile>" << std::endl;
  file_vtu.close( );

  std::cout << "done." << std::endl;

  return true;
}

void besthea::mesh::triangular_surface_mesh::map_to_unit_sphere( ) {
  sc x[ 3 ];
  sc norm;
  for ( lo i_node = 0; i_node < _n_nodes; ++i_node ) {
    get_node( i_node, x );
    norm = std::sqrt( x[ 0 ] * x[ 0 ] + x[ 1 ] * x[ 1 ] + x[ 2 ] * x[ 2 ] );
    x[ 0 ] /= norm;
    x[ 1 ] /= norm;
    x[ 2 ] /= norm;
    set_node( i_node, x );
  }
}
