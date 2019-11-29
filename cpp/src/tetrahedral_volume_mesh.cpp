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

#include "besthea/tetrahedral_volume_mesh.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

besthea::mesh::tetrahedral_volume_mesh::tetrahedral_volume_mesh(
  const std::string & file )
  : _n_nodes( 0 ),
    _n_elements( 0 ),
    _n_surface_elements( 0 ),
    _n_surface_edges( 0 ) {
  load( file );
}

besthea::mesh::tetrahedral_volume_mesh::~tetrahedral_volume_mesh( ) {
}

void besthea::mesh::tetrahedral_volume_mesh::print_info( ) const {
  std::cout << "besthea::mesh::tetrahedral_volume_mesh" << std::endl;
  std::cout << "  elements: " << _n_elements << ", nodes: " << _n_nodes
            << ", edges: " << _n_edges << std::endl;
  std::cout << "  surface elements: " << _n_surface_elements
            << ", surface edges: " << _n_surface_edges << std::endl;
}

bool besthea::mesh::tetrahedral_volume_mesh::load( const std::string & file ) {
  std::ifstream filestream( file.c_str( ) );

  if ( !filestream.is_open( ) ) {
    std::cerr << "File could not be opened!" << std::endl;
    return false;
  }

  lo dummy;

  filestream >> dummy;  // dimension (3)
  filestream >> dummy;  // nodes per element (4)
  filestream >> _n_nodes;

  _nodes.resize( 3 * _n_nodes );

  for ( lo i = 0; i < _n_nodes; ++i ) {
    filestream >> _nodes[ 3 * i ];
    filestream >> _nodes[ 3 * i + 1 ];
    filestream >> _nodes[ 3 * i + 2 ];
  }

  std::string line;
  std::stringstream linestream;
  lo n1, n2;

  filestream >> _n_surface_elements;
  filestream.ignore( std::numeric_limits< std::streamsize >::max( ), '\n' );
  _surface_elements.resize( _n_surface_elements * 3 );
  _surface_orientation.first.resize( _n_surface_elements );
  _surface_orientation.second.resize( _n_surface_elements );

  for ( lo i = 0; i < _n_surface_elements; ++i ) {
    std::getline( filestream, line );
    linestream.clear( );
    linestream.str( line );
    linestream >> _surface_elements[ 3 * i ];
    linestream >> _surface_elements[ 3 * i + 1 ];
    linestream >> _surface_elements[ 3 * i + 2 ];

    if ( linestream >> n1 >> n2 ) {
      _surface_orientation.first[ i ] = n1;
      _surface_orientation.second[ i ] = n2;
    } else {
      _surface_orientation.first[ i ] = 0;
      _surface_orientation.second[ i ] = 1;
    }
  }

  filestream >> _n_elements;
  filestream.ignore( std::numeric_limits< std::streamsize >::max( ), '\n' );
  _elements.resize( _n_elements * 4 );

  for ( lo i = 0; i < _n_elements; ++i ) {
    std::getline( filestream, line );
    linestream.clear( );
    linestream.str( line );
    linestream >> _elements[ 4 * i ];
    linestream >> _elements[ 4 * i + 1 ];
    linestream >> _elements[ 4 * i + 2 ];
    linestream >> _elements[ 4 * i + 3 ];
  }

  filestream.close( );

  init_areas( );
  init_edges( );

  return true;
}

void besthea::mesh::tetrahedral_volume_mesh::scale( sc factor ) {
  linear_algebra::coordinates< 3 > centroid;
  get_centroid( centroid );
  for ( lo i_node = 0; i_node < _n_nodes; ++i_node ) {
    _nodes[ 3 * i_node ]
      = ( _nodes[ 3 * i_node ] - centroid[ 0 ] ) * factor + centroid[ 0 ];
    _nodes[ 3 * i_node + 1 ]
      = ( _nodes[ 3 * i_node + 1 ] - centroid[ 1 ] ) * factor + centroid[ 1 ];
    _nodes[ 3 * i_node + 2 ]
      = ( _nodes[ 3 * i_node + 2 ] - centroid[ 2 ] ) * factor + centroid[ 2 ];
  }
  init_areas( );
}

void besthea::mesh::tetrahedral_volume_mesh::refine( int level ) {
  lo new_n_nodes, new_n_elements;
  linear_algebra::coordinates< 3 > x1, x2;
  linear_algebra::indices< 2 > edge;
  linear_algebra::indices< 4 > element;
  linear_algebra::indices< 6 > edges;
  lo node1, node2, node3, node4, node12, node13, node14, node23, node24, node34;

  for ( int l = 0; l < level; ++l ) {
    // allocate new arrays
    new_n_nodes = _n_nodes + _n_edges;
    new_n_elements = 8 * _n_elements;
    std::vector< sc > new_nodes( _nodes );
    new_nodes.resize( 3 * new_n_nodes );
    std::vector< lo > new_elements;
    new_elements.resize( 6 * new_n_elements );

    // loop through edges to insert new nodes
#pragma omp parallel for private( x1, x2, edge )
    for ( lo i = 0; i < _n_edges; ++i ) {
      get_edge( i, edge );
      get_node( edge[ 0 ], x1 );
      get_node( edge[ 1 ], x2 );
      new_nodes[ 3 * ( _n_nodes + i ) ] = ( x1[ 0 ] + x2[ 0 ] ) / 2.0;
      new_nodes[ 3 * ( _n_nodes + i ) + 1 ] = ( x1[ 1 ] + x2[ 1 ] ) / 2.0;
      new_nodes[ 3 * ( _n_nodes + i ) + 2 ] = ( x1[ 2 ] + x2[ 2 ] ) / 2.0;
    }

    // loop through old elements to define new elements as in Bey, J. Computing
    //   (1995) 55: 355. https://doi.org/10.1007/BF02238487 node1
#pragma omp parallel for private( element, edges, node1, node2, node3, node4, \
  node12, node13, node14, node23, node24, node34 )
    for ( lo i = 0; i < _n_elements; ++i ) {
      get_element( i, element );
      get_edges( i, edges );

      node1 = element[ 0 ];
      node2 = element[ 1 ];
      node3 = element[ 2 ];
      node4 = element[ 3 ];
      node12 = _n_nodes + edges[ 0 ];
      node13 = _n_nodes + edges[ 1 ];
      node14 = _n_nodes + edges[ 2 ];
      node23 = _n_nodes + edges[ 3 ];
      node24 = _n_nodes + edges[ 4 ];
      node34 = _n_nodes + edges[ 5 ];

      // first element
      new_elements[ 32 * i ] = node1;
      new_elements[ 32 * i + 1 ] = node12;
      new_elements[ 32 * i + 2 ] = node13;
      new_elements[ 32 * i + 3 ] = node14;
      // second element
      new_elements[ 32 * i + 4 ] = node12;
      new_elements[ 32 * i + 5 ] = node2;
      new_elements[ 32 * i + 6 ] = node23;
      new_elements[ 32 * i + 7 ] = node24;
      // third element
      new_elements[ 32 * i + 8 ] = node13;
      new_elements[ 32 * i + 9 ] = node23;
      new_elements[ 32 * i + 10 ] = node3;
      new_elements[ 32 * i + 11 ] = node34;
      // fourth element
      new_elements[ 32 * i + 12 ] = node14;
      new_elements[ 32 * i + 13 ] = node24;
      new_elements[ 32 * i + 14 ] = node34;
      new_elements[ 32 * i + 15 ] = node4;
      // fifth element
      new_elements[ 32 * i + 16 ] = node12;
      new_elements[ 32 * i + 17 ] = node13;
      new_elements[ 32 * i + 18 ] = node14;
      new_elements[ 32 * i + 19 ] = node24;
      // sixth element
      new_elements[ 32 * i + 20 ] = node12;
      new_elements[ 32 * i + 21 ] = node13;
      new_elements[ 32 * i + 22 ] = node23;
      new_elements[ 32 * i + 23 ] = node24;
      // seventh element
      new_elements[ 32 * i + 24 ] = node13;
      new_elements[ 32 * i + 25 ] = node14;
      new_elements[ 32 * i + 26 ] = node24;
      new_elements[ 32 * i + 27 ] = node34;
      // eighth element
      new_elements[ 32 * i + 28 ] = node13;
      new_elements[ 32 * i + 29 ] = node23;
      new_elements[ 32 * i + 30 ] = node24;
      new_elements[ 32 * i + 31 ] = node34;
    }

    // update the mesh
    _n_nodes = new_n_nodes;
    _n_elements = new_n_elements;
    _nodes.swap( new_nodes );
    _elements.swap( new_elements );

    init_edges( );
  }

  init_areas( );
}

void besthea::mesh::tetrahedral_volume_mesh::init_areas( ) {
  _areas.resize( _n_elements );

  linear_algebra::coordinates< 3 > x21;
  linear_algebra::coordinates< 3 > x31;
  linear_algebra::coordinates< 3 > x41;
  linear_algebra::coordinates< 3 > cross;
  sc dot;

  for ( lo i_elem = 0; i_elem < _n_elements; ++i_elem ) {
    x21[ 0 ] = _nodes[ 3 * _elements[ 4 * i_elem + 1 ] ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] ];
    x21[ 1 ] = _nodes[ 3 * _elements[ 4 * i_elem + 1 ] + 1 ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] + 1 ];
    x21[ 2 ] = _nodes[ 3 * _elements[ 4 * i_elem + 1 ] + 2 ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] + 2 ];

    x31[ 0 ] = _nodes[ 3 * _elements[ 4 * i_elem + 2 ] ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] ];
    x31[ 1 ] = _nodes[ 3 * _elements[ 4 * i_elem + 2 ] + 1 ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] + 1 ];
    x31[ 2 ] = _nodes[ 3 * _elements[ 4 * i_elem + 2 ] + 2 ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] + 2 ];

    x41[ 0 ] = _nodes[ 3 * _elements[ 4 * i_elem + 3 ] ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] ];
    x41[ 1 ] = _nodes[ 3 * _elements[ 4 * i_elem + 3 ] + 1 ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] + 1 ];
    x41[ 2 ] = _nodes[ 3 * _elements[ 4 * i_elem + 3 ] + 2 ]
      - _nodes[ 3 * _elements[ 4 * i_elem ] + 2 ];

    cross[ 0 ] = x21[ 1 ] * x31[ 2 ] - x21[ 2 ] * x31[ 1 ];
    cross[ 1 ] = x21[ 2 ] * x31[ 0 ] - x21[ 0 ] * x31[ 2 ];
    cross[ 2 ] = x21[ 0 ] * x31[ 1 ] - x21[ 1 ] * x31[ 0 ];

    dot = x41[ 0 ] * cross[ 0 ] + x41[ 1 ] * cross[ 1 ] + x41[ 2 ] * cross[ 2 ];

    _areas[ i_elem ] = std::abs( dot ) / 6.0;
  }
}

void besthea::mesh::tetrahedral_volume_mesh::init_edges( ) {
  // allocate class variables
  _element_to_edges.clear( );
  _element_to_edges.resize( 6 * _n_elements );
  _surface_element_to_edges.clear( );
  _surface_element_to_edges.resize( 3 * _n_surface_elements );

  // allocate aux. variables
  linear_algebra::indices< 4 > element;
  std::vector< std::vector< lo > > local_edges;
  // highest index does not have any neighbours with higher index
  local_edges.resize( _n_nodes - 1 );
  std::vector< std::pair< lo, lo > > element_to_edges_tmp;
  element_to_edges_tmp.resize( 6 * _n_elements );

  typedef typename std::vector< lo >::iterator it;
  it second_node_position;

  // iterate through elements and insert edges with index of starting point
  // lower than ending point
  for ( lo i_elem = 0; i_elem < _n_elements; ++i_elem ) {
    get_element( i_elem, element );

    for ( lo i_node_1 = 0; i_node_1 < 4; ++i_node_1 ) {
      for ( lo i_node_2 = 0; i_node_2 < 4; ++i_node_2 ) {
        // add only if not present already
        // first check - order nodes
        if ( element[ i_node_1 ] < element[ i_node_2 ] ) {
          // second check
          second_node_position
            = std::find< it, lo >( local_edges[ element[ i_node_1 ] ].begin( ),
              local_edges[ element[ i_node_1 ] ].end( ), element[ i_node_2 ] );
          // not found
          if ( second_node_position
            == local_edges[ element[ i_node_1 ] ].end( ) ) {
            local_edges[ element[ i_node_1 ] ].push_back( element[ i_node_2 ] );
          }
        }
      }
    }
  }

  it edge_position;
  lo f, s;
  lo counter;

  // iterate through elements, find its edges in vector edges and
  // fill the mapping vector from element to edges
  // we number the edges within an element as
  // node 0 -- node 1
  // node 0 -- node 2
  // node 0 -- node 3
  // node 1 -- node 2
  // node 1 -- node 3
  // node 2 -- node 3
  for ( lo i_elem = 0; i_elem < _n_elements; ++i_elem ) {
    counter = 0;
    get_element( i_elem, element );

    for ( lo i_node_1 = 0; i_node_1 < 3; ++i_node_1 ) {
      for ( lo i_node_2 = i_node_1 + 1; i_node_2 < 4; ++i_node_2 ) {
        if ( element[ i_node_1 ] < element[ i_node_2 ] ) {
          f = element[ i_node_1 ];
          s = element[ i_node_2 ];
        } else {
          f = element[ i_node_2 ];
          s = element[ i_node_1 ];
        }

        edge_position = std::find< it, lo >(
          local_edges[ f ].begin( ), local_edges[ f ].end( ), s );
        // counter = i_node_1 * 4 - ( i_node_1 + 1 ) * i_node_1 / 2 + i_node_2 -
        // i_node_1 - 1
        element_to_edges_tmp[ 6 * i_elem + counter ]
          = std::pair< lo, lo >( f, edge_position - local_edges[ f ].begin( ) );
        ++counter;
      }
    }
  }

  std::vector< lo > offsets( _n_nodes - 1 );
  lo offset = 0;
  for ( lo i = 0; i < _n_nodes - 1; ++i ) {
    offsets[ i ] = offset;
    offset += local_edges[ i ].size( );
  }
  _n_edges = offset;

  for ( lo i_elem = 0; i_elem < _n_elements; ++i_elem ) {
    for ( lo i_edge = 0; i_edge < 6; ++i_edge ) {
      _element_to_edges[ 6 * i_elem + i_edge ]
        = element_to_edges_tmp[ 6 * i_elem + i_edge ].second
        + offsets[ element_to_edges_tmp[ 6 * i_elem + i_edge ].first ];
    }
  }

  linear_algebra::indices< 3 > surface_element;
  std::vector< std::vector< lo > > surface_local_edges;
  surface_local_edges.resize( _n_nodes - 1 );
  std::vector< std::pair< lo, lo > > surface_element_to_edges_tmp;
  surface_element_to_edges_tmp.resize( 3 * _n_surface_elements );
  lo modulo_3[ 4 ] = { 0, 1, 2, 0 };

  // iterate through surface elements and insert edges with index of starting
  // point lower than ending point
  for ( lo i = 0; i < _n_surface_elements; ++i ) {
    get_surface_element( i, surface_element );
    for ( lo i_node = 0; i_node < 3; ++i_node ) {
      if ( surface_element[ i_node ]
        < surface_element[ modulo_3[ i_node + 1 ] ] ) {
        surface_local_edges[ surface_element[ i_node ] ].push_back(
          surface_element[ modulo_3[ i_node + 1 ] ] );
      }
    }
  }

  // iterate through surface elements, find its edges in vector edges and
  // fill the mapping vector from element to edges
  // we number the edges within an element as
  // node 0 -- node 1
  // node 1 -- node 2
  // node 2 -- node 0
  for ( lo i_elem = 0; i_elem < _n_surface_elements; ++i_elem ) {
    get_surface_element( i_elem, surface_element );

    for ( lo i_node = 0; i_node < 3; ++i_node ) {
      if ( surface_element[ i_node ]
        < surface_element[ modulo_3[ i_node + 1 ] ] ) {
        f = surface_element[ i_node ];
        s = surface_element[ modulo_3[ i_node + 1 ] ];
      } else {
        f = surface_element[ modulo_3[ i_node + 1 ] ];
        s = surface_element[ i_node ];
      }

      edge_position = std::find< it, lo >(
        local_edges[ f ].begin( ), local_edges[ f ].end( ), s );
      surface_element_to_edges_tmp[ 3 * i_elem + i_node ]
        = std::pair< lo, lo >( f, edge_position - local_edges[ f ].begin( ) );
    }
  }

  _n_surface_edges = 0;
  for ( lo i = 0; i < _n_nodes - 1; ++i ) {
    _n_surface_edges += surface_local_edges[ i ].size( );
  }

  for ( lo i_elem = 0; i_elem < _n_surface_elements; ++i_elem ) {
    for ( lo i_edge = 0; i_edge < 3; ++i_edge ) {
      _surface_element_to_edges[ 3 * i_elem + i_edge ]
        = surface_element_to_edges_tmp[ 3 * i_elem + i_edge ].second
        + offsets[ surface_element_to_edges_tmp[ 3 * i_elem + i_edge ].first ];
    }
  }

  // finally convert edges from vector of std::pairs to 1D vector of lo
  _edges.clear( );
  _edges.resize( 2 * _n_edges );

  for ( lo i = 0; i < _n_nodes - 1; ++i ) {
    for ( std::vector< lo >::size_type j = 0; j < local_edges[ i ].size( );
          ++j ) {
      _edges[ 2 * ( offsets[ i ] + j ) ] = i;
      _edges[ 2 * ( offsets[ i ] + j ) + 1 ] = local_edges[ i ][ j ];
    }
  }
}

void besthea::mesh::tetrahedral_volume_mesh::get_centroid(
  linear_algebra::coordinates< 3 > & centroid ) {
  linear_algebra::coordinates< 3 > x;
  centroid[ 0 ] = centroid[ 1 ] = centroid[ 2 ] = 0.0;
  for ( lo i_node = 0; i_node < _n_nodes; ++i_node ) {
    get_node( i_node, x );
    centroid[ 0 ] += x[ 0 ];
    centroid[ 1 ] += x[ 1 ];
    centroid[ 2 ] += x[ 2 ];
  }
  centroid[ 0 ] /= _n_nodes;
  centroid[ 1 ] /= _n_nodes;
  centroid[ 2 ] /= _n_nodes;
}

bool besthea::mesh::tetrahedral_volume_mesh::print_ensight_case(
  const std::string & directory, const std::vector< std::string > * node_labels,
  const std::vector< std::string > * element_labels, lo n_timesteps,
  sc timestep_size ) const {
  std::string filename = directory + "/output.case";

  std::ofstream case_file( filename.c_str( ) );

  if ( !case_file.is_open( ) ) {
    std::cout << "File '" << filename << "' could not be opened!" << std::endl;
    return false;
  }

  // std::cout << "Printing '" << filename << "' ... ";
  // std::cout.flush( );

  case_file << "FORMAT\n"
            << "type: ensight gold\n\n"

            << "GEOMETRY\n"
            << "model: mesh.geo"
            << "\n\n";

  int n_nodal = node_labels ? node_labels->size( ) : 0;
  int n_element = element_labels ? element_labels->size( ) : 0;

  if ( n_nodal > 0 || n_element > 0 ) {
    case_file << "VARIABLE\n";
  }

  for ( lo i = 0; i < n_nodal; ++i ) {
    case_file << "scalar per node: " << ( *node_labels )[ i ] << " node_data_"
              << i;
    if ( n_timesteps > 0 ) {
      case_file << ".****";
    }
    case_file << "\n";
  }

  for ( lo i = 0; i < n_element; ++i ) {
    case_file << "scalar per element: " << ( *element_labels )[ i ]
              << " elem_data_" << i;
    if ( n_timesteps > 0 ) {
      case_file << ".****";
    }
    case_file << "\n";
  }

  if ( n_timesteps > 0 ) {
    case_file << "\n"
              << "TIME\n"
              << "time set: 1\n"
              << "number of steps: " << n_timesteps << "\n"
              << "filename start number: 0\n"
              << "filename increment: 1\n"
              << "time values:\n";

    sc t = 0.5 * timestep_size;
    for ( lo i = 0; i < n_timesteps; ++i ) {
      case_file << std::setw( 12 ) << std::setprecision( 5 ) << t << "\n";
      t += timestep_size;
    }
  }

  case_file.close( );

  // std::cout << "done." << std::endl;

  return true;
}

bool besthea::mesh::tetrahedral_volume_mesh::print_ensight_geometry(
  const std::string & directory ) const {
  std::string filename = directory + "/mesh.geo";

  std::ofstream geometry_file(
    filename.c_str( ), std::ios::out | std::ios::binary );

  if ( !geometry_file.is_open( ) ) {
    std::cout << "File '" << filename << "' could not be opened!" << std::endl;
    return false;
  }

  // std::cout << "Printing '" << filename << "' ... ";
  // std::cout.flush( );

  std::vector< std::string > strings
    = { "C Binary", "Decription line 1", "Decription line 2", "node id off",
        "element id off", "part", "Description", "coordinates", "tetra4" };

  for ( auto & line : strings ) {
    line.resize( 80, ' ' );
  }

  for ( size_t i = 0; i < 5; ++i ) {
    geometry_file.write( strings[ i ].c_str( ), 80 );
  }

  geometry_file.write( strings[ 5 ].c_str( ), 80 );

  int part_number = 1;
  geometry_file.write(
    reinterpret_cast< const char * >( &part_number ), sizeof( int ) );

  geometry_file.write( strings[ 6 ].c_str( ), 80 );
  geometry_file.write( strings[ 7 ].c_str( ), 80 );

  geometry_file.write(
    reinterpret_cast< const char * >( &_n_nodes ), sizeof( int ) );

  // Write coordinates of nodes:
  // first all of x coordinates then y's and z's

  // x-coordinates
  for ( lo i = 0; i < _n_nodes; ++i ) {
    float x = static_cast< float >( _nodes[ 3 * i + 0 ] );
    geometry_file.write(
      reinterpret_cast< const char * >( &x ), sizeof( float ) );
  }

  // y-coordinates
  for ( lo i = 0; i < _n_nodes; ++i ) {
    float y = static_cast< float >( _nodes[ 3 * i + 1 ] );
    geometry_file.write(
      reinterpret_cast< const char * >( &y ), sizeof( float ) );
  }

  // z-coordinates
  for ( lo i = 0; i < _n_nodes; ++i ) {
    float z = static_cast< float >( _nodes[ 3 * i + 2 ] );
    geometry_file.write(
      reinterpret_cast< const char * >( &z ), sizeof( float ) );
  }

  geometry_file.write( strings[ 8 ].c_str( ), 80 );

  geometry_file.write(
    reinterpret_cast< const char * >( &_n_elements ), sizeof( int ) );

  for ( lo i = 0; i < _n_elements; ++i ) {
    // ensight indices start at ONE not zero
    for ( lo offset = 0; offset < 4; ++offset ) {
      int element_index = _elements[ 4 * i + offset ] + 1;
      geometry_file.write(
        reinterpret_cast< const char * >( &element_index ), sizeof( int ) );
    }
  }

  geometry_file.close( );

  // std::cout << "done." << std::endl;

  return true;
}

bool besthea::mesh::tetrahedral_volume_mesh::print_ensight_datafiles(
  const std::string & directory, const std::vector< std::string > * node_labels,
  const std::vector< linear_algebra::vector * > * node_data,
  const std::vector< std::string > * element_labels,
  const std::vector< linear_algebra::vector * > * element_data,
  std::optional< lo > timestep ) const {
  int n_nodal = node_labels ? node_labels->size( ) : 0;
  int n_element = element_labels ? element_labels->size( ) : 0;

  for ( lo i = 0; i < n_nodal; ++i ) {
    std::stringstream filename;
    filename << directory << "/node_data_" << i;

    if ( timestep ) {
      filename << '.' << std::setw( 4 ) << std::setfill( '0' )
               << timestep.value( );
    }

    std::ofstream data_file(
      filename.str( ), std::ios::out | std::ios::binary );

    if ( !data_file.is_open( ) ) {
      std::cout << "File '" << filename.str( ) << "' could not be opened!"
                << std::endl;
      return false;
    }

    // std::cout << "Printing '" << filename.str( ) << "' ... ";
    // std::cout.flush( );

    std::vector< std::string > strings
      = { ( *node_labels )[ i ], "part", "coordinates" };

    for ( auto & line : strings ) {
      line.resize( 80, ' ' );
    }

    data_file.write( strings[ 0 ].c_str( ), 80 );
    data_file.write( strings[ 1 ].c_str( ), 80 );

    int part_number = 1;
    data_file.write(
      reinterpret_cast< const char * >( &part_number ), sizeof( int ) );

    data_file.write( strings[ 2 ].c_str( ), 80 );

    for ( lo j = 0; j < _n_nodes; ++j ) {
      float value = ( *node_data )[ i ]->get( j );
      data_file.write(
        reinterpret_cast< const char * >( &value ), sizeof( float ) );
    }

    data_file.close( );

    // std::cout << "done." << std::endl;
  }

  for ( lo i = 0; i < n_element; ++i ) {
    std::stringstream filename;
    filename << directory << "/elem_data_" << i;

    if ( timestep ) {
      filename << '.' << std::setw( 4 ) << std::setfill( '0' )
               << timestep.value( );
    }

    std::ofstream data_file(
      filename.str( ), std::ios::out | std::ios::binary );

    if ( !data_file.is_open( ) ) {
      std::cout << "File '" << filename.str( ) << "' could not be opened!"
                << std::endl;
      return false;
    }

    // std::cout << "Printing '" << filename.str( ) << "' ... ";
    // std::cout.flush( );

    std::vector< std::string > strings
      = { ( *element_labels )[ i ], "part", "tetra4" };

    for ( auto & line : strings ) {
      line.resize( 80, ' ' );
    }

    data_file.write( strings[ 0 ].c_str( ), 80 );
    data_file.write( strings[ 1 ].c_str( ), 80 );

    int part_number = 1;
    data_file.write(
      reinterpret_cast< const char * >( &part_number ), sizeof( int ) );

    data_file.write( strings[ 2 ].c_str( ), 80 );

    for ( lo j = 0; j < _n_elements; ++j ) {
      float value = ( *element_data )[ i ]->get( j );
      data_file.write(
        reinterpret_cast< const char * >( &value ), sizeof( float ) );
    }

    data_file.close( );

    // std::cout << "done." << std::endl;
  }

  return true;
}

bool besthea::mesh::tetrahedral_volume_mesh::print_ensight(
  const std::string & directory, const std::vector< std::string > * node_labels,
  const std::vector< linear_algebra::vector * > * node_data,
  const std::vector< std::string > * element_labels,
  const std::vector< linear_algebra::vector * > * element_data ) const {
  return print_ensight_case( directory, node_labels, element_labels )
    && print_ensight_geometry( directory )
    && print_ensight_datafiles(
      directory, node_labels, node_data, element_labels, element_data );
}
