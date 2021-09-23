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

#include "besthea/tetrahedral_spacetime_mesh.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

besthea::mesh::tetrahedral_spacetime_mesh::tetrahedral_spacetime_mesh(
  const besthea::mesh::spacetime_tensor_mesh & stmesh )
  : _n_nodes( stmesh.get_n_nodes( ) ),
    _n_elements( 3 * stmesh.get_n_elements( ) ) {
  stmesh.get_nodes( _nodes );
  _elements.reserve( _n_elements );
  _normals.reserve( 3 * _n_elements );

  // based on http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.35.4908

  int map[ 6 ][ 6 ]
    = { { 0, 1, 2, 3, 4, 5 }, { 1, 2, 0, 4, 5, 3 }, { 2, 0, 1, 5, 3, 4 },
        { 3, 5, 4, 0, 2, 1 }, { 4, 3, 5, 1, 0, 2 }, { 5, 4, 3, 2, 1, 0 } };

  besthea::linear_algebra::indices< 6 > e;
  besthea::linear_algebra::coordinates< 3 > n;
  for ( lo i = 0; i < stmesh.get_n_elements( ); ++i ) {
    stmesh.get_element( i, e );
    stmesh.get_spatial_normal_using_spatial_element_index(
      stmesh.get_space_element_index( i ), n );

    int * cmap = map[ std::min_element( e.begin( ), e.end( ) ) - e.begin( ) ];

    if ( std::min( e[ cmap[ 1 ] ], e[ cmap[ 5 ] ] )
      < std::min( e[ cmap[ 2 ] ], e[ cmap[ 4 ] ] ) ) {
      // first
      _elements.push_back( e[ cmap[ 0 ] ] );
      _elements.push_back( e[ cmap[ 1 ] ] );
      _elements.push_back( e[ cmap[ 2 ] ] );
      _elements.push_back( e[ cmap[ 5 ] ] );
      // second
      _elements.push_back( e[ cmap[ 0 ] ] );
      _elements.push_back( e[ cmap[ 1 ] ] );
      _elements.push_back( e[ cmap[ 5 ] ] );
      _elements.push_back( e[ cmap[ 4 ] ] );
    } else {
      // first
      _elements.push_back( e[ cmap[ 0 ] ] );
      _elements.push_back( e[ cmap[ 1 ] ] );
      _elements.push_back( e[ cmap[ 2 ] ] );
      _elements.push_back( e[ cmap[ 4 ] ] );
      // second
      _elements.push_back( e[ cmap[ 0 ] ] );
      _elements.push_back( e[ cmap[ 4 ] ] );
      _elements.push_back( e[ cmap[ 2 ] ] );
      _elements.push_back( e[ cmap[ 5 ] ] );
    }
    // third
    _elements.push_back( e[ cmap[ 0 ] ] );
    _elements.push_back( e[ cmap[ 4 ] ] );
    _elements.push_back( e[ cmap[ 5 ] ] );
    _elements.push_back( e[ cmap[ 3 ] ] );

    // copy spatial normals
    for ( int j = 0; j < 3; ++j ) {
      _normals.push_back( n[ 0 ] );
      _normals.push_back( n[ 1 ] );
      _normals.push_back( n[ 2 ] );
    }
  }

  init_areas( );
  init_edges( );
  init_node_to_elements( );
}

besthea::mesh::tetrahedral_spacetime_mesh::tetrahedral_spacetime_mesh(
  const std::vector< sc > & nodes, const std::vector< lo > & elements,
  const std::vector< sc > & normals )
  : _n_nodes( nodes.size( ) / 4 ),
    _nodes( nodes ),
    _n_elements( elements.size( ) / 4 ),
    _elements( elements ),
    _normals( normals ) {
  init_areas( );
  init_edges( );
  init_node_to_elements( );
}

besthea::mesh::tetrahedral_spacetime_mesh::~tetrahedral_spacetime_mesh( ) {
}

void besthea::mesh::tetrahedral_spacetime_mesh::print_info( ) const {
  std::cout << "besthea::mesh::tetrahedral_spacetime_mesh" << std::endl;
  std::cout << "  elements: " << _n_elements << ", nodes: " << _n_nodes
            << ", edges: " << _n_edges << std::endl;
}
/*
bool besthea::mesh::tetrahedral_spacetime_mesh::load(
  const std::string & file ) {
  std::ifstream filestream( file.c_str( ) );

  if ( !filestream.is_open( ) ) {
    std::cerr << "File could not be opened!" << std::endl;
    return false;
  }

  lo dummy;

  filestream >> dummy;  // dimension (4)
  filestream >> dummy;  // nodes per element (4)
  filestream >> _n_nodes;

  _nodes.resize( 4 * _n_nodes );

  for ( lo i = 0; i < _n_nodes; ++i ) {
    filestream >> _nodes[ 4 * i ];
    filestream >> _nodes[ 4 * i + 1 ];
    filestream >> _nodes[ 4 * i + 2 ];
    filestream >> _nodes[ 4 * i + 3 ];
  }

  std::string line;
  std::stringstream linestream;

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
  init_node_to_elements( );

  return true;
}
*/
void besthea::mesh::tetrahedral_spacetime_mesh::scale_in_space( sc factor ) {
  linear_algebra::coordinates< 3 > centroid;
  get_spatial_centroid( centroid );
  for ( lo i_node = 0; i_node < _n_nodes; ++i_node ) {
    _nodes[ 4 * i_node ]
      = ( _nodes[ 4 * i_node ] - centroid[ 0 ] ) * factor + centroid[ 0 ];
    _nodes[ 4 * i_node + 1 ]
      = ( _nodes[ 4 * i_node + 1 ] - centroid[ 1 ] ) * factor + centroid[ 1 ];
    _nodes[ 4 * i_node + 2 ]
      = ( _nodes[ 4 * i_node + 2 ] - centroid[ 2 ] ) * factor + centroid[ 2 ];
  }
  init_areas( );
}

void besthea::mesh::tetrahedral_spacetime_mesh::get_spatial_centroid(
  linear_algebra::coordinates< 3 > & centroid ) {
  linear_algebra::coordinates< 4 > x;
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

void besthea::mesh::tetrahedral_spacetime_mesh::refine( int level ) {
  if ( level < 1 )
    return;

  lo new_n_nodes, new_n_elements;
  linear_algebra::coordinates< 4 > x1, x2;
  linear_algebra::coordinates< 3 > n;
  linear_algebra::indices< 2 > edge;
  linear_algebra::indices< 4 > element;
  linear_algebra::indices< 6 > edges;
  lo node1, node2, node3, node4, node12, node13, node14, node23, node24, node34;

  for ( int l = 0; l < level; ++l ) {
    // allocate new arrays
    new_n_nodes = _n_nodes + _n_edges;
    new_n_elements = 8 * _n_elements;
    std::vector< sc > new_nodes( _nodes );
    new_nodes.resize( 4 * new_n_nodes );
    std::vector< lo > new_elements;
    new_elements.resize( 6 * new_n_elements );
    std::vector< sc > new_normals;
    new_normals.resize( 3 * new_n_elements );

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
      new_nodes[ 4 * ( _n_nodes + i ) ] = ( x1[ 0 ] + x2[ 0 ] ) / 2.0;
      new_nodes[ 4 * ( _n_nodes + i ) + 1 ] = ( x1[ 1 ] + x2[ 1 ] ) / 2.0;
      new_nodes[ 4 * ( _n_nodes + i ) + 2 ] = ( x1[ 2 ] + x2[ 2 ] ) / 2.0;
      new_nodes[ 4 * ( _n_nodes + i ) + 3 ] = ( x1[ 3 ] + x2[ 3 ] ) / 2.0;
    }

    // loop through old elements to define new elements as in Bey, J. Computing
    //   (1995) 55: 355. https://doi.org/10.1007/BF02238487
#pragma omp parallel for private( element, edges, n, node1, node2, node3, \
  node4, node12, node13, node14, node23, node24, node34 )
    for ( lo i = 0; i < _n_elements; ++i ) {
      get_element( i, element );
      get_edges( i, edges );
      get_spatial_normal( i, n );

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

      // copy spatial normals
      for ( int j = 0; j < 8; ++j ) {
        new_normals[ 24 * i + 3 * j ] = n[ 0 ];
        new_normals[ 24 * i + 3 * j + 1 ] = n[ 1 ];
        new_normals[ 24 * i + 3 * j + 2 ] = n[ 2 ];
      }
    }

    // update the mesh
    _n_nodes = new_n_nodes;
    _n_elements = new_n_elements;
    _nodes.swap( new_nodes );
    _elements.swap( new_elements );
    _normals.swap( new_normals );

    init_edges( );
  }

  init_areas( );
  init_node_to_elements( );
}

void besthea::mesh::tetrahedral_spacetime_mesh::init_areas( ) {
  _areas.resize( _n_elements );

  linear_algebra::coordinates< 4 > x21;
  linear_algebra::coordinates< 4 > x31;
  linear_algebra::coordinates< 4 > x41;
  sc dot11, dot12, dot13, dot22, dot23, dot33, det;

  for ( lo i_elem = 0; i_elem < _n_elements; ++i_elem ) {
    x21[ 0 ] = _nodes[ 4 * _elements[ 4 * i_elem + 1 ] ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] ];
    x21[ 1 ] = _nodes[ 4 * _elements[ 4 * i_elem + 1 ] + 1 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 1 ];
    x21[ 2 ] = _nodes[ 4 * _elements[ 4 * i_elem + 1 ] + 2 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 2 ];
    x21[ 3 ] = _nodes[ 4 * _elements[ 4 * i_elem + 1 ] + 3 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 3 ];

    x31[ 0 ] = _nodes[ 4 * _elements[ 4 * i_elem + 2 ] ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] ];
    x31[ 1 ] = _nodes[ 4 * _elements[ 4 * i_elem + 2 ] + 1 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 1 ];
    x31[ 2 ] = _nodes[ 4 * _elements[ 4 * i_elem + 2 ] + 2 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 2 ];
    x31[ 3 ] = _nodes[ 4 * _elements[ 4 * i_elem + 2 ] + 3 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 3 ];

    x41[ 0 ] = _nodes[ 4 * _elements[ 4 * i_elem + 3 ] ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] ];
    x41[ 1 ] = _nodes[ 4 * _elements[ 4 * i_elem + 3 ] + 1 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 1 ];
    x41[ 2 ] = _nodes[ 4 * _elements[ 4 * i_elem + 3 ] + 2 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 2 ];
    x41[ 3 ] = _nodes[ 4 * _elements[ 4 * i_elem + 3 ] + 3 ]
      - _nodes[ 4 * _elements[ 4 * i_elem ] + 3 ];

    // volume of the 3d parallelogram in 4d
    dot11 = x21[ 0 ] * x21[ 0 ] + x21[ 1 ] * x21[ 1 ] + x21[ 2 ] * x21[ 2 ]
      + x21[ 3 ] * x21[ 3 ];
    dot12 = x21[ 0 ] * x31[ 0 ] + x21[ 1 ] * x31[ 1 ] + x21[ 2 ] * x31[ 2 ]
      + x21[ 3 ] * x31[ 3 ];
    dot13 = x21[ 0 ] * x41[ 0 ] + x21[ 1 ] * x41[ 1 ] + x21[ 2 ] * x41[ 2 ]
      + x21[ 3 ] * x41[ 3 ];
    dot22 = x31[ 0 ] * x31[ 0 ] + x31[ 1 ] * x31[ 1 ] + x31[ 2 ] * x31[ 2 ]
      + x31[ 3 ] * x31[ 3 ];
    dot23 = x31[ 0 ] * x41[ 0 ] + x31[ 1 ] * x41[ 1 ] + x31[ 2 ] * x41[ 2 ]
      + x31[ 3 ] * x41[ 3 ];
    dot33 = x41[ 0 ] * x41[ 0 ] + x41[ 1 ] * x41[ 1 ] + x41[ 2 ] * x41[ 2 ]
      + x41[ 3 ] * x41[ 3 ];

    det = dot11 * dot22 * dot33 + 2.0 * dot12 * dot23 * dot13
      - dot13 * dot13 * dot22 - dot12 * dot12 * dot33 - dot11 * dot23 * dot23;

    _areas[ i_elem ] = std::sqrt( det ) / 6.0;
  }
}

void besthea::mesh::tetrahedral_spacetime_mesh::init_edges( ) {
  // allocate class variables
  _element_to_edges.clear( );
  _element_to_edges.resize( 6 * _n_elements );

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

void besthea::mesh::tetrahedral_spacetime_mesh::init_node_to_elements( ) {
  _node_to_elements.clear( );
  _node_to_elements.resize( _n_nodes );

  linear_algebra::indices< 4 > element;
  for ( lo i_elem = 0; i_elem < _n_elements; ++i_elem ) {
    get_element( i_elem, element );
    _node_to_elements[ element[ 0 ] ].push_back( i_elem );
    _node_to_elements[ element[ 1 ] ].push_back( i_elem );
    _node_to_elements[ element[ 2 ] ].push_back( i_elem );
    _node_to_elements[ element[ 3 ] ].push_back( i_elem );
  }
}
/*
void besthea::mesh::tetrahedral_spacetime_mesh::init_faces( ) {
  // allocate aux. variables
  linear_algebra::indices< 6 > element_edges;
  linear_algebra::indices< 4 > element;
  std::vector< std::vector< lo > > local_faces;
  // highest index does not have any neighbours with higher index
  local_faces.resize( _n_edges );
  std::vector< std::vector< lo > > offsets;
  offsets.resize(
    _n_edges );  // offsets of nodes in edges vector defining a face
  // edge + certain nodes = face:
  lo face2edge[] = { 0, 2, 4, 3, 4, 5, 1, 2, 5, 0, 1,
    3 };  // [0,2,4] - edges of 1st face, [ 3,4,5] - 2nd, etc
  lo edge2node[] = { 3, 1, 0, 3, 2, 1, 3, 2, 0, 2, 1,
    3 };  // nodes associated with given edge forming a face
  linear_algebra::indices< 3 > face_edges;
  std::vector< std::pair< lo, lo > > element_to_faces_tmp;
  element_to_faces_tmp.resize( 4 * _n_elements );

  linear_algebra::indices< 2 > edge;

  typedef typename std::vector< lo >::iterator it;
  it node_position;

  // iterate over elements
  for ( lo i = 0; i < _n_elements; ++i ) {
    get_edges( i, element_edges );
    get_element( i, element );

    // iterate over its faces
    for ( lo j = 0; j < 4; ++j ) {
      face_edges[ 0 ] = element_edges[ face2edge[ j * 3 ] ];
      face_edges[ 1 ] = element_edges[ face2edge[ j * 3 + 1 ] ];
      face_edges[ 2 ] = element_edges[ face2edge[ j * 3 + 2 ] ];

      // find the edge with smallest global index
      lo min_val = _n_edges;
      lo min_idx = 0;
      for ( lo k = 0; k < 3; ++k ) {
        if ( face_edges[ k ] < min_val ) {
          min_idx = k;
          min_val = face_edges[ k ];
        }
      }

      // add the couple min_val (edge) + element[edge2node[j*3+k]] (node) to the
      // list of faces if it is not yet present
      node_position = std::find< it, lo >( local_faces[ min_val ].begin( ),
        local_faces[ min_val ].end( ),
        element[ edge2node[ j * 3 + min_idx ] ] );
      // not found
      if ( node_position == local_faces[ min_val ].end( ) ) {
        local_faces[ min_val ].push_back(
          element[ edge2node[ j * 3 + min_idx ] ] );
      }
      node_position = std::find< it, lo >( local_faces[ min_val ].begin( ),
        local_faces[ min_val ].end( ),
        element[ edge2node[ j * 3 + min_idx ] ] );

      // store temporary mapping from element to edge-node pair
      element_to_faces_tmp[ 4 * i + j ] = std::pair< lo, lo >( min_val,
        std::distance( local_faces[ min_val ].begin( ), node_position ) );
    }
  }

  lo offset = 0;
  for ( lo i = 0; i < _n_edges; ++i ) {
    if ( local_faces[ i ].size( ) != 0 ) {
      for ( auto it_face = local_faces[ i ].begin( );
            it_face != local_faces[ i ].end( ); ++it_face ) {
        offsets[ i ].push_back( offset );
        offset++;
        get_edge( i, edge );
        _faces.push_back( edge[ 0 ] );
        _faces.push_back( edge[ 1 ] );
        _faces.push_back( *it_face );
      }
    }
  }
  _faces.resize( offset );
  _n_faces = _faces.size( );

  _element_to_faces.resize( 4 * _n_elements );
  for ( lo i = 0; i < _n_elements; ++i ) {
    for ( lo j = 0; j < 4; ++j ) {
      std::pair< lo, lo > idx = element_to_faces_tmp[ 4 * i + j ];
      // std::cout << idx.first << " " << idx.second << std::endl;
      lo current_offset = offsets[ idx.first ][ idx.second ];
      // std::cout << offset << std::endl;
      _element_to_faces[ 4 * i + j ] = current_offset;
    }
  }

  //  for ( lo i = 0; i < _n_elements; ++i ) {
  //    for ( lo j = 0; j < 4; ++j ) {
  //      std::cout << _element_to_faces[ 4 * i + j ] << " ";
  //    }
  //    std::cout << std::endl;
  //  }
  //
  //  for ( lo i = 0; i < _n_faces; ++i ) {
  //    for ( lo j = 0; j < 3; ++j ) {
  //      std::cout << _faces[ 3 * i + j ] << " ";
  //    }
  //    std::cout << std::endl;
  //  }
}
*/
