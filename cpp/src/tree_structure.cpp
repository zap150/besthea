/*
 * Copyright 2020, VSB - Technical University of Ostrava and Graz University of
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

#include "besthea/tree_structure.h"

#include <fstream> //for ofstream and ifstream

template <>
void besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  vector_2_tree( const std::vector<char> & tree_vector, 
  besthea::mesh::scheduling_time_cluster & root, lou & position ) {
  using scheduling_time_cluster = besthea::mesh::scheduling_time_cluster;
  // get the cluster data of root
  lo level = root.get_level( );
  sc center = root.get_center( );
  sc half_size = root.get_half_size( );
  // determine status of the children of root and create them accordingly
  char left_child_status = tree_vector[ position++ ];
  char right_child_status = tree_vector[ position++ ];
  lo child_counter = 0;
  scheduling_time_cluster * left_cluster = nullptr;
  scheduling_time_cluster * right_cluster = nullptr;
  if ( left_child_status > 0 ) {
    child_counter++;
    left_cluster = new scheduling_time_cluster(
    center - half_size / 2.0, half_size / 2.0, &root, level + 1 );
  }
  if ( right_child_status > 0 ) {
    child_counter++;
    right_cluster = new scheduling_time_cluster(
    center + half_size / 2.0, half_size / 2.0, &root, level + 1 );
  }
  // add the newly created clusters to the root
  root.set_n_children( child_counter );
  if ( left_cluster != nullptr ) {
    root.add_child( left_cluster );
  }
  if ( right_cluster != nullptr ) {
    root.add_child( right_cluster );
  }
  // call the routine recursively for non-leaf children or update the depth of
  // the cluster tree if a leaf is encountered
  if ( left_child_status == 1 ) {
    vector_2_tree( tree_vector, *left_cluster, position );
  } else {
    if ( level + 2 > _levels ) {
      _levels = level + 2;
    }
  }
  if ( right_child_status == 1 ) {
    vector_2_tree( tree_vector, *right_cluster, position );
  } else {
    if ( level + 2 > _levels ) {
      _levels = level + 2;
    }
  }
}

template < class cluster_type >
void besthea::mesh::tree_structure< cluster_type >::vector_2_tree( 
  const std::vector<char> & tree_vector, cluster_type & root, 
  lou & position ) {
  std::cout << "vector_2_tree: NOT IMPLEMENTED!" << std::endl;
}


template < class cluster_type >
besthea::mesh::tree_structure< cluster_type >::tree_structure( 
  const std::string filename )
  : _levels( 0 ) {
    std::cout << "Constructor NOT IMPLEMENTED!" << std::endl;
  // // load tree structure from file
  // std::vector< char > tree_vector = load_tree_structure( filename );
  // // create tree structure from vector 
  // if ( ( tree_vector.size( ) > 0 ) && ( tree_vector[ 0 ] != 0 ) ) {
  //   sc center = 0.5 * ( _mesh.get_end( ) + _mesh.get_start( ) );
  //   sc half_size = 0.5 * ( _mesh.get_end( ) - _mesh.get_start( ) );
  //   _root = new time_cluster(
  //     center, half_size, 0, nullptr, 0, _mesh );
  //   _real_max_levels = 1;
  //   if ( tree_vector[ 0 ] == 1 ) {
  //     lou position = 1;
  //     vector_2_tree( tree_vector, *_root, position );
  //   }
  // } else {
  //   _root = nullptr;
  // }
  // _levels = _real_max_levels;
  // _paddings = std::move( std::vector< sc > (_levels, -1.0) );
  // collect_leaves( *_root );
}

template <>
besthea::mesh::tree_structure< besthea::mesh::scheduling_time_cluster >::
  tree_structure( const std::string filename )
  : _levels( 0 ) {
  // load tree structure from file
  std::vector< char > tree_vector = load_tree_structure( filename );
  // create tree structure from vector 
  if ( ( tree_vector.size( ) > 0 ) && ( tree_vector[ 0 ] != 0 ) ) {
    sc center = 0.5; //TODO: correct this
    sc half_size = 0.5; //TODO: correct this
    _root = new scheduling_time_cluster( center, half_size, nullptr, 0 );
      _levels = 1;
    if ( tree_vector[ 0 ] == 1 ) {
      lou position = 1;
      vector_2_tree( tree_vector, *_root, position );
    }
  } else {
    _root = nullptr;
  }
  collect_leaves( *_root );
}

template < class cluster_type >
std::vector< char > besthea::mesh::tree_structure< cluster_type >::
  tree_structure::compute_tree_structure( ) const {
  std::vector< char > tree_vector;
  if ( _root == nullptr ) {
    tree_vector.push_back( 0 );
  }
  else if ( _root->get_n_children( ) == 0 ) {
    tree_vector.push_back( 2 );
  }
  else {
    tree_vector.push_back( 1 );
    tree_2_vector( *_root, tree_vector );
  }
  return tree_vector;
}

template < class cluster_type >
void besthea::mesh::tree_structure< cluster_type >::print_tree_structure( 
  const std::string filename ) const
{
  const std::vector< char > tree_vector = compute_tree_structure( );
  std::ofstream file_out( filename.c_str( ), std::ios::binary );
  if ( file_out.is_open( ) )
  {
    lou n_chars = tree_vector.size( );
    const char * tree_vector_data = tree_vector.data( );
    file_out.write( tree_vector_data, n_chars );
    file_out.close();
  } else {
    std::cout << "Error. Could not open the output file for printing the tree \
                  structure." << std::endl;
  }
}

template < class cluster_type >
std::vector< char > besthea::mesh::tree_structure< cluster_type >::
  load_tree_structure( const std::string filename ) const {
  std::vector< char > tree_vector;
  std::ifstream read_file;
  read_file.open( filename.c_str( ) );
  if ( read_file.is_open( ) ) {
    //determine the number of chars to be received
    read_file.seekg( 0, read_file.end );
    lo n_chars = read_file.tellg( );
    read_file.seekg( 0 );
    tree_vector.resize( n_chars );
    //load all chars from file
    read_file.read( tree_vector.data( ), n_chars );
    read_file.close();
  } else {
    std::cout << "Error. Could not open the input file for reading the tree \
                  structure." << std::endl;
  }
  return tree_vector;
}

template < class cluster_type >
void besthea::mesh::tree_structure< cluster_type >::tree_2_vector( 
  const cluster_type & root, std::vector<char> & tree_vector ) const {
  // get the children of root and determine if they are leaves or not 
  // WARNING: it is assumed that root always has two children; this assumption
  // is reasonable if the method is called for a non-leaf cluster in the tree,
  // since the tree is a full binary tree by construction (in build tree)
  const std::vector< cluster_type * > * children = root.get_children( );
  char left_child_status = 
    ( ( *children )[ 0 ]->get_n_children( ) > 0 ) ? 1 : 2;
  char right_child_status = 
    ( ( *children )[ 1 ]->get_n_children( ) > 0 ) ? 1 : 2;
  tree_vector.push_back( left_child_status );
  tree_vector.push_back( right_child_status );
  if ( left_child_status == 1 ) {
    tree_2_vector( *( *children )[ 0 ], tree_vector );
  } 
  if ( right_child_status == 1 ) {
    tree_2_vector( *( *children )[ 1 ], tree_vector );
  }
}

template < class cluster_type >
void besthea::mesh::tree_structure< cluster_type >::collect_leaves( 
  cluster_type & root ) {
  if ( root.get_n_children( ) == 0 ) {
    _leaves.push_back( &root );
  } else {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      collect_leaves( **it );
    }
  }
}

template class besthea::mesh::tree_structure<
  besthea::mesh::scheduling_time_cluster >;