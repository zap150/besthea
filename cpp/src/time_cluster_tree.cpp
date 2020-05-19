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

#include "besthea/time_cluster_tree.h"

#include <fstream> //for ofstream and ifstream

besthea::mesh::time_cluster_tree::time_cluster_tree(
  const temporal_mesh & mesh, lo levels, lo n_min_elems )
  : _mesh( mesh ),
    _levels( levels ),
    _real_max_levels( 0 ),
    _n_min_elems( n_min_elems ),
    _n_max_elems_leaf( 0 ),
    _paddings( _levels, -1.0 ) {
  sc center = ( _mesh.get_end( ) + _mesh.get_start( ) ) / 2;
  sc half_size = ( _mesh.get_end( ) - _mesh.get_start( ) ) / 2;

  _root = new time_cluster(
    center, half_size, _mesh.get_n_elements( ), nullptr, 0, _mesh );

  for ( lo i = 0; i < _mesh.get_n_elements( ); ++i ) {
    _root->add_element( i );
  }

  this->build_tree( *_root, 1 );
  this->compute_padding( *_root );
  _levels = std::min( _levels, _real_max_levels );

  _paddings.resize( _levels );
  _paddings.shrink_to_fit( );
  collect_leaves( *_root );
}

besthea::mesh::time_cluster_tree::time_cluster_tree( 
  const temporal_mesh & mesh, const std::string filename )
  : _mesh( mesh ),
    _levels( 0 ),
    _real_max_levels( 0 ),
    _n_min_elems( -1 ),
    _n_max_elems_leaf( -1 ) {
  // load tree structure from file
  std::vector< char > tree_vector = load_tree_structure( filename );
  // create tree structure from vector 
  if ( ( tree_vector.size( ) > 0 ) && ( tree_vector[ 0 ] != 0 ) ) {
    sc center = 0.5 * ( _mesh.get_end( ) + _mesh.get_start( ) );
    sc half_size = 0.5 * ( _mesh.get_end( ) - _mesh.get_start( ) );
    _root = new time_cluster(
      center, half_size, 0, nullptr, 0, _mesh );
    _real_max_levels = 1;
    if ( tree_vector[ 0 ] == 1 ) {
      lou position = 1;
      vector_2_tree( tree_vector, *_root, position );
    }
  } else {
    _root = nullptr;
  }
  _levels = _real_max_levels;
  _paddings = std::move( std::vector< sc > (_levels, -1.0) );
  collect_leaves( *_root );
}

std::vector< char > besthea::mesh::time_cluster_tree::
  compute_tree_structure( ) const {
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

void besthea::mesh::time_cluster_tree::print_tree_structure( 
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

std::vector< char > besthea::mesh::time_cluster_tree::load_tree_structure( 
  const std::string filename ) const {
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

void besthea::mesh::time_cluster_tree::build_tree(
  time_cluster & root, lo level ) {
  // stop recursion if maximum number of levels is reached or root contains less
  // than _n_min_elems elements
  if ( level > _levels - 1 || root.get_n_elements( ) < _n_min_elems ) {
    root.set_n_children( 0 );
    if ( root.get_n_elements( ) > _n_max_elems_leaf ) {
      _n_max_elems_leaf = root.get_n_elements( );
    }
    if ( level > _real_max_levels ) {
      _real_max_levels = level;
    }
    return;
  }

  sc center = root.get_center( );
  sc half_size = root.get_half_size( );

  sc el_centroid;

  lo root_n_elems = root.get_n_elements( );
  lo n_left = 0;
  lo n_right = 0;
  lo elem_idx = 0;

  // count the number of elements in each subcluster
  for ( lo i = 0; i < root_n_elems; ++i ) {
    elem_idx = root.get_element( i );
    el_centroid = _mesh.get_centroid( elem_idx );
    if ( el_centroid <= center ) {
      ++n_left;
    } else {
      ++n_right;
    }
  }

  time_cluster * left_cluster = new time_cluster(
    center - half_size / 2, half_size / 2, n_left, &root, level, _mesh );

  time_cluster * right_cluster = new time_cluster(
    center + half_size / 2, half_size / 2, n_right, &root, level, _mesh );

  // add elements to each subcluster
  for ( lo i = 0; i < root_n_elems; ++i ) {
    elem_idx = root.get_element( i );
    el_centroid = _mesh.get_centroid( elem_idx );
    if ( el_centroid <= center ) {
      left_cluster->add_element( elem_idx );
    } else {
      right_cluster->add_element( elem_idx );
    }
  }

  root.set_n_children( 2 );
  root.add_child( left_cluster );
  this->build_tree( *left_cluster, level + 1 );
  root.add_child( right_cluster );
  this->build_tree( *right_cluster, level + 1 );
}

void besthea::mesh::time_cluster_tree::tree_2_vector( const time_cluster & root, 
  std::vector<char> & tree_vector ) const {
  // get the children of root and determine if they are leaves or not 
  // WARNING: it is assumed that root always has two children; this assumption
  // is reasonable if the method is called for a non-leaf cluster in the tree,
  // since the tree is a full binary tree by construction (in build tree)
  const std::vector< time_cluster * > * children = root.get_children( );
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

void besthea::mesh::time_cluster_tree::vector_2_tree( 
  const std::vector<char> & tree_vector, time_cluster & root, 
  lou & position ) {
  // get the cluster data of root
  lo level = root.get_level( );
  sc center = root.get_center( );
  sc half_size = root.get_half_size( );
  // determine status of the children of root and create them accordingly
  char left_child_status = tree_vector[ position++ ];
  char right_child_status = tree_vector[ position++ ];
  lo child_counter = 0;
  time_cluster * left_cluster = nullptr;
  time_cluster * right_cluster = nullptr;
  if ( left_child_status > 0 ) {
    child_counter++;
    left_cluster = new time_cluster(
    center - half_size / 2.0, half_size / 2.0, 0, &root, level + 1, _mesh );
  }
  if ( right_child_status > 0 ) {
    child_counter++;
    right_cluster = new time_cluster(
    center + half_size / 2.0, half_size / 2.0, 0, &root, level + 1, _mesh );
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
    if ( level + 2 > _real_max_levels ) {
      _real_max_levels = level + 2;
    }
  }
  if ( right_child_status == 1 ) {
    vector_2_tree( tree_vector, *right_cluster, position );
  } else {
    if ( level + 2 > _real_max_levels ) {
      _real_max_levels = level + 2;
    }
  }
}

sc besthea::mesh::time_cluster_tree::compute_padding( time_cluster & root ) {
  std::vector< time_cluster * > * children = root.get_children( );
  sc padding = -1.0;
  sc tmp_padding;

  if ( children != nullptr ) {
    // for non-leaf clusters, find the largest padding of its descendants
    for ( auto it = children->begin( ); it != children->end( ); ++it ) {
      tmp_padding = this->compute_padding( **it );
      if ( tmp_padding > padding ) {
        padding = tmp_padding;
      }
    }
    if ( padding > _paddings[ root.get_level( ) ] ) {
      _paddings[ root.get_level( ) ] = padding;
    }
  } else {
    // for leaf clusters, compute padding directly
    padding = root.compute_padding( );
    if ( padding > _paddings[ root.get_level( ) ] ) {
      _paddings[ root.get_level( ) ] = padding;
    }
  }

  return padding;
}

void besthea::mesh::time_cluster_tree::collect_leaves( time_cluster & root ) {
  if ( root.get_n_children( ) == 0 ) {
    _leaves.push_back( &root );
  } else {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      collect_leaves( **it );
    }
  }
}