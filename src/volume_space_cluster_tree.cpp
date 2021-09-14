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

#include "besthea/volume_space_cluster_tree.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>

besthea::mesh::volume_space_cluster_tree::volume_space_cluster_tree(
  const tetrahedral_volume_mesh & mesh, lo levels, lo n_min_elems,
  bool print_warnings )
  : _mesh( mesh ),
    _max_n_levels( levels ),
    _real_n_levels( 0 ),
    _n_min_elems( n_min_elems ),
    _n_max_elems_leaf( 0 ),
    _non_empty_clusters_per_level( _max_n_levels ),
    _paddings( _max_n_levels, 0.0 ),
    _n_nonempty_clusters( 0 ) {
  sc xmin, xmax, ymin, ymax, zmin, zmax;
  compute_cubic_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );
  _bounding_box_size.resize( 3 );
  _bounding_box_size[ 0 ] = ( xmax - xmin ) / 2.0;
  _bounding_box_size[ 1 ] = ( ymax - ymin ) / 2.0;
  _bounding_box_size[ 2 ] = ( zmax - zmin ) / 2.0;

  vector_type center
    = { ( xmin + xmax ) / 2.0, ( ymin + ymax ) / 2.0, ( zmin + zmax ) / 2.0 };
  vector_type half_sizes = { std::abs( xmax - xmin ) / 2.0,
    std::abs( ymax - ymin ) / 2.0, std::abs( zmax - zmin ) / 2.0 };

  _idx_2_coord = { { 1, 1, 1 }, { 0, 1, 1 }, { 0, 0, 1 }, { 1, 0, 1 },
    { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 0 }, { 1, 0, 0 } };

  // create a root cluster and call the recursive tree building routine
  std::vector< slou > coordinates = { 0, 0, 0, 0 };
  _root = new volume_space_cluster( center, half_sizes, _mesh.get_n_elements( ),
    nullptr, 0, 0, coordinates, _mesh );
  _coord_2_cluster.insert(
    std::pair< std::vector< slou >, volume_space_cluster * >(
      coordinates, _root ) );

  for ( lo i = 0; i < _mesh.get_n_elements( ); ++i ) {
    _root->add_element( i );
  }

  _non_empty_clusters_per_level[ 0 ].push_back( _root );
  ++_n_nonempty_clusters;
  this->build_tree( *_root );
  this->compute_padding( *_root );
  _paddings.shrink_to_fit( );

  if ( print_warnings ) {
    // check whether there are clusters which are padded a lot (more than 50 %)
    // and print a warning if necessary
    bool extensive_padding = false;
    sc current_cluster_half_size = half_sizes[ 0 ];
    lo i = 0;
    while ( extensive_padding == false && i < _real_n_levels ) {
      extensive_padding = ( current_cluster_half_size / 2.0 < _paddings[ i ] );
      if ( extensive_padding ) {
        std::cout << "Warning: Level " << i << ": padding = " << _paddings[ i ]
                  << ", cluster half size = " << current_cluster_half_size
                  << std::endl;
      }
      current_cluster_half_size /= 2.0;
      ++i;
    }
    if ( extensive_padding ) {
      std::cout << "Warning: Extensive padding detected in construction of "
                   "volume space tree!"
                << std::endl;
    }
  }

  // collect all clusters without descendants
  collect_leaf_descendants( *_root, _leaves );

  initialize_levelwise_cluster_grids( );
}

void besthea::mesh::volume_space_cluster_tree::build_tree(
  volume_space_cluster & root ) {
  // stop recursion if maximum number of tree levels is reached
  if ( root.get_level( ) == _max_n_levels - 1
    || root.get_n_elements( ) < _n_min_elems ) {
    root.set_n_children( 0 );
    root.compute_node_mapping( );

    if ( root.get_n_elements( ) > _n_max_elems_leaf ) {
      _n_max_elems_leaf = root.get_n_elements( );
    }
    if ( root.get_level( ) + 1 > _real_n_levels ) {
      _real_n_levels = root.get_level( ) + 1;
    }

    return;
  }

  // allocate children's and temporary data
  vector_type center( 3 );
  vector_type half_size( 3 );
  linear_algebra::coordinates< 3 > el_centroid;
  root.get_center( center );
  root.get_half_size( half_size );
  lo elem_idx = 0;
  lo oct_sizes[ 8 ] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  volume_space_cluster * clusters[ 8 ];

  // first count the number of elements in octants for data preallocation
  std::vector< lo > clusters_of_elements( root.get_n_elements( ) );
  for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
    elem_idx = root.get_element( i );

    _mesh.get_centroid( elem_idx, el_centroid );

    if ( el_centroid[ 0 ] >= center( 0 ) && el_centroid[ 1 ] >= center( 1 )
      && el_centroid[ 2 ] >= center( 2 ) ) {
      ++oct_sizes[ 0 ];
      clusters_of_elements[ i ] = 0;
    } else if ( el_centroid[ 0 ] < center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] >= center( 2 ) ) {
      ++oct_sizes[ 1 ];
      clusters_of_elements[ i ] = 1;
    } else if ( el_centroid[ 0 ] < center( 0 ) && el_centroid[ 1 ] < center( 1 )
      && el_centroid[ 2 ] >= center( 2 ) ) {
      ++oct_sizes[ 2 ];
      clusters_of_elements[ i ] = 2;
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] < center( 1 ) && el_centroid[ 2 ] >= center( 2 ) ) {
      ++oct_sizes[ 3 ];
      clusters_of_elements[ i ] = 3;
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      ++oct_sizes[ 4 ];
      clusters_of_elements[ i ] = 4;
    } else if ( el_centroid[ 0 ] < center( 0 )
      && el_centroid[ 1 ] >= center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      ++oct_sizes[ 5 ];
      clusters_of_elements[ i ] = 5;
    } else if ( el_centroid[ 0 ] < center( 0 ) && el_centroid[ 1 ] < center( 1 )
      && el_centroid[ 2 ] < center( 2 ) ) {
      ++oct_sizes[ 6 ];
      clusters_of_elements[ i ] = 6;
    } else if ( el_centroid[ 0 ] >= center( 0 )
      && el_centroid[ 1 ] < center( 1 ) && el_centroid[ 2 ] < center( 2 ) ) {
      ++oct_sizes[ 7 ];
      clusters_of_elements[ i ] = 7;
    }
  }

  lo n_clusters = 0;
  vector_type new_center( 3 );
  vector_type new_half_size( 3 );
  for ( short i = 0; i < 8; ++i ) {
    if ( oct_sizes[ i ] > 0 ) {
      root.compute_suboctant( i, new_center, new_half_size );
      ++n_clusters;

      slou coord_x
        = 2 * root.get_box_coordinate( )[ 1 ] + _idx_2_coord[ i ][ 0 ];
      slou coord_y
        = 2 * root.get_box_coordinate( )[ 2 ] + _idx_2_coord[ i ][ 1 ];
      slou coord_z
        = 2 * root.get_box_coordinate( )[ 3 ] + _idx_2_coord[ i ][ 2 ];

      std::vector< slou > coordinates
        = { static_cast< slou >( root.get_level( ) + 1 ), coord_x, coord_y,
            coord_z };
      clusters[ i ] = new volume_space_cluster( new_center, new_half_size,
        oct_sizes[ i ], &root, root.get_level( ) + 1, i, coordinates, _mesh );
      _non_empty_clusters_per_level[ root.get_level( ) + 1 ].push_back(
        clusters[ i ] );
      _coord_2_cluster.insert(
        std::pair< std::vector< slou >, volume_space_cluster * >(
          coordinates, clusters[ i ] ) );
      ++_n_nonempty_clusters;
    } else {
      clusters[ i ] = nullptr;
    }
  }

  // next assign elements to clusters
  for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
    clusters[ clusters_of_elements[ i ] ]->add_element( root.get_element( i ) );
  }

  // finally, add new children to cluster's children list and call the method
  // recursively on them
  root.set_n_children( n_clusters );

  for ( lo i = 0; i < 8; ++i ) {
    if ( clusters[ i ] != nullptr ) {
      root.add_child( clusters[ i ] );
      this->build_tree( *clusters[ i ] );
    }
  }

  // shrink internal data storage for child clusters
  root.shrink_children( );
}

void besthea::mesh::volume_space_cluster_tree::compute_cubic_bounding_box(
  sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax ) {
  xmin = ymin = zmin = std::numeric_limits< sc >::max( );
  xmax = ymax = zmax = std::numeric_limits< sc >::min( );

  linear_algebra::coordinates< 3 > node;
  for ( lo i = 0; i < _mesh.get_n_nodes( ); ++i ) {
    _mesh.get_node( i, node );

    if ( node[ 0 ] < xmin )
      xmin = node[ 0 ];
    if ( node[ 0 ] > xmax )
      xmax = node[ 0 ];

    if ( node[ 1 ] < ymin )
      ymin = node[ 1 ];
    if ( node[ 1 ] > ymax )
      ymax = node[ 1 ];

    if ( node[ 2 ] < zmin )
      zmin = node[ 2 ];
    if ( node[ 2 ] > zmax )
      zmax = node[ 2 ];
  }

  // turn the bounding box into a cube:
  // determine the side lengths and their maxima
  sc x_side_length = xmax - xmin;
  sc y_side_length = ymax - ymin;
  sc z_side_length = zmax - zmin;
  sc max_side_length = x_side_length;
  if ( y_side_length > max_side_length ) {
    max_side_length = y_side_length;
  }
  if ( z_side_length > max_side_length ) {
    max_side_length = z_side_length;
  }
  // adapt the bounding box if necessary in each dimension. this is done by
  // extending the box to the right
  if ( max_side_length > x_side_length ) {
    // add side difference to xmax
    xmax += max_side_length - x_side_length;
  }
  if ( max_side_length > y_side_length ) {
    // add side difference to ymax
    ymax += max_side_length - y_side_length;
  }
  if ( max_side_length > z_side_length ) {
    // add side difference to zmax
    zmax += max_side_length - z_side_length;
  }
}

sc besthea::mesh::volume_space_cluster_tree::compute_padding(
  volume_space_cluster & root ) {
  const std::vector< volume_space_cluster * > * children = root.get_children( );
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

void besthea::mesh::volume_space_cluster_tree::find_neighbors( const lo level,
  const std::vector< slou > & target_grid_coordinates, slou limit,
  std::vector< volume_space_cluster * > & neighbors ) const {
  // search for clusters in a local neighborhood of the given grid coordinates.
  // determine the search bounds first
  slou max_coordinate = (slou) ( 1 << level );
  slou i_x_low = (slou) ( (lo) target_grid_coordinates[ 0 ] - limit > 0
      ? target_grid_coordinates[ 0 ] - limit
      : 0 );
  slou i_x_up
    = (slou) ( target_grid_coordinates[ 0 ] + limit + 1 < max_coordinate
        ? target_grid_coordinates[ 0 ] + limit + 1
        : max_coordinate );
  slou i_y_low = (slou) ( (lo) target_grid_coordinates[ 1 ] - limit > 0
      ? target_grid_coordinates[ 1 ] - limit
      : 0 );
  slou i_y_up
    = (slou) ( target_grid_coordinates[ 1 ] + limit + 1 < max_coordinate
        ? target_grid_coordinates[ 1 ] + limit + 1
        : max_coordinate );
  slou i_z_low = (slou) ( (lo) target_grid_coordinates[ 2 ] - limit > 0
      ? target_grid_coordinates[ 2 ] - limit
      : 0 );
  slou i_z_up
    = (slou) ( target_grid_coordinates[ 2 ] + limit + 1 < max_coordinate
        ? target_grid_coordinates[ 2 ] + limit + 1
        : max_coordinate );
  // compute the strides to switch from cluster coordinates to indices in the
  // grid vectors
  lo x_stride = 1 << ( 2 * level );
  lo y_stride = 1 << level;
  // now go through the grid vector at the current level and add all encountered
  // clusters to two auxiliary vectors.
  std::vector< volume_space_cluster * > neighbors_coarser_level;
  for ( slou i_x = i_x_low; i_x < i_x_up; ++i_x ) {
    for ( slou i_y = i_y_low; i_y < i_y_up; ++i_y ) {
      for ( slou i_z = i_z_low; i_z < i_z_up; ++i_z ) {
        lo grid_index = x_stride * i_x + y_stride * i_y + i_z;
        volume_space_cluster * neighbor
          = _levelwise_cluster_grids[ level ][ grid_index ];
        if ( neighbor != nullptr ) {
          // add the neighbor to the appropriate list
          if ( neighbor->get_level( ) == level ) {
            // neighbors from the same level can be added directly to the list
            // of neighbors
            neighbors.push_back( neighbor );
          } else {
            // coarser neighbors are added uniquely to a separate list
            bool neighbor_is_new = true;
            auto previous_neighbor_it = neighbors_coarser_level.begin( );
            while ( neighbor_is_new
              && previous_neighbor_it != neighbors_coarser_level.end( ) ) {
              if ( *previous_neighbor_it == neighbor ) {
                neighbor_is_new = false;
              }
              ++previous_neighbor_it;
            }
            if ( neighbor_is_new ) {
              neighbors_coarser_level.push_back( neighbor );
            }
          }
        }
      }
    }
  }
  // add the coarser neighbors to the output list
  for ( auto coarse_neighbor : neighbors_coarser_level ) {
    neighbors.push_back( coarse_neighbor );
  }
}

bool besthea::mesh::volume_space_cluster_tree::print_tree(
  const std::string & directory, bool include_padding, lo level,
  std::optional< lo > suffix ) const {
  std::stringstream file;
  file << directory << "/tree.vtu";

  if ( suffix ) {
    file << '.' << std::setw( 4 ) << std::setfill( '0' ) << suffix.value( );
  }

  std::ofstream file_vtu( file.str( ).c_str( ) );

  file_vtu.setf( std::ios::showpoint | std::ios::scientific );
  file_vtu.precision( 6 );

  if ( !file_vtu.is_open( ) ) {
    std::cout << "File could not be opened!" << std::endl;
    return false;
  }

  lo n_clusters = 0;
  if ( level == -1 ) {
    n_clusters = _n_nonempty_clusters;
  } else {
    n_clusters = _non_empty_clusters_per_level[ level ].size( );
  }

  std::cout << "Printing '" << file.str( ) << "' ... ";
  std::cout.flush( );

  file_vtu << "<?xml version=\"1.0\"?>" << std::endl;
  file_vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\">"
           << std::endl;
  file_vtu << "  <UnstructuredGrid>" << std::endl;

  file_vtu << "    <Piece NumberOfPoints=\"" << n_clusters
           << "\" NumberOfCells=\"" << 0 << "\">" << std::endl;

  file_vtu << "      <Points>" << std::endl;
  file_vtu << "        <DataArray type=\"Float32\" Name=\"points\""
              " NumberOfComponents=\"3\" format=\"ascii\">"
           << std::endl;

  vector_type center( 3 );

  for ( auto it = _non_empty_clusters_per_level.begin( );
        it != _non_empty_clusters_per_level.end( ); ++it ) {
    if ( level != -1
      && std::distance( _non_empty_clusters_per_level.begin( ), it )
        != level ) {
      continue;
    }
    for ( auto itt = ( *it ).begin( ); itt != ( *it ).end( ); ++itt ) {
      ( *itt )->get_center( center );
      file_vtu << "          " << static_cast< float >( center[ 0 ] ) << " "
               << static_cast< float >( center[ 1 ] ) << " "
               << static_cast< float >( center[ 2 ] ) << std::endl;
    }
  }

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu << "      </Points>" << std::endl;

  file_vtu << "      <Cells>" << std::endl;
  file_vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\""
              " format=\"ascii\">"
           << std::endl;

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu
    << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"
    << std::endl;

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu
    << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    << std::endl;

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu << "      </Cells>" << std::endl;

  file_vtu << "      <PointData Vectors=\"Half_sizes\">" << std::endl;

  file_vtu << "        <DataArray type=\"Float32\" Name=\"Half_sizes\" "
              "format=\"ascii\" NumberOfComponents=\"3\">"
           << std::endl;
  ;

  vector_type half_size( 3 );
  sc x_half_size, y_half_size, z_half_size;

  for ( auto it = _non_empty_clusters_per_level.begin( );
        it != _non_empty_clusters_per_level.end( ); ++it ) {
    if ( level != -1
      && std::distance( _non_empty_clusters_per_level.begin( ), it )
        != level ) {
      continue;
    }
    for ( auto itt = ( *it ).begin( ); itt != ( *it ).end( ); ++itt ) {
      ( *itt )->get_half_size( half_size );

      x_half_size = half_size[ 0 ];
      y_half_size = half_size[ 1 ];
      z_half_size = half_size[ 2 ];

      if ( include_padding ) {
        x_half_size += _paddings[ std::distance(
          _non_empty_clusters_per_level.begin( ), it ) ];
        y_half_size += _paddings[ std::distance(
          _non_empty_clusters_per_level.begin( ), it ) ];
        z_half_size += _paddings[ std::distance(
          _non_empty_clusters_per_level.begin( ), it ) ];
      }

      file_vtu << "          " << static_cast< float >( 2.0 * x_half_size )
               << " " << static_cast< float >( 2.0 * y_half_size ) << " "
               << static_cast< float >( 2.0 * z_half_size ) << std::endl;
    }
  }

  file_vtu << "        </DataArray>" << std::endl;
  file_vtu << "      </PointData>" << std::endl;

  file_vtu << "    </Piece>" << std::endl;
  file_vtu << "  </UnstructuredGrid>" << std::endl;
  file_vtu << "</VTKFile>" << std::endl;
  file_vtu.close( );

  std::cout << "done." << std::endl;

  return true;
}

void besthea::mesh::volume_space_cluster_tree::collect_leaf_descendants(
  volume_space_cluster & root,
  std::vector< volume_space_cluster * > & leaves ) const {
  if ( root.get_n_children( ) == 0 ) {
    leaves.push_back( &root );
  } else {
    for ( auto it = root.get_children( )->begin( );
          it != root.get_children( )->end( ); ++it ) {
      collect_leaf_descendants( **it, leaves );
    }
  }
}

void besthea::mesh::volume_space_cluster_tree::initialize_moment_contributions(
  volume_space_cluster & current_cluster, lou contribution_size ) {
  current_cluster.resize_moments( contribution_size );
  if ( current_cluster.get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster.get_children( ) ) {
      initialize_moment_contributions( *child, contribution_size );
    }
  }
}

void besthea::mesh::volume_space_cluster_tree::clear_moment_contributions(
  volume_space_cluster & current_cluster ) const {
  current_cluster.clear_moments( );
  if ( current_cluster.get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster.get_children( ) ) {
      clear_moment_contributions( *child );
    }
  }
}

void besthea::mesh::volume_space_cluster_tree::
  initialize_levelwise_cluster_grids( ) {
  _levelwise_cluster_grids.resize( _max_n_levels );
  lo n_grid_clusters = 1;
  for ( lo i = 0; i < _max_n_levels; ++i ) {
    // resize all grid vectors and initialize them with nullptrs.
    _levelwise_cluster_grids[ i ].resize( n_grid_clusters );
    std::fill( _levelwise_cluster_grids[ i ].begin( ),
      _levelwise_cluster_grids[ i ].end( ), nullptr );
    n_grid_clusters *= 8;
  }
  // fill the cluster grids recursively by a tree traversal
  fill_levelwise_cluster_grids_recursively( *_root );
}

void besthea::mesh::volume_space_cluster_tree::
  fill_levelwise_cluster_grids_recursively(
    volume_space_cluster & current_cluster ) {
  std::vector< slou > cluster_coords = current_cluster.get_box_coordinate( );
  // the first entry of cluster_coords is the level, the other ones the grid
  // coordinates
  lo x_stride = 1 << ( 2 * cluster_coords[ 0 ] );
  lo y_stride = 1 << ( cluster_coords[ 0 ] );
  _levelwise_cluster_grids[ cluster_coords[ 0 ] ][ cluster_coords[ 3 ]
    + cluster_coords[ 2 ] * y_stride + x_stride * cluster_coords[ 1 ] ]
    = &current_cluster;
  if ( current_cluster.get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster.get_children( ) ) {
      fill_levelwise_cluster_grids_recursively( *child );
    }
  } else if ( cluster_coords[ 0 ] < _max_n_levels - 1 ) {
    // fill the entries of cluster grids which would correspond to descendants
    // of the current cluster.
    // remember the grid coordinates of the previous level in each step (only
    // the last 3 box coordinates need to be remembered)
    std::vector< std::vector< slou > > parent_level_coordinates
      = { { cluster_coords[ 1 ], cluster_coords[ 2 ], cluster_coords[ 3 ] } };
    for ( lo child_level = cluster_coords[ 0 ] + 1; child_level < _max_n_levels;
          ++child_level ) {
      // compute the coordinates of all descendants that the clusters in
      // parent_level_coordinates would have
      std::vector< std::vector< slou > > child_level_coordinates;
      child_level_coordinates.reserve( 8 * parent_level_coordinates.size( ) );
      for ( size_t i = 0; i < parent_level_coordinates.size( ); ++i ) {
        for ( slou i_x = 0; i_x < 2; ++i_x ) {
          for ( slou i_y = 0; i_y < 2; ++i_y ) {
            for ( slou i_z = 0; i_z < 2; ++i_z ) {
              slou child_x_coord = static_cast< slou >(
                2 * parent_level_coordinates[ i ][ 0 ] + i_x );
              slou child_y_coord = static_cast< slou >(
                2 * parent_level_coordinates[ i ][ 1 ] + i_y );
              slou child_z_coord = static_cast< slou >(
                2 * parent_level_coordinates[ i ][ 2 ] + i_z );
              child_level_coordinates.push_back(
                { child_x_coord, child_y_coord, child_z_coord } );
            }
          }
        }
      }
      lo x_stride = 1 << ( 2 * child_level );
      lo y_stride = 1 << ( child_level );
      for ( size_t i = 0; i < child_level_coordinates.size( ); ++i ) {
        std::vector< slou > child_coords = child_level_coordinates[ i ];
        _levelwise_cluster_grids[ child_level ][ child_coords[ 2 ]
          + child_coords[ 1 ] * y_stride + child_coords[ 0 ] * x_stride ]
          = &current_cluster;
      }
      parent_level_coordinates.swap( child_level_coordinates );
    }
  }
}
