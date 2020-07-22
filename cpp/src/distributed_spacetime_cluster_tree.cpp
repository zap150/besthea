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

#include "besthea/distributed_spacetime_cluster_tree.h"

#include <algorithm>
#include <cmath>
#include <vector>

besthea::mesh::distributed_spacetime_cluster_tree::
  distributed_spacetime_cluster_tree(
    const distributed_spacetime_tensor_mesh & spacetime_mesh, lo levels,
    lo n_min_elems, sc st_coeff, lo spatial_nearfield_limit, MPI_Comm * comm )
  : _max_levels( levels ),
    _real_max_levels( 0 ),
    _spacetime_mesh( spacetime_mesh ),
    _n_min_elems( n_min_elems ),
    _s_t_coeff( st_coeff ),
    _spatial_nearfield_limit( spatial_nearfield_limit ),
    _comm( comm ) {
  MPI_Comm_rank( *_comm, &_my_rank );
  MPI_Comm_size( *_comm, &_n_processes );

  sc xmin, xmax, ymin, ymax, zmin, zmax;
  // this is now a local computation since space mesh is replicated on all MPI
  // processe
  compute_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );
  _bounding_box_size.resize( 4 );
  _bounding_box_size[ 0 ] = ( xmax - xmin ) / 2.0;
  _bounding_box_size[ 1 ] = ( ymax - ymin ) / 2.0;
  _bounding_box_size[ 2 ] = ( zmax - zmin ) / 2.0;
  _bounding_box_size[ 3 ]
    = ( _spacetime_mesh.get_end( ) - _spacetime_mesh.get_start( ) ) / 2.0;

  vector_type space_center
    = { ( xmin + xmax ) / 2.0, ( ymin + ymax ) / 2.0, ( zmin + zmax ) / 2.0 };
  sc time_center
    = ( _spacetime_mesh.get_end( ) + _spacetime_mesh.get_start( ) ) / 2.0;

  vector_type space_half_sizes = { std::abs( xmax - xmin ) / 2.0,
    std::abs( ymax - ymin ) / 2.0, std::abs( zmax - zmin ) / 2.0 };
  sc time_half_size
    = ( _spacetime_mesh.get_end( ) - _spacetime_mesh.get_start( ) ) / 2.0;

  std::vector< slou > coordinates = { 0, 0, 0, 0, 0 };
  _root = new general_spacetime_cluster( space_center, time_center,
    space_half_sizes, time_half_size, spacetime_mesh.get_n_elements( ), nullptr,
    0, 0, coordinates, 0, 0, 0, _spacetime_mesh, -1, false );

  std::vector< lo > elems_in_clusters;

  build_tree( _root );
}

void besthea::mesh::distributed_spacetime_cluster_tree::build_tree(
  general_spacetime_cluster * root ) {
  tree_structure * dist_tree = _spacetime_mesh.get_distribution_tree( );
  lo dist_tree_depth = dist_tree->get_levels( );
  lo dist_tree_depth_coll;

  MPI_Allreduce( &dist_tree_depth, &dist_tree_depth_coll, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );
  //  if ( _my_rank == 1 ) {
  //    dist_tree->print_tree_human_readable( 2, true );
  //    dist_tree->print( );
  //  }
  std::vector< lo > n_elems_per_subdivisioning;
  n_elems_per_subdivisioning.resize( 0 );

  lo n_space_div = 0;
  lo n_time_div = 1;
  std::vector< general_spacetime_cluster * > clusters;
  std::vector< scheduling_time_cluster * > scheduling_clusters;
  lo level;
  for ( level = 0; level < dist_tree_depth_coll - 1; ++level ) {
    get_n_elements_in_subdivisioning(
      *_root, n_space_div, n_time_div, n_elems_per_subdivisioning );

    clusters.resize( 0 );
    scheduling_clusters.resize( 0 );
    collect_clusters_on_level( *root, level, clusters );
    collect_scheduling_clusters_on_level(
      *dist_tree->get_root( ), level + 1, scheduling_clusters );
    for ( auto it : clusters ) {
      split_cluster( *it, scheduling_clusters, level % 2, n_space_div,
        n_time_div, n_elems_per_subdivisioning );
    }

    // level++;
    n_time_div++;
    if ( level % 2 == 0 ) {
      n_space_div++;
    }
  }

  std::vector< general_spacetime_cluster * > my_leaves;
  collect_my_leaves( *_root, my_leaves );
  for ( auto it : my_leaves ) {
    fill_elements( *it );

    build_subtree( *it, false );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  get_n_elements_in_subdivisioning( general_spacetime_cluster & root,
    lo n_space_div, lo n_time_div, std::vector< lo > & elems_in_clusters ) {
  lo n_space_clusters = 1;
  lo n_time_clusters = 1;

  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }
  for ( lo i = 0; i < n_time_div; ++i ) {
    n_time_clusters *= 2;
  }
  lo n_clusters
    = n_space_clusters * n_space_clusters * n_space_clusters * n_time_clusters;

  elems_in_clusters.resize( n_clusters );
  std::vector< lo > loc_elems_in_clusters( n_clusters );
  linear_algebra::coordinates< 4 > centroid;
  lo pos_x, pos_y, pos_z, pos_t;

  vector_type space_center;
  space_center.resize( 3 );
  vector_type half_size;
  half_size.resize( 3 );
  sc time_center, time_half_size;
  root.get_center( space_center, time_center );
  root.get_half_size( half_size, time_half_size );

  // doing it this complicated to avoid inconsistencies with tree
  // assembly due to rounding errors
  std::vector< sc > timesteps( 0 );
  timesteps.reserve( n_time_clusters );
  std::vector< sc > steps_x( 0 );
  steps_x.reserve( n_space_clusters );
  std::vector< sc > steps_y( 0 );
  steps_y.reserve( n_space_clusters );
  std::vector< sc > steps_z( 0 );
  steps_z.reserve( n_space_clusters );

  decompose_line( time_center, time_half_size, time_center - time_half_size,
    n_time_div, 0, timesteps );

  decompose_line( space_center[ 0 ], half_size[ 0 ],
    space_center[ 0 ] - half_size[ 0 ], n_space_div, 0, steps_x );
  decompose_line( space_center[ 1 ], half_size[ 1 ],
    space_center[ 1 ] - half_size[ 1 ], n_space_div, 0, steps_y );
  decompose_line( space_center[ 2 ], half_size[ 2 ],
    space_center[ 2 ] - half_size[ 2 ], n_space_div, 0, steps_z );

  timesteps.push_back( time_center + time_half_size );
  steps_x.push_back( centroid[ 0 ] + half_size[ 0 ] + 1.0 );
  steps_y.push_back( centroid[ 1 ] + half_size[ 1 ] + 1.0 );
  steps_z.push_back( centroid[ 2 ] + half_size[ 2 ] + 1.0 );

  // assign slices to tree nodes
  const std::vector< sc > & slices = _spacetime_mesh.get_slices( );
  std::vector< sc > starts(
    timesteps.size( ) - 1, std::numeric_limits< sc >::infinity( ) );
  std::vector< sc > ends( timesteps.size( ) - 1, -1.0 );
  sc center;

  for ( lo i = 0; i < timesteps.size( ) - 1; ++i ) {
    for ( lo j = 0; j < slices.size( ) - 1; ++j ) {
      center = ( slices[ j ] + slices[ j + 1 ] ) / 2.0;
      if ( center > timesteps[ i ] && center <= timesteps[ i + 1 ] ) {
        if ( slices[ j ] < starts[ i ] ) {
          starts[ i ] = slices[ j ];
        }
        if ( slices[ j + 1 ] > ends[ i ] ) {
          ends[ i ] = slices[ j + 1 ];
        }
      }
    }
  }

  timesteps[ 0 ] -= 1.0;
  sc delta_x = ( 2 * half_size[ 0 ] ) / n_space_clusters;
  sc delta_y = ( 2 * half_size[ 1 ] ) / n_space_clusters;
  sc delta_z = ( 2 * half_size[ 2 ] ) / n_space_clusters;

  for ( lo i = 0; i < _spacetime_mesh.get_my_mesh( )->get_n_elements( ); ++i ) {
    _spacetime_mesh.get_my_mesh( )->get_centroid( i, centroid );
    pos_x
      = ( centroid[ 0 ] - ( space_center[ 0 ] - half_size[ 0 ] ) ) / delta_x;
    pos_y
      = ( centroid[ 1 ] - ( space_center[ 1 ] - half_size[ 1 ] ) ) / delta_y;
    pos_z
      = ( centroid[ 2 ] - ( space_center[ 2 ] - half_size[ 2 ] ) ) / delta_z;

    lo start, end;

    for ( lo i = 0; i < timesteps.size( ) - 1; ++i ) {
      if ( centroid[ 3 ] > starts[ i ] && centroid[ 3 ] <= ends[ i ] ) {
        pos_t = i;
        break;
      }
    }

    start = pos_x > 0 ? pos_x - 1 : pos_x;
    end = pos_x < steps_x.size( ) - 2 ? pos_x + 1 : pos_x;
    for ( lo i = start; i <= end; ++i ) {
      if ( ( centroid[ 0 ] >= steps_x[ i ] )
        && ( centroid[ 0 ] < steps_x[ i + 1 ] ) ) {
        pos_x = i;
        break;
      }
    }

    start = pos_y > 0 ? pos_y - 1 : pos_y;
    end = pos_y < steps_y.size( ) - 2 ? pos_y + 1 : pos_y;
    for ( lo i = start; i <= end; ++i ) {
      if ( ( centroid[ 1 ] >= steps_y[ i ] )
        && ( centroid[ 1 ] < steps_y[ i + 1 ] ) ) {
        pos_y = i;
        break;
      }
    }
    start = pos_z > 0 ? pos_z - 1 : pos_z;
    end = pos_z < steps_z.size( ) - 2 ? pos_z + 1 : pos_z;
    for ( lo i = start; i <= end; ++i ) {
      if ( ( centroid[ 2 ] >= steps_z[ i ] )
        && ( centroid[ 2 ] < steps_z[ i + 1 ] ) ) {
        pos_z = i;
        break;
      }
    }

    lo pos = pos_t * n_space_clusters * n_space_clusters * n_space_clusters
      + pos_x * n_space_clusters * n_space_clusters + pos_y * n_space_clusters
      + pos_z;
    loc_elems_in_clusters.at( pos )++;
  }

  MPI_Allreduce( loc_elems_in_clusters.data( ), elems_in_clusters.data( ),
    loc_elems_in_clusters.size( ), get_index_type< lo >::MPI_LO( ), MPI_SUM,
    *_comm );
}

void besthea::mesh::distributed_spacetime_cluster_tree::decompose_line(
  sc center, sc half_size, sc left_bound, lo n_ref, lo curr_level,
  std::vector< sc > & steps ) {
  if ( curr_level == n_ref - 1 ) {
    steps.push_back( left_bound );
    steps.push_back( center );
    return;
  } else if ( n_ref == 0 ) {
    steps.push_back( left_bound );
    return;
  }

  decompose_line( center - half_size / 2.0, half_size / 2.0, left_bound, n_ref,
    curr_level + 1, steps );
  decompose_line( center + half_size / 2.0, half_size / 2.0, center, n_ref,
    curr_level + 1, steps );
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  collect_clusters_on_level( general_spacetime_cluster & root, lo level,
    std::vector< general_spacetime_cluster * > & clusters ) {
  if ( root.get_level( ) == level ) {
    clusters.push_back( &root );
    return;
  }

  std::vector< general_spacetime_cluster * > * children = root.get_children( );
  if ( children != nullptr ) {
    for ( auto it : *children ) {
      collect_clusters_on_level( *it, level, clusters );
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::
  collect_scheduling_clusters_on_level( scheduling_time_cluster & root,
    lo level, std::vector< scheduling_time_cluster * > & clusters ) {
  if ( root.get_level( ) == level ) {
    clusters.push_back( &root );
    return;
  }

  std::vector< scheduling_time_cluster * > * children = root.get_children( );
  if ( children != nullptr ) {
    for ( auto it : *children ) {
      collect_scheduling_clusters_on_level( *it, level, clusters );
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::split_cluster(
  general_spacetime_cluster & cluster,
  std::vector< scheduling_time_cluster * > & my_clusters_on_level,
  bool split_space, lo n_space_div, lo n_time_div,
  std::vector< lo > & elems_in_clusters ) {
  lo n_space_clusters = 1;
  lo n_time_clusters = 1;
  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }
  for ( lo i = 0; i < n_time_div; ++i ) {
    n_time_clusters *= 2;
  }

  const std::vector< slou > parent_coord = cluster.get_box_coordinate( );

  vector_type space_center( 3 );
  vector_type space_half_size( 3 );
  sc time_center, time_half_size;
  cluster.get_center( space_center, time_center );
  cluster.get_half_size( space_half_size, time_half_size );

  sc new_time_center, new_time_half_size;
  lo coord_t;

  lo pos;

  // first, create left temporal cluster
  new_time_half_size = time_half_size / 2.0;
  new_time_center = time_center - new_time_half_size;
  coord_t = 2 * parent_coord[ 4 ];

  bool is_my_cluster = false;
  lo owner = -1;
  for ( auto it : my_clusters_on_level ) {
    if ( ( std::abs( new_time_center - it->get_center( ) )
           < it->get_half_size( ) )
      && ( _spacetime_mesh.get_my_mesh( )->get_temporal_mesh( )->get_end( )
        >= it->get_center( ) - it->get_half_size( ) ) ) {
      is_my_cluster = true;
      owner = it->get_process_id( );
    }
  }

  if ( is_my_cluster ) {
    if ( split_space ) {
      vector_type new_spat_center( 3 );
      vector_type new_spat_half_size( 3 );
      for ( short i = 0; i < 8; ++i ) {
        cluster.compute_spatial_suboctant(
          i, new_spat_center, new_spat_half_size );

        slou coord_x
          = 2 * cluster.get_box_coordinate( )[ 1 ] + _idx_2_coord[ i ][ 0 ];
        slou coord_y
          = 2 * cluster.get_box_coordinate( )[ 2 ] + _idx_2_coord[ i ][ 1 ];
        slou coord_z
          = 2 * cluster.get_box_coordinate( )[ 3 ] + _idx_2_coord[ i ][ 2 ];
        std::vector< slou > coordinates
          = { static_cast< slou >( cluster.get_level( ) ), coord_x, coord_y,
              coord_z, coord_t };

        pos = coord_t * n_space_clusters * n_space_clusters * n_space_clusters
          + coord_x * n_space_clusters * n_space_clusters
          + coord_y * n_space_clusters + coord_z;

        if ( elems_in_clusters[ pos ] > 0 ) {
          general_spacetime_cluster * left_child
            = new general_spacetime_cluster( new_spat_center, new_time_center,
              new_spat_half_size, new_time_half_size, elems_in_clusters[ pos ],
              &cluster, cluster.get_level( ) + 1, i, coordinates, 0,
              n_space_div, n_time_div, _spacetime_mesh, owner, false );
          cluster.add_child( left_child );
        }
      }
    } else {
      slou coord_x = parent_coord[ 1 ];
      slou coord_y = parent_coord[ 2 ];
      slou coord_z = parent_coord[ 3 ];
      std::vector< slou > coordinates
        = { static_cast< slou >( cluster.get_level( ) ), coord_x, coord_y,
            coord_z, coord_t };
      pos = coord_t * n_space_clusters * n_space_clusters * n_space_clusters
        + coord_x * n_space_clusters * n_space_clusters
        + coord_y * n_space_clusters + coord_z;

      if ( elems_in_clusters[ pos ] > 0 ) {
        general_spacetime_cluster * left_child = new general_spacetime_cluster(
          space_center, new_time_center, space_half_size, new_time_half_size,
          elems_in_clusters[ pos ], &cluster, cluster.get_level( ) + 1,
          cluster.get_spatial_octant( ), coordinates, 1, n_space_div,
          n_time_div, _spacetime_mesh, owner, false );
        cluster.add_child( left_child );
      }
    }
  }

  // next, right temporal cluster
  new_time_center = time_center + new_time_half_size;
  coord_t = 2 * parent_coord[ 4 ] + 1;

  is_my_cluster = false;
  for ( auto it : my_clusters_on_level ) {
    if ( ( std::abs( new_time_center - it->get_center( ) )
           < it->get_half_size( ) )
      && ( _spacetime_mesh.get_my_mesh( )->get_temporal_mesh( )->get_end( )
        >= it->get_center( ) - it->get_half_size( ) ) ) {
      is_my_cluster = true;
      owner = it->get_process_id( );
    }
  }

  if ( is_my_cluster ) {
    if ( split_space ) {
      vector_type new_spat_center( 3 );
      vector_type new_spat_half_size( 3 );
      for ( short i = 0; i < 8; ++i ) {
        cluster.compute_spatial_suboctant(
          i, new_spat_center, new_spat_half_size );

        slou coord_x
          = 2 * cluster.get_box_coordinate( )[ 1 ] + _idx_2_coord[ i ][ 0 ];
        slou coord_y
          = 2 * cluster.get_box_coordinate( )[ 2 ] + _idx_2_coord[ i ][ 1 ];
        slou coord_z
          = 2 * cluster.get_box_coordinate( )[ 3 ] + _idx_2_coord[ i ][ 2 ];
        std::vector< slou > coordinates
          = { static_cast< slou >( cluster.get_level( ) ), coord_x, coord_y,
              coord_z, coord_t };

        pos = coord_t * n_space_clusters * n_space_clusters * n_space_clusters
          + coord_x * n_space_clusters * n_space_clusters
          + coord_y * n_space_clusters + coord_z;

        if ( elems_in_clusters[ pos ] > 0 ) {
          general_spacetime_cluster * right_child
            = new general_spacetime_cluster( new_spat_center, new_time_center,
              new_spat_half_size, new_time_half_size, elems_in_clusters[ pos ],
              &cluster, cluster.get_level( ) + 1, i, coordinates, 1,
              n_space_div, n_time_div, _spacetime_mesh, owner, false );
          cluster.add_child( right_child );
        }
      }
    } else {
      slou coord_x = parent_coord[ 1 ];
      slou coord_y = parent_coord[ 2 ];
      slou coord_z = parent_coord[ 3 ];
      std::vector< slou > coordinates
        = { static_cast< slou >( cluster.get_level( ) ), coord_x, coord_y,
            coord_z, coord_t };

      pos = coord_t * n_space_clusters * n_space_clusters * n_space_clusters
        + coord_x * n_space_clusters * n_space_clusters
        + coord_y * n_space_clusters + coord_z;

      if ( elems_in_clusters[ pos ] > 0 ) {
        general_spacetime_cluster * right_child = new general_spacetime_cluster(
          space_center, new_time_center, space_half_size, new_time_half_size,
          elems_in_clusters[ pos ], &cluster, cluster.get_level( ) + 1,
          cluster.get_spatial_octant( ), coordinates, 0, n_space_div,
          n_time_div, _spacetime_mesh, owner, false );
        cluster.add_child( right_child );
      }
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::compute_bounding_box(
  sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax ) {
  // only local computation since spatial mesh is now duplicated
  xmin = ymin = zmin = std::numeric_limits< sc >::max( );
  xmax = ymax = zmax = std::numeric_limits< sc >::min( );

  linear_algebra::coordinates< 4 > node;
  for ( lo i = 0; i < _spacetime_mesh.get_my_mesh( )->get_n_nodes( ); ++i ) {
    _spacetime_mesh.get_my_mesh( )->get_node( i, node );

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
}

void besthea::mesh::distributed_spacetime_cluster_tree::collect_my_leaves(
  general_spacetime_cluster & root,
  std::vector< general_spacetime_cluster * > & leaves ) {
  std::vector< general_spacetime_cluster * > * children = root.get_children( );

  if ( children != nullptr ) {
    for ( auto it : *children ) {
      collect_my_leaves( *it, leaves );
    }
  } else if ( _my_rank == root.get_process_id( ) ) {
    leaves.push_back( &root );
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::fill_elements(
  general_spacetime_cluster & cluster ) {
  lo n_space_div, n_time_div;
  cluster.get_n_divs( n_space_div, n_time_div );
  lo n_space_clusters = 1;
  for ( lo i = 0; i < n_space_div; ++i ) {
    n_space_clusters *= 2;
  }

  const std::vector< slou > coord = cluster.get_box_coordinate( );
  const std::vector< sc > & slices = _spacetime_mesh.get_slices( );

  vector_type space_center;
  space_center.resize( 3 );
  vector_type half_size;
  half_size.resize( 3 );
  sc time_center, time_half_size;
  cluster.get_center( space_center, time_center );
  cluster.get_half_size( half_size, time_half_size );
  spacetime_tensor_mesh * my_mesh = _spacetime_mesh.get_my_mesh( );
  linear_algebra::coordinates< 4 > centroid;

  // modify clusters according to the slices
  sc cluster_start = time_center - time_half_size;
  sc cluster_end = time_center + time_half_size;
  sc start_time = std::numeric_limits< sc >::infinity( );
  sc end_time = -1.0;
  sc slice_center;
  for ( lo j = 0; j < slices.size( ) - 1; ++j ) {
    slice_center = ( slices[ j ] + slices[ j + 1 ] ) / 2.0;
    if ( slice_center > cluster_start && slice_center <= cluster_end ) {
      if ( slices[ j ] < start_time ) {
        start_time = slices[ j ];
      }
      if ( slices[ j + 1 ] > end_time ) {
        end_time = slices[ j + 1 ];
      }
    }
  }

  sc new_center = ( start_time + end_time ) / 2.0;
  sc new_half_size = ( end_time - start_time ) / 2.0;
  cluster.set_time_center( new_center );
  cluster.set_time_half_size( new_half_size );

  sc left = space_center[ 0 ] - half_size[ 0 ];
  sc right = space_center[ 0 ] + half_size[ 0 ];
  sc front = space_center[ 1 ] - half_size[ 1 ];
  sc back = space_center[ 1 ] + half_size[ 1 ];
  sc bottom = space_center[ 2 ] - half_size[ 2 ];
  sc top = space_center[ 2 ] + half_size[ 2 ];
  sc beginning = new_center - new_half_size;
  sc end = new_center + new_half_size;

  if ( coord[ 1 ] == n_space_clusters - 1 ) {
    right += 1.0;
  }
  if ( coord[ 2 ] == n_space_clusters - 1 ) {
    back += 1.0;
  }

  if ( coord[ 3 ] == n_space_clusters - 1 ) {
    top += 1.0;
  }
  if ( coord[ 4 ] == 0 ) {
    beginning -= 1.0;
  }

  lo n_elems = 0;

  for ( lo i = 0; i < my_mesh->get_n_elements( ); ++i ) {
    my_mesh->get_centroid( i, centroid );

    if ( ( centroid[ 0 ] >= left ) && ( centroid[ 0 ] < right )
      && ( centroid[ 1 ] >= front ) && ( centroid[ 1 ] < back )
      && ( centroid[ 2 ] >= bottom ) && ( centroid[ 2 ] < top )
      && ( centroid[ 3 ] > beginning ) && ( centroid[ 3 ] <= end ) ) {
      n_elems++;
    }
  }
  cluster.reserve_elements( n_elems );

  for ( lo i = 0; i < my_mesh->get_n_elements( ); ++i ) {
    my_mesh->get_centroid( i, centroid );

    if ( ( centroid[ 0 ] >= left ) && ( centroid[ 0 ] < right )
      && ( centroid[ 1 ] >= front ) && ( centroid[ 1 ] < back )
      && ( centroid[ 2 ] >= bottom ) && ( centroid[ 2 ] < top )
      && ( centroid[ 3 ] > beginning ) && ( centroid[ 3 ] <= end ) ) {
      cluster.add_element( _spacetime_mesh.local_2_global( i ) );
    }
  }
}

void besthea::mesh::distributed_spacetime_cluster_tree::build_subtree(
  general_spacetime_cluster & root, bool split_space ) {
  if ( root.get_level( ) + 1 > _max_levels - 1
    || root.get_n_elements( ) < _n_min_elems ) {
    root.set_n_children( 0 );
    return;
  }

  spacetime_tensor_mesh * my_mesh = _spacetime_mesh.get_my_mesh( );
  const std::vector< slou > parent_coord = root.get_box_coordinate( );

  vector_type space_center( 3 );
  vector_type space_half_size( 3 );
  sc time_center, time_half_size;
  root.get_center( space_center, time_center );
  root.get_half_size( space_half_size, time_half_size );

  lo n_space_div, n_time_div;
  root.get_n_divs( n_space_div, n_time_div );

  linear_algebra::coordinates< 4 > el_centroid;
  lo elem_idx = 0;
  lo oct_sizes[ 16 ];
  for ( lo i = 0; i < 16; ++i ) {
    oct_sizes[ i ] = 0;
  }
  general_spacetime_cluster * clusters[ 16 ];

  sc new_time_center, new_time_half_size;

  vector_type new_space_center( 3 );
  vector_type new_space_half_size( 3 );

  slou coord_x, coord_y, coord_z, coord_t;

  if ( split_space ) {
    for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
      elem_idx = _spacetime_mesh.global_2_local( root.get_element( i ) );
      my_mesh->get_centroid( elem_idx, el_centroid );

      if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 0 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 1 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 2 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 3 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 4 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 5 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 6 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        ++oct_sizes[ 7 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 8 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 9 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 10 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 11 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 12 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 13 ];
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 14 ];
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        ++oct_sizes[ 15 ];
      }
    }
    lo n_clusters = 0;

    for ( short i = 0; i < 8; ++i ) {
      root.compute_spatial_suboctant(
        i, new_space_center, new_space_half_size );

      coord_x = 2 * root.get_box_coordinate( )[ 1 ] + _idx_2_coord[ i ][ 0 ];
      coord_y = 2 * root.get_box_coordinate( )[ 2 ] + _idx_2_coord[ i ][ 1 ];
      coord_z = 2 * root.get_box_coordinate( )[ 3 ] + _idx_2_coord[ i ][ 2 ];

      std::vector< slou > coordinates
        = { static_cast< slou >( root.get_level( ) + 1 ), coord_x, coord_y,
            coord_z, coord_t };
      // std::cout << oct_sizes[ i ] << " " << oct_sizes[ i + 8 ] << std::endl;
      if ( oct_sizes[ i ] > 0 ) {
        ++n_clusters;
        new_time_half_size = time_half_size / 2.0;
        new_time_center = time_center - new_time_half_size;
        coord_t = 2 * parent_coord[ 4 ];
        clusters[ i ]
          = new general_spacetime_cluster( new_space_center, new_time_center,
            new_space_half_size, new_time_half_size, oct_sizes[ i ], &root,
            root.get_level( ) + 1, i, coordinates, 0, n_space_div + 1,
            n_time_div + 1, _spacetime_mesh, root.get_process_id( ), true );
      } else {
        clusters[ i ] = nullptr;
      }
      if ( oct_sizes[ i + 8 ] > 0 ) {
        ++n_clusters;
        new_time_half_size = time_half_size / 2.0;
        new_time_center = time_center + new_time_half_size;
        coord_t = 2 * parent_coord[ 4 ] + 1;
        clusters[ i + 8 ]
          = new general_spacetime_cluster( new_space_center, new_time_center,
            new_space_half_size, new_time_half_size, oct_sizes[ i ], &root,
            root.get_level( ) + 1, i, coordinates, 1, n_space_div + 1,
            n_time_div + 1, _spacetime_mesh, root.get_process_id( ), true );
      } else {
        clusters[ i + 8 ] = nullptr;
      }
    }

    // finally, assign elements to clusters
    for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
      elem_idx = _spacetime_mesh.global_2_local( root.get_element( i ) );
      my_mesh->get_centroid( elem_idx, el_centroid );

      if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 0 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 1 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 2 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 3 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 4 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 5 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 6 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] < time_center ) {
        clusters[ 7 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 8 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 9 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 10 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] >= space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 11 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 12 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] >= space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 13 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] < space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 14 ]->add_element( root.get_element( i ) );
      } else if ( el_centroid[ 0 ] >= space_center( 0 )
        && el_centroid[ 1 ] < space_center( 1 )
        && el_centroid[ 2 ] < space_center( 2 )
        && el_centroid[ 3 ] >= time_center ) {
        clusters[ 15 ]->add_element( root.get_element( i ) );
      }
    }

    root.set_n_children( n_clusters );

    for ( lo i = 0; i < 16; ++i ) {
      if ( clusters[ i ] != nullptr ) {
        root.add_child( clusters[ i ] );

        build_subtree( *clusters[ i ], !split_space );
      }
    }
  } else {
    for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
      // get elem idx in local mesh indexing
      elem_idx = _spacetime_mesh.global_2_local( root.get_element( i ) );
      my_mesh->get_centroid( elem_idx, el_centroid );
      if ( el_centroid[ 3 ] >= time_center ) {
        oct_sizes[ 1 ] += 1;
      } else {
        oct_sizes[ 0 ] += 1;
      }
    }

    lo n_clusters = 0;
    coord_x = parent_coord[ 1 ];
    coord_y = parent_coord[ 2 ];
    coord_z = parent_coord[ 3 ];
    coord_t = 2 * parent_coord[ 4 ];
    std::vector< slou > coordinates
      = { static_cast< slou >( root.get_level( ) + 1 ), coord_x, coord_y,
          coord_z, coord_t };

    new_time_half_size = time_half_size / 2.0;
    new_time_center = time_center - new_time_half_size;
    general_spacetime_cluster *left_child, *right_child;

    // left temporal cluster
    if ( oct_sizes[ 0 ] > 0 ) {
      n_clusters++;
      left_child = new general_spacetime_cluster( space_center, new_time_center,
        space_half_size, new_time_half_size, oct_sizes[ 0 ], &root,
        root.get_level( ) + 1, root.get_spatial_octant( ), coordinates, 0,
        n_space_div, n_time_div + 1, _spacetime_mesh, root.get_process_id( ),
        true );
    }

    // right temporal cluster
    new_time_center = time_center + new_time_half_size;
    coord_t = 2 * parent_coord[ 4 ] + 1;
    coordinates[ 4 ] = coord_t;
    if ( oct_sizes[ 1 ] > 0 ) {
      n_clusters++;
      right_child = new general_spacetime_cluster( space_center,
        new_time_center, space_half_size, new_time_half_size, oct_sizes[ 0 ],
        &root, root.get_level( ) + 1, root.get_spatial_octant( ), coordinates,
        1, n_space_div, n_time_div + 1, _spacetime_mesh, root.get_process_id( ),
        true );
    }

    for ( lo i = 0; i < root.get_n_elements( ); ++i ) {
      // get elem idx in local mesh indexing
      elem_idx = _spacetime_mesh.global_2_local( root.get_element( i ) );
      my_mesh->get_centroid( elem_idx, el_centroid );
      if ( el_centroid[ 3 ] >= time_center ) {
        right_child->add_element( root.get_element( i ) );
      } else {
        left_child->add_element( root.get_element( i ) );
      }
    }
    root.set_n_children( n_clusters );

    root.add_child( left_child );
    build_subtree( *left_child, !split_space );
    root.add_child( right_child );
    build_subtree( *right_child, !split_space );
  }
}
