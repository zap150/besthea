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
    const distributed_spacetime_tensor_mesh & spacetime_mesh, lo time_levels,
    lo n_min_time_elems, lo n_min_space_elems, sc st_coeff,
    lo spatial_nearfield_limit, MPI_Comm * comm )
  : _levels( 0 ),
    _spacetime_mesh( spacetime_mesh ),
    _s_t_coeff( st_coeff ),
    _spatial_nearfield_limit( spatial_nearfield_limit ),
    _comm( comm ) {
  MPI_Comm_rank( *_comm, &_my_rank );
  MPI_Comm_size( *_comm, &_n_processes );

  //  // get the initial time cluster tree
  //  tree_structure * t_dist_tree = spacetime_mesh.get_distribution_tree( );
  //
  //  // get the spatial mesh (assuming so far it is the same for all processes)
  //  triangular_surface_mesh * space_mesh
  //    = spacetime_mesh.get_my_mesh( )->get_spatial_surface_mesh( );
  //
  //  sc xmin, xmax, ymin, ymax, zmin, zmax;
  //  space_mesh->compute_bounding_box( xmin, xmax, ymin, ymax, zmin, zmax );
  //
  //  sc max_half_size = std::max( { ( xmax - xmin ) / 2.0, ( ymax - ymin )
  //  / 2.0,
  //                       ( zmax - zmin ) / 2.0 } )
  //    / 2.0;
  //
  //  sc delta = spacetime_mesh.get_end( ) - spacetime_mesh.get_start( );
  //
  //  _start_spatial_level = 0;
  //
  //  // determine the number of initial octasections that has to be performed
  //  to
  //  // get to the level of the spatial tree satisfying the condition
  //  // h_x^l \approx st_coeff sqrt(delta)
  //  while ( max_half_size > st_coeff * sqrt( delta ) ) {
  //    max_half_size *= 0.5;
  //    _start_spatial_level += 1;
  //  }
  //
  //  lo n_t_levels = t_dist_tree->get_levels( );
  //  lo n_s_levels = n_t_levels / 2;
  //  if ( n_t_levels % 2 ) {
  //    ++n_s_levels;
  //  }
  //
  //  space_cluster_tree * space_tree = new space_cluster_tree(
  //    *spacetime_mesh.get_my_mesh( )->get_spatial_surface_mesh( ),
  //    _start_spatial_level + n_s_levels, n_min_space_elems );
  //
  //  // determine for which temporal level the first spatial refinement is
  //  needed _start_temporal_level = 0; if ( _start_spatial_level == 0 ) {
  //    while ( max_half_size <= st_coeff * sqrt( delta ) ) {
  //      delta *= 0.5;
  //      _start_temporal_level += 1;
  //    }
  //    // shift _start_temporal_level if necessary to guarantee levels as in
  //    // Messner's work
  //    if ( ( n_t_levels - _start_temporal_level ) % 2 ) {
  //      _start_temporal_level -= 1;
  //    }
  //  }
  //
  //  // next, we assemble their combination into a space-time tree
  //  // the level -1 node is always the combination of the whole space & time
  //  // root is duplicated on all processes
  //  _time_root = new time_cluster( ) _root = new spacetime_cluster(
  //    *_space_tree->get_root( ), *_time_tree->get_root( ), nullptr, -1 );

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

  //_root = new general_spacetime_cluster( space_center, time_center,
  //space_half_sizes, time_half_size, nullptr, -1,  )
}

void besthea::mesh::distributed_spacetime_cluster_tree::build_tree(
  general_spacetime_cluster * root, lo level, bool split_space ) {
}

void besthea::mesh::distributed_spacetime_cluster_tree::compute_bounding_box(
  sc & xmin, sc & xmax, sc & ymin, sc & ymax, sc & zmin, sc & zmax ) {
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
