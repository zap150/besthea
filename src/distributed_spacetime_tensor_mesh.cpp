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

#include "besthea/distributed_spacetime_tensor_mesh.h"

#include "besthea/distributed_block_vector.h"
#include "besthea/scheduling_time_cluster.h"
#include "besthea/tree_structure.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>

besthea::mesh::distributed_spacetime_tensor_mesh::
  distributed_spacetime_tensor_mesh( const std::string & decomposition_file,
    const std::string & tree_file, const std::string & cluster_bounds_file,
    const std::string & distribution_file, const bool enable_m2t_and_s2l,
    MPI_Comm * comm, lo & status )
  : _my_mesh( nullptr ),
    _space_mesh( nullptr ),
    _time_mesh( nullptr ),
    _dist_tree( nullptr ),
    _comm( comm ) {
  MPI_Comm_rank( *_comm, &_my_rank );
  MPI_Comm_size( *_comm, &_n_processes );

  load( decomposition_file, tree_file, cluster_bounds_file, distribution_file,
    enable_m2t_and_s2l, status );

  // check whether some process produced a warning
  if ( _my_rank == 0 ) {
    std::vector< lo > all_load_status( _n_processes );
    MPI_Gather( &status, 1, get_index_type< lo >::MPI_LO( ),
      all_load_status.data( ), 1, get_index_type< lo >::MPI_LO( ), 0, *_comm );
    bool first_warning = true;
    for ( lo i = 0; i < _n_processes; ++i ) {
      if ( all_load_status[ i ] == -1 ) {
        if ( first_warning ) {
          std::cout
            << "WARNING: nearfield mesh was not connected for process: ";
          first_warning = false;
        }
        std::cout << i << " ";
      }
    }
    if ( !first_warning ) {
      std::cout << std::endl;
    }
  } else {
    MPI_Gather( &status, 1, get_index_type< lo >::MPI_LO( ), nullptr, 1,
      get_index_type< lo >::MPI_LO( ), 0, *_comm );
  }

  // get global data
  MPI_Barrier( *_comm );
  lo n_elems_array[ 2 ];
  n_elems_array[ 0 ] = _my_mesh->get_n_elements( );
  n_elems_array[ 1 ] = _my_mesh->get_n_temporal_elements( );
  MPI_Allreduce( MPI_IN_PLACE, n_elems_array, 2,
    get_index_type< lo >::MPI_LO( ), MPI_SUM, *_comm );
  _n_global_elements = n_elems_array[ 0 ];
  _n_global_time_elements = n_elems_array[ 1 ];
}

besthea::mesh::distributed_spacetime_tensor_mesh::
  ~distributed_spacetime_tensor_mesh( ) {
  if ( _my_mesh != nullptr ) {
    delete _my_mesh;
  }
  if ( _space_mesh != nullptr ) {
    delete _space_mesh;
  }
  if ( _time_mesh != nullptr ) {
    delete _time_mesh;
  }
  if ( _dist_tree != nullptr ) {
    delete _dist_tree;
  }
  if ( _nearfield_time_mesh != nullptr ) {
    delete _nearfield_time_mesh;
  }
  if ( _nearfield_mesh != nullptr ) {
    delete _nearfield_mesh;
  }
}

besthea::mesh::tree_structure const *
besthea::mesh::distributed_spacetime_tensor_mesh::get_distribution_tree( )
  const {
  return _dist_tree;
}

besthea::mesh::tree_structure *
besthea::mesh::distributed_spacetime_tensor_mesh::get_distribution_tree( ) {
  return _dist_tree;
}

void besthea::mesh::distributed_spacetime_tensor_mesh::find_slices_to_load(
  std::set< lo > & nearfield_slice_indices,
  std::set< lo > & local_slice_indices ) const {
  for ( auto leaf_cluster : _dist_tree->get_leaves( ) ) {
    lo cluster_owner = leaf_cluster->get_process_id( );
    if ( cluster_owner == _my_rank ) {
      // add all the indices of slices in the local cluster to the corresponding
      // set
      const std::vector< lo > * slices = leaf_cluster->get_time_slices( );
      if ( slices != nullptr ) {
        for ( auto idx : *slices ) {
          local_slice_indices.insert( idx );
        }
      }
      // remember that this cluster's mesh is available
      leaf_cluster->set_mesh_availability( true );

      // if a cluster in the nearfield is not local, add its slices to the
      // corresponding set
      std::vector< scheduling_time_cluster * > * nearfield
        = leaf_cluster->get_nearfield_list( );
      for ( auto nearfield_cluster : *nearfield ) {
        lo nearfield_cluster_owner = nearfield_cluster->get_process_id( );
        if ( nearfield_cluster_owner != _my_rank ) {
          const std::vector< lo > * nearfield_slices
            = nearfield_cluster->get_time_slices( );
          for ( auto idx : *nearfield_slices ) {
            nearfield_slice_indices.insert( idx );
          }
          // remember that this cluster's mesh is available
          nearfield_cluster->set_mesh_availability( true );
        }
      }
    }
  }
}

bool besthea::mesh::distributed_spacetime_tensor_mesh::load(
  const std::string & decomposition_file, const std::string & tree_file,
  const std::string & cluster_bounds_file,
  const std::string & distribution_file, const bool enable_m2t_and_s2l,
  lo & status ) {
  // load the file with basic description of the decomposed mesh
  std::ifstream filestream( decomposition_file.c_str( ) );

  if ( !filestream.is_open( ) ) {
    std::cout << "File could not be opened!" << std::endl;
    _n_meshes = 0;
    return false;
  }

  // load the boundary of the time interval

  filestream >> _t_start;
  filestream >> _t_end;

  filestream >> _n_meshes;  // read total number of time slices

  // read the slices definition
  _slices.resize( _n_meshes + 1 );
  sc slice_node;
  for ( lo i = 0; i < _n_meshes + 1; ++i ) {
    filestream >> slice_node;
    _slices[ i ] = slice_node;
  }

  // read and reconstruct temporal tree and distribution of clusters
  _dist_tree = new tree_structure(
    tree_file, cluster_bounds_file, _my_rank, enable_m2t_and_s2l );
  _dist_tree->load_process_assignments( distribution_file );
  _dist_tree->assign_slices_to_clusters( _slices );

  std::set< lo > local_slice_indices;
  std::set< lo > nearfield_slice_indices;
  find_slices_to_load( nearfield_slice_indices, local_slice_indices );

  // ensure that the nearfield slices are connected
  if ( !nearfield_slice_indices.empty( ) ) {
    auto first_it = nearfield_slice_indices.begin( );
    auto second_it = nearfield_slice_indices.begin( );
    ++second_it;
    while ( second_it != nearfield_slice_indices.end( ) ) {
      if ( ( *second_it ) != ( *first_it ) + 1 ) {
        // insert the missing nearfield slice, and set status to -1 to warn the
        // user
        nearfield_slice_indices.insert( first_it, ( *first_it ) + 1 );
        status = -1;
        ++first_it;
      } else {
        ++first_it;
        ++second_it;
      }
    }
  }

  _n_meshes_per_rank = local_slice_indices.size( );

  lo local_start_mesh = *local_slice_indices.begin( );
  lo nearfield_start_mesh = *nearfield_slice_indices.begin( );
  std::vector< sc > my_time_nodes;
  std::vector< sc > my_nearfield_time_nodes;

  lo current_idx;
  std::string t_file_path;
  std::string s_file_path;

  std::set< lo >::iterator next_slice_to_load
    = nearfield_slice_indices.begin( );
  bool is_local_mesh = false;
  // check whether there are nearfield slices to load
  if ( next_slice_to_load == nearfield_slice_indices.end( ) ) {
    _my_nearfield_start_idx = 0;
    next_slice_to_load = local_slice_indices.begin( );
    is_local_mesh = true;
  }

  lo mesh_idx = 0;
  while ( ( mesh_idx < _n_meshes )
    && ( next_slice_to_load != local_slice_indices.end( ) ) ) {
    filestream >> current_idx;
    filestream >> t_file_path;
    filestream >> s_file_path;

    if ( mesh_idx == *next_slice_to_load ) {
      // check whether the mesh is a nearfield mesh or a local one
      if ( !is_local_mesh ) {  // nearfield mesh
        if ( mesh_idx == nearfield_start_mesh ) {
          _my_nearfield_start_idx = current_idx;
        }
        std::ifstream temp_file( t_file_path.c_str( ) );
        if ( !temp_file.is_open( ) ) {
          std::cout << "File " << t_file_path << " could not be opened!"
                    << std::endl;
          return false;
        }
        lo dummy;

        temp_file >> dummy;  // dimension (1)
        temp_file >> dummy;  // nodes per element (2)

        lo n_nodes;
        sc node;
        temp_file >> n_nodes;
        for ( lo i_node = 0; i_node < n_nodes; ++i_node ) {
          temp_file >> node;
          // avoid double entries in nearfield time nodes vector by adding the
          // first node of a mesh only if the mesh is the first mesh.
          if ( i_node != 0 || mesh_idx == nearfield_start_mesh ) {
            my_nearfield_time_nodes.push_back( node );
          }
        }
        temp_file.close( );
      } else {  // local mesh
        if ( mesh_idx == local_start_mesh ) {
          _local_start_idx = current_idx;
        }
        std::ifstream temp_file( t_file_path.c_str( ) );
        if ( !temp_file.is_open( ) ) {
          std::cout << "File " << t_file_path << " could not be opened!"
                    << std::endl;
          return false;
        }
        lo dummy;

        temp_file >> dummy;  // dimension (1)
        temp_file >> dummy;  // nodes per element (2)

        lo n_nodes;
        sc node;
        temp_file >> n_nodes;
        for ( lo i_node = 0; i_node < n_nodes; ++i_node ) {
          temp_file >> node;
          // again, avoid double entries
          if ( i_node != 0 || mesh_idx == local_start_mesh ) {
            my_time_nodes.push_back( node );
          }
        }

        temp_file.close( );
      }
      ++next_slice_to_load;
      if ( next_slice_to_load == nearfield_slice_indices.end( ) ) {
        next_slice_to_load = local_slice_indices.begin( );
        is_local_mesh = true;
      }
    }
    ++mesh_idx;
  }

  _time_mesh = new temporal_mesh( my_time_nodes );
  _space_mesh = new triangular_surface_mesh( s_file_path );
  _my_mesh = new spacetime_tensor_mesh( *_space_mesh, *_time_mesh );

  if ( my_nearfield_time_nodes.size( ) > 0 ) {
    _nearfield_time_mesh = new temporal_mesh( my_nearfield_time_nodes );
    _nearfield_mesh
      = new spacetime_tensor_mesh( *_space_mesh, *_nearfield_time_mesh );
  } else {
    _nearfield_time_mesh = nullptr;
    _nearfield_mesh = nullptr;
  }

  filestream.close( );
  _dist_tree->reduce_2_essential( );

  return true;
}

std::vector< lo >
besthea::mesh::distributed_spacetime_tensor_mesh::get_my_timesteps( ) const {
  lo n_nearfield_elems = 0;
  if ( _nearfield_mesh != nullptr ) {
    n_nearfield_elems = _nearfield_mesh->get_n_temporal_elements( );
  }
  lo n_local_elems = _my_mesh->get_n_temporal_elements( );
  std::vector< lo > my_timesteps( n_nearfield_elems + n_local_elems );

  if ( _nearfield_mesh != nullptr ) {
    for ( lo i = _my_nearfield_start_idx;
          i < _my_nearfield_start_idx + n_nearfield_elems; ++i ) {
      my_timesteps[ i - _my_nearfield_start_idx ] = i;
    }
  }
  for ( lo i = _local_start_idx; i < _local_start_idx + n_local_elems; ++i ) {
    my_timesteps[ n_nearfield_elems + i - _local_start_idx ] = i;
  }
  return my_timesteps;
}

bool besthea::mesh::distributed_spacetime_tensor_mesh::print_ensight_case(
  const std::string & directory, const std::vector< std::string > * node_labels,
  const std::vector< std::string > * element_labels,
  const std::vector< lo > * selected_timesteps, const int root_process ) const {
  // let the root process gather the centers of all selected timesteps
  // in the distributed mesh
  std::vector< lo > const * timesteps_to_consider;
  std::vector< lo > help_vector;
  if ( selected_timesteps == nullptr ) {
    help_vector.resize( _n_global_time_elements );
    for ( lo i = 0; i < _n_global_time_elements; ++i ) {
      help_vector[ i ] = i;
    }
    timesteps_to_consider = &help_vector;
  } else {
    timesteps_to_consider = selected_timesteps;
  }

  std::vector< lo > my_part_of_selected_timesteps;
  for ( lou i = 0; i < timesteps_to_consider->size( ); ++i ) {
    if ( ( *timesteps_to_consider )[ i ] >= _local_start_idx
      && ( *timesteps_to_consider )[ i ]
        < _local_start_idx + _time_mesh->get_n_elements( ) ) {
      my_part_of_selected_timesteps.push_back( global_2_local_time(
        _local_start_idx, ( *timesteps_to_consider )[ i ] ) );
    }
  }
  int n_my_selected_timesteps
    = static_cast< int >( my_part_of_selected_timesteps.size( ) );

  std::vector< sc > my_selected_time_centers( n_my_selected_timesteps, 0.0 );
  for ( lo i = 0; i < n_my_selected_timesteps; ++i ) {
    my_selected_time_centers[ i ]
      = _time_mesh->get_centroid( my_part_of_selected_timesteps[ i ] );
  }

  std::vector< sc > all_selected_time_centers;
  std::vector< int > individual_sizes;
  std::vector< int > displacements;
  if ( _my_rank == root_process ) {
    all_selected_time_centers.resize( timesteps_to_consider->size( ) );
    individual_sizes.resize( _n_processes );
  }
  // first, determine how many selected timesteps each process has
  MPI_Gather( &n_my_selected_timesteps, 1, MPI_INT, individual_sizes.data( ), 1,
    MPI_INT, root_process, *_comm );
  // compute the displacement for the right access in all_selected_time_centers
  if ( _my_rank == root_process ) {
    displacements.resize( _n_processes );
    displacements[ 0 ] = 0;
    for ( lo i = 1; i < _n_processes; ++i ) {
      displacements[ i ] = displacements[ i - 1 ] + individual_sizes[ i - 1 ];
    }
  }
  // now get the centers of the selected timesteps from each process
  MPI_Gatherv( my_selected_time_centers.data( ),
    my_selected_time_centers.size( ), get_scalar_type< sc >::MPI_SC( ),
    all_selected_time_centers.data( ), individual_sizes.data( ),
    displacements.data( ), get_scalar_type< sc >::MPI_SC( ), root_process,
    *_comm );

  bool return_value = true;
  if ( _my_rank == root_process ) {
    std::string filename = directory + "/output.case";

    std::ofstream case_file( filename.c_str( ) );

    if ( !case_file.is_open( ) ) {
      std::cout << "File '" << filename << "' could not be opened!"
                << std::endl;
      return_value = false;
    } else {
      lo n_timesteps = timesteps_to_consider->size( );

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
        case_file << "scalar per node: " << ( *node_labels )[ i ]
                  << " node_data_" << i;
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

        for ( lo i = 0; i < n_timesteps; ++i ) {
          case_file << std::setw( 12 ) << std::setprecision( 5 )
                    << all_selected_time_centers[ i ] << "\n";
        }
      }

      case_file.close( );
    }
  }
  // the root process tells the other processes if the operation was successfull
  MPI_Bcast( &return_value, 1, MPI_CXX_BOOL, root_process, *_comm );
  return return_value;
}

bool besthea::mesh::distributed_spacetime_tensor_mesh::print_ensight_datafiles(
  const std::string & directory, const std::vector< std::string > * node_labels,
  const std::vector< linear_algebra::distributed_block_vector * > * node_data,
  const std::vector< std::string > * element_labels,
  const std::vector< linear_algebra::distributed_block_vector * > *
    element_data,
  const std::vector< lo > * selected_timesteps, const int root_process ) const {
  bool return_value = true;
  lo n_nodal = node_data ? node_data->size( ) : 0;
  lo n_elem = element_data ? element_data->size( ) : 0;
  std::vector< linear_algebra::vector * > node_data_for_one_ts;
  std::vector< linear_algebra::vector * > elem_data_for_one_ts;
  std::vector< linear_algebra::vector > root_node_buffers;
  std::vector< linear_algebra::vector > root_element_buffers;

  lo n_global_space_nodes = _space_mesh->get_n_nodes( );
  lo n_global_space_elements = _space_mesh->get_n_elements( );

  if ( _my_rank == root_process ) {
    node_data_for_one_ts.resize( n_nodal );
    elem_data_for_one_ts.resize( n_elem );
    root_node_buffers.resize( n_nodal );
    for ( lo i = 0; i < n_nodal; ++i ) {
      root_node_buffers[ i ].resize( n_global_space_nodes );
    }
    root_element_buffers.resize( n_elem );
    for ( lo i = 0; i < n_elem; ++i ) {
      root_element_buffers[ i ].resize( n_global_space_elements );
    }
  }

  std::vector< lo > const * timesteps_to_consider;
  std::vector< lo > help_vector;
  if ( selected_timesteps == nullptr ) {
    help_vector.resize( _n_global_time_elements );
    for ( lo i = 0; i < _n_global_time_elements; ++i ) {
      help_vector[ i ] = i;
    }
    timesteps_to_consider = &help_vector;
  } else {
    timesteps_to_consider = selected_timesteps;
  }

  lou ts = 0;
  while ( return_value == true && ts < timesteps_to_consider->size( ) ) {
    // get the appropriate blocks of the distributed nodal vectors (using MPI
    // communication if necessary)
    lo curr_timestep = ( *timesteps_to_consider )[ ts ];
    for ( lo i = 0; i < n_nodal; ++i ) {
      if ( _my_rank == root_process ) {
        if ( ( *node_data )[ i ]->am_i_owner( curr_timestep ) ) {
          node_data_for_one_ts[ i ]
            = &( *node_data )[ i ]->get_block( curr_timestep );
        } else {
          MPI_Status receive_status;
          MPI_Recv( root_node_buffers[ i ].data( ), n_global_space_nodes,
            get_scalar_type< sc >::MPI_SC( ),
            ( *node_data )[ i ]->get_primary_owner( curr_timestep ), ts, *_comm,
            &receive_status );
          node_data_for_one_ts[ i ] = &root_node_buffers[ i ];
        }
      } else {
        std::vector< int > curr_owners
          = ( *node_data )[ i ]->get_owners( )[ curr_timestep ];
        bool root_process_is_owner = false;
        for ( lou k = 0; k < curr_owners.size( ); ++k ) {
          if ( curr_owners[ k ] == root_process ) {
            root_process_is_owner = true;
          }
        }
        if ( !root_process_is_owner
          && _my_rank
            == ( *node_data )[ i ]->get_primary_owner( curr_timestep ) ) {
          MPI_Send( ( *node_data )[ i ]->get_block( curr_timestep ).data( ),
            n_global_space_nodes, get_scalar_type< sc >::MPI_SC( ),
            root_process, ts, *_comm );
        }
      }
    }
    // same for the element vectors
    for ( lo i = 0; i < n_elem; ++i ) {
      if ( _my_rank == root_process ) {
        if ( ( *element_data )[ i ]->am_i_owner( curr_timestep ) ) {
          elem_data_for_one_ts[ i ]
            = &( *element_data )[ i ]->get_block( curr_timestep );
        } else {
          MPI_Status receive_status;
          MPI_Recv( root_element_buffers[ i ].data( ), n_global_space_elements,
            get_scalar_type< sc >::MPI_SC( ),
            ( *element_data )[ i ]->get_primary_owner( curr_timestep ), ts,
            *_comm, &receive_status );
          elem_data_for_one_ts[ i ] = &root_element_buffers[ i ];
        }
      } else {
        std::vector< int > curr_owners
          = ( *element_data )[ i ]->get_owners( )[ curr_timestep ];
        bool root_process_is_owner = false;
        for ( lou k = 0; k < curr_owners.size( ); ++k ) {
          if ( curr_owners[ k ] == root_process ) {
            root_process_is_owner = true;
          }
        }
        if ( !root_process_is_owner
          && _my_rank
            == ( *element_data )[ i ]->get_primary_owner( curr_timestep ) ) {
          MPI_Send( ( *element_data )[ i ]->get_block( curr_timestep ).data( ),
            n_global_space_elements, get_scalar_type< sc >::MPI_SC( ),
            root_process, ts, *_comm );
        }
      }
    }
    // print the datafiles for the current timestep.
    // note: while there may be jumps between selected timesteps, we still
    // number the resulting files consecutively (not with respect to the
    // original indices of the timesteps)
    if ( _my_rank == root_process ) {
      return_value
        = _space_mesh->print_ensight_datafiles( directory, node_labels,
          &node_data_for_one_ts, element_labels, &elem_data_for_one_ts, ts );
    }
    MPI_Bcast( &return_value, 1, MPI_CXX_BOOL, 0, *_comm );
    ++ts;
  }
  return return_value;
}
