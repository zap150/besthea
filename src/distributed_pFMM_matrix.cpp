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

#include "besthea/distributed_pFMM_matrix.h"

#include "besthea/auxiliary_routines.h"
#include "besthea/distributed_fast_spacetime_be_assembler.h"
#include "besthea/fmm_routines.h"
#include "besthea/quadrature.h"
#include "besthea/timer.h"

#include <assert.h>
#include <filesystem>
#include <mkl_rci.h>
#include <set>
#include <sstream>

using besthea::linear_algebra::full_matrix;
using besthea::mesh::distributed_spacetime_cluster_tree;
using besthea::mesh::general_spacetime_cluster;
using besthea::mesh::scheduling_time_cluster;

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply( const block_vector & /*x*/,
  block_vector & /*y*/, bool /*trans*/, sc /*alpha*/, sc /*beta*/ ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED for standard block vectors. Please use "
               "distributed block vectors!"
            << std::endl;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::set_trees( distributed_spacetime_cluster_tree *
    distributed_spacetime_tree ) {
  _distributed_spacetime_tree = distributed_spacetime_tree;
  _scheduling_tree_structure
    = _distributed_spacetime_tree->get_distribution_tree( );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::determine_clusters_with_nearfield_operations( ) {
  // determine the clusters, for which nearfield operations have to be executed.
  determine_clusters_with_nearfield_operations_recursively(
    _distributed_spacetime_tree->get_root( ) );
  _clusters_with_nearfield_operations.shrink_to_fit( );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::initialize_nearfield_containers( ) {
  if ( _clusters_with_nearfield_operations.size( ) == 0 ) {
    determine_clusters_with_nearfield_operations( );
  }
  // for all clusters with nearfield operations, create a proper entry in the
  // map _clusterwise_nf_matrices
  for ( auto cluster : _clusters_with_nearfield_operations ) {
    if ( cluster->get_n_children( ) == 0 ) {
      // standard nearfield operations are executed for leaf clusters and the
      // clusters in their nearfield
      _clusterwise_nf_matrices.insert(
        { cluster, std::vector< full_matrix * >( ) } );
      _clusterwise_nf_matrices[ cluster ].resize(
        cluster->get_nearfield_list( )->size( ), nullptr );
    }
    // for clusters in the spatially admissible nearfield list nearfield
    // operations compressed by aca have to be executed.
    auto spat_adm_nearfield_list
      = cluster->get_spatially_admissible_nearfield_list( );
    if ( spat_adm_nearfield_list != nullptr ) {
      _clusterwise_spat_adm_nf_matrix_pairs.insert(
        { cluster, std::vector< std::pair< lo, matrix * > >( ) } );
      _clusterwise_spat_adm_nf_matrix_pairs[ cluster ].reserve(
        spat_adm_nearfield_list->size( ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  determine_clusters_with_nearfield_operations_recursively(
    mesh::general_spacetime_cluster * current_cluster ) {
  if ( current_cluster->get_process_id( ) == _my_rank ) {
    if ( current_cluster->get_n_children( ) == 0
      || current_cluster->get_spatially_admissible_nearfield_list( )
        != nullptr ) {
      _clusters_with_nearfield_operations.push_back( current_cluster );
    }
  }
  // call the routine recursively for all children
  if ( current_cluster->get_n_children( ) > 0 ) {
    for ( auto child : *current_cluster->get_children( ) ) {
      determine_clusters_with_nearfield_operations_recursively( child );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::prepare_fmm( ) {
  _scheduling_tree_structure->init_fmm_lists(
    *_scheduling_tree_structure->get_root( ), _m_list, _m2l_list, _l_list,
    _m2t_list, _s2l_list, _n_list );
  // sort the lists according to the priorities of the operations. E.g. clusters
  // in the m-list should be sorted from bottom to top, from right to left.
  _m_list.sort(
    _scheduling_tree_structure->compare_clusters_bottom_up_right_2_left );
  _m2l_list.sort(
    _scheduling_tree_structure->compare_clusters_top_down_right_2_left );
  _l_list.sort(
    _scheduling_tree_structure->compare_clusters_top_down_right_2_left );
  _s2l_list.sort(
    _scheduling_tree_structure->compare_clusters_top_down_right_2_left );
  _m2t_list.sort(
    _scheduling_tree_structure->compare_clusters_top_down_right_2_left );
  // the n-list is sorted in the routine
  // sort_clusters_in_n_list_and_associated_st_targets

  // allocate the timers
  for ( auto & it : _m_task_times ) {
    it.reserve( 2 * _m_list.size( ) );
  }
  for ( auto & it : _m2l_task_times ) {
    it.reserve( 2 * _m2l_list.size( ) );
  }
  for ( auto & it : _l_task_times ) {
    it.reserve( 2 * _l_list.size( ) );
  }
  for ( auto & it : _m2t_task_times ) {
    it.reserve( 2 * _m2t_list.size( ) );
  }
  for ( auto & it : _s2l_task_times ) {
    it.reserve( 2 * _s2l_list.size( ) );
  }
  for ( auto & it : _n_task_times ) {
    it.reserve( 2 * _n_list.size( ) );
  }
  for ( auto & it : _m_subtask_times ) {
    it.reserve( 4 * _m_list.size( ) );
  }
  for ( auto & it : _m2l_subtask_times ) {
    it.reserve( 4 * _m2l_list.size( ) );
  }
  for ( auto & it : _l_subtask_times ) {
    it.reserve( 4 * _l_list.size( ) );
  }
  for ( auto & it : _m2t_subtask_times ) {
    it.reserve( 4 * _m2t_list.size( ) );
  }
  for ( auto & it : _s2l_subtask_times ) {
    it.reserve( 4 * _s2l_list.size( ) );
  }
  for ( auto & it : _n_subtask_times ) {
    it.reserve( 4 * _n_list.size( ) );
  }
  for ( auto & it : _mpi_send_m2l_m2t_or_s2l ) {
    it.reserve( _m2l_list.size( ) );
  }
  for ( auto & it : _mpi_send_m_parent ) {
    it.reserve( _m_list.size( ) );
  }
  for ( auto & it : _mpi_send_l_children ) {
    it.reserve( _l_list.size( ) );
  }
  for ( auto & it : _mpi_recv_m2l_m2t_or_s2l ) {
    it.reserve( _m2l_list.size( ) );
  }
  for ( auto & it : _mpi_recv_m_parent ) {
    it.reserve( _m_list.size( ) );
  }
  for ( auto & it : _mpi_recv_l_children ) {
    it.reserve( _l_list.size( ) );
  }

  // set the positions of clusters in the lists
  lo counter = 0;
  for ( auto clst : _m_list ) {
    clst->set_pos_in_m_list( counter );
    counter++;
  }
  counter = 0;
  for ( auto clst : _l_list ) {
    clst->set_pos_in_l_list( counter );
    counter++;
  }
  counter = 0;
  for ( auto clst : _m2l_list ) {
    clst->set_pos_in_m2l_list( counter );
    counter++;
  }
  counter = 0;
  for ( auto clst : _s2l_list ) {
    clst->set_pos_in_s2l_list( counter );
    counter++;
  }

  // fill the receive list by determining all incoming data.
  // check for receive operations in the upward path
  for ( auto it = _m_list.begin( ); it != _m_list.end( ); ++it ) {
    std::vector< scheduling_time_cluster * > * children
      = ( *it )->get_children( );
    if ( children != nullptr ) {
      for ( auto it_child = children->begin( ); it_child != children->end( );
            ++it_child ) {
        // if the current cluster has a child which is handled by a different
        // process p the current process has to receive the processed moments
        // from p.
        if ( ( *it_child )->get_process_id( ) != _my_rank ) {
          _receive_data_information.push_back(
            { *it, ( *it_child )->get_process_id( ) } );
          ( *it )->add_receive_buffer( ( *it_child )->get_process_id( ) );
        }
      }
    }
  }
  _n_moments_to_receive_upward = _receive_data_information.size( );

  // check for receive operations in the interaction phase
  for ( auto it = _m2l_list.begin( ); it != _m2l_list.end( ); ++it ) {
    std::vector< scheduling_time_cluster * > * interaction_list
      = ( *it )->get_interaction_list( );
    // interaction list is never empty for clusters in the m2l_list, so it is
    // not checked, whether the pointer is nullptr
    for ( auto it_src = interaction_list->begin( );
          it_src != interaction_list->end( ); ++it_src ) {
      // if the source cluster is handled by a different process p the current
      // process has to receive its moments from p.
      if ( ( *it_src )->get_process_id( ) != _my_rank ) {
        _receive_data_information.push_back(
          { *it_src, ( *it_src )->get_process_id( ) } );
      }
    }
  }

  // check for receive operations for m2t operations
  for ( auto it = _m2t_list.begin( ); it != _m2t_list.end( ); ++it ) {
    std::vector< scheduling_time_cluster * > * clusters_m2t_list
      = ( *it )->get_m2t_list( );
    // clusters_m2t_list is never empty for clusters in the m2t-list!
    for ( auto it_src = clusters_m2t_list->begin( );
          it_src != clusters_m2t_list->end( ); ++it_src ) {
      // if the source cluster is handled by a different process p the current
      // process has to receive its moments from p.
      if ( ( *it_src )->get_process_id( ) != _my_rank ) {
        _receive_data_information.push_back(
          { *it_src, ( *it_src )->get_process_id( ) } );
      }
    }
  }

  // the moments of each cluster have to be received only once -> find and
  // eliminate double entries in the second part of the receive vector
  std::sort( _receive_data_information.begin( ) + _n_moments_to_receive_upward,
    _receive_data_information.end( ),
    [ & ]( const std::pair< scheduling_time_cluster *, lo > pair_one,
      const std::pair< scheduling_time_cluster *, lo > pair_two ) {
      return _scheduling_tree_structure->compare_clusters_top_down_right_2_left(
        pair_one.first, pair_two.first );
    } );
  auto new_end = std::unique(
    _receive_data_information.begin( ) + _n_moments_to_receive_upward,
    _receive_data_information.end( ),
    [ & ]( const std::pair< scheduling_time_cluster *, lo > pair_one,
      const std::pair< scheduling_time_cluster *, lo > pair_two ) {
      return pair_one.first == pair_two.first;
    } );
  _receive_data_information.resize(
    std::distance( _receive_data_information.begin( ), new_end ) );
  _n_moments_to_receive_m2l_or_m2t
    = _receive_data_information.size( ) - _n_moments_to_receive_upward;
  lo n_st_moments = _receive_data_information.size( );

  // check for receive operations for s2l operations (spatial moments)
  for ( auto it = _s2l_list.begin( ); it != _s2l_list.end( ); ++it ) {
    std::vector< mesh::scheduling_time_cluster * > * cluster_s2l_list
      = ( *it )->get_s2l_list( );
    for ( auto s2l_source_cluster : *cluster_s2l_list ) {
      if ( s2l_source_cluster->get_n_children( ) == 0
        && s2l_source_cluster->get_process_id( ) != _my_rank ) {
        _receive_data_information.push_back(
          { s2l_source_cluster, s2l_source_cluster->get_process_id( ) } );
      }
    }
  }
  // spatial moments of each cluster have to be received only once -> find and
  // eliminate double entries in the third part of the receive vector
  std::sort( _receive_data_information.begin( ) + n_st_moments,
    _receive_data_information.end( ),
    [ & ]( const std::pair< scheduling_time_cluster *, lo > pair_one,
      const std::pair< scheduling_time_cluster *, lo > pair_two ) {
      return _scheduling_tree_structure->compare_clusters_top_down_right_2_left(
        pair_one.first, pair_two.first );
    } );
  new_end = std::unique( _receive_data_information.begin( ) + n_st_moments,
    _receive_data_information.end( ),
    [ & ]( const std::pair< scheduling_time_cluster *, lo > pair_one,
      const std::pair< scheduling_time_cluster *, lo > pair_two ) {
      return pair_one.first == pair_two.first;
    } );
  _receive_data_information.resize(
    std::distance( _receive_data_information.begin( ), new_end ) );
  _n_spatial_moments_to_receive
    = _receive_data_information.size( ) - n_st_moments;

  // check for receive operations in the downward path
  for ( auto it = _l_list.begin( ); it != _l_list.end( ); ++it ) {
    scheduling_time_cluster * parent = ( *it )->get_parent( );
    // if the parent cluster is handled by a different process p the current
    // process has to receive its local contributions from p.
    if ( parent->get_process_id( ) != _my_rank
      && parent->get_process_id( ) != -1 ) {
      _receive_data_information.push_back(
        { parent, parent->get_process_id( ) } );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
besthea::linear_algebra::full_matrix *
besthea::linear_algebra::distributed_pFMM_matrix< kernel_type, target_space,
  source_space >::create_nearfield_matrix( lou nf_cluster_index,
  lou source_index ) {
  general_spacetime_cluster * target_cluster
    = _clusters_with_nearfield_operations[ nf_cluster_index ];
  general_spacetime_cluster * source_cluster
    = ( *( target_cluster->get_nearfield_list( ) ) )[ source_index ];
  lo n_dofs_source = source_cluster->get_n_dofs< source_space >( );
  lo n_dofs_target = target_cluster->get_n_dofs< target_space >( );
  full_matrix * local_matrix = new full_matrix( n_dofs_target, n_dofs_source );

  _clusterwise_nf_matrices[ target_cluster ][ source_index ] = local_matrix;

  return local_matrix;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  insert_spatially_admissible_nearfield_matrix(
    lou nf_cluster_index, lou source_index, matrix * nf_matrix ) {
  general_spacetime_cluster * target_cluster
    = _clusters_with_nearfield_operations[ nf_cluster_index ];
  _clusterwise_spat_adm_nf_matrix_pairs[ target_cluster ].push_back(
    { source_index, nf_matrix } );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::initialize_spatial_m2m_coeffs( ) {
  lo n_space_levels = _distributed_spacetime_tree->get_global_n_space_levels( );
  vector_type root_half_size( 3, false );
  sc dummy_val;
  _distributed_spacetime_tree->get_root( )->get_half_size(
    root_half_size, dummy_val );

  compute_spatial_m2m_coeffs( n_space_levels, _spat_order, root_half_size[ 0 ],
    _distributed_spacetime_tree->get_spatial_paddings_per_spatial_level( ),
    _m2m_coeffs_s_dim_0_left, _m2m_coeffs_s_dim_0_right );
  // @todo Due to the cubic bounding boxes the m2m coefficients are the same
  // for all dimensions. Get rid of redundant vectors
  _m2m_coeffs_s_dim_1_left = _m2m_coeffs_s_dim_0_left;
  _m2m_coeffs_s_dim_1_right = _m2m_coeffs_s_dim_0_right;
  _m2m_coeffs_s_dim_2_left = _m2m_coeffs_s_dim_0_left;
  _m2m_coeffs_s_dim_2_right = _m2m_coeffs_s_dim_0_right;
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_chebyshev( ) {
  // first, resize the relevant containers of the class that are used to store
  // chebyshev nodes and evaluated chebyshev polynomials
  _cheb_nodes_integrate.resize( _m2l_integration_order + 1 );
  _all_poly_vals_integrate.resize(
    ( _m2l_integration_order + 1 ) * ( _spat_order + 1 ) );

  // initialize the Chebyshev nodes
  for ( lo i = 0; i <= _m2l_integration_order; ++i ) {
    _cheb_nodes_integrate[ i ] = std::cos(
      M_PI * ( 2 * i + 1 ) / ( 2 * ( _m2l_integration_order + 1 ) ) );
  }

  // evaluate Chebyshev polynomials for all degrees <= _spat_order for
  // integrals
  _chebyshev.evaluate( _cheb_nodes_integrate, _all_poly_vals_integrate );

  if ( _cheb_nodes_sum_coll.size( )
    != (lou) _cheb_nodes_integrate.size( ) * _cheb_nodes_integrate.size( ) ) {
    _cheb_nodes_sum_coll.resize(
      _cheb_nodes_integrate.size( ) * _cheb_nodes_integrate.size( ) );
  }
  lo counter = 0;

  for ( lo mu = 0; mu < _cheb_nodes_integrate.size( ); ++mu ) {
    for ( lo nu = 0; nu < _cheb_nodes_integrate.size( ); ++nu ) {
      _cheb_nodes_sum_coll[ counter ]
        = _cheb_nodes_integrate[ mu ] - _cheb_nodes_integrate[ nu ];
      ++counter;
    }
  }

  if ( _all_poly_vals_mult_coll.size( )
    != (lou) ( _spat_order + 1 ) * ( _spat_order + 1 )
      * _cheb_nodes_integrate.size( ) * _cheb_nodes_integrate.size( ) ) {
    _all_poly_vals_mult_coll.resize( ( _spat_order + 1 ) * ( _spat_order + 1 )
      * _cheb_nodes_integrate.size( ) * _cheb_nodes_integrate.size( ) );
  }

  counter = 0;

  for ( lo alpha = 0; alpha <= _spat_order; ++alpha ) {
    for ( lo beta = 0; beta <= _spat_order; ++beta ) {
      for ( lo mu = 0; mu < _cheb_nodes_integrate.size( ); ++mu ) {
        for ( lo nu = 0; nu < _cheb_nodes_integrate.size( ); ++nu ) {
          _all_poly_vals_mult_coll[ counter ]
            = _all_poly_vals_integrate[ alpha * _cheb_nodes_integrate.size( )
                + mu ]
            * _all_poly_vals_integrate[ beta * _cheb_nodes_integrate.size( )
              + nu ];
          ++counter;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::print_information( const int root_process,
  const bool print_tree_information ) {
  // first print some information of the underlying distributed space time
  // tree
  if ( print_tree_information ) {
    _distributed_spacetime_tree->print_information( root_process );
  }

  // compute the number of nearfield entries on each process (total, spatially
  // admissible, spatially admissible compressed) and separately those
  // corresponding to time-separated clusters
  std::vector< long long > loc_lvlwise_tot_nf_entries_uncompressed;
  std::vector< long long > loc_lvlwise_spat_adm_nf_entries_no_compr;
  std::vector< long long > loc_lvlwise_spat_adm_nf_entries_compr;
  std::vector< long long > loc_lvlwise_time_separated_nf_entries;
  long long loc_max_n_nf_entries_per_st_target;
  sc loc_mean_n_nf_entries_per_st_target;
  count_nearfield_entries_levelwise( loc_lvlwise_tot_nf_entries_uncompressed,
    loc_lvlwise_spat_adm_nf_entries_no_compr,
    loc_lvlwise_spat_adm_nf_entries_compr, loc_max_n_nf_entries_per_st_target,
    loc_mean_n_nf_entries_per_st_target );
  bool using_m2t_and_s2l_operations
    = _distributed_spacetime_tree->get_distribution_tree( )
        ->supports_m2t_and_s2l_operations( );
  if ( !using_m2t_and_s2l_operations ) {
    count_time_separated_nearfield_entries_levelwise(
      loc_lvlwise_time_separated_nf_entries );
  }

  long long n_discarded_blocks, n_tot_size_discarded_blocks,
    n_compressed_blocks, n_tot_size_compressed_blocks, n_uncompressed_blocks,
    n_tot_size_uncompressed_blocks;
  if ( _aca_eps > 0.0 ) {
    // collect_information_of_spatially_admissible_nearfield_operations( )
    collect_information_spatially_admissible_nearfield_operations(
      n_discarded_blocks, n_tot_size_discarded_blocks, n_compressed_blocks,
      n_tot_size_compressed_blocks, n_uncompressed_blocks,
      n_tot_size_uncompressed_blocks );
    if ( _my_rank == root_process ) {
      MPI_Reduce( MPI_IN_PLACE, &n_discarded_blocks, 1, MPI_LONG_LONG_INT,
        MPI_SUM, root_process, *_comm );
      MPI_Reduce( MPI_IN_PLACE, &n_tot_size_discarded_blocks, 1,
        MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
      MPI_Reduce( MPI_IN_PLACE, &n_compressed_blocks, 1, MPI_LONG_LONG_INT,
        MPI_SUM, root_process, *_comm );
      MPI_Reduce( MPI_IN_PLACE, &n_tot_size_compressed_blocks, 1,
        MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
      MPI_Reduce( MPI_IN_PLACE, &n_uncompressed_blocks, 1, MPI_LONG_LONG_INT,
        MPI_SUM, root_process, *_comm );
      MPI_Reduce( MPI_IN_PLACE, &n_tot_size_uncompressed_blocks, 1,
        MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    } else {
      MPI_Reduce( &n_discarded_blocks, nullptr, 1, MPI_LONG_LONG_INT, MPI_SUM,
        root_process, *_comm );
      MPI_Reduce( &n_tot_size_discarded_blocks, nullptr, 1, MPI_LONG_LONG_INT,
        MPI_SUM, root_process, *_comm );
      MPI_Reduce( &n_compressed_blocks, nullptr, 1, MPI_LONG_LONG_INT, MPI_SUM,
        root_process, *_comm );
      MPI_Reduce( &n_tot_size_compressed_blocks, nullptr, 1, MPI_LONG_LONG_INT,
        MPI_SUM, root_process, *_comm );
      MPI_Reduce( &n_uncompressed_blocks, nullptr, 1, MPI_LONG_LONG_INT,
        MPI_SUM, root_process, *_comm );
      MPI_Reduce( &n_tot_size_uncompressed_blocks, nullptr, 1,
        MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    }
  }

  lo n_global_tree_levels = _distributed_spacetime_tree->get_n_levels( );
  // gather the number of nearfield entries at the root process
  int n_processes;
  MPI_Comm_size( *_comm, &n_processes );
  long long * all_loc_tot_nf_entries = nullptr;
  long long * all_loc_spat_adm_nf_entries = nullptr;
  long long * all_loc_compr_spat_adm_nf_entries = nullptr;
  long long * all_loc_separated_entries = nullptr;
  long long * all_max_n_nf_entries_st_targets = nullptr;
  sc * all_mean_n_nf_entries_st_targets = nullptr;
  if ( _my_rank == root_process ) {
    all_loc_tot_nf_entries
      = new long long[ n_processes * n_global_tree_levels ];
    all_loc_spat_adm_nf_entries
      = new long long[ n_processes * n_global_tree_levels ];
    all_loc_compr_spat_adm_nf_entries
      = new long long[ n_processes * n_global_tree_levels ];
    all_loc_separated_entries
      = new long long[ n_processes * n_global_tree_levels ];
    all_max_n_nf_entries_st_targets = new long long[ n_processes ];
    all_mean_n_nf_entries_st_targets = new sc[ n_processes ];
  }
  MPI_Gather( loc_lvlwise_tot_nf_entries_uncompressed.data( ),
    n_global_tree_levels, MPI_LONG_LONG_INT, all_loc_tot_nf_entries,
    n_global_tree_levels, MPI_LONG_LONG_INT, root_process, *_comm );
  MPI_Gather( loc_lvlwise_spat_adm_nf_entries_no_compr.data( ),
    n_global_tree_levels, MPI_LONG_LONG_INT, all_loc_spat_adm_nf_entries,
    n_global_tree_levels, MPI_LONG_LONG_INT, root_process, *_comm );
  MPI_Gather( loc_lvlwise_spat_adm_nf_entries_compr.data( ),
    n_global_tree_levels, MPI_LONG_LONG_INT, all_loc_compr_spat_adm_nf_entries,
    n_global_tree_levels, MPI_LONG_LONG_INT, root_process, *_comm );
  MPI_Gather( &loc_max_n_nf_entries_per_st_target, 1, MPI_LONG_LONG_INT,
    all_max_n_nf_entries_st_targets, 1, MPI_LONG_LONG_INT, root_process,
    *_comm );
  MPI_Gather( &loc_mean_n_nf_entries_per_st_target, 1,
    get_scalar_type< sc >::MPI_SC( ), all_mean_n_nf_entries_st_targets, 1,
    get_scalar_type< sc >::MPI_SC( ), root_process, *_comm );
  if ( !using_m2t_and_s2l_operations ) {
    MPI_Gather( loc_lvlwise_time_separated_nf_entries.data( ),
      n_global_tree_levels, MPI_LONG_LONG_INT, all_loc_separated_entries,
      n_global_tree_levels, MPI_LONG_LONG_INT, root_process, *_comm );
  }

  // count the fmm operations levelwise
  std::vector< long long > n_s2m_operations, n_m2m_operations, n_m2l_operations,
    n_l2l_operations, n_l2t_operations, n_s2l_operations, n_m2t_operations;
  count_fmm_operations_levelwise( n_s2m_operations, n_m2m_operations,
    n_m2l_operations, n_l2l_operations, n_l2t_operations, n_s2l_operations,
    n_m2t_operations );
  // collect the numbers of operations at the root process via reduce
  // operations
  if ( _my_rank == root_process ) {
    MPI_Reduce( MPI_IN_PLACE, n_s2m_operations.data( ), n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_m2m_operations.data( ), n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_m2l_operations.data( ), n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_l2l_operations.data( ), n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_l2t_operations.data( ), n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_s2l_operations.data( ), n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, n_m2t_operations.data( ), n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
  } else {
    MPI_Reduce( n_s2m_operations.data( ), nullptr, n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_m2m_operations.data( ), nullptr, n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_m2l_operations.data( ), nullptr, n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_l2l_operations.data( ), nullptr, n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_l2t_operations.data( ), nullptr, n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_s2l_operations.data( ), nullptr, n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
    MPI_Reduce( n_m2t_operations.data( ), nullptr, n_global_tree_levels,
      MPI_LONG_LONG_INT, MPI_SUM, root_process, *_comm );
  }
  // count the number of allocated moments (own and received) and
  // local contributions
  lo local_n_moments( 0 ), local_n_moments_receive( 0 ),
    local_n_local_contributions( 0 ), local_n_spat_moments( 0 ),
    local_n_spat_local_contributions( 0 );
  _scheduling_tree_structure->count_number_of_contributions(
    _scheduling_tree_structure->get_root( ), local_n_moments,
    local_n_moments_receive, local_n_local_contributions, local_n_spat_moments,
    local_n_spat_local_contributions );
  lo * all_n_moments = nullptr;
  lo * all_n_moments_receive = nullptr;
  lo * all_n_local_contributions = nullptr;
  lo * all_n_spat_moments = nullptr;
  lo * all_n_spat_local_contributions = nullptr;
  if ( _my_rank == root_process ) {
    all_n_moments = new lo[ n_processes ];
    all_n_moments_receive = new lo[ n_processes ];
    all_n_local_contributions = new lo[ n_processes ];
    all_n_spat_moments = new lo[ n_processes ];
    all_n_spat_local_contributions = new lo[ n_processes ];
  }
  // gather the computed numbers at the root process
  MPI_Gather( &local_n_moments, 1, get_index_type< lo >::MPI_LO( ),
    all_n_moments, 1, get_index_type< lo >::MPI_LO( ), root_process, *_comm );
  MPI_Gather( &local_n_moments_receive, 1, get_index_type< lo >::MPI_LO( ),
    all_n_moments_receive, 1, get_index_type< lo >::MPI_LO( ), root_process,
    *_comm );
  MPI_Gather( &local_n_local_contributions, 1, get_index_type< lo >::MPI_LO( ),
    all_n_local_contributions, 1, get_index_type< lo >::MPI_LO( ), root_process,
    *_comm );
  MPI_Gather( &local_n_spat_moments, 1, get_index_type< lo >::MPI_LO( ),
    all_n_spat_moments, 1, get_index_type< lo >::MPI_LO( ), root_process,
    *_comm );
  MPI_Gather( &local_n_spat_local_contributions, 1,
    get_index_type< lo >::MPI_LO( ), all_n_spat_local_contributions, 1,
    get_index_type< lo >::MPI_LO( ), root_process, *_comm );
  // postprocessing and printing on the root process
  if ( _my_rank == root_process ) {
    std::cout << "Farfield part:" << std::endl;
    lo start_space_refinement
      = _distributed_spacetime_tree->get_start_space_refinement( );
    std::cout << "number of s2m operations: " << std::endl;
    for ( lo i = 2; i < n_global_tree_levels; ++i ) {
      std::cout << "level " << i << ": " << n_s2m_operations[ i ] << std::endl;
    }
    std::cout << "number of m2m operations: " << std::endl;
    for ( lo i = 2; i < n_global_tree_levels; ++i ) {
      std::cout << "level " << i << ": " << n_m2m_operations[ i ];
      if ( i >= start_space_refinement
        && ( ( i - start_space_refinement ) ) % 2 == 0 ) {
        std::cout << " spacetime m2m" << std::endl;
      } else {
        std::cout << " temporal m2m" << std::endl;
      }
    }
    std::cout << "number of m2l operations: " << std::endl;
    for ( lo i = 2; i < n_global_tree_levels; ++i ) {
      std::cout << "level " << i << ": " << n_m2l_operations[ i ] << std::endl;
    }
    std::cout << "number of l2l operations: " << std::endl;
    for ( lo i = 2; i < n_global_tree_levels; ++i ) {
      std::cout << "level " << i << ": " << n_l2l_operations[ i ];
      if ( i >= start_space_refinement
        && ( ( i - start_space_refinement ) ) % 2 == 0 ) {
        std::cout << " spacetime l2l" << std::endl;
      } else {
        std::cout << " temporal l2l" << std::endl;
      }
    }
    std::cout << "number of l2t operations: " << std::endl;
    for ( lo i = 2; i < n_global_tree_levels; ++i ) {
      std::cout << "level " << i << ": " << n_l2t_operations[ i ] << std::endl;
    }
    if ( using_m2t_and_s2l_operations ) {
      std::cout << "number of s2l operations: " << std::endl;
      for ( lo i = 2; i < n_global_tree_levels; ++i ) {
        std::cout << "level " << i << ": " << n_s2l_operations[ i ]
                  << std::endl;
      }
      std::cout << "number of m2t operations: " << std::endl;
      for ( lo i = 2; i < n_global_tree_levels; ++i ) {
        std::cout << "level " << i << ": " << n_m2t_operations[ i ]
                  << std::endl;
      }
    }
    std::cout << "########################" << std::endl;
    std::cout << "rough memory estimates: " << std::endl;
    std::cout << "########################" << std::endl;
    lo n_target_dofs
      = _distributed_spacetime_tree->get_mesh( ).get_n_dofs< target_space >( );
    lo n_source_dofs
      = _distributed_spacetime_tree->get_mesh( ).get_n_dofs< source_space >( );
    // print first the estimates for the farfield part
    std::cout << "Required storage for moments and local contributions"
              << std::endl;
    sc total_storage_contributions = 0.0;
    for ( int i = 0; i < n_processes; ++i ) {
      sc local_storage_contributions
        = ( all_n_moments[ i ] + all_n_moments_receive[ i ]
            + all_n_local_contributions[ i ] )
        * 8. * _contribution_size / 1024. / 1024. / 1024.;
      if ( using_m2t_and_s2l_operations ) {
        local_storage_contributions
          += ( all_n_spat_moments[ i ] + all_n_spat_local_contributions[ i ] )
          * 8. * _spat_contribution_size / 1024. / 1024. / 1024.;
      }
      total_storage_contributions += local_storage_contributions;
      std::cout << "process " << i
                << ": storage for moment and local contributions";
      if ( using_m2t_and_s2l_operations ) {
        std::cout << " (including spatial moment and local contributions)";
      }
      std::cout << ": " << local_storage_contributions << " GiB." << std::endl;
    }
    std::cout << "total storage for moments and local contributions";
    if ( using_m2t_and_s2l_operations ) {
      std::cout << " (including spatial moment and local contributions)";
    }
    std::cout << ": " << total_storage_contributions << " GiB." << std::endl;
    // next, estimate the storage for the nearfield part
    std::vector< sc > global_levelwise_nearfield_storage(
      n_global_tree_levels, 0.0 );
    std::vector< sc > global_levelwise_nearfield_storage_spat_adm_no_compr(
      n_global_tree_levels, 0.0 );
    std::vector< sc > global_levelwise_nearfield_storage_spat_adm_compr(
      n_global_tree_levels, 0.0 );
    std::vector< sc > global_levelwise_time_separated_nearfield_storage(
      n_global_tree_levels, 0.0 );
    std::vector< sc > nearfield_storage_on_processes( n_processes, 0.0 );
    std::vector< sc > nearfield_storage_on_processes_spat_adm_no_compr(
      n_processes, 0.0 );
    std::vector< sc > nearfield_storage_on_processes_spat_adm_compr(
      n_processes, 0.0 );
    std::vector< sc > time_separated_nearfield_storage_on_processes(
      n_processes, 0.0 );
    // add up the nearfield entries processwise or levelwise for the respective
    // counters (and all of them for a total count)
    long long total_nearfield_entries = 0;
    long long total_nearfield_entries_spat_adm_no_compr = 0;
    long long total_nearfield_entries_spat_adm_compr = 0;
    // in case that m2t and s2l operations are used, summing time separated
    // nearfield storage is obsolete, because all entries of the corresponding
    // arrays are zero. still it is done, since the effort is negligible.
    long long total_nearfield_entries_time_separated_part = 0;
    for ( lo i = 0; i < n_processes; ++i ) {
      for ( lo j = 0; j < n_global_tree_levels; ++j ) {
        global_levelwise_nearfield_storage[ j ]
          += all_loc_tot_nf_entries[ n_global_tree_levels * i + j ];
        global_levelwise_nearfield_storage_spat_adm_no_compr[ j ]
          += all_loc_spat_adm_nf_entries[ n_global_tree_levels * i + j ];
        global_levelwise_nearfield_storage_spat_adm_compr[ j ]
          += all_loc_compr_spat_adm_nf_entries[ n_global_tree_levels * i + j ];
        global_levelwise_time_separated_nearfield_storage[ j ]
          += all_loc_separated_entries[ n_global_tree_levels * i + j ];
        nearfield_storage_on_processes[ i ]
          += all_loc_tot_nf_entries[ n_global_tree_levels * i + j ];
        nearfield_storage_on_processes_spat_adm_no_compr[ i ]
          += all_loc_spat_adm_nf_entries[ n_global_tree_levels * i + j ];
        nearfield_storage_on_processes_spat_adm_compr[ i ]
          += all_loc_compr_spat_adm_nf_entries[ n_global_tree_levels * i + j ];
        time_separated_nearfield_storage_on_processes[ i ]
          += all_loc_separated_entries[ n_global_tree_levels * i + j ];
        total_nearfield_entries
          += all_loc_tot_nf_entries[ n_global_tree_levels * i + j ];
        total_nearfield_entries_spat_adm_no_compr
          += all_loc_spat_adm_nf_entries[ n_global_tree_levels * i + j ];
        total_nearfield_entries_spat_adm_compr
          += all_loc_compr_spat_adm_nf_entries[ n_global_tree_levels * i + j ];
        total_nearfield_entries_time_separated_part
          += all_loc_separated_entries[ n_global_tree_levels * i + j ];
      }
    }
    // compute required memory for storing nearfield entries in GiB levelwise,
    // processwise and total
    for ( lo i = 0; i < n_processes; ++i ) {
      nearfield_storage_on_processes[ i ] *= 8. / 1024. / 1024. / 1024.;
      nearfield_storage_on_processes_spat_adm_no_compr[ i ]
        *= 8. / 1024. / 1024. / 1024.;
      nearfield_storage_on_processes_spat_adm_compr[ i ]
        *= 8. / 1024. / 1024. / 1024.;
      time_separated_nearfield_storage_on_processes[ i ]
        *= 8. / 1024. / 1024. / 1024.;
    }
    for ( lo i = 0; i < n_global_tree_levels; ++i ) {
      global_levelwise_nearfield_storage[ i ] *= 8. / 1024. / 1024. / 1024.;
      global_levelwise_nearfield_storage_spat_adm_no_compr[ i ]
        *= 8. / 1024. / 1024. / 1024.;
      global_levelwise_nearfield_storage_spat_adm_compr[ i ]
        *= 8. / 1024. / 1024. / 1024.;
      global_levelwise_time_separated_nearfield_storage[ i ]
        *= 8. / 1024. / 1024. / 1024.;
    }
    long long total_nf_entries_compr = total_nearfield_entries
      - total_nearfield_entries_spat_adm_no_compr
      + total_nearfield_entries_spat_adm_compr;
    sc total_storage_nearfield
      = total_nearfield_entries * 8. / 1024. / 1024. / 1024.;
    sc total_storage_spat_adm_nf_no_compr
      = total_nearfield_entries_spat_adm_no_compr * 8. / 1024. / 1024. / 1024.;
    sc total_storage_spat_adm_nf_compr
      = total_nearfield_entries_spat_adm_compr * 8. / 1024. / 1024. / 1024.;
    sc total_storage_time_separated_part
      = total_nearfield_entries_time_separated_part * 8. / 1024. / 1024.
      / 1024.;

    std::cout
      << "Required storage for nearfield matrices per process (uncompressed): "
      << std::endl;
    for ( lo i = 0; i < n_processes; ++i ) {
      std::cout << "process " << i << ": nearfield_matrices: "
                << nearfield_storage_on_processes[ i ] << " GiB";
      if ( !using_m2t_and_s2l_operations ) {
        std::cout << "; 'time separated': "
                  << time_separated_nearfield_storage_on_processes[ i ]
                  << " GiB";
      }
      if ( _aca_eps > 0.0 ) {
        std::cout << "; spatially admissible uncompressed: "
                  << nearfield_storage_on_processes_spat_adm_no_compr[ i ]
                  << " GiB, compressed: "
                  << nearfield_storage_on_processes_spat_adm_compr[ i ]
                  << " GiB";
      }
      std::cout << std::endl;
    }
    std::cout << "Required storage (global) per level (uncompressed):"
              << std::endl;
    for ( lo i = 0; i < n_global_tree_levels; ++i ) {
      std::cout << "level " << i << ": nearfield matrices: "
                << global_levelwise_nearfield_storage[ i ] << " GiB";
      if ( !using_m2t_and_s2l_operations ) {
        std::cout << "; 'time separated': "
                  << global_levelwise_time_separated_nearfield_storage[ i ]
                  << " GiB";
      }
      if ( _aca_eps > 0.0 ) {
        std::cout << "; spatially admissible uncompressed: "
                  << global_levelwise_nearfield_storage_spat_adm_no_compr[ i ]
                  << " GiB, compressed: "
                  << global_levelwise_nearfield_storage_spat_adm_compr[ i ]
                  << " GiB";
      }
      std::cout << std::endl;
    }
    std::cout << "Total nearfield storage (uncompressed): "
              << total_storage_nearfield << " GiB";
    if ( !using_m2t_and_s2l_operations ) {
      std::cout << "; 'time separated': " << total_storage_time_separated_part
                << " GiB";
    }
    if ( _aca_eps > 0 ) {
      std::cout << "; spatially admissible uncompressed: "
                << total_storage_spat_adm_nf_no_compr
                << " GiB, compressed: " << total_storage_spat_adm_nf_compr
                << " GiB";
    }
    std::cout << std::endl;
    sc total_storage_nf_compr = total_storage_nearfield
      - total_storage_spat_adm_nf_no_compr + total_storage_spat_adm_nf_compr;
    if ( total_storage_spat_adm_nf_compr > 0 ) {
      std::cout << "Total nearfield storage compressed: "
                << total_storage_nf_compr << " GiB" << std::endl;
    }
    std::cout << "Nearfield ratio (uncompressed): "
              << (sc) total_nearfield_entries
        / ( n_target_dofs * n_source_dofs );
    if ( !using_m2t_and_s2l_operations ) {
      std::cout << ", for time separated part: "
                << (sc) total_nearfield_entries_time_separated_part
          / ( n_target_dofs * n_source_dofs );
    }
    if ( _aca_eps > 0.0 ) {
      std::cout << ", for spatially admissible part: "
                << (sc) total_nearfield_entries_spat_adm_no_compr
          / ( n_target_dofs * n_source_dofs );
    }
    std::cout << std::endl;
    if ( total_storage_spat_adm_nf_compr > 0 ) {
      std::cout << "Nearfield ratio compressed: "
                << (sc) total_nf_entries_compr
          / ( n_target_dofs * n_source_dofs )
                << std::endl;
      std::cout << "Compression rate for spatially admissible part: "
                << 1
          - ( (sc) total_storage_spat_adm_nf_compr
            / (sc) total_storage_spat_adm_nf_no_compr )
                << std::endl;
    }
    if ( n_compressed_blocks > 0 || n_discarded_blocks > 0 ) {
      std::cout << "Compression details: " << std::endl;
      std::cout << "Total amount of spatially admissible blocks: "
                << n_compressed_blocks + n_uncompressed_blocks
          + n_discarded_blocks;
      std::cout << ", size: "
                << ( (sc) ( n_tot_size_uncompressed_blocks
                     + n_tot_size_compressed_blocks
                     + n_tot_size_discarded_blocks ) )
          * 8.0 / ( 1024. * 1024. * 1024. )
                << " GiB" << std::endl;
      std::cout << "Compressed blocks: " << n_compressed_blocks << ", size: "
                << (sc) n_tot_size_compressed_blocks * 8.0
          / ( 1024. * 1024. * 1024. )
                << " GiB" << std::endl;
      std::cout << "Uncompressed blocks: " << n_uncompressed_blocks
                << ", size: "
                << (sc) n_tot_size_uncompressed_blocks * 8.0
          / ( 1024. * 1024. * 1024. )
                << " GiB" << std::endl;
      std::cout << "Discarded blocks: " << n_discarded_blocks << ", size: "
                << (sc) n_tot_size_discarded_blocks * 8.0
          / ( 1024. * 1024. * 1024. )
                << " GiB" << std::endl;
    }

    std::cout << "storage per allocated vector (source): "
              << n_source_dofs * 8. / 1024. / 1024. / 1024. << " GiB."
              << std::endl;
    std::cout << "storage per allocated vector (target): "
              << n_target_dofs * 8. / 1024. / 1024. / 1024. << " GiB."
              << std::endl;
    std::cout << "Max. and mean number of nearfield matrix entries per st "
                 "target per process: "
              << std::endl;
    for ( lo i = 0; i < n_processes; ++i ) {
      std::cout << "process " << i
                << ": max n entries: " << all_max_n_nf_entries_st_targets[ i ]
                << " mean n entries: " << all_mean_n_nf_entries_st_targets[ i ]
                << std::endl;
    }

    std::cout << std::endl;

    delete[] all_loc_tot_nf_entries;
    delete[] all_loc_spat_adm_nf_entries;
    delete[] all_loc_compr_spat_adm_nf_entries;
    delete[] all_loc_separated_entries;
    delete[] all_n_moments;
    delete[] all_n_moments_receive;
    delete[] all_n_local_contributions;
    delete[] all_n_spat_moments;
    delete[] all_n_spat_local_contributions;

    delete[] all_max_n_nf_entries_st_targets;
    delete[] all_mean_n_nf_entries_st_targets;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  analyze_spatially_admissible_nearfield_operations(
    std::string & output_file_base, const int root_process ) const {
  std::vector< std::vector< long long > > disc_blocks_per_time_level,
    disc_blocks_per_aux_space_level, comp_blocks_per_time_level,
    comp_blocks_per_aux_space_level, uncomp_blocks_per_time_level,
    uncomp_blocks_per_aux_space_level;
  long long other_disc_blocks, other_comp_blocks, other_uncomp_blocks;
  // count the spatially admissible nf operations grid-wise sorted by their
  // status (discarded/compressed) on each process
  count_spatially_admissible_nearfield_operations_gridwise(
    disc_blocks_per_time_level, disc_blocks_per_aux_space_level,
    other_disc_blocks, comp_blocks_per_time_level,
    comp_blocks_per_aux_space_level, other_comp_blocks,
    uncomp_blocks_per_time_level, uncomp_blocks_per_aux_space_level,
    other_uncomp_blocks );

  // communicate the results (inefficient way, but not relevant)
  lo outer_size_per_time_vectors = disc_blocks_per_time_level.size( );
  lo inner_size_per_time_vectors = disc_blocks_per_time_level[ 0 ].size( );
  lo outer_size_per_aux_space_level_vectors
    = disc_blocks_per_aux_space_level.size( );
  lo inner_size_per_aux_space_level_vectors
    = 343;  // warning: hard coded here and in the counting routine
  MPI_Allreduce( MPI_IN_PLACE, &outer_size_per_time_vectors, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );
  MPI_Allreduce( MPI_IN_PLACE, &outer_size_per_aux_space_level_vectors, 1,
    get_index_type< lo >::MPI_LO( ), MPI_MAX, *_comm );
  // each process fills up its own vector if it is shorter than those of the
  // other processes
  if ( outer_size_per_time_vectors > (lo) disc_blocks_per_time_level.size( ) ) {
    lo old_size = disc_blocks_per_time_level.size( );
    disc_blocks_per_time_level.resize( outer_size_per_time_vectors );
    comp_blocks_per_time_level.resize( outer_size_per_time_vectors );
    uncomp_blocks_per_time_level.resize( outer_size_per_time_vectors );
    for ( lo i = old_size; i < outer_size_per_time_vectors; ++i ) {
      disc_blocks_per_time_level[ i ].resize( inner_size_per_time_vectors, 0 );
      comp_blocks_per_time_level[ i ].resize( inner_size_per_time_vectors, 0 );
      uncomp_blocks_per_time_level[ i ].resize(
        inner_size_per_time_vectors, 0 );
    }
  }
  if ( outer_size_per_aux_space_level_vectors
    > (lo) disc_blocks_per_aux_space_level.size( ) ) {
    lo old_size = disc_blocks_per_aux_space_level.size( );
    disc_blocks_per_aux_space_level.resize(
      outer_size_per_aux_space_level_vectors );
    comp_blocks_per_aux_space_level.resize(
      outer_size_per_aux_space_level_vectors );
    uncomp_blocks_per_aux_space_level.resize(
      outer_size_per_aux_space_level_vectors );
    for ( lo i = old_size; i < outer_size_per_aux_space_level_vectors; ++i ) {
      disc_blocks_per_aux_space_level[ i ].resize(
        inner_size_per_aux_space_level_vectors, 0 );
      comp_blocks_per_aux_space_level[ i ].resize(
        inner_size_per_aux_space_level_vectors, 0 );
      uncomp_blocks_per_aux_space_level[ i ].resize(
        inner_size_per_aux_space_level_vectors, 0 );
    }
  }
  // collect the total numbers at the root process via reduce operations
  if ( _my_rank == root_process ) {
    // Reduce operations for root process
    // first for the "per time" vectors
    for ( lo i = 0; i < outer_size_per_time_vectors; ++i ) {
      MPI_Reduce( MPI_IN_PLACE, disc_blocks_per_time_level[ i ].data( ),
        inner_size_per_time_vectors, MPI_LONG_LONG_INT, MPI_SUM, root_process,
        *_comm );
      MPI_Reduce( MPI_IN_PLACE, comp_blocks_per_time_level[ i ].data( ),
        inner_size_per_time_vectors, MPI_LONG_LONG_INT, MPI_SUM, root_process,
        *_comm );
      MPI_Reduce( MPI_IN_PLACE, uncomp_blocks_per_time_level[ i ].data( ),
        inner_size_per_time_vectors, MPI_LONG_LONG_INT, MPI_SUM, root_process,
        *_comm );
    }
    // same for the "per aux space level" vectors
    for ( lo i = 0; i < outer_size_per_aux_space_level_vectors; ++i ) {
      MPI_Reduce( MPI_IN_PLACE, disc_blocks_per_aux_space_level[ i ].data( ),
        inner_size_per_aux_space_level_vectors, MPI_LONG_LONG_INT, MPI_SUM,
        root_process, *_comm );
      MPI_Reduce( MPI_IN_PLACE, comp_blocks_per_aux_space_level[ i ].data( ),
        inner_size_per_aux_space_level_vectors, MPI_LONG_LONG_INT, MPI_SUM,
        root_process, *_comm );
      MPI_Reduce( MPI_IN_PLACE, uncomp_blocks_per_aux_space_level[ i ].data( ),
        inner_size_per_aux_space_level_vectors, MPI_LONG_LONG_INT, MPI_SUM,
        root_process, *_comm );
    }
    // communicate also the numbers of other discarded/compressed blocks
    MPI_Reduce( MPI_IN_PLACE, &other_disc_blocks, 1, MPI_LONG_LONG_INT, MPI_SUM,
      root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, &other_comp_blocks, 1, MPI_LONG_LONG_INT, MPI_SUM,
      root_process, *_comm );
    MPI_Reduce( MPI_IN_PLACE, &other_uncomp_blocks, 1, MPI_LONG_LONG_INT,
      MPI_SUM, root_process, *_comm );
  } else {
    // Reduce operations for non-root processes
    // first for the "per time" vectors
    for ( lo i = 0; i < outer_size_per_time_vectors; ++i ) {
      MPI_Reduce( disc_blocks_per_time_level[ i ].data( ), nullptr,
        inner_size_per_time_vectors, MPI_LONG_LONG_INT, MPI_SUM, root_process,
        *_comm );
      MPI_Reduce( comp_blocks_per_time_level[ i ].data( ), nullptr,
        inner_size_per_time_vectors, MPI_LONG_LONG_INT, MPI_SUM, root_process,
        *_comm );
      MPI_Reduce( uncomp_blocks_per_time_level[ i ].data( ), nullptr,
        inner_size_per_time_vectors, MPI_LONG_LONG_INT, MPI_SUM, root_process,
        *_comm );
    }
    // same for the "per aux space level" vectors
    for ( lo i = 0; i < outer_size_per_aux_space_level_vectors; ++i ) {
      MPI_Reduce( disc_blocks_per_aux_space_level[ i ].data( ), nullptr,
        inner_size_per_aux_space_level_vectors, MPI_LONG_LONG_INT, MPI_SUM,
        root_process, *_comm );
      MPI_Reduce( comp_blocks_per_aux_space_level[ i ].data( ), nullptr,
        inner_size_per_aux_space_level_vectors, MPI_LONG_LONG_INT, MPI_SUM,
        root_process, *_comm );
      MPI_Reduce( uncomp_blocks_per_aux_space_level[ i ].data( ), nullptr,
        inner_size_per_aux_space_level_vectors, MPI_LONG_LONG_INT, MPI_SUM,
        root_process, *_comm );
    }
    // communicate also the numbers of other discarded/compressed blocks
    MPI_Reduce( &other_disc_blocks, nullptr, 1, MPI_LONG_LONG_INT, MPI_SUM,
      root_process, *_comm );
    MPI_Reduce( &other_comp_blocks, nullptr, 1, MPI_LONG_LONG_INT, MPI_SUM,
      root_process, *_comm );
    MPI_Reduce( &other_uncomp_blocks, nullptr, 1, MPI_LONG_LONG_INT, MPI_SUM,
      root_process, *_comm );
  }

  // print the results
  if ( _my_rank == root_process ) {
    lo n_time_levels
      = _distributed_spacetime_tree->get_distribution_tree( )->get_levels( );
    lo n_trunc_space
      = _distributed_spacetime_tree->get_spatial_nearfield_limit( );
    lo edge_length = 2 * n_trunc_space + 1;
    lo edge_length_aux_s = 7;
    std::cout << "printing discarded blocks, levelwise per time level: "
              << std::endl;
    // two auxiliary variables in loops:
    bool all_zeros;
    std::string next_output_base;
    // print grid analysis for discarded blocks for each temporal level to files
    for ( lo i = 0; i < n_time_levels; ++i ) {
      next_output_base
        = output_file_base + "_disc_t_lev_" + std::to_string( i ) + "_same_t";
      all_zeros = print_integers_in_cubic_grid(
        disc_blocks_per_time_level[ 2 * i ], edge_length, next_output_base );
      if ( all_zeros ) {
        std::cout << "time level " << i
                  << ", nf for same time cluster, all zeros" << std::endl;
      }

      next_output_base
        = output_file_base + "_disc_t_lev_" + std::to_string( i ) + "_prev_t";
      all_zeros
        = print_integers_in_cubic_grid( disc_blocks_per_time_level[ 2 * i + 1 ],
          edge_length, next_output_base );
      if ( all_zeros ) {
        std::cout << "time level " << i
                  << ", nf for previous time cluster, all zeros" << std::endl;
      }
    }
    // same for auxiliary refined blocks
    std::cout << "printing discarded blocks, levelwise per aux space level: "
              << std::endl;
    for ( lou i = 0; i < disc_blocks_per_aux_space_level.size( ) / 2; ++i ) {
      next_output_base = output_file_base + "_disc_aux_s_lev_"
        + std::to_string( i + 1 ) + "_same_t";
      all_zeros = print_integers_in_cubic_grid(
        disc_blocks_per_aux_space_level[ 2 * i ], edge_length_aux_s,
        next_output_base );
      if ( all_zeros ) {
        std::cout << "aux level " << i + 1
                  << ", nf for same time cluster, all zeros" << std::endl;
      }

      next_output_base = output_file_base + "_disc_aux_s_lev_"
        + std::to_string( i + 1 ) + "_prev_t";
      all_zeros = print_integers_in_cubic_grid(
        disc_blocks_per_aux_space_level[ 2 * i + 1 ], edge_length_aux_s,
        next_output_base );
      if ( all_zeros ) {
        std::cout << "aux level " << i + 1
                  << ", nf for previous time cluster, all zeros" << std::endl;
      }
    }
    std::cout << "number of discarded blocks not included above: "
              << other_disc_blocks << std::endl;

    // ######## same for compressed blocks: ################

    std::cout << "print compressed blocks, levelwise per time level: "
              << std::endl;
    for ( lo i = 0; i < n_time_levels; ++i ) {
      next_output_base
        = output_file_base + "_comp_t_lev_" + std::to_string( i ) + "_same_t";
      all_zeros = print_integers_in_cubic_grid(
        comp_blocks_per_time_level[ 2 * i ], edge_length, next_output_base );
      if ( all_zeros ) {
        std::cout << "time level " << i
                  << ", nf for same time cluster, all zeros" << std::endl;
      }

      next_output_base
        = output_file_base + "_comp_t_lev_" + std::to_string( i ) + "_prev_t";
      all_zeros
        = print_integers_in_cubic_grid( comp_blocks_per_time_level[ 2 * i + 1 ],
          edge_length, next_output_base );
      if ( all_zeros ) {
        std::cout << "time level " << i
                  << ", nf for previous time cluster, all zeros" << std::endl;
      }
    }
    std::cout << "compressed blocks, levelwise per aux space level: "
              << std::endl;
    for ( lou i = 0; i < comp_blocks_per_aux_space_level.size( ) / 2; ++i ) {
      next_output_base = output_file_base + "_comp_aux_s_lev_"
        + std::to_string( i + 1 ) + "_same_t";
      all_zeros = print_integers_in_cubic_grid(
        comp_blocks_per_aux_space_level[ 2 * i ], edge_length_aux_s,
        next_output_base );
      if ( all_zeros ) {
        std::cout << "aux level " << i + 1
                  << ", nf for same time cluster, all zeros" << std::endl;
      }

      next_output_base = output_file_base + "_comp_aux_s_lev_"
        + std::to_string( i + 1 ) + "_prev_t";
      all_zeros = print_integers_in_cubic_grid(
        comp_blocks_per_aux_space_level[ 2 * i + 1 ], edge_length_aux_s,
        next_output_base );
      if ( all_zeros ) {
        std::cout << "aux level " << i + 1
                  << ", nf for previous time cluster, all zeros" << std::endl;
      }
    }
    std::cout << "number of compressed blocks not included above: "
              << other_comp_blocks << std::endl;

    // ######## same for uncompressed blocks: ################

    std::cout << "print uncompressed blocks, levelwise per time level: "
              << std::endl;
    for ( lo i = 0; i < n_time_levels; ++i ) {
      next_output_base
        = output_file_base + "_uncomp_t_lev_" + std::to_string( i ) + "_same_t";
      all_zeros = print_integers_in_cubic_grid(
        uncomp_blocks_per_time_level[ 2 * i ], edge_length, next_output_base );
      if ( all_zeros ) {
        std::cout << "time level " << i
                  << ", nf for same time cluster, all zeros" << std::endl;
      }

      next_output_base
        = output_file_base + "_uncomp_t_lev_" + std::to_string( i ) + "_prev_t";
      all_zeros = print_integers_in_cubic_grid(
        uncomp_blocks_per_time_level[ 2 * i + 1 ], edge_length,
        next_output_base );
      if ( all_zeros ) {
        std::cout << "time level " << i
                  << ", nf for previous time cluster, all zeros" << std::endl;
      }
    }
    std::cout << "uncompressed blocks, levelwise per aux space level: "
              << std::endl;
    for ( lou i = 0; i < uncomp_blocks_per_aux_space_level.size( ) / 2; ++i ) {
      next_output_base = output_file_base + "_uncomp_aux_s_lev_"
        + std::to_string( i + 1 ) + "_same_t";
      all_zeros = print_integers_in_cubic_grid(
        uncomp_blocks_per_aux_space_level[ 2 * i ], edge_length_aux_s,
        next_output_base );
      if ( all_zeros ) {
        std::cout << "aux level " << i + 1
                  << ", nf for same time cluster, all zeros" << std::endl;
      }

      next_output_base = output_file_base + "_uncomp_aux_s_lev_"
        + std::to_string( i + 1 ) + "_prev_t";
      all_zeros = print_integers_in_cubic_grid(
        uncomp_blocks_per_aux_space_level[ 2 * i + 1 ], edge_length_aux_s,
        next_output_base );
      if ( all_zeros ) {
        std::cout << "aux level " << i + 1
                  << ", nf for previous time cluster, all zeros" << std::endl;
      }
    }
    std::cout << "number of uncompressed blocks not included above: "
              << other_uncomp_blocks << std::endl;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::call_m2m_operations( scheduling_time_cluster *
                                                       t_cluster,
  bool verbose, const std::string & verbose_file ) const {
  scheduling_time_cluster * parent_cluster = t_cluster->get_parent( );
  // m2m operations are only executed if the parent is active in the upward
  // path
  if ( parent_cluster->is_active_in_upward_path( ) ) {
    if ( verbose ) {
#pragma omp critical( verbose )
      {
        std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "call M2M for cluster " << t_cluster->get_global_index( )
                  << " at level " << t_cluster->get_level( ) << std::endl;
          outfile.close( );
        }
      }
    }
    slou configuration = t_cluster->get_configuration( );
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = parent_cluster->get_associated_spacetime_clusters( );
    lou n_associated_leaves = parent_cluster->get_n_associated_leaves( );

    // call the m2m operations for all non-leaf spacetime clusters which are
    // associated with the parent scheduling time cluster (note: by construction
    // these are not auxiliary clusters!)

    // there is an implicit taskgroup after this taskloop
#pragma omp taskloop
    for ( lou i = n_associated_leaves;
          i < associated_spacetime_clusters->size( ); ++i ) {
      if ( _measure_tasks ) {
        _m_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      general_spacetime_cluster * current_cluster
        = ( *associated_spacetime_clusters )[ i ];

      apply_grouped_m2m_operation( current_cluster, configuration );
      if ( _measure_tasks ) {
        _m_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_grouped_m2m_operation( mesh::general_spacetime_cluster *
                                                 parent_cluster,
  slou child_configuration ) const {
  sc * parent_moment = parent_cluster->get_pointer_to_moment( );

  std::vector< general_spacetime_cluster * > * children
    = parent_cluster->get_children( );
  // ###############################################
  // ### compute the matrix for the temporal m2m ###
  full_matrix temporal_m2m_matrix( _temp_order + 1, _temp_order + 1 );
  lo child_idx = 0;
  // find a child with the right temporal configuration. Note: such a child
  // exists, otherwise the routine is not executed
  while ( ( *children )[ child_idx ]->get_temporal_configuration( )
    != child_configuration ) {
    ++child_idx;
  }
  // get the temporal data of the parent and child cluster.
  sc parent_time_center = parent_cluster->get_time_center( );
  sc parent_time_half_size = parent_cluster->get_time_half_size( );
  sc child_time_center = ( *children )[ child_idx ]->get_time_center( );
  sc child_time_half_size = ( *children )[ child_idx ]->get_time_half_size( );

  const vector_type & nodes = _lagrange.get_nodes( );
  vector_type nodes_child( _temp_order + 1, false );
  vector_type values_lagrange( _temp_order + 1, false );
  // transform the nodes from [-1, 1] to the child interval and then back to
  // [-1, 1] with the transformation of the parent interval:
  for ( lo j = 0; j <= _temp_order; ++j ) {
    nodes_child[ j ] = ( child_time_center + (child_time_half_size) *nodes[ j ]
                         - parent_time_center )
      / parent_time_half_size;
  }
  // compute entries of the m2m matrix (by evaluating lagrange polynomials)
  for ( lo j = 0; j <= _temp_order; ++j ) {
    _lagrange.evaluate( j, nodes_child, values_lagrange );
    for ( lo k = 0; k <= _temp_order; ++k )
      temporal_m2m_matrix.set( j, k, values_lagrange[ k ] );
  }
  // ###############################################
  // determine whether a space-time m2m operation is necessary or only a
  // temporal m2m operation
  lo n_space_div_parent, n_space_div_children, dummy;
  parent_cluster->get_n_divs( n_space_div_parent, dummy );
  ( *children )[ child_idx ]->get_n_divs( n_space_div_children, dummy );
  bool temporal_only = ( n_space_div_parent == n_space_div_children );
  if ( temporal_only ) {
    // execute only temporal m2m operation for children with correct
    // configuration
    for ( auto child : *children ) {
      if ( child->get_temporal_configuration( ) == child_configuration ) {
        const sc * child_moment = child->get_pointer_to_moment( );
        apply_temporal_m2m_operation(
          child_moment, temporal_m2m_matrix, parent_moment );
      }
    }
  } else {
    // apply the spatial m2m operation to all child moments and store the
    // results in a buffer
    // @todo use buffer as input argument to avoid reallocation.
    std::vector< sc > buffer_array(
      ( _temp_order + 1 ) * _spat_contribution_size );
    for ( int i = 0; i < ( _temp_order + 1 ) * _spat_contribution_size; ++i ) {
      buffer_array[ i ] = 0.0;
    }
    for ( auto child : *children ) {
      short child_octant, current_configuration;
      child->get_position( child_octant, current_configuration );
      if ( current_configuration == child_configuration ) {
        const sc * child_moment = child->get_pointer_to_moment( );
        apply_spatial_m2m_operation(
          child_moment, n_space_div_parent, child_octant, buffer_array );
      }
    }
    // apply the temporal m2m operation to the buffer and add the result to
    // the parent moment
    apply_temporal_m2m_operation(
      buffer_array.data( ), temporal_m2m_matrix, parent_moment );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_temporal_m2m_operation( const sc *
                                                                child_moment,
  const full_matrix & temporal_m2m_matrix, sc * parent_moment ) const {
  // call the appropriate cblas routine for matrix matrix multiplication.
  lo n_rows_m2m_matrix = temporal_m2m_matrix.get_n_rows( );
  lo n_cols_moment = _spat_contribution_size;
  lo n_cols_m2m_matrix = n_rows_m2m_matrix;
  lo lda = n_rows_m2m_matrix;
  lo ldb = n_cols_m2m_matrix;
  sc alpha = 1.0;
  sc beta = 1.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_m2m_matrix,
    n_cols_moment, n_cols_m2m_matrix, alpha, temporal_m2m_matrix.data( ), lda,
    child_moment, ldb, beta, parent_moment, n_rows_m2m_matrix );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_spatial_m2m_operation( const sc *
                                                               child_moment,
  const lo n_space_div_parent, const slou octant,
  std::vector< sc > & output_array ) const {
  const vector_type * m2m_coeffs_s_dim_0;
  const vector_type * m2m_coeffs_s_dim_1;
  const vector_type * m2m_coeffs_s_dim_2;
  switch ( octant ) {
    case 0:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 1:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 2:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 3:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 4:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 5:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 6:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 7:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    default:  // default case should never be used, program will crash!
      assert( octant < 8 );
      m2m_coeffs_s_dim_0 = nullptr;
      m2m_coeffs_s_dim_1 = nullptr;
      m2m_coeffs_s_dim_2 = nullptr;
  }

  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  // initialize auxiliary matrices lambda_1/2 for intermediate results with 0
  // @todo use buffers to avoid allocations
  full_matrix lambda_1( _temp_order + 1, n_coeffs_s, true );
  full_matrix lambda_2( _temp_order + 1, n_coeffs_s, true );

  for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
    lo child_index = 0;
    for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
      for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0; ++alpha1 ) {
        lo alpha2;
        for ( alpha2 = 0; alpha2 <= beta2; ++alpha2 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            lambda_1( b,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 )
              += ( *m2m_coeffs_s_dim_2 )[ beta2 * ( _spat_order + 1 ) + alpha2 ]
              * child_moment[ b + child_index * ( _temp_order + 1 ) ];
          }
          ++child_index;
        }
        // correction needed for skipped entries of child_moment
        child_index += _spat_order + 1 - alpha0 - alpha1 - alpha2;
      }
      // correction for current index; necessary since alpha1 does not run
      // until _spat_order - alpha0 as it does in stored child_moment
      child_index += ( ( beta2 + 1 ) * beta2 ) / 2;
    }
  }

  // compute intermediate result lambda_2 not exploiting zero entries for the
  // sake of better readability
  for ( lo beta1 = 0; beta1 <= _spat_order; ++beta1 ) {
    for ( lo beta2 = 0; beta2 <= _spat_order - beta1; ++beta2 ) {
      for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
        for ( lo alpha1 = 0; alpha1 <= beta1; ++alpha1 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            lambda_2( b,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                + ( _spat_order + 1 ) * beta2 + alpha0 )
              += ( *m2m_coeffs_s_dim_1 )[ beta1 * ( _spat_order + 1 ) + alpha1 ]
              * lambda_1( b,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                  + ( _spat_order + 1 ) * alpha0 + alpha1 );
          }
        }
      }
    }
  }

  lo output_array_index = 0;
  for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
    for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
        for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
          for ( lo b = 0; b <= _temp_order; ++b ) {
            output_array[ b + output_array_index * ( _temp_order + 1 ) ]
              += ( *m2m_coeffs_s_dim_0 )[ beta0 * ( _spat_order + 1 ) + alpha0 ]
              * lambda_2( b,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                  + ( _spat_order + 1 ) * beta2 + alpha0 );
          }
        }
        ++output_array_index;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::call_m2l_operations( scheduling_time_cluster *
                                                       src_cluster,
  scheduling_time_cluster * tar_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( verbose ) {
#pragma omp critical( verbose )
    {
      std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "call M2L for source " << src_cluster->get_global_index( )
                << " and target " << tar_cluster->get_global_index( )
                << " at level " << src_cluster->get_level( ) << std::endl;
        outfile.close( );
      }
    }
  }

  // execute an m2l operation for each non-auxiliary spacetime cluster (which
  // are those with local contributions) associated with tar_cluster and each
  // source in its interaction list, whose temporal component is src_cluster
  // (global_time_index coincides with global index of src_cluster)

  std::vector< general_spacetime_cluster * > * associated_spacetime_targets
    = tar_cluster->get_associated_spacetime_clusters( );
  lou n_relevant_clusters
    = tar_cluster->get_n_st_clusters_w_local_contributions( );
  // by construction, the relevant st_clusters are the first ones in the list of
  // associated_spacetime_clusters.
  // there is an implicit taskgroup associated with this taskloop
#pragma omp taskloop
  for ( lou i = 0; i < n_relevant_clusters; ++i ) {
    if ( _measure_tasks ) {
      _m2l_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
    //      for ( auto spacetime_tar : *associated_spacetime_targets ) {
    std::vector< general_spacetime_cluster * > * spacetime_interaction_list
      = ( *associated_spacetime_targets )[ i ]->get_interaction_list( );
    if ( spacetime_interaction_list != nullptr ) {
      for ( auto spacetime_src : *spacetime_interaction_list ) {
        if ( spacetime_src->get_global_time_index( )
          == src_cluster->get_global_index( ) ) {
          apply_m2l_operation(
            spacetime_src, ( *associated_spacetime_targets )[ i ] );
        }
      }
    }
    if ( _measure_tasks ) {
      _m2l_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_m2l_operation( const mesh::general_spacetime_cluster *
                                         src_cluster,
  mesh::general_spacetime_cluster * tar_cluster ) const {
  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2l coefficients
  vector_type buffer_for_gaussians( ( _spat_order + 1 ) * ( _spat_order + 1 )
    * ( _temp_order + 1 ) * ( _temp_order + 1 ) );
  // buffer to store m2l coefficients.
  vector_type buffer_for_coeffs( ( _spat_order + 1 ) * ( _spat_order + 1 )
    * ( _temp_order + 1 ) * ( _temp_order + 1 ) );
  // buffer matrices to store intermediate m2l results.
  lo thread_num = omp_get_thread_num( );

  _aux_buffer_0[ thread_num ].fill( 0.0 );
  _aux_buffer_1[ thread_num ].fill( 0.0 );

  // get geometrical data of the clusters
  sc src_half_size_time;
  vector_type half_size_space( 3, false );
  src_cluster->get_half_size( half_size_space, src_half_size_time );
  sc tar_half_size_time = tar_cluster->get_time_half_size( );
  sc src_center_time, tar_center_time;
  vector_type src_center_space( 3, false );
  vector_type tar_center_space( 3, false );
  src_cluster->get_center( src_center_space, src_center_time );
  tar_cluster->get_center( tar_center_space, tar_center_time );

  // initialize temporal interpolation nodes in source and target cluster
  vector_type src_time_nodes( _temp_order + 1, false );
  vector_type tar_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    tar_time_nodes[ i ]
      = tar_center_time + tar_half_size_time * time_nodes[ i ];
    src_time_nodes[ i ]
      = src_center_time + src_half_size_time * time_nodes[ i ];
  }

  // get spatial properties ( difference of cluster, half length )
  vector_type center_diff_space( tar_center_space );
  for ( lo i = 0; i < 3; ++i ) {
    center_diff_space[ i ] -= src_center_space[ i ];
  }

  sc padding_space = _distributed_spacetime_tree
                       ->get_spatial_paddings( )[ src_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // compute coupling coefficients for dimension 2
  compute_m2l_coupling_coeffs( src_time_nodes, tar_time_nodes,
    half_size_space[ 2 ], center_diff_space[ 2 ], buffer_for_gaussians,
    buffer_for_coeffs );

  const sc * src_moment = src_cluster->get_pointer_to_moment( );
  sc * tar_local = tar_cluster->get_pointer_to_local_contribution( );
  // efficient m2l operation similar to Tausch, 2009, p. 3558
  // help variables for accessing right values in coefficient buffer
  lo hlp_acs_alpha
    = ( _spat_order + 1 ) * ( _temp_order + 1 ) * ( _temp_order + 1 );
  lo hlp_acs_beta = ( _temp_order + 1 ) * ( _temp_order + 1 );
  lo hlp_acs_a = ( _temp_order + 1 );

  sc * aux_buffer_0_data = _aux_buffer_0[ thread_num ].data( );
  sc * buffer_for_coeffs_data = buffer_for_coeffs.data( );

  // compute first intermediate product and store it in aux_buffer_0
  lo buffer_0_index = 0;
  for ( lo alpha2 = 0; alpha2 <= _spat_order; ++alpha2 ) {
    lo moment_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order - alpha2; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - alpha2 - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          // no need for reduction, in a single inner cycle data are written on
          // unique positions
          for ( lo a = 0; a <= _temp_order; ++a ) {
#pragma omp simd aligned( aux_buffer_0_data, buffer_for_coeffs_data \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
            for ( lo b = 0; b <= _temp_order; ++b ) {
              aux_buffer_0_data[ buffer_0_index * hlp_acs_beta + hlp_acs_a * a
                + b ]
                += buffer_for_coeffs_data[ alpha2 * hlp_acs_alpha
                     + beta2 * hlp_acs_beta + a * hlp_acs_a + b ]
                * src_moment[ b + moment_index * ( _temp_order + 1 ) ];
            }
          }
          ++moment_index;
        }
        ++buffer_0_index;
      }
      // correction for moment index; this is necessary since beta1 does not
      // run until _spat_order - beta0 as it does in src_moment;
      moment_index += ( ( alpha2 + 1 ) * alpha2 ) / 2;
    }
  }
  // update coefficients and compute 2nd intermediate product in aux_buffer_1
  compute_m2l_coupling_coeffs( src_time_nodes, tar_time_nodes,
    half_size_space[ 1 ], center_diff_space[ 1 ], buffer_for_gaussians,
    buffer_for_coeffs );

  sc * aux_buffer_1_data = _aux_buffer_1[ thread_num ].data( );

  lo buffer_1_index = 0;
  for ( lo alpha1 = 0; alpha1 <= _spat_order; ++alpha1 ) {
    buffer_0_index = 0;
    for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha1; ++alpha2 ) {
      for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
        for ( lo beta1 = 0; beta1 <= _spat_order - beta0 - alpha2; ++beta1 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
#pragma omp simd aligned(                                      \
  aux_buffer_1_data, buffer_for_coeffs_data, aux_buffer_0_data \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
            for ( lo b = 0; b <= _temp_order; ++b ) {
              aux_buffer_1_data[ buffer_1_index * hlp_acs_beta + hlp_acs_a * a
                + b ]
                += buffer_for_coeffs_data[ alpha1 * hlp_acs_alpha
                     + beta1 * hlp_acs_beta + a * hlp_acs_a + b ]
                * aux_buffer_0_data[ buffer_0_index * hlp_acs_beta
                  + hlp_acs_a * a + b ];
            }
          }
          ++buffer_0_index;
        }
        ++buffer_1_index;
      }
      // correction for buffer_0 index; this is necessary since beta0 does not
      // run until _spat_order - alpha2 as it does in aux_buffer_0;
      buffer_0_index += ( ( alpha1 + 1 ) * alpha1 ) / 2;
    }
  }

  // update coefficients and update targets local contribution with m2l result
  compute_m2l_coupling_coeffs( src_time_nodes, tar_time_nodes,
    half_size_space[ 0 ], center_diff_space[ 0 ], buffer_for_gaussians,
    buffer_for_coeffs );
  sc val = 0;
  int local_index = 0;
  for ( lo alpha0 = 0; alpha0 <= _spat_order; ++alpha0 ) {
    buffer_1_index = 0;
    for ( lo alpha1 = 0; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
      for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha0 - alpha1; ++alpha2 ) {
        for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            val = 0;
#pragma omp simd aligned( buffer_for_coeffs_data, aux_buffer_1_data : \
                  DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH) reduction( + : val)
            for ( lo b = 0; b <= _temp_order; ++b ) {
              val += buffer_for_coeffs_data[ alpha0 * hlp_acs_alpha
                       + beta0 * hlp_acs_beta + a * hlp_acs_a + b ]
                * aux_buffer_1_data[ buffer_1_index * hlp_acs_beta
                  + hlp_acs_a * a + b ];
            }
            tar_local[ a + local_index * ( _temp_order + 1 ) ] += val;
          }
          ++buffer_1_index;
        }
        ++local_index;
      }
      // correction for buffer_1 index; this is necessary since alpha0 does
      // not run until _spat_order - alpha1 as it does in aux_buffer_1;
      buffer_1_index += ( ( alpha0 + 1 ) * alpha0 ) / 2;
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::call_l2l_operations( scheduling_time_cluster *
                                                       t_cluster,
  bool verbose, const std::string & verbose_file ) const {
  if ( verbose ) {
#pragma omp critical( verbose )
    {
      std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "call L2L for cluster " << t_cluster->get_global_index( )
                << " at level " << t_cluster->get_level( ) << std::endl;
        outfile.close( );
      }
    }
  }
  scheduling_time_cluster * parent_cluster = t_cluster->get_parent( );
  slou configuration = t_cluster->get_configuration( );
  std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
    = parent_cluster->get_associated_spacetime_clusters( );
  lou n_associated_leaves = parent_cluster->get_n_associated_leaves( );
  // call the l2l operations for all non-leaf spacetime clusters which are
  // associated with the parent scheduling time cluster (note: there cannot be
  // auxiliary space-time clusters associated with parent_cluster by
  // construction).
  // there is an implicit taskgroup associated with this taskloop
#pragma omp taskloop
  for ( lou i = n_associated_leaves; i < associated_spacetime_clusters->size( );
        ++i ) {
    if ( _measure_tasks ) {
      _l_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
    apply_grouped_l2l_operation(
      ( *associated_spacetime_clusters )[ i ], configuration );
    if ( _measure_tasks ) {
      _l_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_grouped_l2l_operation( mesh::general_spacetime_cluster *
                                                 parent_cluster,
  slou child_configuration ) const {
  sc * parent_local_contribution
    = parent_cluster->get_pointer_to_local_contribution( );
  std::vector< general_spacetime_cluster * > * children
    = parent_cluster->get_children( );
  // ###############################################
  // ### compute the matrix for the temporal l2l ###
  full_matrix temporal_l2l_matrix( _temp_order + 1, _temp_order + 1 );
  lo child_idx = 0;
  // find a child with the right temporal configuration. Note: such a child
  // exists, otherwise the routine is not executed
  while ( ( *children )[ child_idx ]->get_temporal_configuration( )
    != child_configuration ) {
    ++child_idx;
  }
  // get the temporal data of the parent and child cluster.
  sc parent_time_center = parent_cluster->get_time_center( );
  sc parent_time_half_size = parent_cluster->get_time_half_size( );
  sc child_time_center = ( *children )[ child_idx ]->get_time_center( );
  sc child_time_half_size = ( *children )[ child_idx ]->get_time_half_size( );

  const vector_type & nodes = _lagrange.get_nodes( );
  vector_type nodes_child( _temp_order + 1, false );
  vector_type values_lagrange( _temp_order + 1, false );
  // transform the nodes from [-1, 1] to the child interval and then back to
  // [-1, 1] with the transformation of the parent interval:
  for ( lo j = 0; j <= _temp_order; ++j ) {
    nodes_child[ j ] = ( child_time_center + child_time_half_size * nodes[ j ]
                         - parent_time_center )
      / parent_time_half_size;
  }
  // compute entries of the l2l matrix (by evaluating lagrange polynomials)
  for ( lo j = 0; j <= _temp_order; ++j ) {
    _lagrange.evaluate( j, nodes_child, values_lagrange );
    for ( lo k = 0; k <= _temp_order; ++k )
      temporal_l2l_matrix.set( k, j, values_lagrange[ k ] );
  }
  // ###############################################
  // determine whether a space-time l2l operation is necessary or only a
  // temporal l2l operation
  lo n_space_div_parent, n_space_div_children, dummy;
  parent_cluster->get_n_divs( n_space_div_parent, dummy );
  ( *children )[ 0 ]->get_n_divs( n_space_div_children, dummy );
  bool temporal_only = ( n_space_div_parent == n_space_div_children );
  if ( temporal_only ) {
    // execute only temporal l2l operation
    for ( auto child : *children ) {
      if ( child->get_temporal_configuration( ) == child_configuration ) {
        sc * child_local_contribution
          = child->get_pointer_to_local_contribution( );
        apply_temporal_l2l_operation( parent_local_contribution,
          temporal_l2l_matrix, child_local_contribution );
      }
    }
  } else {
    // apply the temporal l2l operation to the parent's local contribution and
    // store the results in a buffer
    // @todo use buffer as input argument to avoid reallocation.
    std::vector< sc > buffer_array(
      ( _temp_order + 1 ) * _spat_contribution_size );
    for ( int i = 0; i < ( _temp_order + 1 ) * _spat_contribution_size; ++i ) {
      buffer_array[ i ] = 0.0;
    }
    apply_temporal_l2l_operation(
      parent_local_contribution, temporal_l2l_matrix, buffer_array.data( ) );

    for ( auto child : *children ) {
      short child_octant, current_configuration;
      child->get_position( child_octant, current_configuration );
      if ( current_configuration == child_configuration ) {
        sc * child_local_contribution
          = child->get_pointer_to_local_contribution( );
        apply_spatial_l2l_operation( buffer_array.data( ), n_space_div_parent,
          child_octant, child_local_contribution );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_temporal_l2l_operation( const sc *
                                                  parent_local_contribution,
  const full_matrix & temporal_l2l_matrix,
  sc * child_local_contribution ) const {
  // call the appropriate cblas routine for matrix matrix multiplication.
  lo n_rows_l2l_matrix = temporal_l2l_matrix.get_n_rows( );
  lo n_cols_moment = _spat_contribution_size;
  lo n_cols_l2l_matrix = n_rows_l2l_matrix;
  lo lda = n_rows_l2l_matrix;
  lo ldb = n_cols_l2l_matrix;
  sc alpha = 1.0;
  sc beta = 1.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_l2l_matrix,
    n_cols_moment, n_cols_l2l_matrix, alpha, temporal_l2l_matrix.data( ), lda,
    parent_local_contribution, ldb, beta, child_local_contribution,
    n_rows_l2l_matrix );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_spatial_l2l_operation( const sc *
                                                               parent_local,
  const lo n_space_div_parent, const slou octant, sc * child_local ) const {
  const vector_type * m2m_coeffs_s_dim_0;
  const vector_type * m2m_coeffs_s_dim_1;
  const vector_type * m2m_coeffs_s_dim_2;
  switch ( octant ) {
    case 0:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 1:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 2:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 3:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ n_space_div_parent ] );
      break;
    case 4:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 5:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 6:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    case 7:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ n_space_div_parent ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ n_space_div_parent ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ n_space_div_parent ] );
      break;
    default:  // default case should never be used, program will crash!
      assert( octant > -1 && octant < 8 );
      m2m_coeffs_s_dim_0 = nullptr;
      m2m_coeffs_s_dim_1 = nullptr;
      m2m_coeffs_s_dim_2 = nullptr;
  }

  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  // initialize auxiliary matrices lambda_1/2 for intermediate results with 0
  // TODO instead of allocating every time use buffers
  full_matrix lambda_1( _temp_order + 1, n_coeffs_s, true );
  full_matrix lambda_2( _temp_order + 1, n_coeffs_s, true );

  for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
    lou parent_index = 0;
    for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
      for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0; ++alpha1 ) {
        // correction for skipped entries of parent_local due to starting
        // point alpha2 = beta2 in the next loop
        parent_index += beta2;
        for ( lo alpha2 = beta2; alpha2 <= _spat_order - alpha0 - alpha1;
              ++alpha2 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            lambda_1( a,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 )
              += ( *m2m_coeffs_s_dim_2 )[ alpha2 * ( _spat_order + 1 ) + beta2 ]
              * parent_local[ a + parent_index * ( _temp_order + 1 ) ];
          }
          ++parent_index;
        }
      }
      // correction for current index; this is necessary since alpha1 does not
      // run until _spat_order - alpha0 as it does in parent_local;
      parent_index += ( ( beta2 + 1 ) * beta2 ) / 2;
    }
  }

  for ( lo beta1 = 0; beta1 <= _spat_order; ++beta1 ) {
    for ( lo beta2 = 0; beta2 <= _spat_order - beta1; ++beta2 ) {
      for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
        for ( lo alpha1 = beta1; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
          for ( lo a = 0; a <= _temp_order; ++a )
            lambda_2( a,
              ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                + ( _spat_order + 1 ) * beta2 + alpha0 )
              += ( *m2m_coeffs_s_dim_1 )[ alpha1 * ( _spat_order + 1 ) + beta1 ]
              * lambda_1( a,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                  + ( _spat_order + 1 ) * alpha0 + alpha1 );
        }
      }
    }
  }

  lou child_index = 0;
  for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
    for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
        for ( lo alpha0 = beta0; alpha0 <= _spat_order - beta1 - beta2;
              ++alpha0 ) {
          for ( lo a = 0; a <= _temp_order; ++a ) {
            child_local[ a + child_index * ( _temp_order + 1 ) ]
              += ( *m2m_coeffs_s_dim_0 )[ alpha0 * ( _spat_order + 1 ) + beta0 ]
              * lambda_2( a,
                ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                  + ( _spat_order + 1 ) * beta2 + alpha0 );
          }
        }
        ++child_index;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_m2t_operations( mesh::scheduling_time_cluster *
                                         t_cluster,
  distributed_block_vector & output_vector, bool verbose,
  const std::string & verbose_file ) const {
  std::vector< general_spacetime_cluster * > * assoc_spacetime_targets
    = t_cluster->get_associated_spacetime_clusters( );
  // first, execute all hybrid m2t operations
  const std::vector< lo > * assoc_hybrid_m2t_tar_indices
    = t_cluster->get_assoc_hybrid_m2t_targets( );
  if ( assoc_hybrid_m2t_tar_indices != nullptr ) {
    if ( verbose ) {
#pragma omp critical( verbose )
      {
        std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "call hybrid M2T for cluster "
                  << t_cluster->get_global_index( ) << " at level "
                  << t_cluster->get_level( ) << std::endl;
          outfile.close( );
        }
      }
    }
#pragma omp taskloop shared( \
  assoc_hybrid_m2t_tar_indices, assoc_spacetime_targets )
    for ( lou i = 0; i < assoc_hybrid_m2t_tar_indices->size( ); ++i ) {
      if ( _measure_tasks ) {
        _m2t_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      general_spacetime_cluster * current_spacetime_target
        = ( *assoc_spacetime_targets )[ (
          *assoc_hybrid_m2t_tar_indices )[ i ] ];
      std::vector< general_spacetime_cluster * > * spacetime_m2t_list
        = current_spacetime_target->get_m2t_list( );
      lou n_hybrid_m2t_ops
        = current_spacetime_target->get_n_hybrid_m2t_operations( );
      // apply an m2ls operation for each source cluster in the target's m2t
      // list
      for ( lou j = 0; j < n_hybrid_m2t_ops; ++j ) {
        general_spacetime_cluster * current_source
          = ( *spacetime_m2t_list )[ j ];
        apply_m2ls_operation< run_count >(
          current_source, current_spacetime_target );
      }
      if ( _measure_tasks ) {
        _m2t_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }

  // next, execute all standard m2t operations
  const std::vector< lo > * assoc_std_m2t_tar_indices
    = t_cluster->get_assoc_standard_m2t_targets( );
  if ( assoc_std_m2t_tar_indices != nullptr ) {
    if ( verbose ) {
#pragma omp critical( verbose )
      {
        std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "call standard M2T for cluster "
                  << t_cluster->get_global_index( ) << " at level "
                  << t_cluster->get_level( ) << std::endl;
          outfile.close( );
        }
      }
    }
    // there is an implicit taskgroup associated with this taskloop
#pragma omp taskloop shared( \
  output_vector, assoc_std_m2t_tar_indices, assoc_spacetime_targets )
    for ( lou i = 0; i < assoc_std_m2t_tar_indices->size( ); ++i ) {
      if ( _measure_tasks ) {
        _m2t_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      general_spacetime_cluster * current_spacetime_target
        = ( *assoc_spacetime_targets )[ ( *assoc_std_m2t_tar_indices )[ i ] ];
      std::vector< general_spacetime_cluster * > * spacetime_m2t_list
        = current_spacetime_target->get_m2t_list( );
      lou n_hybrid_m2t_ops
        = current_spacetime_target->get_n_hybrid_m2t_operations( );
      // construct a local result vector
      vector_type local_result(
        current_spacetime_target->get_n_dofs< target_space >( ), true );
      // apply an m2t operation for each source cluster in the target's m2t list
      // and add the result to the local result vector
      for ( lou j = n_hybrid_m2t_ops; j < spacetime_m2t_list->size( ); ++j ) {
        general_spacetime_cluster * current_source
          = ( *spacetime_m2t_list )[ j ];
        apply_m2t_operation< run_count >(
          current_source, current_spacetime_target, local_result );
      }
      // after all m2t operations for the current clusters are executed, add the
      // local result to the output vector
      output_vector.add_local_part< target_space >(
        current_spacetime_target, local_result );
      if ( _measure_tasks ) {
        _m2t_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_m2t_operation( const mesh::general_spacetime_cluster *
  /*src_cluster*/,
  const mesh::general_spacetime_cluster * /*tar_cluster*/,
  vector_type & /*local_output_vector*/ ) const {
  std::cout << "General M2T operation not implemented!" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_m2t_operation< 0 >( const mesh::general_spacetime_cluster * src_cluster,
    const mesh::general_spacetime_cluster * tar_cluster,
    vector_type & local_output_vector ) const {
  apply_m2t_operation_p0( src_cluster, tar_cluster, local_output_vector );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_m2t_operation< 0 >( const mesh::general_spacetime_cluster * src_cluster,
    const mesh::general_spacetime_cluster * tar_cluster,
    vector_type & local_output_vector ) const {
  apply_m2t_operation_p0( src_cluster, tar_cluster, local_output_vector );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_m2t_operation< 0 >( const mesh::general_spacetime_cluster * src_cluster,
    const mesh::general_spacetime_cluster * tar_cluster,
    vector_type & local_output_vector ) const {
  apply_m2t_operation_p1_normal_drv(
    src_cluster, tar_cluster, local_output_vector );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_m2t_operation_p0( const mesh::general_spacetime_cluster * src_cluster,
    const mesh::general_spacetime_cluster * tar_cluster,
    vector_type & local_output_vector, const lo quad_order_space ) const {
  // get geometrical data of the source cluster
  sc src_half_size_time;
  vector_type half_size_space( 3, false );
  src_cluster->get_half_size( half_size_space, src_half_size_time );
  sc src_center_time;
  vector_type src_center_space( 3, false );
  src_cluster->get_center( src_center_space, src_center_time );
  sc padding_space = _distributed_spacetime_tree
                       ->get_spatial_paddings( )[ src_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // initialize temporal interpolation nodes in the source cluster
  vector_type src_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    src_time_nodes[ i ]
      = src_center_time + src_half_size_time * time_nodes[ i ];
  }

  // get information about the elements in the target cluster
  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = tar_cluster->get_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo n_tar_elems = tar_cluster->get_n_elements( );

  // initialize quadrature data in space
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature, quad_order_space );
  lo s_quad_size = my_quadrature._wy_cheb.size( );
  const sc * quad_space_w = my_quadrature._wy_cheb.data( );
  const sc * y0_quad_elem = my_quadrature._y1.data( );
  const sc * y1_quad_elem = my_quadrature._y2.data( );
  const sc * y2_quad_elem = my_quadrature._y3.data( );
  vector_type quad_point_tar( 3, false );
  linear_algebra::coordinates< 3 > y1, y2, y3;
  // initialize quadrature data in time
  // choose the quadrature order such that polynomials of degree _temp_order are
  // integrated exactly (we use the same degree for s2m and l2t operations)
  lo n_quad_points_time = ( _temp_order + 2 ) / 2;
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_t
    = bem::quadrature::line_x( n_quad_points_time );
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_w
    = bem::quadrature::line_w( n_quad_points_time );
  vector_type quad_time_t_elem( n_quad_points_time );
  vector_type time_node_differences( _temp_order + 1 );

  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2t coefficients
  vector_type buffer_m2t_coeffs( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  // buffer to store m2t coefficients.
  vector_type buffer_coeffs_0( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_coeffs_1( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_coeffs_2( ( _spat_order + 1 ) * ( _temp_order + 1 ) );

  vector_type coupling_coeffs_tensor_product( _contribution_size );
  const sc * coupling_coeffs_tensor_product_data
    = coupling_coeffs_tensor_product.data( );
  std::vector< vector_type * > vector_of_buffers = { &buffer_coeffs_0,
    &buffer_coeffs_1, &buffer_coeffs_2, &buffer_m2t_coeffs };

  // loop over all space-time elements in the target cluster
  for ( lo i_elem = 0; i_elem < n_tar_elems; ++i_elem ) {
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, tar_cluster->get_element( i_elem ) );
    // get the temporal information of the current space-time element
    lo local_elem_idx_time = local_mesh->get_time_element( local_elem_idx );
    sc t_start_elem, t_end_elem, t_size_elem;
    local_mesh->get_temporal_nodes(
      local_elem_idx_time, &t_start_elem, &t_end_elem );
    t_size_elem = t_end_elem - t_start_elem;
    // compute the temporal quadrature points in the element
    for ( slou i = 0; i < n_quad_points_time; ++i ) {
      quad_time_t_elem[ i ] = t_start_elem + t_size_elem * quad_time_t[ i ];
    }
    // get the spatial information of the current space-time element
    lo local_elem_idx_space
      = local_mesh->get_space_element_index( local_elem_idx );
    local_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc s_area_elem = local_mesh->get_spatial_area_using_spatial_index(
      local_elem_idx_space );
    // compute the spatial quadrature points in the element
    triangle_to_geometry( y1, y2, y3, my_quadrature );

    // loop over all quadrature points in space and time and execute the
    // appropriate m2t coupling operations
    for ( lo i_quad_t = 0; i_quad_t < n_quad_points_time; ++i_quad_t ) {
      // update the vector of time node differences
      for ( lo i_t = 0; i_t <= _temp_order; ++i_t ) {
        time_node_differences[ i_t ]
          = quad_time_t_elem[ i_quad_t ] - src_time_nodes[ i_t ];
      }
      for ( lo i_quad_s = 0; i_quad_s < s_quad_size; ++i_quad_s ) {
        // update the quadrature point in space
        quad_point_tar[ 0 ] = y0_quad_elem[ i_quad_s ];
        quad_point_tar[ 1 ] = y1_quad_elem[ i_quad_s ];
        quad_point_tar[ 2 ] = y2_quad_elem[ i_quad_s ];
        compute_single_sided_coupling_coeffs_tensor( time_node_differences,
          half_size_space, src_center_space, quad_point_tar, vector_of_buffers,
          coupling_coeffs_tensor_product );
        // Applying the M2T coefficients to the moments corresponds to
        // a scalar product of 2 vectors
        sc val = 0.0;
        const sc * src_moment = src_cluster->get_pointer_to_moment( );
#pragma omp simd aligned( coupling_coeffs_tensor_product_data :  DATA_ALIGN ) \
                          simdlen( BESTHEA_SIMD_WIDTH) reduction( + : val)
        for ( lo glob_index = 0; glob_index < _contribution_size;
              ++glob_index ) {
          val += coupling_coeffs_tensor_product_data[ glob_index ]
            * src_moment[ glob_index ];
        }
        // finally, update the appropriate part of the local result vector
        local_output_vector[ i_elem ] += quad_time_w[ i_quad_t ]
          * quad_space_w[ i_quad_s ] * s_area_elem * t_size_elem * val;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_m2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * src_cluster,
    const mesh::general_spacetime_cluster * tar_cluster,
    vector_type & local_output_vector ) const {
  // get geometrical data of the source cluster
  sc src_half_size_time;
  vector_type half_size_space( 3, false );
  src_cluster->get_half_size( half_size_space, src_half_size_time );
  sc src_center_time;
  vector_type src_center_space( 3, false );
  src_cluster->get_center( src_center_space, src_center_time );
  sc padding_space = _distributed_spacetime_tree
                       ->get_spatial_paddings( )[ src_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // initialize temporal interpolation nodes in the source cluster
  vector_type src_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    src_time_nodes[ i ]
      = src_center_time + src_half_size_time * time_nodes[ i ];
  }

  // get information about the elements in the target cluster
  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = tar_cluster->get_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo n_tar_elems = tar_cluster->get_n_elements( );
  lo n_tar_space_elems = tar_cluster->get_n_space_elements( );
  lo n_tar_space_nodes = tar_cluster->get_n_space_nodes( );
  const std::vector< lo > & tar_elems_2_local_nodes
    = tar_cluster->get_elems_2_local_nodes( );
  linear_algebra::coordinates< 3 > normal;

  // initialize quadrature data in space
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature, _order_regular );
  lo s_quad_size = my_quadrature._wy_cheb.size( );
  const sc * quad_space_w = my_quadrature._wy_cheb.data( );
  const sc * y0_quad_elem = my_quadrature._y1.data( );
  const sc * y1_quad_elem = my_quadrature._y2.data( );
  const sc * y2_quad_elem = my_quadrature._y3.data( );
  const sc * y0_quad_ref_elem = my_quadrature._y1_ref_cheb.data( );
  const sc * y1_quad_ref_elem = my_quadrature._y2_ref_cheb.data( );
  vector_type quad_point_tar( 3, false );
  linear_algebra::coordinates< 3 > y1, y2, y3;
  // initialize quadrature data in time
  // choose the quadrature order such that polynomials of degree _temp_order are
  // integrated exactly (we use the same degree for s2m and l2t operations)
  lo n_quad_points_time = ( _temp_order + 2 ) / 2;
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_t
    = bem::quadrature::line_x( n_quad_points_time );
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_w
    = bem::quadrature::line_w( n_quad_points_time );
  vector_type quad_time_t_elem( n_quad_points_time );
  vector_type time_node_differences( _temp_order + 1 );

  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2t coefficients
  vector_type buffer_m2t_coeffs_0( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_m2t_coeffs_1( ( _temp_order + 1 ) );
  // buffer to store m2t coefficients.
  vector_type buffer_coeffs_0( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_coeffs_1( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_coeffs_2( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_drv_coeffs_0( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_drv_coeffs_1( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_drv_coeffs_2( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  std::vector< vector_type * > vector_of_buffers
    = { &buffer_coeffs_0, &buffer_coeffs_1, &buffer_coeffs_2,
        &buffer_drv_coeffs_0, &buffer_drv_coeffs_1, &buffer_drv_coeffs_2,
        &buffer_m2t_coeffs_0, &buffer_m2t_coeffs_1 };
  vector_type coupling_coeffs_tensor_product( _contribution_size );
  const sc * coupling_coeffs_tensor_product_data
    = coupling_coeffs_tensor_product.data( );

  // loop over all space-time elements in the target cluster
  lo i_time_elem = 0;
  lo i_space_elem = 0;
  for ( lo i_elem = 0; i_elem < n_tar_elems; ++i_elem ) {
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, tar_cluster->get_element( i_elem ) );
    // get the temporal information of the current space-time element
    lo local_elem_idx_time = local_mesh->get_time_element( local_elem_idx );
    sc t_start_elem, t_end_elem, t_size_elem;
    local_mesh->get_temporal_nodes(
      local_elem_idx_time, &t_start_elem, &t_end_elem );
    t_size_elem = t_end_elem - t_start_elem;
    // compute the temporal quadrature points in the element
    for ( slou i = 0; i < n_quad_points_time; ++i ) {
      quad_time_t_elem[ i ] = t_start_elem + t_size_elem * quad_time_t[ i ];
    }
    // get the spatial information of the current space-time element
    lo local_elem_idx_space
      = local_mesh->get_space_element_index( local_elem_idx );
    local_mesh->get_spatial_normal_using_spatial_element_index(
      local_elem_idx_space, normal );
    local_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc s_area_elem = local_mesh->get_spatial_area_using_spatial_index(
      local_elem_idx_space );
    // compute the spatial quadrature points in the element
    triangle_to_geometry( y1, y2, y3, my_quadrature );

    // loop over all quadrature points in space and time and execute the
    // appropriate m2t coupling operations
    for ( lo i_quad_t = 0; i_quad_t < n_quad_points_time; ++i_quad_t ) {
      // update the vector of time node differences
      for ( lo i_t = 0; i_t <= _temp_order; ++i_t ) {
        time_node_differences[ i_t ]
          = quad_time_t_elem[ i_quad_t ] - src_time_nodes[ i_t ];
      }
      for ( lo i_quad_s = 0; i_quad_s < s_quad_size; ++i_quad_s ) {
        // update the quadrature point in space
        quad_point_tar[ 0 ] = y0_quad_elem[ i_quad_s ];
        quad_point_tar[ 1 ] = y1_quad_elem[ i_quad_s ];
        quad_point_tar[ 2 ] = y2_quad_elem[ i_quad_s ];
        // compute the coupling coefficients and their derivatives for all
        // spatial dimensions
        compute_single_sided_coupling_coeffs_normal_drv_tensor(
          time_node_differences, half_size_space, src_center_space,
          quad_point_tar, normal, vector_of_buffers,
          coupling_coeffs_tensor_product );

        // Applying the M2T coefficients to the moments corresponds to
        // a scalar product of 2 vectors
        sc val = 0.0;
        const sc * src_moment = src_cluster->get_pointer_to_moment( );
#pragma omp simd aligned( coupling_coeffs_tensor_product_data :  DATA_ALIGN ) \
                          simdlen( BESTHEA_SIMD_WIDTH) reduction( + : val)
        for ( lo glob_index = 0; glob_index < _contribution_size;
              ++glob_index ) {
          val += coupling_coeffs_tensor_product_data[ glob_index ]
            * src_moment[ glob_index ];
        }
        // compute the 3 nodal values for the different shape functions
        val *= quad_time_w[ i_quad_t ] * quad_space_w[ i_quad_s ] * s_area_elem
          * t_size_elem * val;
        sc val_1 = val * y0_quad_ref_elem[ i_quad_s ];
        sc val_2 = val * y1_quad_ref_elem[ i_quad_s ];
        sc val_0 = val - val_1 - val_2;

        // update the appropriate positions of the local (nodal) vector by
        // adding the different values to the correct positions
        lo s_node_0_idx
          = tar_cluster->local_spacetime_node_idx_2_local_space_node_idx(
            tar_elems_2_local_nodes[ 6 * i_elem ] );
        lo s_node_1_idx
          = tar_cluster->local_spacetime_node_idx_2_local_space_node_idx(
            tar_elems_2_local_nodes[ 6 * i_elem + 1 ] );
        lo s_node_2_idx
          = tar_cluster->local_spacetime_node_idx_2_local_space_node_idx(
            tar_elems_2_local_nodes[ 6 * i_elem + 2 ] );
        local_output_vector[ i_time_elem * n_tar_space_nodes + s_node_0_idx ]
          += val_0;
        local_output_vector[ i_time_elem * n_tar_space_nodes + s_node_1_idx ]
          += val_1;
        local_output_vector[ i_time_elem * n_tar_space_nodes + s_node_2_idx ]
          += val_2;
      }
    }
    // update the local space and time indices for the next run in the loop.
    i_space_elem++;
    if ( i_space_elem == n_tar_space_elems ) {
      i_time_elem++;
      i_space_elem = 0;
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_m2ls_operation( const mesh::general_spacetime_cluster *
  /*src_cluster*/,
  mesh::general_spacetime_cluster * /*tar_cluster*/ ) const {
  std::cout << "General M2Ls operation not implemented " << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_m2ls_operation< 0 >(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_m2ls_operation_p0_time( src_cluster, tar_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_m2ls_operation< 0 >(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_m2ls_operation_p0_time( src_cluster, tar_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_m2ls_operation< 0 >(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_m2ls_operation_p0_time( src_cluster, tar_cluster );
}

// TODO: add specialization for hypersingular operator

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_m2ls_operation_p0_time(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  lo n_quad_points_time = _temp_order + 1;
  // ATTENTION: When changing n_quad_points_time one has to adapt the size of
  // the auxiliary buffers!
  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2l coefficients
  vector_type buffer_for_gaussians( ( _spat_order + 1 ) * ( _spat_order + 1 )
    * ( _temp_order + 1 ) * n_quad_points_time );
  // buffer to store m2l coefficients.
  vector_type buffer_for_coeffs( ( _spat_order + 1 ) * ( _spat_order + 1 )
    * ( _temp_order + 1 ) * n_quad_points_time );

  // get geometrical data of the clusters
  sc src_half_size_time;
  vector_type half_size_space( 3, false );
  src_cluster->get_half_size( half_size_space, src_half_size_time );
  sc src_center_time, tar_center_time;
  vector_type src_center_space( 3, false );
  vector_type tar_center_space( 3, false );
  src_cluster->get_center( src_center_space, src_center_time );
  tar_cluster->get_center( tar_center_space, tar_center_time );

  // initialize the temporal interpolation nodes in the source cluster
  vector_type src_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    src_time_nodes[ i ]
      = src_center_time + src_half_size_time * time_nodes[ i ];
  }
  // initialize the quadrature points for quadrature in target time intervals
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_t
    = bem::quadrature::line_x( n_quad_points_time );
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_w
    = bem::quadrature::line_w( n_quad_points_time );
  // update n_quad_points_time (necessary, if quadrature rule is not available)
  n_quad_points_time = quad_time_w.size( );

  vector_type quad_points_t_tar_elem( n_quad_points_time );

  vector quad_time_w_unrld( n_quad_points_time * ( _temp_order + 1 ) );

  // get spatial properties ( difference of cluster, half length )
  vector_type center_diff_space( tar_center_space );
  for ( lo i = 0; i < 3; ++i ) {
    center_diff_space[ i ] -= src_center_space[ i ];
  }

  sc padding_space = _distributed_spacetime_tree
                       ->get_spatial_paddings( )[ src_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // get information about the time elements in the target
  // cluster and its mesh
  lo n_tar_time_elems = tar_cluster->get_n_time_elements( );
  lo n_tar_space_elems = tar_cluster->get_n_space_elements( );
  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = tar_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  // m2ls operations have to be executed for each time element in the target
  // cluster individually.
  sc * all_spatial_local_contributions
    = tar_cluster->get_pointer_to_spatial_local_contributions( );
  const sc * src_moment = src_cluster->get_pointer_to_moment( );
  // get the number of the current thread to access the correct buffers in the
  // m2ls operation
  lo thread_num = omp_get_thread_num( );
  for ( lo i_tar_time = 0; i_tar_time < n_tar_time_elems; ++i_tar_time ) {
    // get the address of the spatial local contributions of the current
    // time-step.
    sc * current_spatial_local_contributions = &(
      all_spatial_local_contributions[ i_tar_time * _spat_contribution_size ] );
    // reset the buffers
    _aux_buffer_0[ thread_num ].fill( 0.0 );
    _aux_buffer_1[ thread_num ].fill( 0.0 );

    // get the endpoints of the current target time element
    sc tar_t_elem_start, tar_t_elem_end;
    lo local_tar_elem_idx = distributed_mesh.global_2_local( local_start_idx,
      tar_cluster->get_element( i_tar_time * n_tar_space_elems ) );
    lo local_tar_elem_idx_time
      = local_mesh->get_time_element( local_tar_elem_idx );
    local_mesh->get_temporal_nodes(
      local_tar_elem_idx_time, &tar_t_elem_start, &tar_t_elem_end );
    sc tar_t_elem_size = tar_t_elem_end - tar_t_elem_start;
    // compute the temporal quadrature points in the target time element
    for ( slou i = 0; i < n_quad_points_time; ++i ) {
      quad_points_t_tar_elem[ i ]
        = tar_t_elem_start + tar_t_elem_size * quad_time_t[ i ];
    }
    // update the unrolled quadrature weights
    for ( lo i = 0; i < n_quad_points_time; ++i ) {
      for ( lo j = 0; j <= _temp_order; ++j ) {
        quad_time_w_unrld[ i * ( _temp_order + 1 ) + j ]
          = quad_time_w[ i ] * tar_t_elem_size;
      }
    }

    const sc * quad_time_w_unrl_data = quad_time_w_unrld.data( );

    lo hlp_acs_alpha
      = ( _spat_order + 1 ) * ( _temp_order + 1 ) * n_quad_points_time;
    lo hlp_acs_beta = ( _temp_order + 1 ) * n_quad_points_time;
    lo hlp_acs_a = ( _temp_order + 1 );

    sc * aux_buffer_0_data = _aux_buffer_0[ thread_num ].data( );
    sc * buffer_for_coeffs_data = buffer_for_coeffs.data( );

    // Execute the M2Ls operation for the current target time interval
    // only the last one-dimensional transform is different from a standard
    // M2L
    compute_m2l_coupling_coeffs( src_time_nodes, quad_points_t_tar_elem,
      half_size_space[ 2 ], center_diff_space[ 2 ], buffer_for_gaussians,
      buffer_for_coeffs );

    // compute first intermediate product and store it in aux_buffer_0
    lo buffer_0_index = 0;
    for ( lo alpha2 = 0; alpha2 <= _spat_order; ++alpha2 ) {
      lo moment_index = 0;
      for ( lo beta0 = 0; beta0 <= _spat_order - alpha2; ++beta0 ) {
        for ( lo beta1 = 0; beta1 <= _spat_order - alpha2 - beta0; ++beta1 ) {
          for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
            for ( lo a = 0; a < n_quad_points_time; ++a ) {
              // in a single inner cycle data are written on unique positions
#pragma omp simd aligned( aux_buffer_0_data, buffer_for_coeffs_data \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
              for ( lo b = 0; b <= _temp_order; ++b ) {
                aux_buffer_0_data[ buffer_0_index * hlp_acs_beta + hlp_acs_a * a
                  + b ]
                  += buffer_for_coeffs_data[ alpha2 * hlp_acs_alpha
                       + beta2 * hlp_acs_beta + a * hlp_acs_a + b ]
                  * src_moment[ b + moment_index * ( _temp_order + 1 ) ];
              }
            }
            ++moment_index;
          }
          ++buffer_0_index;
        }
        // correction for moment index; this is necessary since beta1 does not
        // run until _spat_order - beta0 as it does in src_moment;
        moment_index += ( ( alpha2 + 1 ) * alpha2 ) / 2;
      }
    }

    // update coefficients and compute 2nd intermediate product in
    // aux_buffer_1
    compute_m2l_coupling_coeffs( src_time_nodes, quad_points_t_tar_elem,
      half_size_space[ 1 ], center_diff_space[ 1 ], buffer_for_gaussians,
      buffer_for_coeffs );

    sc * aux_buffer_1_data = _aux_buffer_1[ thread_num ].data( );

    lo buffer_1_index = 0;
    for ( lo alpha1 = 0; alpha1 <= _spat_order; ++alpha1 ) {
      buffer_0_index = 0;
      for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha1; ++alpha2 ) {
        for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
          for ( lo beta1 = 0; beta1 <= _spat_order - beta0 - alpha2; ++beta1 ) {
            for ( lo a = 0; a < n_quad_points_time; ++a ) {
#pragma omp simd aligned(                                      \
  aux_buffer_1_data, buffer_for_coeffs_data, aux_buffer_0_data \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
              for ( lo b = 0; b <= _temp_order; ++b ) {
                aux_buffer_1_data[ buffer_1_index * hlp_acs_beta + hlp_acs_a * a
                  + b ]
                  += buffer_for_coeffs_data[ alpha1 * hlp_acs_alpha
                       + beta1 * hlp_acs_beta + a * hlp_acs_a + b ]
                  * aux_buffer_0_data[ buffer_0_index * hlp_acs_beta
                    + hlp_acs_a * a + b ];
              }
            }
            ++buffer_0_index;
          }
          ++buffer_1_index;
        }
        // correction for buffer_0 index; this is necessary since beta0 does
        // not run until _spat_order - alpha2 as it does in aux_buffer_0;
        buffer_0_index += ( ( alpha1 + 1 ) * alpha1 ) / 2;
      }
    }

    // update coefficients and update the target's current spatial local
    // contribution with m2ls result
    compute_m2l_coupling_coeffs( src_time_nodes, quad_points_t_tar_elem,
      half_size_space[ 0 ], center_diff_space[ 0 ], buffer_for_gaussians,
      buffer_for_coeffs );

    int local_index = 0;
    for ( lo alpha0 = 0; alpha0 <= _spat_order; ++alpha0 ) {
      buffer_1_index = 0;
      for ( lo alpha1 = 0; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
        for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha0 - alpha1;
              ++alpha2 ) {
          sc val = 0.0;
          for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2;
                ++beta0 ) {
#pragma omp simd aligned( buffer_for_coeffs_data, quad_time_w_unrl_data, \
        aux_buffer_1_data : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH) \
        reduction( + : val)
            for ( lo ab = 0; ab < n_quad_points_time * ( _temp_order + 1 );
                  ++ab ) {
              val += buffer_for_coeffs_data[ alpha0 * hlp_acs_alpha
                       + beta0 * hlp_acs_beta + ab ]
                * aux_buffer_1_data[ buffer_1_index * hlp_acs_beta + ab ]
                * quad_time_w_unrl_data[ ab ];
            }
            ++buffer_1_index;
          }
          current_spatial_local_contributions[ local_index ] += val;
          ++local_index;
        }
        // correction for buffer_1 index; this is necessary since alpha0 does
        // not run until _spat_order - alpha1 as it does in aux_buffer_1;
        buffer_1_index += ( ( alpha0 + 1 ) * alpha0 ) / 2;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_l2ls_operation( mesh::general_spacetime_cluster *
  /*current_cluster*/ ) const {
  std::cout << "General L2Ls operation not implemented" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply_l2ls_operation< 0 >( mesh::
    general_spacetime_cluster * current_cluster ) const {
  apply_l2ls_operation_p0_time( current_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::apply_l2ls_operation< 0 >( mesh::
    general_spacetime_cluster * current_cluster ) const {
  apply_l2ls_operation_p0_time( current_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply_l2ls_operation< 0 >( mesh::
    general_spacetime_cluster * current_cluster ) const {
  apply_l2ls_operation_p0_time( current_cluster );
}

//! TODO Add specialization for hypersingular operator.

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_l2ls_operation_p0_time( mesh::
    general_spacetime_cluster * current_cluster ) const {
  // get current space-time and spatial local contribution and assemble matrix
  // of integrated Lagrange polynomials in time.
  const sc * st_loc_contribution
    = current_cluster->get_pointer_to_local_contribution( );
  sc * s_loc_contribution
    = current_cluster->get_pointer_to_spatial_local_contributions( );
  full_matrix L;
  compute_lagrange_quadrature( L, current_cluster );
  // interpret the spatial local contribution as a matrix with
  // _spat_contribution_size rows and n_time_elems columns (in column major
  // order), and compute them via a matrix-matrix multiplication using blas.
  // (trans(lambda) * L)
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = 1.0;
  sc beta = 1.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_local,
    n_cols_lagrange, n_rows_lagrange, alpha, st_loc_contribution, lda,
    L.data( ), ldb, beta, s_loc_contribution, n_cols_local );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_ls2ls_operation( mesh::general_spacetime_cluster *
                                           parent_cluster,
  mesh::general_spacetime_cluster * child_cluster ) const {
  // get the necessary cluster information and the spatial local contributions
  lo parent_space_level, dummy;
  parent_cluster->get_n_divs( parent_space_level, dummy );
  short child_octant = child_cluster->get_spatial_octant( );
  lo n_time_elems = parent_cluster->get_n_time_elements( );
  sc * child_s_local
    = child_cluster->get_pointer_to_spatial_local_contributions( );
  const sc * parent_s_local
    = parent_cluster->get_pointer_to_spatial_local_contributions( );

  const vector_type * m2m_coeffs_s_dim_0;
  const vector_type * m2m_coeffs_s_dim_1;
  const vector_type * m2m_coeffs_s_dim_2;
  switch ( child_octant ) {
    case 0:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ parent_space_level ] );
      break;
    case 1:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ parent_space_level ] );
      break;
    case 2:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ parent_space_level ] );
      break;
    case 3:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ parent_space_level ] );
      break;
    case 4:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ parent_space_level ] );
      break;
    case 5:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ parent_space_level ] );
      break;
    case 6:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ parent_space_level ] );
      break;
    case 7:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ parent_space_level ] );
      break;
    default:  // default case should never be used, program will crash!
      assert( child_octant > -1 && child_octant < 8 );
      m2m_coeffs_s_dim_0 = nullptr;
      m2m_coeffs_s_dim_1 = nullptr;
      m2m_coeffs_s_dim_2 = nullptr;
  }

  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  // initialize auxiliary vectors lambda_1/2 for intermediate results with 0
  vector_type lambda_1( n_time_elems * n_coeffs_s, true );
  vector_type lambda_2( n_time_elems * n_coeffs_s, true );

  for ( lo b = 0; b < n_time_elems; ++b ) {
    for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
      lou parent_index = 0;
      for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
        for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0;
              ++alpha1 ) {
          // correction for skipped entries of parent_local due to starting
          // point alpha2 = beta2 in the next loop
          parent_index += beta2;
          for ( lo alpha2 = beta2; alpha2 <= _spat_order - alpha0 - alpha1;
                ++alpha2 ) {
            lambda_1[ b * n_coeffs_s
              + ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
              + ( _spat_order + 1 ) * alpha0 + alpha1 ]
              += ( *m2m_coeffs_s_dim_2 )[ alpha2 * ( _spat_order + 1 ) + beta2 ]
              * parent_s_local[ b * n_coeffs_s + parent_index ];

            ++parent_index;
          }
        }
        // correction for current index; this is necessary since alpha1 does not
        // run until _spat_order - alpha0 as it does in parent_local;
        parent_index += ( ( beta2 + 1 ) * beta2 ) / 2;
      }
    }
  }

  for ( lo b = 0; b < n_time_elems; ++b ) {
    for ( lo beta1 = 0; beta1 <= _spat_order; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta1; ++beta2 ) {
        for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
          for ( lo alpha1 = beta1; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
            lambda_2[ b * n_coeffs_s
              + ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
              + ( _spat_order + 1 ) * beta2 + alpha0 ]
              += ( *m2m_coeffs_s_dim_1 )[ alpha1 * ( _spat_order + 1 ) + beta1 ]
              * lambda_1[ b * n_coeffs_s
                + ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 ];
          }
        }
      }
    }
  }

  for ( lo b = 0; b < n_time_elems; ++b ) {
    lou child_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          for ( lo alpha0 = beta0; alpha0 <= _spat_order - beta1 - beta2;
                ++alpha0 ) {
            child_s_local[ b * n_coeffs_s + child_index ]
              += ( *m2m_coeffs_s_dim_0 )[ alpha0 * ( _spat_order + 1 ) + beta0 ]
              * lambda_2[ b * n_coeffs_s
                + ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                + ( _spat_order + 1 ) * beta2 + alpha0 ];
          }
          ++child_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_ls2t_operation( distributed_block_vector & /*output_vector*/,
    const mesh::general_spacetime_cluster *
    /*tar_cluster*/,
    const mesh::general_spacetime_cluster * /*tar_element_cluster*/
  ) const {
  std::cout << "General Ls2T operation not implemented." << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_ls2t_operation< 0 >( distributed_block_vector & output_vector,
    const mesh::general_spacetime_cluster * tar_cluster,
    const mesh::general_spacetime_cluster * tar_element_cluster ) const {
  apply_ls2t_operation_p0( output_vector, tar_cluster, tar_element_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_ls2t_operation< 0 >( distributed_block_vector & output_vector,
    const mesh::general_spacetime_cluster * tar_cluster,
    const mesh::general_spacetime_cluster * tar_element_cluster ) const {
  apply_ls2t_operation_p0( output_vector, tar_cluster, tar_element_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_ls2t_operation< 0 >( distributed_block_vector & output_vector,
    const mesh::general_spacetime_cluster * tar_cluster,
    const mesh::general_spacetime_cluster * tar_element_cluster ) const {
  apply_ls2t_operation_p1_normal_drv(
    output_vector, tar_cluster, tar_element_cluster );
}

// TODO: add template specialization for hypersingular operator

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_ls2t_operation_p0( distributed_block_vector &
                                             output_vector,
  const mesh::general_spacetime_cluster * tar_cluster,
  const mesh::general_spacetime_cluster * tar_element_cluster ) const {
  // determine the actual tar element cluster (in case that tar_element_cluster
  // == nullptr)
  const mesh::general_spacetime_cluster * actual_tar_elem_cluster
    = tar_element_cluster;
  if ( actual_tar_elem_cluster == nullptr ) {
    actual_tar_elem_cluster = tar_cluster;
  }

  lo n_time_elems = actual_tar_elem_cluster->get_n_time_elements( );
  lo n_space_elements = actual_tar_elem_cluster->get_n_space_elements( );

  full_matrix T;
  compute_chebyshev_quadrature_p0( T, actual_tar_elem_cluster, tar_cluster );
  // T has dimensions (n_space_elements, _spat_contribution_size)
  // multiply T by the spatial local contributions to get the local result
  // and add this to the output vector

  vector local_result_vector( n_space_elements * n_time_elems, true );
  const sc * all_spatial_local_contributions
    = tar_cluster->get_pointer_to_spatial_local_contributions( );

  for ( lo i_tar_time = 0; i_tar_time < n_time_elems; ++i_tar_time ) {
    const sc * current_spatial_local_contributions = &(
      all_spatial_local_contributions[ _spat_contribution_size * i_tar_time ] );
    sc * current_local_result_vector
      = &( local_result_vector.data( )[ i_tar_time * n_space_elements ] );
    // blas routine is called explicitly here, since we work with raw vectors
    cblas_dgemv( CblasColMajor, CblasNoTrans, n_space_elements,
      _spat_contribution_size, 1.0, T.data( ), n_space_elements,
      current_spatial_local_contributions, 1, 0.0, current_local_result_vector,
      1 );
  }
  // add the results to the correct positions of the output vector
  output_vector.add_local_part< target_space >(
    actual_tar_elem_cluster, local_result_vector );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_ls2t_operation_p1_normal_drv( distributed_block_vector &
                                                        output_vector,
  const mesh::general_spacetime_cluster * tar_cluster,
  const mesh::general_spacetime_cluster * tar_element_cluster ) const {
  // determine the actual tar element cluster (in case that tar_element_cluster
  // == nullptr)
  const mesh::general_spacetime_cluster * actual_tar_elem_cluster
    = tar_element_cluster;
  if ( actual_tar_elem_cluster == nullptr ) {
    actual_tar_elem_cluster = tar_cluster;
  }

  lo n_time_elems = actual_tar_elem_cluster->get_n_time_elements( );
  lo n_space_nodes = actual_tar_elem_cluster->get_n_space_nodes( );

  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1(
    T_drv, tar_element_cluster, tar_cluster );
  // T_drv has dimensions (n_space_nodes, _spat_contribution_size)
  // multiply T by the spatial local contributions to get the local result
  // and add this to the output vector

  vector local_result_vector( n_space_nodes * n_time_elems, true );
  const sc * all_spatial_local_contributions
    = tar_cluster->get_pointer_to_spatial_local_contributions( );

  for ( lo i_tar_time = 0; i_tar_time < n_time_elems; ++i_tar_time ) {
    const sc * current_spatial_local_contributions = &(
      all_spatial_local_contributions[ _spat_contribution_size * i_tar_time ] );
    sc * current_local_result_vector
      = &( local_result_vector.data( )[ i_tar_time * n_space_nodes ] );
    // blas routine is called explicitly here, since we work with raw vectors
    cblas_dgemv( CblasColMajor, CblasNoTrans, n_space_nodes,
      _spat_contribution_size, 1.0, T_drv.data( ), n_space_nodes,
      current_spatial_local_contributions, 1, 0.0, current_local_result_vector,
      1 );
  }
  // add the results to the correct positions of the output vector
  output_vector.add_local_part< target_space >(
    actual_tar_elem_cluster, local_result_vector );
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_s2l_operations( const distributed_block_vector & sources,
  mesh::scheduling_time_cluster * t_cluster, bool verbose,
  const std::string & verbose_file ) const {
  // execute first all standard s2l operations
  const std::vector< general_spacetime_cluster * > * assoc_spacetime_targets
    = t_cluster->get_associated_spacetime_clusters( );
  const std::vector< lo > * assoc_std_s2l_tar_indices
    = t_cluster->get_assoc_standard_s2l_targets( );
  if ( assoc_std_s2l_tar_indices != nullptr ) {
    if ( verbose ) {
#pragma omp critical( verbose )
      {
        std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "call standard S2L operations for cluster "
                  << t_cluster->get_global_index( ) << " at level "
                  << t_cluster->get_level( ) << std::endl;
          outfile.close( );
        }
      }
    }
#pragma omp taskloop shared( \
  sources, assoc_spacetime_targets, assoc_std_s2l_tar_indices )
    for ( lou i = 0; i < assoc_std_s2l_tar_indices->size( ); ++i ) {
      if ( _measure_tasks ) {
        _s2l_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      general_spacetime_cluster * current_spacetime_target
        = ( *assoc_spacetime_targets )[ ( *assoc_std_s2l_tar_indices )[ i ] ];
      std::vector< general_spacetime_cluster * > * spacetime_s2l_list
        = current_spacetime_target->get_s2l_list( );
      lou n_hybrid_s2l_ops
        = current_spacetime_target->get_n_hybrid_s2l_operations( );
      // apply an s2l operation for each source cluster in the target's s2l list
      for ( lou j = n_hybrid_s2l_ops; j < spacetime_s2l_list->size( ); ++j ) {
        general_spacetime_cluster * current_source
          = ( *spacetime_s2l_list )[ j ];
        apply_s2l_operation< run_count >(
          sources, current_source, current_spacetime_target );
      }

      if ( _measure_tasks ) {
        _s2l_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }

  // next, execute all hybrid s2l operations
  const std::vector< lo > * assoc_hybrid_s2l_tar_indices
    = t_cluster->get_assoc_hybrid_s2l_targets( );
  if ( assoc_hybrid_s2l_tar_indices != nullptr ) {
    if ( verbose ) {
#pragma omp critical( verbose )
      {
        std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
        if ( outfile.is_open( ) ) {
          outfile << "call hybrid S2L operations for cluster "
                  << t_cluster->get_global_index( ) << " at level "
                  << t_cluster->get_level( ) << std::endl;
          outfile.close( );
        }
      }
    }
#pragma omp taskloop shared( \
  assoc_spacetime_targets, assoc_hybrid_s2l_tar_indices )
    for ( lou i = 0; i < assoc_hybrid_s2l_tar_indices->size( ); ++i ) {
      if ( _measure_tasks ) {
        _s2l_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
      general_spacetime_cluster * current_spacetime_target
        = ( *assoc_spacetime_targets )[ (
          *assoc_hybrid_s2l_tar_indices )[ i ] ];
      std::vector< general_spacetime_cluster * > * spacetime_s2l_list
        = current_spacetime_target->get_s2l_list( );
      lo n_hybrid_s2l_ops
        = current_spacetime_target->get_n_hybrid_s2l_operations( );
      // apply an s2l operation for each source cluster in the target's s2l list
      for ( lo j = 0; j < n_hybrid_s2l_ops; ++j ) {
        general_spacetime_cluster * current_source
          = ( *spacetime_s2l_list )[ j ];
        apply_ms2l_operation< run_count >(
          current_source, current_spacetime_target );
      }
      if ( _measure_tasks ) {
        _s2l_subtask_times.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2l_operation( const distributed_block_vector &
  /*src_vector*/,
  const mesh::general_spacetime_cluster * /*src_cluster*/,
  mesh::general_spacetime_cluster * /*tar_cluster*/ ) const {
  std::cout << "General S2L operation not implemented!" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_s2l_operation< 0 >( const distributed_block_vector & src_vector,
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_s2l_operation_p0( src_vector, src_cluster, tar_cluster );
}

//! template specialization for double layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2l_operation< 0 >( const distributed_block_vector & src_vector,
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_s2l_operation_p1_normal_drv( src_vector, src_cluster, tar_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_s2l_operation< 0 >( const distributed_block_vector & src_vector,
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_s2l_operation_p0( src_vector, src_cluster, tar_cluster );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2l_operation_p0( const distributed_block_vector &
                                            src_vector,
  const mesh::general_spacetime_cluster * src_cluster,
  mesh::general_spacetime_cluster * tar_cluster,
  const lo quad_order_space ) const {
  // get information about the elements in the source cluster
  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = src_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * src_mesh;
  lo src_start_idx;
  if ( src_cluster->get_elements_are_local( ) ) {
    src_mesh = distributed_mesh.get_local_mesh( );
    src_start_idx = distributed_mesh.get_local_start_idx( );
  } else {
    src_mesh = distributed_mesh.get_nearfield_mesh( );
    src_start_idx = distributed_mesh.get_nearfield_start_idx( );
  }
  lo n_src_elems = src_cluster->get_n_elements( );

  // get the local part of the src vector whose entries are related to the
  // elements of the source cluster.
  vector_type local_source_vector( n_src_elems, false );
  src_vector.get_local_part< source_space >( src_cluster, local_source_vector );

  // get geometrical data of the target cluster
  sc tar_half_size_time;
  vector_type half_size_space( 3, false );
  tar_cluster->get_half_size( half_size_space, tar_half_size_time );
  sc tar_center_time;
  vector_type tar_center_space( 3, false );
  tar_cluster->get_center( tar_center_space, tar_center_time );
  sc padding_space = _distributed_spacetime_tree
                       ->get_spatial_paddings( )[ tar_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // initialize temporal interpolation nodes in the target cluster
  vector_type tar_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    tar_time_nodes[ i ]
      = tar_center_time + tar_half_size_time * time_nodes[ i ];
  }

  // initialize quadrature data in space
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature, quad_order_space );
  lo s_quad_size = my_quadrature._wy_cheb.size( );
  const sc * quad_space_w = my_quadrature._wy_cheb.data( );
  const sc * y0_quad_elem = my_quadrature._y1.data( );
  const sc * y1_quad_elem = my_quadrature._y2.data( );
  const sc * y2_quad_elem = my_quadrature._y3.data( );
  vector_type quad_point_src( 3, false );
  linear_algebra::coordinates< 3 > y1, y2, y3;
  // initialize quadrature data in time
  // choose the quadrature order such that polynomials of degree _temp_order
  // are integrated exactly (we use the same degree for s2m and l2t
  // operations)
  lo n_quad_points_time = ( _temp_order + 2 ) / 2;
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_t
    = bem::quadrature::line_x( n_quad_points_time );
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_w
    = bem::quadrature::line_w( n_quad_points_time );
  vector_type quad_time_t_elem( n_quad_points_time );
  vector_type time_node_differences( _temp_order + 1 );

  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2t coefficients
  vector_type buffer_m2t_coeffs( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  // buffer to store m2t coefficients.
  vector_type buffer_coeffs_0( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_coeffs_1( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_coeffs_2( ( _spat_order + 1 ) * ( _temp_order + 1 ) );

  vector_type coupling_coeffs_tensor_product( _contribution_size );
  const sc * coupling_coeffs_tensor_product_data
    = coupling_coeffs_tensor_product.data( );
  std::vector< vector_type * > vector_of_buffers = { &buffer_coeffs_0,
    &buffer_coeffs_1, &buffer_coeffs_2, &buffer_m2t_coeffs };

  // loop over all space-time elements in the target cluster
  for ( lo i_elem = 0; i_elem < n_src_elems; ++i_elem ) {
    lo local_elem_idx = distributed_mesh.global_2_local(
      src_start_idx, src_cluster->get_element( i_elem ) );
    // get the temporal information of the current space-time element
    lo local_elem_idx_time = src_mesh->get_time_element( local_elem_idx );
    sc t_start_elem, t_end_elem, t_size_elem;
    src_mesh->get_temporal_nodes(
      local_elem_idx_time, &t_start_elem, &t_end_elem );
    t_size_elem = t_end_elem - t_start_elem;
    // compute the temporal quadrature points in the element
    for ( slou i = 0; i < n_quad_points_time; ++i ) {
      quad_time_t_elem[ i ] = t_start_elem + t_size_elem * quad_time_t[ i ];
    }

    // get the spatial information of the current space-time element
    lo local_elem_idx_space
      = src_mesh->get_space_element_index( local_elem_idx );
    src_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc s_area_elem
      = src_mesh->get_spatial_area_using_spatial_index( local_elem_idx_space );
    // compute the spatial quadrature points in the element
    triangle_to_geometry( y1, y2, y3, my_quadrature );

    // loop over all quadrature points in space and time and execute the
    // appropriate s2l coupling operations
    for ( lo i_quad_t = 0; i_quad_t < n_quad_points_time; ++i_quad_t ) {
      // update the vector of time node differences
      for ( lo i_t = 0; i_t <= _temp_order; ++i_t ) {
        time_node_differences[ i_t ]
          = tar_time_nodes[ i_t ] - quad_time_t_elem[ i_quad_t ];
      }
      for ( lo i_quad_s = 0; i_quad_s < s_quad_size; ++i_quad_s ) {
        // update the quadrature point in space
        quad_point_src[ 0 ] = y0_quad_elem[ i_quad_s ];
        quad_point_src[ 1 ] = y1_quad_elem[ i_quad_s ];
        quad_point_src[ 2 ] = y2_quad_elem[ i_quad_s ];
        // compute the tensor-product coupling coefficients
        compute_single_sided_coupling_coeffs_tensor( time_node_differences,
          half_size_space, tar_center_space, quad_point_src, vector_of_buffers,
          coupling_coeffs_tensor_product );
        // add weighted coupling coefficients to the local contributions of
        // the target cluster
        sc weight = local_source_vector[ i_elem ] * quad_time_w[ i_quad_t ]
          * quad_space_w[ i_quad_s ] * s_area_elem * t_size_elem;
        sc * tar_local_contributions
          = tar_cluster->get_pointer_to_local_contribution( );
#pragma omp simd aligned( coupling_coeffs_tensor_product_data \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
        for ( lo glob_index = 0; glob_index < _contribution_size;
              ++glob_index ) {
          tar_local_contributions[ glob_index ]
            += weight * coupling_coeffs_tensor_product_data[ glob_index ];
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_s2l_operation_p1_normal_drv(
    const distributed_block_vector & src_vector,
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  // get information about the elements in the source cluster
  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = src_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * src_mesh;
  lo src_start_idx;
  if ( src_cluster->get_elements_are_local( ) ) {
    src_mesh = distributed_mesh.get_local_mesh( );
    src_start_idx = distributed_mesh.get_local_start_idx( );
  } else {
    src_mesh = distributed_mesh.get_nearfield_mesh( );
    src_start_idx = distributed_mesh.get_nearfield_start_idx( );
  }
  lo n_src_elems = src_cluster->get_n_elements( );
  lo n_src_time_elems = src_cluster->get_n_time_elements( );
  lo n_src_space_elems = src_cluster->get_n_space_elements( );
  lo n_src_space_nodes = src_cluster->get_n_space_nodes( );
  const std::vector< lo > & src_elems_2_local_nodes
    = src_cluster->get_elems_2_local_nodes( );
  linear_algebra::coordinates< 3 > normal;

  // get the local part of the src vector whose entries are related to the
  // elements of the source cluster.
  vector_type local_source_vector(
    n_src_time_elems * n_src_space_nodes, false );
  src_vector.get_local_part< source_space >( src_cluster, local_source_vector );

  // get geometrical data of the target cluster
  sc tar_half_size_time;
  vector_type half_size_space( 3, false );
  tar_cluster->get_half_size( half_size_space, tar_half_size_time );
  sc tar_center_time;
  vector_type tar_center_space( 3, false );
  tar_cluster->get_center( tar_center_space, tar_center_time );
  sc padding_space = _distributed_spacetime_tree
                       ->get_spatial_paddings( )[ tar_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // initialize temporal interpolation nodes in the target cluster
  vector_type tar_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    tar_time_nodes[ i ]
      = tar_center_time + tar_half_size_time * time_nodes[ i ];
  }

  // initialize quadrature data in space
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature, _order_regular );
  lo s_quad_size = my_quadrature._wy_cheb.size( );
  const sc * quad_space_w = my_quadrature._wy_cheb.data( );
  const sc * y0_quad_elem = my_quadrature._y1.data( );
  const sc * y1_quad_elem = my_quadrature._y2.data( );
  const sc * y2_quad_elem = my_quadrature._y3.data( );
  const sc * y0_quad_ref_elem = my_quadrature._y1_ref_cheb.data( );
  const sc * y1_quad_ref_elem = my_quadrature._y2_ref_cheb.data( );
  vector_type quad_point_src( 3, false );
  linear_algebra::coordinates< 3 > y1, y2, y3;
  // initialize quadrature data in time
  // choose the quadrature order such that polynomials of degree _temp_order
  // are integrated exactly (we use the same degree for s2m and l2t
  // operations)
  lo n_quad_points_time = ( _temp_order + 2 ) / 2;
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_t
    = bem::quadrature::line_x( n_quad_points_time );
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_w
    = bem::quadrature::line_w( n_quad_points_time );
  vector_type quad_time_t_elem( n_quad_points_time );
  vector_type time_node_differences( _temp_order + 1 );

  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2t coefficients
  vector_type buffer_m2t_coeffs_0( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_m2t_coeffs_1( ( _temp_order + 1 ) );
  // buffer to store m2t coefficients.
  vector_type buffer_coeffs_0( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_coeffs_1( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_coeffs_2( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_drv_coeffs_0( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_drv_coeffs_1( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  vector_type buffer_drv_coeffs_2( ( _spat_order + 1 ) * ( _temp_order + 1 ) );
  std::vector< vector_type * > vector_of_buffers
    = { &buffer_coeffs_0, &buffer_coeffs_1, &buffer_coeffs_2,
        &buffer_drv_coeffs_0, &buffer_drv_coeffs_1, &buffer_drv_coeffs_2,
        &buffer_m2t_coeffs_0, &buffer_m2t_coeffs_1 };
  vector_type coupling_coeffs_tensor_product( _contribution_size );
  const sc * coupling_coeffs_tensor_product_data
    = coupling_coeffs_tensor_product.data( );

  // loop over all space-time elements in the target cluster
  lo i_time_elem = 0;
  lo i_space_elem = 0;
  // loop over all space-time elements in the target cluster
  for ( lo i_elem = 0; i_elem < n_src_elems; ++i_elem ) {
    lo local_elem_idx = distributed_mesh.global_2_local(
      src_start_idx, src_cluster->get_element( i_elem ) );
    // get the temporal information of the current space-time element
    lo local_elem_idx_time = src_mesh->get_time_element( local_elem_idx );
    sc t_start_elem, t_end_elem, t_size_elem;
    src_mesh->get_temporal_nodes(
      local_elem_idx_time, &t_start_elem, &t_end_elem );
    t_size_elem = t_end_elem - t_start_elem;
    // compute the temporal quadrature points in the element
    for ( slou i = 0; i < n_quad_points_time; ++i ) {
      quad_time_t_elem[ i ] = t_start_elem + t_size_elem * quad_time_t[ i ];
    }

    // get the spatial information of the current space-time element
    lo local_elem_idx_space
      = src_mesh->get_space_element_index( local_elem_idx );
    src_mesh->get_spatial_normal_using_spatial_element_index(
      local_elem_idx_space, normal );
    src_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc s_area_elem
      = src_mesh->get_spatial_area_using_spatial_index( local_elem_idx_space );
    // compute the spatial quadrature points in the element
    triangle_to_geometry( y1, y2, y3, my_quadrature );

    // loop over all quadrature points in space and time and execute the
    // appropriate s2l coupling operations
    for ( lo i_quad_t = 0; i_quad_t < n_quad_points_time; ++i_quad_t ) {
      // update the vector of time node differences
      for ( lo i_t = 0; i_t <= _temp_order; ++i_t ) {
        time_node_differences[ i_t ]
          = tar_time_nodes[ i_t ] - quad_time_t_elem[ i_quad_t ];
      }
      for ( lo i_quad_s = 0; i_quad_s < s_quad_size; ++i_quad_s ) {
        // update the quadrature point in space
        quad_point_src[ 0 ] = y0_quad_elem[ i_quad_s ];
        quad_point_src[ 1 ] = y1_quad_elem[ i_quad_s ];
        quad_point_src[ 2 ] = y2_quad_elem[ i_quad_s ];
        // compute the tensor-product coupling coefficients
        compute_single_sided_coupling_coeffs_normal_drv_tensor(
          time_node_differences, half_size_space, tar_center_space,
          quad_point_src, normal, vector_of_buffers,
          coupling_coeffs_tensor_product );

        // get the appropriate nodal entries of the local source vector
        // and multiply them by the value of the corresponding shape
        // function evaluated at the current quadrature point in space
        lo s_node_0_idx
          = src_cluster->local_spacetime_node_idx_2_local_space_node_idx(
            src_elems_2_local_nodes[ 6 * i_elem ] );
        lo s_node_1_idx
          = src_cluster->local_spacetime_node_idx_2_local_space_node_idx(
            src_elems_2_local_nodes[ 6 * i_elem + 1 ] );
        lo s_node_2_idx
          = src_cluster->local_spacetime_node_idx_2_local_space_node_idx(
            src_elems_2_local_nodes[ 6 * i_elem + 2 ] );
        sc val_0 = local_source_vector[ i_time_elem * n_src_space_nodes
                     + s_node_0_idx ]
          * ( (sc) 1.0 - y0_quad_ref_elem[ i_quad_s ]
            - y1_quad_ref_elem[ i_quad_s ] );
        sc val_1 = local_source_vector[ i_time_elem * n_src_space_nodes
                     + s_node_1_idx ]
          * y0_quad_ref_elem[ i_quad_s ];
        sc val_2 = local_source_vector[ i_time_elem * n_src_space_nodes
                     + s_node_2_idx ]
          * y1_quad_ref_elem[ i_quad_s ];
        // compute the weight by which the coupling coefficients are scaled
        sc weight = ( val_0 + val_1 + val_2 ) * quad_time_w[ i_quad_t ]
          * quad_space_w[ i_quad_s ] * s_area_elem * t_size_elem;
        // add weighted coupling coefficients to the local contributions of
        // the target cluster
        sc * tar_local_contributions
          = tar_cluster->get_pointer_to_local_contribution( );
#pragma omp simd aligned( coupling_coeffs_tensor_product_data \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
        for ( lo glob_index = 0; glob_index < _contribution_size;
              ++glob_index ) {
          tar_local_contributions[ glob_index ]
            += weight * coupling_coeffs_tensor_product_data[ glob_index ];
        }
      }
    }
    // update the local space and time indices for the next run in the loop.
    i_space_elem++;
    if ( i_space_elem == n_src_space_elems ) {
      i_time_elem++;
      i_space_elem = 0;
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2ms_operation( const distributed_block_vector &
  /*src_vector*/,
  mesh::general_spacetime_cluster * /*src_cluster*/,
  mesh::general_spacetime_cluster * /*src_geometry_cluster*/ ) const {
  std::cout << "General S2Ms operation not implemented " << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_s2ms_operation< 0 >( const distributed_block_vector & src_vector,
    mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * src_geometry_cluster ) const {
  apply_s2ms_operation_p0( src_vector, src_cluster, src_geometry_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2ms_operation< 0 >( const distributed_block_vector & src_vector,
    mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * src_geometry_cluster ) const {
  apply_s2ms_operation_p1_normal_drv(
    src_vector, src_cluster, src_geometry_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_s2ms_operation< 0 >( const distributed_block_vector & src_vector,
    mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * src_geometry_cluster ) const {
  apply_s2ms_operation_p0( src_vector, src_cluster, src_geometry_cluster );
}

// TODO: add template specialization for hypersingular operator

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2ms_operation_p0( const distributed_block_vector &
                                             src_vector,
  mesh::general_spacetime_cluster * src_cluster,
  mesh::general_spacetime_cluster * src_geometry_cluster ) const {
  lo n_time_elements = src_cluster->get_n_time_elements( );
  lo n_space_elements = src_cluster->get_n_space_elements( );

  sc * all_spatial_moments = src_cluster->get_pointer_to_spatial_moments( );
  // in some cases (auxiliary s2ms operations) a src cluster can have no
  // standard spatial moments but auxiliary spatial moments
  if ( all_spatial_moments == nullptr ) {
    all_spatial_moments = src_cluster->get_pointer_to_aux_spatial_moments( );
  }

  full_matrix T;
  compute_chebyshev_quadrature_p0( T, src_cluster, src_geometry_cluster );

  vector local_src_vector( n_time_elements * n_space_elements, false );
  src_vector.get_local_part< source_space >( src_cluster, local_src_vector );

  for ( lo i_src_time = 0; i_src_time < n_time_elements; ++i_src_time ) {
    sc * current_spatial_moments
      = &( all_spatial_moments[ i_src_time * _spat_contribution_size ] );
    sc * current_local_src_vector
      = &( local_src_vector.data( )[ i_src_time * n_space_elements ] );
    // blas routine is called explicitly here, since we work with raw vectors.
    cblas_dgemv( CblasColMajor, CblasTrans, n_space_elements,
      _spat_contribution_size, 1.0, T.data( ), n_space_elements,
      current_local_src_vector, 1, 0.0, current_spatial_moments, 1 );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_s2ms_operation_p1_normal_drv(
    const distributed_block_vector & src_vector,
    mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * src_geometry_cluster ) const {
  lo n_time_elements = src_cluster->get_n_time_elements( );
  lo n_space_nodes = src_cluster->get_n_space_nodes( );

  sc * all_spatial_moments = src_cluster->get_pointer_to_spatial_moments( );
  // in some cases (auxiliary s2ms operations) a src cluster can have no
  // standard spatial moments but auxiliary spatial moments
  if ( all_spatial_moments == nullptr ) {
    all_spatial_moments = src_cluster->get_pointer_to_aux_spatial_moments( );
  }

  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1(
    T_drv, src_cluster, src_geometry_cluster );

  vector local_src_vector( n_time_elements * n_space_nodes, false );
  src_vector.get_local_part< source_space >( src_cluster, local_src_vector );

  for ( lo i_src_time = 0; i_src_time < n_time_elements; ++i_src_time ) {
    sc * current_spatial_moments
      = &( all_spatial_moments[ i_src_time * _spat_contribution_size ] );
    sc * current_local_src_vector
      = &( local_src_vector.data( )[ i_src_time * n_space_nodes ] );
    // blas routine is called explicitly here, since we work with raw vectors.
    cblas_dgemv( CblasColMajor, CblasTrans, n_space_nodes,
      _spat_contribution_size, 1.0, T_drv.data( ), n_space_nodes,
      current_local_src_vector, 1, 0.0, current_spatial_moments, 1 );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::sum_up_auxiliary_spatial_moments( mesh::
    general_spacetime_cluster * src_cluster ) const {
  sc * all_spatial_moments = src_cluster->get_pointer_to_spatial_moments( );
  const sc * all_aux_spatial_moments
    = src_cluster->get_pointer_to_aux_spatial_moments( );
  lo n_aux_spatial_moments = src_cluster->get_n_aux_spatial_moments( );
  lo n_time_elems = src_cluster->get_n_time_elements( );
  lo aux_mom_offset = 0;
  for ( lo leaf_descendant_idx = 0; leaf_descendant_idx < n_aux_spatial_moments;
        ++leaf_descendant_idx ) {
    for ( lo i = 0; i < _spat_contribution_size * n_time_elems; ++i ) {
      all_spatial_moments[ i ] += all_aux_spatial_moments[ aux_mom_offset + i ];
    }
    aux_mom_offset += _spat_contribution_size * n_time_elems;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_ms2ms_operation( const mesh::general_spacetime_cluster *
                                           child_cluster,
  mesh::general_spacetime_cluster * parent_cluster ) const {
  // get the necessary cluster information and the spatial moments
  lo parent_space_level, dummy;
  parent_cluster->get_n_divs( parent_space_level, dummy );
  short child_octant = child_cluster->get_spatial_octant( );
  lo n_time_elems = parent_cluster->get_n_time_elements( );
  const sc * child_s_moments = child_cluster->get_pointer_to_spatial_moments( );
  sc * parent_s_moments = parent_cluster->get_pointer_to_spatial_moments( );
  // get the proper m2m coefficients (these are the same as used for spatial
  // m2m operations)
  const vector_type * m2m_coeffs_s_dim_0;
  const vector_type * m2m_coeffs_s_dim_1;
  const vector_type * m2m_coeffs_s_dim_2;
  switch ( child_octant ) {
    case 0:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ parent_space_level ] );
      break;
    case 1:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ parent_space_level ] );
      break;
    case 2:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ parent_space_level ] );
      break;
    case 3:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_right[ parent_space_level ] );
      break;
    case 4:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ parent_space_level ] );
      break;
    case 5:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_right[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ parent_space_level ] );
      break;
    case 6:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_left[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ parent_space_level ] );
      break;
    case 7:
      m2m_coeffs_s_dim_0 = &( _m2m_coeffs_s_dim_0_right[ parent_space_level ] );
      m2m_coeffs_s_dim_1 = &( _m2m_coeffs_s_dim_1_left[ parent_space_level ] );
      m2m_coeffs_s_dim_2 = &( _m2m_coeffs_s_dim_2_left[ parent_space_level ] );
      break;
    default:  // default case should never be used, program will crash!
      assert( child_octant < 8 );
      m2m_coeffs_s_dim_0 = nullptr;
      m2m_coeffs_s_dim_1 = nullptr;
      m2m_coeffs_s_dim_2 = nullptr;
  }
  lo n_coeffs_s
    = ( _spat_order + 1 ) * ( _spat_order + 1 ) * ( _spat_order + 1 );
  // initialize auxiliary vectors lambda_1/2 for intermediate results with 0
  vector_type lambda_1( n_time_elems * n_coeffs_s, true );
  vector_type lambda_2( n_time_elems * n_coeffs_s, true );

  for ( lo b = 0; b < n_time_elems; ++b ) {
    for ( lo beta2 = 0; beta2 <= _spat_order; ++beta2 ) {
      lo child_index = 0;
      for ( lo alpha0 = 0; alpha0 <= _spat_order - beta2; ++alpha0 ) {
        for ( lo alpha1 = 0; alpha1 <= _spat_order - beta2 - alpha0;
              ++alpha1 ) {
          lo alpha2;
          for ( alpha2 = 0; alpha2 <= beta2; ++alpha2 ) {
            lambda_1( b * n_coeffs_s
              + ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
              + ( _spat_order + 1 ) * alpha0 + alpha1 )
              += ( *m2m_coeffs_s_dim_2 )[ beta2 * ( _spat_order + 1 ) + alpha2 ]
              * child_s_moments[ b * n_coeffs_s + child_index ];
            ++child_index;
          }
          // correction needed for skipped entries of child_moment
          child_index += _spat_order + 1 - alpha0 - alpha1 - alpha2;
        }
        // correction for current index; necessary since alpha1 does not run
        // until _spat_order - alpha0 as it does in stored child_moment
        child_index += ( ( beta2 + 1 ) * beta2 ) / 2;
      }
    }
  }

  // compute intermediate result lambda_2 not exploiting zero entries for the
  // sake of better readability
  for ( lo b = 0; b < n_time_elems; ++b ) {
    for ( lo beta1 = 0; beta1 <= _spat_order; ++beta1 ) {
      for ( lo beta2 = 0; beta2 <= _spat_order - beta1; ++beta2 ) {
        for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2; ++alpha0 ) {
          for ( lo alpha1 = 0; alpha1 <= beta1; ++alpha1 ) {
            lambda_2( b * n_coeffs_s
              + ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
              + ( _spat_order + 1 ) * beta2 + alpha0 )
              += ( *m2m_coeffs_s_dim_1 )[ beta1 * ( _spat_order + 1 ) + alpha1 ]
              * lambda_1( b * n_coeffs_s
                + ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta2
                + ( _spat_order + 1 ) * alpha0 + alpha1 );
          }
        }
      }
    }
  }

  for ( lo b = 0; b < n_time_elems; ++b ) {
    lo parent_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          for ( lo alpha0 = 0; alpha0 <= _spat_order - beta1 - beta2;
                ++alpha0 ) {
            parent_s_moments[ b * n_coeffs_s + parent_index ]
              += ( *m2m_coeffs_s_dim_0 )[ beta0 * ( _spat_order + 1 ) + alpha0 ]
              * lambda_2( b * n_coeffs_s
                + ( _spat_order + 1 ) * ( _spat_order + 1 ) * beta1
                + ( _spat_order + 1 ) * beta2 + alpha0 );
          }
          ++parent_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_ms2m_operation( mesh::general_spacetime_cluster *
  /*current_cluster*/ ) const {
  std::cout << "General Ms2M operation not implemented " << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply_ms2m_operation< 0 >( mesh::
    general_spacetime_cluster * current_cluster ) const {
  apply_ms2m_operation_p0_time( current_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::apply_ms2m_operation< 0 >( mesh::
    general_spacetime_cluster * current_cluster ) const {
  apply_ms2m_operation_p0_time( current_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply_ms2m_operation< 0 >( mesh::
    general_spacetime_cluster * current_cluster ) const {
  apply_ms2m_operation_p0_time( current_cluster );
}

// TODO: add specialization for hypersingular operator

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_ms2m_operation_p0_time( mesh::
    general_spacetime_cluster * current_cluster ) const {
  // get current moment and spatial moment and assemble matrix of integrated
  // Lagrange polynomials in time.
  sc * st_moment = current_cluster->get_pointer_to_moment( );
  const sc * s_moment = current_cluster->get_pointer_to_spatial_moments( );
  full_matrix L;
  compute_lagrange_quadrature( L, current_cluster );
  lo n_time_elems = current_cluster->get_n_time_elements( );
  // interpret the spatial moment s_moment as a matrix with
  // _spat_contribution_size rows and n_time_elems columns (in column major
  // order), and compute the space-time moment by a matrix-matrix
  // multiplication using blas.
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_rows_s_moments = _spat_contribution_size;
  lo n_columns_s_moments = n_time_elems;
  lo lda = n_rows_lagrange;
  lo ldb = _spat_contribution_size;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans, n_rows_lagrange,
    n_rows_s_moments, n_columns_s_moments, alpha, L.data( ), lda, s_moment, ldb,
    beta, st_moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_ms2l_operation( const mesh::general_spacetime_cluster *
  /*src_cluster*/,
  mesh::general_spacetime_cluster * /*tar_cluster*/ ) const {
  std::cout << "General Ms2L operation not implemented " << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_ms2l_operation< 0 >(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_ms2l_operation_p0_time( src_cluster, tar_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_ms2l_operation< 0 >(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_ms2l_operation_p0_time( src_cluster, tar_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_ms2l_operation< 0 >(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  apply_ms2l_operation_p0_time( src_cluster, tar_cluster );
}

// TODO: add specializations for hypersingular operator

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_ms2l_operation_p0_time(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const {
  lo n_quad_points_time = _temp_order + 1;
  // ATTENTION: When changing n_quad_points_time one has to adapt the size of
  // the auxiliary buffers!
  // allocate some buffers to store intermediate results
  // buffer to store intermediate results in computation of m2l coefficients
  vector_type buffer_for_gaussians( ( _spat_order + 1 ) * ( _spat_order + 1 )
    * ( _temp_order + 1 ) * n_quad_points_time );
  // buffer to store m2l coefficients.
  vector_type buffer_for_coeffs( ( _spat_order + 1 ) * ( _spat_order + 1 )
    * ( _temp_order + 1 ) * n_quad_points_time );

  // get geometrical data of the clusters
  sc tar_half_size_time;
  vector_type half_size_space( 3, false );
  tar_cluster->get_half_size( half_size_space, tar_half_size_time );
  sc src_center_time, tar_center_time;
  vector_type src_center_space( 3, false );
  vector_type tar_center_space( 3, false );
  src_cluster->get_center( src_center_space, src_center_time );
  tar_cluster->get_center( tar_center_space, tar_center_time );

  // initialize the temporal interpolation nodes in the target cluster
  vector_type tar_time_nodes( _temp_order + 1, false );
  const vector_type & time_nodes = _lagrange.get_nodes( );
  for ( lo i = 0; i <= _temp_order; ++i ) {
    tar_time_nodes[ i ]
      = tar_center_time + tar_half_size_time * time_nodes[ i ];
  }

  // initialize the quadrature points for quadrature in target time intervals
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_t
    = bem::quadrature::line_x( n_quad_points_time );
  const std::vector< sc, besthea::allocator_type< sc > > & quad_time_w
    = bem::quadrature::line_w( n_quad_points_time );
  // update n_quad_points_time (necessary, if quadrature rule is not
  // available)
  n_quad_points_time = quad_time_w.size( );

  vector_type quad_points_t_src_elem( n_quad_points_time );

  vector quad_time_w_unrld( n_quad_points_time * ( _temp_order + 1 ) );

  // get spatial properties ( difference of cluster, half length )
  vector_type center_diff_space( tar_center_space );
  for ( lo i = 0; i < 3; ++i ) {
    center_diff_space[ i ] -= src_center_space[ i ];
  }

  sc padding_space = _distributed_spacetime_tree
                       ->get_spatial_paddings( )[ tar_cluster->get_level( ) ];
  for ( lo i = 0; i < 3; ++i ) {
    half_size_space[ i ] += padding_space;
  }

  // get information about the time elements in the source
  // cluster and its mesh
  lo n_src_time_elems = src_cluster->get_n_time_elements( );
  lo n_src_space_elems = src_cluster->get_n_space_elements( );
  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = src_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * src_mesh;
  lo src_mesh_start_idx;
  if ( src_cluster->get_elements_are_local( ) ) {
    src_mesh = distributed_mesh.get_local_mesh( );
    src_mesh_start_idx = distributed_mesh.get_local_start_idx( );
  } else {
    src_mesh = distributed_mesh.get_nearfield_mesh( );
    src_mesh_start_idx = distributed_mesh.get_nearfield_start_idx( );
  }

  // ms2l operations have to be executed for each time element in the source
  // cluster individually.
  const sc * all_spatial_moments
    = src_cluster->get_pointer_to_spatial_moments( );
  sc * tar_local = tar_cluster->get_pointer_to_local_contribution( );
  // get the number of the current thread to access the correct buffers in the
  // ms2l operation
  lo thread_num = omp_get_thread_num( );
  for ( lo i_src_time = 0; i_src_time < n_src_time_elems; ++i_src_time ) {
    // reset the buffers
    _aux_buffer_0[ thread_num ].fill( 0.0 );
    _aux_buffer_1[ thread_num ].fill( 0.0 );
    // get the address of the spatial moments of the current time-step.
    const sc * current_spatial_moments
      = &( all_spatial_moments[ i_src_time * _spat_contribution_size ] );

    // get the endpoints of the current target time element
    sc src_t_elem_start, src_t_elem_end;
    lo local_src_elem_idx = distributed_mesh.global_2_local( src_mesh_start_idx,
      src_cluster->get_element( i_src_time * n_src_space_elems ) );
    lo local_src_elem_idx_time
      = src_mesh->get_time_element( local_src_elem_idx );
    src_mesh->get_temporal_nodes(
      local_src_elem_idx_time, &src_t_elem_start, &src_t_elem_end );
    sc src_t_elem_size = src_t_elem_end - src_t_elem_start;
    // compute the temporal quadrature points in the target time element
    for ( slou i = 0; i < n_quad_points_time; ++i ) {
      quad_points_t_src_elem[ i ]
        = src_t_elem_start + src_t_elem_size * quad_time_t[ i ];
    }
    // update the unrolled quadrature weights
    for ( lo j = 0; j < n_quad_points_time; ++j ) {
      sc quad_weight = quad_time_w[ j ] * src_t_elem_size;
      for ( lo i = 0; i <= _temp_order; ++i ) {
        quad_time_w_unrld[ i * n_quad_points_time + j ] = quad_weight;
      }
    }

    const sc * quad_time_w_unrl_data = quad_time_w_unrld.data( );

    lo hlp_acs_alpha
      = ( _spat_order + 1 ) * ( _temp_order + 1 ) * n_quad_points_time;
    lo hlp_acs_beta = ( _temp_order + 1 ) * n_quad_points_time;
    lo hlp_acs_a = n_quad_points_time;

    sc * aux_buffer_0_data = _aux_buffer_0[ thread_num ].data( );
    sc * buffer_for_coeffs_data = buffer_for_coeffs.data( );

    // Execute the M2Ls operation for the current target time interval
    // only the last one-dimensional transform is different from a standard
    // M2L
    compute_m2l_coupling_coeffs( quad_points_t_src_elem, tar_time_nodes,
      half_size_space[ 2 ], center_diff_space[ 2 ], buffer_for_gaussians,
      buffer_for_coeffs );

    // compute first intermediate product and store it in aux_buffer_0
    lo buffer_0_index = 0;
    for ( lo alpha2 = 0; alpha2 <= _spat_order; ++alpha2 ) {
      lo moment_index = 0;
      for ( lo beta0 = 0; beta0 <= _spat_order - alpha2; ++beta0 ) {
        for ( lo beta1 = 0; beta1 <= _spat_order - alpha2 - beta0; ++beta1 ) {
          for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
#pragma omp simd aligned(                                          \
  aux_buffer_0_data, buffer_for_coeffs_data, quad_time_w_unrl_data \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
            for ( lo ab = 0; ab < n_quad_points_time * ( _temp_order + 1 );
                  ++ab ) {
              aux_buffer_0_data[ buffer_0_index * hlp_acs_beta + ab ]
                += buffer_for_coeffs_data[ alpha2 * hlp_acs_alpha
                     + beta2 * hlp_acs_beta + ab ]
                * quad_time_w_unrl_data[ ab ]
                * current_spatial_moments[ moment_index ];
            }
            ++moment_index;
          }
          ++buffer_0_index;
        }
        // correction for moment index; this is necessary since beta1 does not
        // run until _spat_order - beta0 as it does in src_moment;
        moment_index += ( ( alpha2 + 1 ) * alpha2 ) / 2;
      }
    }

    // update coefficients and compute 2nd intermediate product in
    // aux_buffer_1
    compute_m2l_coupling_coeffs( quad_points_t_src_elem, tar_time_nodes,
      half_size_space[ 1 ], center_diff_space[ 1 ], buffer_for_gaussians,
      buffer_for_coeffs );

    sc * aux_buffer_1_data = _aux_buffer_1[ thread_num ].data( );

    lo buffer_1_index = 0;
    for ( lo alpha1 = 0; alpha1 <= _spat_order; ++alpha1 ) {
      buffer_0_index = 0;
      for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha1; ++alpha2 ) {
        for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2; ++beta0 ) {
          for ( lo beta1 = 0; beta1 <= _spat_order - beta0 - alpha2; ++beta1 ) {
            for ( lo a = 0; a <= _temp_order; ++a ) {
#pragma omp simd aligned(                                      \
  aux_buffer_1_data, buffer_for_coeffs_data, aux_buffer_0_data \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
              for ( lo b = 0; b < n_quad_points_time; ++b ) {
                aux_buffer_1_data[ buffer_1_index * hlp_acs_beta + hlp_acs_a * a
                  + b ]
                  += buffer_for_coeffs_data[ alpha1 * hlp_acs_alpha
                       + beta1 * hlp_acs_beta + a * hlp_acs_a + b ]
                  * aux_buffer_0_data[ buffer_0_index * hlp_acs_beta
                    + hlp_acs_a * a + b ];
              }
            }
            ++buffer_0_index;
          }
          ++buffer_1_index;
        }
        // correction for buffer_0 index; this is necessary since beta0 does
        // not run until _spat_order - alpha2 as it does in aux_buffer_0;
        buffer_0_index += ( ( alpha1 + 1 ) * alpha1 ) / 2;
      }
    }

    // update coefficients and target's local contribution with ms2l result
    compute_m2l_coupling_coeffs( quad_points_t_src_elem, tar_time_nodes,
      half_size_space[ 0 ], center_diff_space[ 0 ], buffer_for_gaussians,
      buffer_for_coeffs );
    int local_index = 0;
    for ( lo alpha0 = 0; alpha0 <= _spat_order; ++alpha0 ) {
      buffer_1_index = 0;
      for ( lo alpha1 = 0; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
        for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha0 - alpha1;
              ++alpha2 ) {
          for ( lo beta0 = 0; beta0 <= _spat_order - alpha1 - alpha2;
                ++beta0 ) {
            for ( lo a = 0; a <= _temp_order; ++a ) {
              sc val = 0.0;
#pragma omp simd aligned( buffer_for_coeffs_data, aux_buffer_1_data : \
                  DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH) reduction( + : val)
              for ( lo b = 0; b < n_quad_points_time; ++b ) {
                val += buffer_for_coeffs_data[ alpha0 * hlp_acs_alpha
                         + beta0 * hlp_acs_beta + a * hlp_acs_a + b ]
                  * aux_buffer_1_data[ buffer_1_index * hlp_acs_beta
                    + hlp_acs_a * a + b ];
              }
              tar_local[ a + local_index * ( _temp_order + 1 ) ] += val;
            }
            ++buffer_1_index;
          }
          ++local_index;
        }
        // correction for buffer_1 index; this is necessary since alpha0 does
        // not run until _spat_order - alpha1 as it does in aux_buffer_1;
        buffer_1_index += ( ( alpha0 + 1 ) * alpha0 ) / 2;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_l2t_operations( mesh::scheduling_time_cluster *
                                         t_cluster,
  distributed_block_vector & output_vector, bool verbose,
  const std::string & verbose_file ) const {
  // execute only for associated spacetime leaves
  if ( t_cluster->get_n_associated_leaves( ) > 0 ) {
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = t_cluster->get_associated_spacetime_clusters( );
    lo max_relative_space_level_ls2t
      = t_cluster->get_max_relative_space_level_ls2t( );
    // if max_relative_space_level_ls2t is negative (i.e. it was not
    // initialized explicitly) there are no spatially refined clusters for
    // which spatial local contributions have to be evaluated, so we can
    // evaluate local contributions of original leaf clusters directly.
    if ( max_relative_space_level_ls2t < 0 ) {
      if ( verbose ) {
#pragma omp critical( verbose )
        {
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << "call direct L2T for cluster "
                    << t_cluster->get_global_index( ) << " at level "
                    << t_cluster->get_level( ) << std::endl;
            outfile.close( );
          }
        }
      }
      // there is an implicit taskgroup after this taskloop
#pragma omp taskloop shared( output_vector, associated_spacetime_clusters )
      for ( lou i = 0; i < t_cluster->get_n_associated_leaves( ); ++i ) {
        if ( _measure_tasks ) {
          _l_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        apply_l2t_operation< run_count >(
          ( *associated_spacetime_clusters )[ i ], output_vector );
        if ( _measure_tasks ) {
          _l_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
      }
    } else {
      if ( verbose ) {
#pragma omp critical( verbose )
        {
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << "call L2T for cluster " << t_cluster->get_global_index( )
                    << " at level " << t_cluster->get_level( )
                    << " via spatially refined clusters" << std::endl;
            outfile.close( );
          }
        }
      }
      // first, transform the space-time local contributions into spatial ones
      const std::vector< lo > * n_leaves_and_aux_cluster_per_level
        = t_cluster->get_n_associated_leaves_and_aux_clusters_per_level( );
#pragma omp taskloop shared( n_leaves_and_aux_cluster_per_level )
      for ( lo i = 0; i < ( *n_leaves_and_aux_cluster_per_level )[ 0 ]; ++i ) {
        if ( _measure_tasks ) {
          _l_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        general_spacetime_cluster * current_cluster
          = ( *associated_spacetime_clusters )[ i ];
        apply_l2ls_operation< run_count >( current_cluster );
        if ( _measure_tasks ) {
          _l_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
      }

      // pass spatial local contributions downwards by ls2ls operations
      // (spatial local contribution to spatial local contribution)
      lo offset = 0;
      for ( lo rel_space_level = 0;
            rel_space_level < max_relative_space_level_ls2t;
            ++rel_space_level ) {
#pragma omp taskloop shared( n_leaves_and_aux_cluster_per_level, output_vector )
        for ( lo i = 0;
              i < ( *n_leaves_and_aux_cluster_per_level )[ rel_space_level ];
              ++i ) {
          if ( _measure_tasks ) {
            _l_subtask_times.at( omp_get_thread_num( ) )
              .push_back( _global_timer.get_time_from_start< time_type >( ) );
          }
          general_spacetime_cluster * current_cluster
            = ( *associated_spacetime_clusters )[ offset + i ];
          // if current_cluster is not a leaf, apply an ls2ls operation for
          // each of its children; otherwise ls2t operations.
          if ( current_cluster->get_n_children( ) > 0 ) {
            for ( auto child : *current_cluster->get_children( ) ) {
              apply_ls2ls_operation( current_cluster, child );
            }
          } else {
            apply_ls2t_operation< run_count >( output_vector, current_cluster );
          }
          if ( _measure_tasks ) {
            _l_subtask_times.at( omp_get_thread_num( ) )
              .push_back( _global_timer.get_time_from_start< time_type >( ) );
          }
        }
        // update offset to access clusters at next spatial level
        offset += ( *n_leaves_and_aux_cluster_per_level )[ rel_space_level ];
      }

      // finally, apply ls2t operations for all cluster with the maximal
      // relative space level or for their leaf descendants
      // determine first the leaves at level max_relative_space_level_ls2t
      std::vector< lo > leaf_indices_at_max_rel_space_level_ls2t;
      for ( lo i = 0; i < ( *n_leaves_and_aux_cluster_per_level )
                        [ max_relative_space_level_ls2t ];
            ++i ) {
        general_spacetime_cluster * current_cluster
          = ( *associated_spacetime_clusters )[ offset + i ];
        if ( current_cluster->get_n_children( ) == 0 ) {
          leaf_indices_at_max_rel_space_level_ls2t.push_back( offset + i );
        }
      }
      // now, execute all direct Ls2T operations in parallel
#pragma omp taskloop shared( \
  output_vector, leaf_indices_at_max_rel_space_level_ls2t )
      for ( lou i = 0; i < leaf_indices_at_max_rel_space_level_ls2t.size( );
            ++i ) {
        if ( _measure_tasks ) {
          _l_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        general_spacetime_cluster * current_cluster
          = ( *associated_spacetime_clusters )
            [ leaf_indices_at_max_rel_space_level_ls2t[ i ] ];
        apply_ls2t_operation< run_count >( output_vector, current_cluster );
        if ( _measure_tasks ) {
          _l_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
      }
      // if there are auxiliary Ls2T operations to execute, do this at last.
      // (i.e. Ls2T operations for large clusters via their leaf descendants)
      const std::vector< std::pair< lo, general_spacetime_cluster * > > *
        assoc_aux_ls2t_cluster_pairs
        = t_cluster->get_assoc_aux_ls2t_cluster_pairs( );
      if ( assoc_aux_ls2t_cluster_pairs != nullptr ) {
#pragma omp taskloop shared( assoc_aux_ls2t_cluster_pairs, output_vector )
        for ( lou i = 0; i < assoc_aux_ls2t_cluster_pairs->size( ); ++i ) {
          if ( _measure_tasks ) {
            _l_subtask_times.at( omp_get_thread_num( ) )
              .push_back( _global_timer.get_time_from_start< time_type >( ) );
          }

          lo current_target_index
            = ( *assoc_aux_ls2t_cluster_pairs )[ i ].first;
          general_spacetime_cluster * current_coarse_target_cluster
            = ( *assoc_aux_ls2t_cluster_pairs )[ i ].second;
          general_spacetime_cluster * current_target_cluster
            = ( *associated_spacetime_clusters )[ current_target_index ];
          apply_ls2t_operation< run_count >( output_vector,
            current_coarse_target_cluster, current_target_cluster );

          if ( _measure_tasks ) {
            _l_subtask_times.at( omp_get_thread_num( ) )
              .push_back( _global_timer.get_time_from_start< time_type >( ) );
          }
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::check_for_received_data( std::vector< MPI_Request > &
                                             array_of_requests,
  std::vector< int > & array_of_indices, int & outcount ) const {
  MPI_Testsome( _receive_data_information.size( ), array_of_requests.data( ),
    &outcount, array_of_indices.data( ), MPI_STATUSES_IGNORE );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  find_cluster_in_l_list( std::list< scheduling_time_cluster * > & l_list,
    std::list< scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const {
  it_next_cluster = l_list.begin( );
  // a cluster is ready if its parents downward path status equals 2
  while ( status != 2 && it_next_cluster != l_list.end( ) ) {
    if ( ( *it_next_cluster )->get_parent( )->get_downward_path_status( ) == 2 )
      status = 2;
    else
      ++it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  find_cluster_in_m_list( std::list< scheduling_time_cluster * > & m_list,
    std::list< scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const {
  it_next_cluster = m_list.begin( );
  // a cluster is ready if its upward path counter equals 0
  while ( status != 1 && it_next_cluster != m_list.end( ) ) {
    if ( ( *it_next_cluster )->get_upward_path_counter( ) == 0 )
      status = 1;
    else
      ++it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  find_cluster_in_m2l_list(
    std::list< mesh::scheduling_time_cluster * > & m2l_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const {
  // go through the m2l list and search for a ready cluster, i.e. a
  // cluster whose number of ready m2l source equals the size of its
  // interaction list.
  it_next_cluster = m2l_list.begin( );
  while ( status != 4 && it_next_cluster != m2l_list.end( ) ) {
    if ( ( *it_next_cluster )->get_n_ready_m2l_sources( )
      == ( *it_next_cluster )->get_interaction_list( )->size( ) )
      status = 4;
    else
      ++it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  find_cluster_in_s2l_list(
    std::list< mesh::scheduling_time_cluster * > & s2l_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const {
  // go through the s2l list and search for a ready cluster, i.e. a
  // cluster whose s2l execution status = 1.
  it_next_cluster = s2l_list.begin( );
  while ( status != 3 && it_next_cluster != s2l_list.end( ) ) {
    if ( ( *it_next_cluster )->get_s2l_execution_status( ) == 1 )
      status = 3;
    else
      ++it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  find_cluster_in_m2t_list(
    std::list< mesh::scheduling_time_cluster * > & m2t_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const {
  it_next_cluster = m2t_list.begin( );
  // a cluster is ready if number of ready m2t sources is equal to the
  // size of the m2t_list of the cluster (NOTE: the cluster's m2t-list is
  // not directly related to the input variable m2t_list)
  while ( status != 5 && it_next_cluster != m2t_list.end( ) ) {
    if ( ( *it_next_cluster )->get_n_ready_m2t_sources( )
      == ( *it_next_cluster )->get_m2t_list( )->size( ) )
      status = 5;
    else
      ++it_next_cluster;
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::provide_moments_for_m2l_or_m2t( scheduling_time_cluster *
                                                    src_cluster,
  bool verbose, const std::string & verbose_file ) const {
  std::vector< scheduling_time_cluster * > * send_list
    = src_cluster->get_send_list( );
  std::vector< scheduling_time_cluster * > * m2t_send_list
    = src_cluster->get_m2t_send_list( );
  // go through the send and diagonal send list and see whether there are
  // target clusters handled by other processes.
  std::set< lo > process_send_list;
  std::vector< scheduling_time_cluster * >::iterator list_iterator;
  std::vector< scheduling_time_cluster * >::iterator list_end;
  bool in_m2t_send_list;
  if ( send_list != nullptr ) {
    list_iterator = send_list->begin( );
    in_m2t_send_list = false;
    if ( m2t_send_list == nullptr ) {
      list_end = send_list->end( );
    } else {
      list_end = m2t_send_list->end( );
    }
  } else if ( m2t_send_list != nullptr ) {
    list_iterator = m2t_send_list->begin( );
    in_m2t_send_list = true;
    list_end = m2t_send_list->end( );
  }
  if ( send_list != nullptr || m2t_send_list != nullptr ) {
    while ( list_iterator != list_end ) {
      lo tar_process_id = ( *list_iterator )->get_process_id( );
      if ( tar_process_id == _my_rank ) {
        if ( in_m2t_send_list ) {
          ( *list_iterator )->update_n_ready_m2t_sources( );
        } else {
          ( *list_iterator )->update_n_ready_m2l_sources( );
        }
      } else if ( process_send_list.count( tar_process_id ) == 0 ) {
        if ( verbose ) {
#pragma omp critical( verbose )
          {
            std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
            if ( outfile.is_open( ) ) {
              outfile << "send for m2t or m2l: data from source ";
              outfile << src_cluster->get_global_index( ) << " at level "
                      << src_cluster->get_level( ) << " to process "
                      << tar_process_id << std::endl;
              outfile.close( );
            }
          }
        }
        lo tag = 2 * src_cluster->get_global_index( );
        sc * moment_buffer = src_cluster->get_associated_moments( );
        // determine the number of associated clusters with moments
        lo n_clusters_with_moments
          = src_cluster->get_n_st_clusters_w_moments( );
        int buffer_size = n_clusters_with_moments * _contribution_size;

        if ( _measure_tasks ) {
          _mpi_send_m2l_m2t_or_s2l.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        MPI_Request req;
        MPI_Isend( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ),
          tar_process_id, tag, *_comm, &req );
        MPI_Request_free( &req );
        process_send_list.insert( tar_process_id );
      }
      ++list_iterator;
      if ( !in_m2t_send_list && m2t_send_list != nullptr
        && list_iterator == send_list->end( ) ) {
        list_iterator = m2t_send_list->begin( );
        in_m2t_send_list = true;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  provide_spatial_moments_for_hybrid_s2l(
    mesh::scheduling_time_cluster * src_cluster, bool verbose,
    const std::string & verbose_file ) const {
  std::vector< mesh::scheduling_time_cluster * > * s2l_send_list
    = src_cluster->get_s2l_send_list( );
  std::set< lo > process_send_list;
  // go through the s2l send list and send spatial moments to those processes
  // handling the target clusters.
  if ( s2l_send_list != nullptr ) {
    for ( auto s2l_tar_cluster : *s2l_send_list ) {
      lo tar_process_id = s2l_tar_cluster->get_process_id( );
      if ( tar_process_id == _my_rank ) {
        s2l_tar_cluster->set_s2l_execution_status( 1 );
      } else if ( process_send_list.count( tar_process_id ) == 0 ) {
        if ( verbose ) {
#pragma omp critical( verbose )
          {
            std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
            if ( outfile.is_open( ) ) {
              outfile << "send spatial moments for s2l: data from source ";
              outfile << src_cluster->get_global_index( ) << " at level "
                      << src_cluster->get_level( ) << " to process "
                      << tar_process_id << std::endl;
              outfile.close( );
            }
          }
        }
        // as unique tag we use 2 * the global index the left child of the
        // sending cluster would have. (note: sending cluster is always a
        // leave!)
        lo tag = 2 * ( 2 * src_cluster->get_global_index( ) + 1 );
        sc * spatial_moment_buffer
          = src_cluster->get_associated_spatial_moments( );
        int buffer_size = src_cluster->get_n_st_clusters_w_spatial_moments( )
          * _spat_contribution_size;
        if ( _measure_tasks ) {
          _mpi_send_m2l_m2t_or_s2l.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        MPI_Request req;
        MPI_Isend( spatial_moment_buffer, buffer_size,
          get_scalar_type< sc >::MPI_SC( ), tar_process_id, tag, *_comm, &req );
        MPI_Request_free( &req );
        process_send_list.insert( tar_process_id );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::provide_moments_to_parents( scheduling_time_cluster *
                                                child_cluster,
  bool verbose, const std::string & verbose_file ) const {
  scheduling_time_cluster * parent_cluster = child_cluster->get_parent( );
  // moments have to be provided only if the parent is active in the
  // upward path
  if ( parent_cluster->is_active_in_upward_path( ) ) {
    lo parent_process_id = parent_cluster->get_process_id( );
    if ( parent_process_id == _my_rank ) {
      // update local dependency counter
      parent_cluster->reduce_upward_path_counter( );
    } else if ( parent_process_id != -1 ) {
      if ( verbose ) {
#pragma omp critical( verbose )
        {
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << "send upward: from source "
                    << child_cluster->get_global_index( ) << " at level "
                    << child_cluster->get_level( ) << " to process "
                    << parent_process_id << std::endl;
            outfile.close( );
          }
        }
      }
      lo tag = 2 * parent_cluster->get_global_index( );
      sc * moment_buffer = parent_cluster->get_associated_moments( );
      int buffer_size
        = parent_cluster->get_n_st_clusters_w_moments( ) * _contribution_size;

      if ( _measure_tasks ) {
        _mpi_send_m_parent.at( omp_get_thread_num( ) )
          .push_back( _global_timer.get_time_from_start< time_type >( ) );
      }

      MPI_Request req;
      MPI_Isend( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ),
        parent_process_id, tag, *_comm, &req );
      MPI_Request_free( &req );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  provide_local_contributions_to_children(
    scheduling_time_cluster * parent_cluster, bool verbose,
    const std::string & verbose_file ) const {
  std::vector< scheduling_time_cluster * > * children
    = parent_cluster->get_children( );
  if ( children != nullptr ) {
    for ( auto child : *children ) {
      lo child_process_id = child->get_process_id( );
      if ( child_process_id != _my_rank ) {
        if ( verbose ) {
#pragma omp critical( verbose )
          {
            std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
            if ( outfile.is_open( ) ) {
              outfile << "send downward: from source "
                      << parent_cluster->get_global_index( ) << " at level "
                      << parent_cluster->get_level( ) << " to process "
                      << child_process_id << std::endl;
              outfile.close( );
            }
          }
        }
        lo tag = 2 * parent_cluster->get_global_index( ) + 1;
        sc * local_contribution_buffer
          = parent_cluster->get_associated_local_contributions( );
        int buffer_size
          = parent_cluster->get_n_st_clusters_w_local_contributions( )
          * _contribution_size;

        if ( _measure_tasks ) {
          _mpi_send_l_children.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }

        MPI_Request req;
        MPI_Isend( local_contribution_buffer, buffer_size,
          get_scalar_type< sc >::MPI_SC( ), child_process_id, tag, *_comm,
          &req );
        MPI_Request_free( &req );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::start_receive_operations( std::vector< MPI_Request > &
    array_of_requests ) const {
  // start the receive operations for the moments in the upward path
  for ( lou i = 0; i < _n_moments_to_receive_upward; ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    scheduling_time_cluster * receive_cluster
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( );
    sc * moment_buffer
      = receive_cluster->get_extraneous_moment_pointer( source_id );
    // note: by construction there cannot be any auxiliary clusters associated
    // with the parent cluster, so the buffer size is correct.
    int buffer_size
      = receive_cluster->get_associated_spacetime_clusters( )->size( )
      * _contribution_size;
    MPI_Irecv( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ),
      source_id, tag, *_comm, &array_of_requests[ i ] );
  }

  // start the receive operations for the moments needed for m2l
  for ( lou i = _n_moments_to_receive_upward;
        i < _n_moments_to_receive_upward + _n_moments_to_receive_m2l_or_m2t;
        ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    scheduling_time_cluster * receive_cluster
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( );
    sc * moment_buffer = receive_cluster->get_associated_moments( );
    lo n_clusters_with_moments
      = receive_cluster->get_n_st_clusters_w_moments( );
    int buffer_size = n_clusters_with_moments * _contribution_size;
    MPI_Irecv( moment_buffer, buffer_size, get_scalar_type< sc >::MPI_SC( ),
      source_id, tag, *_comm, &array_of_requests[ i ] );
  }

  // start the receive operations for the spatial moments
  for ( lou i = _n_moments_to_receive_upward + _n_moments_to_receive_m2l_or_m2t;
        i < _n_moments_to_receive_upward + _n_moments_to_receive_m2l_or_m2t
          + _n_spatial_moments_to_receive;
        ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    mesh::scheduling_time_cluster * receive_cluster
      = _receive_data_information[ i ].first;
    // as unique tag we use 2 * the global index the left child of the receive
    // cluster would have. (note: receive cluster is always a leave!)
    lo tag = 2 * ( 2 * receive_cluster->get_global_index( ) + 1 );
    sc * spatial_moment_buffer
      = receive_cluster->get_associated_spatial_moments( );
    int buffer_size = receive_cluster->get_n_st_clusters_w_spatial_moments( )
      * _spat_contribution_size;
    MPI_Irecv( spatial_moment_buffer, buffer_size,
      get_scalar_type< sc >::MPI_SC( ), source_id, tag, *_comm,
      &array_of_requests[ i ] );
  }

  // start the receive operations for the local contributions
  for ( lou i = _n_moments_to_receive_upward + _n_moments_to_receive_m2l_or_m2t
          + _n_spatial_moments_to_receive;
        i < _receive_data_information.size( ); ++i ) {
    lo source_id = _receive_data_information[ i ].second;
    scheduling_time_cluster * receive_cluster
      = _receive_data_information[ i ].first;
    lo tag = 2 * receive_cluster->get_global_index( ) + 1;
    sc * local_contribution_buffer
      = receive_cluster->get_associated_local_contributions( );
    int buffer_size
      = receive_cluster->get_n_st_clusters_w_local_contributions( )
      * _contribution_size;
    MPI_Irecv( local_contribution_buffer, buffer_size,
      get_scalar_type< sc >::MPI_SC( ), source_id, tag, *_comm,
      &array_of_requests[ i ] );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_chebyshev_quadrature_p0( full_matrix &
                                                                   T,
  const general_spacetime_cluster * source_elem_cluster,
  const general_spacetime_cluster * source_geom_cluster ) const {
  lo n_space_elems = source_elem_cluster->get_n_space_elements( );
  T.resize( n_space_elems, _spat_contribution_size );
  // get some info on the source geometry cluster
  const general_spacetime_cluster * actual_geom_cluster = source_geom_cluster;
  if ( actual_geom_cluster == nullptr ) {
    actual_geom_cluster = source_elem_cluster;
  }

  vector_type geom_cluster_s_center( 3 );
  vector_type geom_cluster_s_half( 3 );
  sc dummy;
  actual_geom_cluster->get_center( geom_cluster_s_center, dummy );
  actual_geom_cluster->get_half_size( geom_cluster_s_half, dummy );
  sc padding = _distributed_spacetime_tree
                 ->get_spatial_paddings( )[ actual_geom_cluster->get_level( ) ];
  sc start_0 = geom_cluster_s_center[ 0 ] - geom_cluster_s_half[ 0 ] - padding;
  sc end_0 = geom_cluster_s_center[ 0 ] + geom_cluster_s_half[ 0 ] + padding;
  sc start_1 = geom_cluster_s_center[ 1 ] - geom_cluster_s_half[ 1 ] - padding;
  sc end_1 = geom_cluster_s_center[ 1 ] + geom_cluster_s_half[ 1 ] + padding;
  sc start_2 = geom_cluster_s_center[ 2 ] - geom_cluster_s_half[ 2 ] - padding;
  sc end_2 = geom_cluster_s_center[ 2 ] + geom_cluster_s_half[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature, _order_regular );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature
  // points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );

  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = source_elem_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  for ( lo i = 0; i < n_space_elems; ++i ) {
    // we use that the elements in the cluster are tensor products of
    // spatial elements and timesteps, and are sorted w.r.t. the
    // timesteps. In particular we get all spatial elements in the cluster
    // by considering the first n_space_elems spacetime elements.
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, source_elem_cluster->get_element( i ) );
    lo local_elem_idx_space
      = local_mesh->get_space_element_index( local_elem_idx );

    local_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc elem_area = local_mesh->get_spatial_area_using_spatial_index(
      local_elem_idx_space );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          sc quad = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            quad += cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ] * wy[ j ];
          }
          quad *= elem_area;
          T.set( i, current_index, quad );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::compute_normal_drv_chebyshev_quadrature_p1( full_matrix &
                                                                T_drv,
  const general_spacetime_cluster * source_elem_cluster,
  const mesh::general_spacetime_cluster * source_geom_cluster ) const {
  lo n_space_elems = source_elem_cluster->get_n_space_elements( );
  lo n_space_nodes = source_elem_cluster->get_n_space_nodes( );
  T_drv.resize( n_space_nodes, _spat_contribution_size );
  T_drv.fill( 0.0 );
  // get some info on the source geometry cluster
  const general_spacetime_cluster * actual_geom_cluster = source_geom_cluster;
  if ( actual_geom_cluster == nullptr ) {
    actual_geom_cluster = source_elem_cluster;
  }

  vector_type geom_cluster_s_center( 3 );
  vector_type geom_cluster_s_half( 3 );
  sc dummy;
  actual_geom_cluster->get_center( geom_cluster_s_center, dummy );
  actual_geom_cluster->get_half_size( geom_cluster_s_half, dummy );
  sc padding = _distributed_spacetime_tree
                 ->get_spatial_paddings( )[ actual_geom_cluster->get_level( ) ];
  sc start_0 = geom_cluster_s_center[ 0 ] - geom_cluster_s_half[ 0 ] - padding;
  sc end_0 = geom_cluster_s_center[ 0 ] + geom_cluster_s_half[ 0 ] + padding;
  sc start_1 = geom_cluster_s_center[ 1 ] - geom_cluster_s_half[ 1 ] - padding;
  sc end_1 = geom_cluster_s_center[ 1 ] + geom_cluster_s_half[ 1 ] + padding;
  sc start_2 = geom_cluster_s_center[ 2 ] - geom_cluster_s_half[ 2 ] - padding;
  sc end_2 = geom_cluster_s_center[ 2 ] + geom_cluster_s_half[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature, _order_regular );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature
  // points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );
  // same for evaluations of scaled derivatives of Chebyshev polynomials
  vector_type cheb_drv_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_drv_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_drv_dim_2( ( _spat_order + 1 ) * size_quad );

  sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  sc * y2_ref = my_quadrature._y2_ref_cheb.data( );

  linear_algebra::coordinates< 3 > grad;
  const std::vector< lo > & elems_2_local_nodes
    = source_elem_cluster->get_elems_2_local_nodes( );
  linear_algebra::coordinates< 3 > normal;

  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = source_elem_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  for ( lo i = 0; i < n_space_elems; ++i ) {
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, source_elem_cluster->get_element( i ) );
    lo local_elem_idx_space
      = local_mesh->get_space_element_index( local_elem_idx );
    local_mesh->get_spatial_normal_using_spatial_element_index(
      local_elem_idx_space, normal );
    local_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc elem_area = local_mesh->get_spatial_area_using_spatial_index(
      local_elem_idx_space );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y1_polynomial, cheb_drv_dim_0 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y2_polynomial, cheb_drv_dim_1 );
    _chebyshev.evaluate_derivative(
      my_quadrature._y3_polynomial, cheb_drv_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          sc value1 = 0.0;
          sc value2 = 0.0;
          sc value3 = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            // compute the gradient of the tensor-product chebyshev
            // polynomials using the chain rule.
            grad[ 0 ] = cheb_drv_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ]
              / ( geom_cluster_s_half[ 0 ] + padding );
            grad[ 1 ] = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_drv_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ]
              / ( geom_cluster_s_half[ 1 ] + padding );
            grad[ 2 ] = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_drv_dim_2[ beta2 * size_quad + j ]
              / ( geom_cluster_s_half[ 2 ] + padding );
            sc weighted_normal_derivative
              = wy[ j ] * elem_area * normal.dot( grad );
            value1 += weighted_normal_derivative
              * ( (sc) 1.0 - y1_ref[ j ] - y2_ref[ j ] );
            value2 += weighted_normal_derivative * y1_ref[ j ];
            value3 += weighted_normal_derivative * y2_ref[ j ];
          }

          T_drv.add( source_elem_cluster
                       ->local_spacetime_node_idx_2_local_space_node_idx(
                         elems_2_local_nodes[ 6 * i ] ),
            current_index, _alpha * value1 );
          T_drv.add( source_elem_cluster
                       ->local_spacetime_node_idx_2_local_space_node_idx(
                         elems_2_local_nodes[ 6 * i + 1 ] ),
            current_index, _alpha * value2 );
          T_drv.add( source_elem_cluster
                       ->local_spacetime_node_idx_2_local_space_node_idx(
                         elems_2_local_nodes[ 6 * i + 2 ] ),
            current_index, _alpha * value3 );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_chebyshev_times_normal_quadrature_p1_along_dimension(
    full_matrix & T_normal_along_dim, const slou dim,
    const mesh::general_spacetime_cluster * source_cluster ) const {
  lo n_space_elems = source_cluster->get_n_space_elements( );
  lo n_space_nodes = source_cluster->get_n_space_nodes( );
  T_normal_along_dim.resize( n_space_nodes, _spat_contribution_size );
  T_normal_along_dim.fill( 0.0 );
  // get some info on the current cluster
  vector_type cluster_center_space( 3 );
  vector_type cluster_half_space( 3 );
  sc dummy;
  source_cluster->get_center( cluster_center_space, dummy );
  source_cluster->get_half_size( cluster_half_space, dummy );
  sc padding = _distributed_spacetime_tree
                 ->get_spatial_paddings( )[ source_cluster->get_level( ) ];
  sc start_0 = cluster_center_space[ 0 ] - cluster_half_space[ 0 ] - padding;
  sc end_0 = cluster_center_space[ 0 ] + cluster_half_space[ 0 ] + padding;
  sc start_1 = cluster_center_space[ 1 ] - cluster_half_space[ 1 ] - padding;
  sc end_1 = cluster_center_space[ 1 ] + cluster_half_space[ 1 ] + padding;
  sc start_2 = cluster_center_space[ 2 ] - cluster_half_space[ 2 ] - padding;
  sc end_2 = cluster_center_space[ 2 ] + cluster_half_space[ 2 ] + padding;

  // init quadrature data
  quadrature_wrapper my_quadrature;
  init_quadrature_polynomials( my_quadrature, _order_regular );
  lo size_quad = my_quadrature._wy_cheb.size( );
  sc * wy = my_quadrature._wy_cheb.data( );
  linear_algebra::coordinates< 3 > y1, y2, y3;

  // for storing the result of the Chebyshev evaluation in quadrature
  // points
  vector_type cheb_dim_0( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_1( ( _spat_order + 1 ) * size_quad );
  vector_type cheb_dim_2( ( _spat_order + 1 ) * size_quad );

  sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  sc * y2_ref = my_quadrature._y2_ref_cheb.data( );

  const std::vector< lo > & elems_2_local_nodes
    = source_cluster->get_elems_2_local_nodes( );
  linear_algebra::coordinates< 3 > normal;

  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = source_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  for ( lo i = 0; i < n_space_elems; ++i ) {
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, source_cluster->get_element( i ) );
    lo local_elem_idx_space
      = local_mesh->get_space_element_index( local_elem_idx );
    local_mesh->get_spatial_normal_using_spatial_element_index(
      local_elem_idx_space, normal );
    local_mesh->get_spatial_nodes_using_spatial_element_index(
      local_elem_idx_space, y1, y2, y3 );
    sc elem_area = local_mesh->get_spatial_area_using_spatial_index(
      local_elem_idx_space );

    triangle_to_geometry( y1, y2, y3, my_quadrature );

    cluster_to_polynomials(
      my_quadrature, start_0, end_0, start_1, end_1, start_2, end_2 );

    _chebyshev.evaluate( my_quadrature._y1_polynomial, cheb_dim_0 );
    _chebyshev.evaluate( my_quadrature._y2_polynomial, cheb_dim_1 );
    _chebyshev.evaluate( my_quadrature._y3_polynomial, cheb_dim_2 );

    lo current_index = 0;
    for ( lo beta0 = 0; beta0 <= _spat_order; ++beta0 ) {
      for ( lo beta1 = 0; beta1 <= _spat_order - beta0; ++beta1 ) {
        for ( lo beta2 = 0; beta2 <= _spat_order - beta0 - beta1; ++beta2 ) {
          sc value1 = 0.0;
          sc value2 = 0.0;
          sc value3 = 0.0;
          for ( lo j = 0; j < size_quad; ++j ) {
            sc weight_poly = cheb_dim_0[ beta0 * size_quad + j ]
              * cheb_dim_1[ beta1 * size_quad + j ]
              * cheb_dim_2[ beta2 * size_quad + j ] * wy[ j ] * elem_area;
            value1 += weight_poly * ( (sc) 1.0 - y1_ref[ j ] - y2_ref[ j ] );
            value2 += weight_poly * y1_ref[ j ];
            value3 += weight_poly * y2_ref[ j ];
          }
          T_normal_along_dim.add(
            source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
              elems_2_local_nodes[ 6 * i ] ),
            current_index, value1 * normal[ dim ] );
          T_normal_along_dim.add(
            source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
              elems_2_local_nodes[ 6 * i + 1 ] ),
            current_index, value2 * normal[ dim ] );
          T_normal_along_dim.add(
            source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
              elems_2_local_nodes[ 6 * i + 2 ] ),
            current_index, value3 * normal[ dim ] );
          ++current_index;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou dim >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_chebyshev_times_p1_surface_curls_along_dimension(
    full_matrix & T_curl_along_dim,
    const mesh::general_spacetime_cluster * source_cluster ) const {
  // get the chebyshev quadratures for p0 functions
  full_matrix T;
  compute_chebyshev_quadrature_p0( T, source_cluster );
  // get the surface curls along the current dimension
  std::vector< sc > surf_curls_curr_dim;
  source_cluster->compute_surface_curls_p1_along_dim< dim >(
    surf_curls_curr_dim );
  // get cluster information and resize T_curl_along_dim
  lo n_space_elements = source_cluster->get_n_space_elements( );
  lo n_space_nodes = source_cluster->get_n_space_nodes( );
  const std::vector< lo > & elems_2_local_nodes
    = source_cluster->get_elems_2_local_nodes( );
  T_curl_along_dim.resize( n_space_nodes, _spat_contribution_size );
  T_curl_along_dim.fill( 0.0 );
  // compute T_curl_along_dim from T and surface_curls
  for ( lo i_beta = 0; i_beta < _spat_contribution_size; ++i_beta ) {
    for ( lo i_space_el = 0; i_space_el < n_space_elements; ++i_space_el ) {
      T_curl_along_dim.add(
        source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
          elems_2_local_nodes[ 6 * i_space_el ] ),
        i_beta,
        surf_curls_curr_dim[ 3 * i_space_el ] * T.get( i_space_el, i_beta ) );
      T_curl_along_dim.add(
        source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
          elems_2_local_nodes[ 6 * i_space_el + 1 ] ),
        i_beta,
        surf_curls_curr_dim[ 3 * i_space_el + 1 ]
          * T.get( i_space_el, i_beta ) );
      T_curl_along_dim.add(
        source_cluster->local_spacetime_node_idx_2_local_space_node_idx(
          elems_2_local_nodes[ 6 * i_space_el + 2 ] ),
        i_beta,
        surf_curls_curr_dim[ 3 * i_space_el + 2 ]
          * T.get( i_space_el, i_beta ) );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_lagrange_quadrature( full_matrix & L,
  const mesh::general_spacetime_cluster * source_cluster ) const {
  lo n_temp_elems = source_cluster->get_n_time_elements( );
  lo n_spat_elems = source_cluster->get_n_space_elements( );
  L.resize( _temp_order + 1, n_temp_elems );

  // choose the quadrature order such that polynomials of degree _temp_order
  // are integrated exactly
  lo quad_order = ( _temp_order + 2 ) / 2;
  const std::vector< sc, besthea::allocator_type< sc > > & line_t
    = bem::quadrature::line_x( quad_order );
  const std::vector< sc, besthea::allocator_type< sc > > & line_w
    = bem::quadrature::line_w( quad_order );

  vector_type eval_points( line_t.size( ) );
  vector_type evaluation( line_t.size( ) );

  sc cluster_t_start = source_cluster->get_time_center( )
    - source_cluster->get_time_half_size( );
  sc cluster_t_end = source_cluster->get_time_center( )
    + source_cluster->get_time_half_size( );
  sc cluster_size = cluster_t_end - cluster_t_start;

  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = source_cluster->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  linear_algebra::coordinates< 1 > elem_t_start;
  linear_algebra::coordinates< 1 > elem_t_end;
  for ( lo i = 0; i < n_temp_elems; ++i ) {
    // we use that the elements in the cluster are tensor products of
    // spatial elements and timesteps, and are sorted w.r.t. the
    // timesteps. In particular we get all temporal elements in the
    // cluster by considering every n_spat_elems spacetime element.
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, source_cluster->get_element( i * n_spat_elems ) );
    lo local_elem_idx_time = local_mesh->get_time_element( local_elem_idx );

    local_mesh->get_temporal_nodes(
      local_elem_idx_time, elem_t_start, elem_t_end );
    sc elem_size = elem_t_end[ 0 ] - elem_t_start[ 0 ];
    // compute the quadrature points in the current element in relative
    // coordinates with respect to the time cluster and transform them to
    // [-1,1]
    for ( std::vector< sc, besthea::allocator_type< sc > >::size_type j = 0;
          j < line_t.size( ); ++j ) {
      eval_points[ j ] = -1.0
        + 2.0
          * ( elem_t_start[ 0 ] + elem_size * line_t[ j ] - cluster_t_start )
          / cluster_size;
    }

    for ( lo j = 0; j <= _temp_order; ++j ) {
      _lagrange.evaluate( j, eval_points, evaluation );
      sc quad = 0.0;
      for ( lo k = 0; k < eval_points.size( ); ++k ) {
        quad += evaluation[ k ] * line_w[ k ];
      }
      quad *= elem_size;
      L.set( j, i, quad );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_lagrange_drv_quadrature( full_matrix &
                                                                   L_drv,
  const mesh::general_spacetime_cluster * source_cluster ) const {
  lo n_temp_elems = source_cluster->get_n_time_elements( );
  lo n_spat_elems = source_cluster->get_n_space_elements( );
  L_drv.resize( _temp_order + 1, n_temp_elems );

  vector_type eval_points( 2 );
  vector_type evaluation( 2 );

  sc cluster_t_start = source_cluster->get_time_center( )
    - source_cluster->get_time_half_size( );
  sc cluster_t_end = source_cluster->get_time_center( )
    + source_cluster->get_time_half_size( );
  sc cluster_size = cluster_t_end - cluster_t_start;

  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = source_cluster->get_mesh( );
  // this function should only be called for clusters which are local,
  // i.e. whose elements are in the local mesh
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );

  linear_algebra::coordinates< 1 > elem_t_start;
  linear_algebra::coordinates< 1 > elem_t_end;
  for ( lo i = 0; i < n_temp_elems; ++i ) {
    // we use that the elements in the cluster are tensor products of
    // spatial elements and timesteps, and are sorted w.r.t. the
    // timesteps. In particular we get all temporal elements in the
    // cluster by considering every n_spat_elems spacetime element.
    lo local_elem_idx = distributed_mesh.global_2_local(
      local_start_idx, source_cluster->get_element( i * n_spat_elems ) );
    lo local_elem_idx_time = local_mesh->get_time_element( local_elem_idx );

    local_mesh->get_temporal_nodes(
      local_elem_idx_time, elem_t_start, elem_t_end );
    // compute the end points of the current element in relative
    // coordinates with respect to the time cluster and transform them to
    // [-1,1]
    eval_points[ 0 ]
      = -1.0 + 2.0 * ( elem_t_start[ 0 ] - cluster_t_start ) / cluster_size;
    eval_points[ 1 ]
      = -1.0 + 2.0 * ( elem_t_end[ 0 ] - cluster_t_start ) / cluster_size;
    for ( lo j = 0; j <= _temp_order; ++j ) {
      _lagrange.evaluate( j, eval_points, evaluation );
      L_drv.set( j, i, evaluation[ 1 ] - evaluation[ 0 ] );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::compute_m2l_coupling_coeffs( const vector_type &
                                                               src_time_nodes,
  const vector_type & tar_time_nodes, const sc half_size, const sc center_diff,
  vector_type & buffer_for_gaussians, vector_type & coupling_coeffs ) const {
  // evaluate the gaussian kernel for the numerical integration
  sc h_alpha = half_size * half_size / ( 4.0 * _alpha );
  sc scaled_center_diff = center_diff / half_size;
  lou index_gaussian = 0;

  sc * buffer_for_gaussians_data = buffer_for_gaussians.data( );
  const sc * cheb_nodes_sum_coll_data = _cheb_nodes_sum_coll.data( );
  const sc * all_poly_vals_mult_coll_data = _all_poly_vals_mult_coll.data( );

  for ( lo a = 0; a < tar_time_nodes.size( ); ++a ) {
    for ( lo b = 0; b < src_time_nodes.size( ); ++b ) {
      sc h_delta_ab = h_alpha / ( tar_time_nodes[ a ] - src_time_nodes[ b ] );
      lou i = 0;
#pragma omp simd aligned( cheb_nodes_sum_coll_data, buffer_for_gaussians_data \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
      for ( i = 0; i < _cheb_nodes_sum_coll.size( ); ++i ) {
        buffer_for_gaussians_data[ index_gaussian + i ] = std::exp( -h_delta_ab
          * ( scaled_center_diff + cheb_nodes_sum_coll_data[ i ] )
          * ( scaled_center_diff + cheb_nodes_sum_coll_data[ i ] ) );
      }
      index_gaussian += i;
    }
  }

  // compute the numerical integrals
  sc mul_factor = 4.0 / ( _cheb_nodes_sum_coll.size( ) );
  lou index_integral = 0;

  for ( lo alpha = 0; alpha <= _spat_order; ++alpha ) {
    for ( lo beta = 0; beta <= _spat_order; ++beta ) {
      index_gaussian = 0;
      for ( lo a = 0; a < tar_time_nodes.size( ); ++a ) {
        for ( lo b = 0; b < src_time_nodes.size( ); ++b ) {
          sc val = 0.0;

          lo start_idx
            = alpha * ( _spat_order + 1 ) * _cheb_nodes_sum_coll.size( )
            + beta * _cheb_nodes_sum_coll.size( );
          const sc * curr_ptr = all_poly_vals_mult_coll_data;  // + start_idx;
          std::vector< sc, besthea::allocator_type< sc > >::size_type idx;
#pragma omp simd aligned( buffer_for_gaussians_data,curr_ptr : \
           DATA_ALIGN ) reduction( + : val ) simdlen( BESTHEA_SIMD_WIDTH )
          for ( idx = 0; idx < _cheb_nodes_sum_coll.size( ); ++idx ) {
            val += buffer_for_gaussians_data[ index_gaussian + idx ]
              * curr_ptr[ start_idx + idx ];
          }
          index_gaussian += idx;
          coupling_coeffs[ index_integral ] = val;

          sc mul_factor_ab = mul_factor
            / std::sqrt( 4.0 * M_PI * _alpha
              * ( tar_time_nodes[ a ] - src_time_nodes[ b ] ) );
          // In the multiplicative factor a factor of 2 (gamma) is used
          // for all alpha and beta. For alpha == 0 or beta == 0 a
          // correction is required) an attempt to compute this in a
          // separate loop with precomputed mul_factor_ab was slower
          if ( alpha == 0 ) {
            mul_factor_ab *= 0.5;
          }
          if ( beta == 0 ) {
            mul_factor_ab *= 0.5;
          }
          coupling_coeffs[ index_integral ] *= mul_factor_ab;

          ++index_integral;
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::compute_single_sided_coupling_coeffs( const vector_type &
                                                          time_node_differences,
  const sc half_size, const sc center, const sc eval_point_space,
  vector_type & buffer_for_gaussians, vector_type & coupling_coeffs ) const {
  // evaluate the gaussian kernel for the numerical integration
  sc h_alpha = half_size * half_size / ( 4.0 * _alpha );

  sc * buffer_for_gaussians_data = buffer_for_gaussians.data( );
  const sc * cheb_nodes_integrate_data = _cheb_nodes_integrate.data( );
  const sc * all_poly_vals_integrate_data = _all_poly_vals_integrate.data( );

  sc transformed_eval_point = ( eval_point_space - center ) / half_size;

  lou index_gaussian = 0;
  for ( lo a = 0; a <= _temp_order; ++a ) {
    sc h_delta_a = h_alpha / ( time_node_differences[ a ] );
#pragma omp simd aligned( cheb_nodes_integrate_data, buffer_for_gaussians_data \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
    for ( lo i = 0; i < _cheb_nodes_integrate.size( ); ++i ) {
      buffer_for_gaussians_data[ index_gaussian + i ] = std::exp( -h_delta_a
        * ( transformed_eval_point - cheb_nodes_integrate_data[ i ] )
        * ( transformed_eval_point - cheb_nodes_integrate_data[ i ] ) );
    }
    index_gaussian += _cheb_nodes_integrate.size( );
  }

  // compute the numerical integrals
  sc mul_factor = 2.0 / ( _cheb_nodes_integrate.size( ) );
  lou index_coeff = 0;

  for ( lo alpha = 0; alpha <= _spat_order; ++alpha ) {
    index_gaussian = 0;
    for ( lo a = 0; a <= _temp_order; ++a ) {
      sc val = 0.0;

      lo index_poly = alpha * _cheb_nodes_integrate.size( );
      const sc * curr_ptr = all_poly_vals_integrate_data;
#pragma omp simd aligned( buffer_for_gaussians_data, curr_ptr : DATA_ALIGN ) \
                         reduction( + : val ) simdlen( BESTHEA_SIMD_WIDTH )
      for ( lo idx = 0; idx < _cheb_nodes_integrate.size( ); ++idx ) {
        val += buffer_for_gaussians_data[ index_gaussian + idx ]
          * curr_ptr[ index_poly + idx ];
      }
      index_gaussian += _cheb_nodes_integrate.size( );

      sc mul_factor_a = mul_factor
        / std::sqrt( 4.0 * M_PI * _alpha * ( time_node_differences[ a ] ) );
      // In the multiplicative factor a factor of 2 (gamma) is used for
      // all alpha. For alpha == 0 a correction is required
      if ( alpha == 0 ) {
        mul_factor_a *= 0.5;
      }
      coupling_coeffs[ index_coeff ] = mul_factor_a * val;

      ++index_coeff;
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_single_sided_coupling_coeffs_tensor(
    const vector_type & time_node_differences,
    const vector_type & half_size_space, const vector_type & center_space,
    const vector_type eval_point_space,
    const std::vector< vector_type * > & vector_of_buffers,
    vector_type & coupling_coeffs_tensor ) const {
  for ( lo i = 0; i < 3; ++i ) {
    compute_single_sided_coupling_coeffs( time_node_differences,
      half_size_space[ i ], center_space[ i ], eval_point_space[ i ],
      *vector_of_buffers[ 3 ], *vector_of_buffers[ i ] );
  }
  const sc * buffer_coeffs_0_data = vector_of_buffers[ 0 ]->data( );
  const sc * buffer_coeffs_1_data = vector_of_buffers[ 1 ]->data( );
  const sc * buffer_coeffs_2_data = vector_of_buffers[ 2 ]->data( );
  sc * aux_buffer_assembly_data = vector_of_buffers[ 3 ]->data( );
  sc * coupling_coeffs_tensor_data = coupling_coeffs_tensor.data( );
  // compute the tensor product coupling coefficients
  lo spat_index = 0;
  for ( lo alpha0 = 0; alpha0 <= _spat_order; ++alpha0 ) {
    for ( lo alpha1 = 0; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
#pragma omp simd aligned(                                              \
  buffer_coeffs_0_data, buffer_coeffs_1_data, aux_buffer_assembly_data \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
      for ( lo a = 0; a <= _temp_order; ++a ) {
        aux_buffer_assembly_data[ a ]
          = buffer_coeffs_0_data[ alpha0 * ( _temp_order + 1 ) + a ]
          * buffer_coeffs_1_data[ alpha1 * ( _temp_order + 1 ) + a ];
      }
      for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha0 - alpha1; ++alpha2 ) {
#pragma omp simd aligned(                                                     \
  buffer_coeffs_2_data, aux_buffer_assembly_data, coupling_coeffs_tensor_data \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
        for ( lo a = 0; a <= _temp_order; ++a ) {
          coupling_coeffs_tensor_data[ spat_index * ( _temp_order + 1 ) + a ]
            = aux_buffer_assembly_data[ a ]
            * buffer_coeffs_2_data[ alpha2 * ( _temp_order + 1 ) + a ];
        }
        spat_index++;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_single_sided_coupling_coeffs_drv_space(
    const vector_type & time_node_differences, const sc half_size,
    const sc center, const sc eval_point_space,
    vector_type & buffer_for_drv_gaussians,
    vector_type & coupling_coeffs ) const {
  // evaluate the gaussian kernel's derivative for the numerical integration
  sc h_alpha = half_size * half_size / ( 4.0 * _alpha );

  sc * buffer_for_drv_gaussians_data = buffer_for_drv_gaussians.data( );
  const sc * cheb_nodes_integrate_data = _cheb_nodes_integrate.data( );
  const sc * all_poly_vals_integrate_data = _all_poly_vals_integrate.data( );

  sc transformed_eval_point = ( eval_point_space - center ) / half_size;

  lou index_gaussian = 0;
  for ( lo a = 0; a <= _temp_order; ++a ) {
    sc h_delta_a = h_alpha / ( time_node_differences[ a ] );
#pragma omp simd aligned(                                  \
  cheb_nodes_integrate_data, buffer_for_drv_gaussians_data \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
    for ( lo i = 0; i < _cheb_nodes_integrate.size( ); ++i ) {
      // the missing factor 2/half_size from the derivative is multiplied
      // later
      buffer_for_drv_gaussians_data[ index_gaussian + i ]
        = -std::exp( -h_delta_a
            * ( transformed_eval_point - cheb_nodes_integrate_data[ i ] )
            * ( transformed_eval_point - cheb_nodes_integrate_data[ i ] ) )
        * h_delta_a
        * ( transformed_eval_point - cheb_nodes_integrate_data[ i ] );
    }
    index_gaussian += _cheb_nodes_integrate.size( );
  }

  // compute the numerical integrals
  sc mul_factor = 4.0 / ( _cheb_nodes_integrate.size( ) * half_size );
  // the multiplication factor includes 2/half_size from the derivative of
  // the gaussian
  lou index_coeff = 0;

  for ( lo alpha = 0; alpha <= _spat_order; ++alpha ) {
    index_gaussian = 0;
    for ( lo a = 0; a <= _temp_order; ++a ) {
      sc val = 0.0;

      lo index_poly = alpha * _cheb_nodes_integrate.size( );
      const sc * curr_ptr = all_poly_vals_integrate_data;
#pragma omp simd aligned( buffer_for_drv_gaussians_data,curr_ptr : \
                DATA_ALIGN ) reduction( + : val ) simdlen( BESTHEA_SIMD_WIDTH )
      for ( lo idx = 0; idx < _cheb_nodes_integrate.size( ); ++idx ) {
        val += buffer_for_drv_gaussians_data[ index_gaussian + idx ]
          * curr_ptr[ index_poly + idx ];
      }
      index_gaussian += _cheb_nodes_integrate.size( );

      sc mul_factor_a = mul_factor
        / std::sqrt( 4.0 * M_PI * _alpha * ( time_node_differences[ a ] ) );
      // In the multiplicative factor a factor of 2 (gamma) is used for
      // all alpha. For alpha == 0 a correction is required
      if ( alpha == 0 ) {
        mul_factor_a *= 0.5;
      }
      coupling_coeffs[ index_coeff ] = mul_factor_a * val;

      ++index_coeff;
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  compute_single_sided_coupling_coeffs_normal_drv_tensor(
    const vector_type & time_node_differences,
    const vector_type & half_size_space, const vector_type & center_space,
    const vector_type eval_point_space,
    linear_algebra::coordinates< 3 > & normal,
    const std::vector< vector_type * > & vector_of_buffers,
    vector_type & coupling_coeffs_tensor ) const {
  for ( lo i = 0; i < 3; ++i ) {
    compute_single_sided_coupling_coeffs( time_node_differences,
      half_size_space[ i ], center_space[ i ], eval_point_space[ i ],
      *vector_of_buffers[ 6 ], *vector_of_buffers[ i ] );
    compute_single_sided_coupling_coeffs_drv_space( time_node_differences,
      half_size_space[ i ], center_space[ i ], eval_point_space[ i ],
      *vector_of_buffers[ 6 ], *vector_of_buffers[ 3 + i ] );
  }
  const sc * buffer_coeffs_0_data = vector_of_buffers[ 0 ]->data( );
  const sc * buffer_coeffs_1_data = vector_of_buffers[ 1 ]->data( );
  const sc * buffer_coeffs_2_data = vector_of_buffers[ 2 ]->data( );
  const sc * buffer_drv_coeffs_0_data = vector_of_buffers[ 3 ]->data( );
  const sc * buffer_drv_coeffs_1_data = vector_of_buffers[ 4 ]->data( );
  const sc * buffer_drv_coeffs_2_data = vector_of_buffers[ 5 ]->data( );
  sc * aux_buffer_assembly_0_data = vector_of_buffers[ 6 ]->data( );
  sc * aux_buffer_assembly_1_data = vector_of_buffers[ 7 ]->data( );
  sc * coupling_coeffs_tensor_data = coupling_coeffs_tensor.data( );
  // compute the tensor product coupling coefficients
  lo spat_index = 0;
  for ( lo alpha0 = 0; alpha0 <= _spat_order; ++alpha0 ) {
    for ( lo alpha1 = 0; alpha1 <= _spat_order - alpha0; ++alpha1 ) {
#pragma omp simd aligned( buffer_coeffs_0_data, buffer_drv_coeffs_0_data, \
                          buffer_coeffs_1_data, buffer_drv_coeffs_1_data, \
                          aux_buffer_assembly_0_data                      \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
      for ( lo a = 0; a <= _temp_order; ++a ) {
        aux_buffer_assembly_0_data[ a ]
          = buffer_drv_coeffs_0_data[ alpha0 * ( _temp_order + 1 ) + a ]
            * buffer_coeffs_1_data[ alpha1 * ( _temp_order + 1 ) + a ]
            * normal[ 0 ]
          + buffer_coeffs_0_data[ alpha0 * ( _temp_order + 1 ) + a ]
            * buffer_drv_coeffs_1_data[ alpha1 * ( _temp_order + 1 ) + a ]
            * normal[ 1 ];
      }
#pragma omp simd aligned(                                                \
  buffer_coeffs_0_data, buffer_coeffs_1_data, aux_buffer_assembly_1_data \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
      for ( lo a = 0; a <= _temp_order; ++a ) {
        aux_buffer_assembly_1_data[ a ]
          = buffer_coeffs_0_data[ alpha0 * ( _temp_order + 1 ) + a ]
          * buffer_coeffs_1_data[ alpha1 * ( _temp_order + 1 ) + a ]
          * normal[ 2 ];
      }
      for ( lo alpha2 = 0; alpha2 <= _spat_order - alpha0 - alpha1; ++alpha2 ) {
#pragma omp simd aligned(                                                     \
  buffer_coeffs_2_data, buffer_drv_coeffs_2_data, aux_buffer_assembly_0_data, \
  aux_buffer_assembly_1_data, coupling_coeffs_tensor_data                     \
  : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
        for ( lo a = 0; a <= _temp_order; ++a ) {
          coupling_coeffs_tensor_data[ spat_index * ( _temp_order + 1 ) + a ]
            = aux_buffer_assembly_0_data[ a ]
              * buffer_coeffs_2_data[ alpha2 * ( _temp_order + 1 ) + a ]
            + aux_buffer_assembly_1_data[ a ]
              * buffer_drv_coeffs_2_data[ alpha2 * ( _temp_order + 1 ) + a ];
        }
        spat_index++;
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::reset_scheduling_clusters_dependency_data( )
  const {
  // reset the upward path counters of the clusters in the _m_list.
  for ( scheduling_time_cluster * t_cluster : _m_list ) {
    t_cluster->set_upward_path_counter( t_cluster->get_n_children( ) );
  }

  // reset the downward path status of cluster recursively
  reset_downward_path_status_recursively(
    _distributed_spacetime_tree->get_distribution_tree( )->get_root( ) );

  // reset the m2l counter and number of ready m2l sources of all the
  // clusters in the _m2l_list.
  for ( scheduling_time_cluster * t_cluster : _m2l_list ) {
    t_cluster->set_m2l_counter( 0 );
    t_cluster->reset_n_ready_m2l_sources( );
  }
  // reset the number of ready m2t sources of all clusters in the
  // _m2t_list
  for ( scheduling_time_cluster * t_cluster : _m2t_list ) {
    t_cluster->reset_n_ready_m2t_sources( );
    t_cluster->set_m2t_execution_status( 0 );
  }
  // reset the s2l counter of all clusters in the _s2l_list.
  for ( scheduling_time_cluster * t_cluster : _s2l_list ) {
    char new_status = 1;
    for ( auto s2l_source_cluster : *t_cluster->get_s2l_list( ) ) {
      // if there is a leaf cluster I in the s2l list of t_cluster, the
      // spatial moments of I will be required for the s2l list operations of
      // t_cluster. (except in exceptional cases, but we will send and receive
      // data even in these cases, even if it is not necessary.) Thus we set
      // the s2l execution status to 0 (and update it during the pfmm
      // procedure).
      if ( s2l_source_cluster->get_n_children( ) == 0 ) {
        new_status = 0;
      }
    }
    t_cluster->set_s2l_execution_status( new_status );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  reset_downward_path_status_recursively(
    scheduling_time_cluster * root ) const {
  // consider all local clusters which are active in the downward path
  if ( root->is_active_in_downward_path( ) ) {
    if ( root->get_parent( )->is_active_in_downward_path( ) ) {
      // the cluster has to execute l2l operations -> status 0
      root->set_downward_path_status( 0 );
    } else {
      // no l2l operations necessary -> status 1
      root->set_downward_path_status( 1 );
    }
  }
  // recursive call for all children
  if ( root->get_n_children( ) > 0 ) {
    for ( auto child : *root->get_children( ) ) {
      reset_downward_path_status_recursively( child );
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::init_quadrature_polynomials( quadrature_wrapper &
                                                 my_quadrature,
  const lo quadrature_order ) const {
  // calling copy constructor of std::vector
  my_quadrature._y1_ref_cheb = bem::quadrature::triangle_x1( quadrature_order );
  my_quadrature._y2_ref_cheb = bem::quadrature::triangle_x2( quadrature_order );
  my_quadrature._wy_cheb = bem::quadrature::triangle_w( quadrature_order );

  lo size = my_quadrature._wy_cheb.size( );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );

  my_quadrature._y1_polynomial.resize( size );
  my_quadrature._y2_polynomial.resize( size );
  my_quadrature._y3_polynomial.resize( size );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::triangle_to_geometry( const linear_algebra::coordinates< 3 > &
                                          x1,
  const linear_algebra::coordinates< 3 > & x2,
  const linear_algebra::coordinates< 3 > & x3,
  quadrature_wrapper & my_quadrature ) const {
  const sc * y1_ref = my_quadrature._y1_ref_cheb.data( );
  const sc * y2_ref = my_quadrature._y2_ref_cheb.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo size = my_quadrature._wy_cheb.size( );

  // x1, x2, x3 are vectors in R^3,
  // y%_mapped are the %th components of the vectors to which y#_ref is
  // mapped
#pragma omp simd aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref \
                          : DATA_ALIGN ) simdlen( BESTHEA_SIMD_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * y1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * y2_ref[ i ];
    y2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * y1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * y2_ref[ i ];
    y3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * y1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * y2_ref[ i ];
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::cluster_to_polynomials( quadrature_wrapper &
                                                          my_quadrature,
  sc start_0, sc end_0, sc start_1, sc end_1, sc start_2, sc end_2 ) const {
  for ( lo i = 0; i < my_quadrature._y1_polynomial.size( ); ++i ) {
    my_quadrature._y1_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y1[ i ] - start_0 ) / ( end_0 - start_0 );
    my_quadrature._y2_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y2[ i ] - start_1 ) / ( end_1 - start_1 );
    my_quadrature._y3_polynomial[ i ]
      = -1.0 + 2.0 * ( my_quadrature._y3[ i ] - start_2 ) / ( end_2 - start_2 );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  count_nearfield_entries_levelwise(
    std::vector< long long > & levelwise_nf_entries_total_uncompressed,
    std::vector< long long > & levelwise_nf_entries_spat_adm_uncompressed,
    std::vector< long long > & levelwise_nf_entries_spat_adm_compressed,
    long long & max_n_nf_entries_per_st_target,
    sc & mean_n_nf_entries_per_st_target ) const {
  levelwise_nf_entries_total_uncompressed.resize(
    _distributed_spacetime_tree->get_n_levels( ) );
  levelwise_nf_entries_spat_adm_uncompressed.resize(
    _distributed_spacetime_tree->get_n_levels( ) );
  levelwise_nf_entries_spat_adm_compressed.resize(
    _distributed_spacetime_tree->get_n_levels( ) );
  for ( lo i = 0; i < _distributed_spacetime_tree->get_n_levels( ); ++i ) {
    levelwise_nf_entries_total_uncompressed[ i ] = 0;
    levelwise_nf_entries_spat_adm_uncompressed[ i ] = 0;
    levelwise_nf_entries_spat_adm_compressed[ i ] = 0;
  }
  long long int n_st_targets_with_nf_ops = 0;
  max_n_nf_entries_per_st_target = 0;
  mean_n_nf_entries_per_st_target = 0.0;
  for ( auto it : _n_list ) {
    lo current_level = it->get_level( );
    const std::vector< general_spacetime_cluster * > * st_targets
      = it->get_associated_spacetime_clusters( );
    for ( lou i = 0; i < st_targets->size( ); ++i ) {
      // increase the counter of st targets and initialize a counter for the
      // nearfield entries for the current target.
      bool counted_this_cluster = false;
      long long int n_nf_entries_current_st_target = 0;
      mesh::general_spacetime_cluster * st_target = ( *st_targets )[ i ];
      if ( st_target->get_n_children( ) == 0 ) {
        // increase the counter of st targets with nf operations
        n_st_targets_with_nf_ops++;
        counted_this_cluster = true;
        lo n_target_dofs = st_target->get_n_dofs< target_space >( );
        std::vector< general_spacetime_cluster * > * st_nearfield_list
          = st_target->get_nearfield_list( );
        for ( lou src_index = 0; src_index < st_nearfield_list->size( );
              ++src_index ) {
          general_spacetime_cluster * st_source
            = ( *st_nearfield_list )[ src_index ];
          levelwise_nf_entries_total_uncompressed[ current_level ]
            += n_target_dofs * st_source->get_n_dofs< source_space >( );
          n_nf_entries_current_st_target
            += n_target_dofs * st_source->get_n_dofs< source_space >( );
        }
      }
      auto st_spat_adm_nearfield_list
        = st_target->get_spatially_admissible_nearfield_list( );
      if ( st_spat_adm_nearfield_list != nullptr ) {
        if ( !counted_this_cluster ) {
          // increase the counter of st targets with nf operations
          n_st_targets_with_nf_ops++;
          counted_this_cluster = true;
        }
        const std::vector< std::pair< lo, matrix * > > &
          spat_adm_nf_matrix_pairs
          = _clusterwise_spat_adm_nf_matrix_pairs.at( st_target );
        bool counted_spat_adm_nf_for_current_st_target = false;
        if ( spat_adm_nf_matrix_pairs.size( ) > 0 ) {
          counted_spat_adm_nf_for_current_st_target = true;
          for ( auto nf_matrix_pair : spat_adm_nf_matrix_pairs ) {
            levelwise_nf_entries_spat_adm_compressed[ current_level ]
              += nf_matrix_pair.second->get_n_stored_entries( );
            n_nf_entries_current_st_target
              += nf_matrix_pair.second->get_n_stored_entries( );
          }
        }
        lo n_target_dofs = st_target->get_n_dofs< target_space >( );
        for ( lou src_index = 0;
              src_index < st_spat_adm_nearfield_list->size( ); ++src_index ) {
          general_spacetime_cluster * st_source
            = ( *st_spat_adm_nearfield_list )[ src_index ];
          levelwise_nf_entries_total_uncompressed[ current_level ]
            += n_target_dofs * st_source->get_n_dofs< source_space >( );
          levelwise_nf_entries_spat_adm_uncompressed[ current_level ]
            += n_target_dofs * st_source->get_n_dofs< source_space >( );
          if ( !counted_spat_adm_nf_for_current_st_target ) {
            n_nf_entries_current_st_target
              += n_target_dofs * st_source->get_n_dofs< source_space >( );
          }
        }
      }
      mean_n_nf_entries_per_st_target += n_nf_entries_current_st_target;
      if ( n_nf_entries_current_st_target > max_n_nf_entries_per_st_target ) {
        max_n_nf_entries_per_st_target = n_nf_entries_current_st_target;
      }
    }
  }
  mean_n_nf_entries_per_st_target /= ( (sc) n_st_targets_with_nf_ops );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  count_time_separated_nearfield_entries_levelwise(
    std::vector< long long > & levelwise_time_separated_nearfield_entries )
    const {
  levelwise_time_separated_nearfield_entries.resize(
    _distributed_spacetime_tree->get_n_levels( ) );
  for ( lo i = 0; i < _distributed_spacetime_tree->get_n_levels( ); ++i ) {
    levelwise_time_separated_nearfield_entries[ i ] = 0;
  }
  for ( auto it : _n_list ) {
    lo current_level = it->get_level( );
    const std::vector< general_spacetime_cluster * > * st_targets
      = it->get_associated_spacetime_clusters( );
    for ( lou i = 0; i < st_targets->size( ); ++i ) {
      general_spacetime_cluster * st_target = ( *st_targets )[ i ];
      if ( st_target->get_n_children( ) == 0 ) {
        sc tar_time_center, tar_time_half_size;
        vector_type dummy( 3, false );
        st_target->get_center( dummy, tar_time_center );
        st_target->get_half_size( dummy, tar_time_half_size );
        lo n_target_dofs = st_target->get_n_dofs< target_space >( );
        std::vector< general_spacetime_cluster * > * st_nearfield_list
          = st_target->get_nearfield_list( );
        for ( lou src_index = 0; src_index < st_nearfield_list->size( );
              ++src_index ) {
          general_spacetime_cluster * st_source
            = ( *st_nearfield_list )[ src_index ];
          sc src_time_center, src_time_half_size;
          st_source->get_center( dummy, src_time_center );
          st_source->get_half_size( dummy, src_time_half_size );
          // only count entries if the clusters are separated in time
          if ( ( tar_time_center - tar_time_half_size
                 - 1e-8 * tar_time_half_size )
            > src_time_center + src_time_half_size ) {
            levelwise_time_separated_nearfield_entries[ current_level ]
              += n_target_dofs * st_source->get_n_dofs< source_space >( );
          }
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  collect_information_spatially_admissible_nearfield_operations(
    long long & n_discarded_blocks, long long & n_tot_size_discarded_blocks,
    long long & n_compressed_blocks, long long & n_tot_size_compressed_blocks,
    long long & n_uncompressed_blocks,
    long long & n_tot_size_uncompressed_blocks ) const {
  n_discarded_blocks = 0;
  n_tot_size_discarded_blocks = 0;
  n_compressed_blocks = 0;
  n_tot_size_compressed_blocks = 0;
  n_uncompressed_blocks = 0;
  n_tot_size_uncompressed_blocks = 0;
  for ( auto it : _n_list ) {
    const std::vector< general_spacetime_cluster * > * st_targets
      = it->get_associated_spacetime_clusters( );
    for ( auto st_target : *st_targets ) {
      auto st_spat_adm_nearfield_list
        = st_target->get_spatially_admissible_nearfield_list( );
      if ( st_spat_adm_nearfield_list != nullptr ) {
        lo n_target_dofs = st_target->get_n_dofs< target_space >( );
        // get the vector of assembled spatially admissible nearfield matrices
        auto spat_adm_nf_matrix_pairs
          = _clusterwise_spat_adm_nf_matrix_pairs.at( st_target );
        lo next_nf_matrix_idx = 0;
        lo next_nf_matrix_src_idx = -1;
        if ( spat_adm_nf_matrix_pairs.size( ) > 0 ) {
          next_nf_matrix_src_idx
            = spat_adm_nf_matrix_pairs[ next_nf_matrix_idx ].first;
        }
        for ( lo src_index = 0;
              src_index < (lo) st_spat_adm_nearfield_list->size( );
              ++src_index ) {
          general_spacetime_cluster * st_source
            = ( *st_spat_adm_nearfield_list )[ src_index ];
          lo n_source_dofs = st_source->get_n_dofs< source_space >( );
          if ( src_index == next_nf_matrix_src_idx ) {
            matrix * nf_matrix
              = spat_adm_nf_matrix_pairs[ next_nf_matrix_idx ].second;
            // check if the compression was successful or not
            if ( nf_matrix->get_rank( ) < std::min(
                   nf_matrix->get_n_rows( ), nf_matrix->get_n_columns( ) ) ) {
              n_compressed_blocks++;
              n_tot_size_compressed_blocks += n_target_dofs * n_source_dofs;
            } else {
              n_uncompressed_blocks++;
              n_tot_size_uncompressed_blocks += n_target_dofs * n_source_dofs;
            }
            if ( (lou) next_nf_matrix_idx
              < spat_adm_nf_matrix_pairs.size( ) - 1 ) {
              next_nf_matrix_idx++;
              next_nf_matrix_src_idx
                = spat_adm_nf_matrix_pairs[ next_nf_matrix_idx ].first;
            }
          } else {
            n_discarded_blocks++;
            n_tot_size_discarded_blocks += n_target_dofs * n_source_dofs;
          }
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  count_spatially_admissible_nearfield_operations_gridwise(
    std::vector< std::vector< long long > > & disc_blocks_per_time_level,
    std::vector< std::vector< long long > > & disc_blocks_per_aux_space_level,
    long long & other_disc_blocks,
    std::vector< std::vector< long long > > & comp_blocks_per_time_level,
    std::vector< std::vector< long long > > & comp_blocks_per_aux_space_level,
    long long & other_comp_blocks,
    std::vector< std::vector< long long > > & uncomp_blocks_per_time_level,
    std::vector< std::vector< long long > > & uncomp_blocks_per_aux_space_level,
    long long & other_uncomp_blocks ) const {
  other_disc_blocks = 0;
  other_comp_blocks = 0;
  other_uncomp_blocks = 0;
  lo n_time_levels
    = _distributed_spacetime_tree->get_distribution_tree( )->get_levels( );
  lo spat_nf_limit
    = _distributed_spacetime_tree->get_spatial_nearfield_limit( );
  // auxiliary variables to access proper positions of clusters in a regular
  // grid later on
  lo y_coord_stride = ( 2 * spat_nf_limit + 1 );
  lo x_coord_stride = y_coord_stride * y_coord_stride;
  lo spat_nf_size = x_coord_stride * y_coord_stride;
  // the values differ for the auxiliary spatial levels (due to the hierarchic
  // subdivision of non-separated nearfield blocks)
  lo y_coord_stride_aux_s = 7;
  lo x_coord_stride_aux_s = 49;
  lo spat_nf_size_aux_s = 343;
  // for each time level we determine grids of spatial nearfield clusters for
  // the same time step and the previous time step separately
  disc_blocks_per_time_level.resize( 2 * n_time_levels );
  comp_blocks_per_time_level.resize( 2 * n_time_levels );
  uncomp_blocks_per_time_level.resize( 2 * n_time_levels );
  for ( lo i = 0; i < 2 * n_time_levels; ++i ) {
    disc_blocks_per_time_level[ i ].resize( spat_nf_size, 0 );
    comp_blocks_per_time_level[ i ].resize( spat_nf_size, 0 );
    uncomp_blocks_per_time_level[ i ].resize( spat_nf_size, 0 );
  }

  lo dummy;
  for ( auto it : _n_list ) {
    lo t_level = it->get_level( );
    lo t_tar_idx = it->get_global_index( );
    const std::vector< mesh::general_spacetime_cluster * > * st_target_list
      = it->get_associated_spacetime_clusters( );
    for ( auto st_target : *st_target_list ) {
      auto st_spat_adm_nearfield_list
        = st_target->get_spatially_admissible_nearfield_list( );
      bool st_target_auxiliary = st_target->is_auxiliary_ref_cluster( );
      lo tar_s_level;
      st_target->get_n_divs( tar_s_level, dummy );
      lo rel_aux_level = 0;
      if ( st_target_auxiliary ) {
        // determine the depth of the current space-time target in the auxiliary
        // refined subtree (i.e. length of the path to the original leaf, from
        // which the current auxiliary cluster was created.)
        rel_aux_level = 1;
        auto current_parent = st_target->get_parent( );
        while ( current_parent->is_auxiliary_ref_cluster( ) ) {
          rel_aux_level += 1;
          current_parent = current_parent->get_parent( );
        }
        if ( disc_blocks_per_aux_space_level.size( )
          < (lou) rel_aux_level * 2 ) {
          disc_blocks_per_aux_space_level.resize( 2 * rel_aux_level );
          disc_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) ].resize(
            spat_nf_size_aux_s, 0 );
          disc_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) + 1 ]
            .resize( spat_nf_size_aux_s, 0 );
          comp_blocks_per_aux_space_level.resize( 2 * rel_aux_level );
          comp_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) ].resize(
            spat_nf_size_aux_s, 0 );
          comp_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) + 1 ]
            .resize( spat_nf_size_aux_s, 0 );
          uncomp_blocks_per_aux_space_level.resize( 2 * rel_aux_level );
          uncomp_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) ].resize(
            spat_nf_size_aux_s, 0 );
          uncomp_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) + 1 ]
            .resize( spat_nf_size_aux_s, 0 );
        }
      }
      std::vector< slou > tar_coords = st_target->get_box_coordinate( );
      if ( st_spat_adm_nearfield_list != nullptr ) {
        // get the vector of assembled spatially admissible nearfield matrices
        auto spat_adm_nf_matrix_pairs
          = _clusterwise_spat_adm_nf_matrix_pairs.at( st_target );
        lo next_nf_matrix_idx = 0;
        lo next_nf_matrix_src_idx = -1;
        if ( spat_adm_nf_matrix_pairs.size( ) > 0 ) {
          next_nf_matrix_src_idx
            = spat_adm_nf_matrix_pairs[ next_nf_matrix_idx ].first;
        }
        for ( lo src_index = 0;
              src_index < (lo) st_spat_adm_nearfield_list->size( );
              ++src_index ) {
          mesh::general_spacetime_cluster * st_source
            = ( *st_spat_adm_nearfield_list )[ src_index ];
          lo t_src_idx = st_source->get_global_time_index( );
          lo src_s_level;
          st_source->get_n_divs( src_s_level, dummy );
          // determine the relative grid position of the source cluster
          // (assuming that its spatial level is the same as the one of the
          // target cluster)
          std::vector< slou > src_coords = st_source->get_box_coordinate( );
          lo x_coord_diff_mod, y_coord_diff_mod, z_coord_diff_mod, pos;
          if ( !st_target_auxiliary ) {
            x_coord_diff_mod
              = spat_nf_limit + (lo) src_coords[ 1 ] - (lo) tar_coords[ 1 ];
            y_coord_diff_mod
              = spat_nf_limit + (lo) src_coords[ 2 ] - (lo) tar_coords[ 2 ];
            z_coord_diff_mod
              = spat_nf_limit + (lo) src_coords[ 3 ] - (lo) tar_coords[ 3 ];
            pos = x_coord_diff_mod * x_coord_stride
              + y_coord_diff_mod * y_coord_stride + z_coord_diff_mod;
          } else {
            x_coord_diff_mod = 3 + (lo) src_coords[ 1 ] - (lo) tar_coords[ 1 ];
            y_coord_diff_mod = 3 + (lo) src_coords[ 2 ] - (lo) tar_coords[ 2 ];
            z_coord_diff_mod = 3 + (lo) src_coords[ 3 ] - (lo) tar_coords[ 3 ];
            pos = x_coord_diff_mod * x_coord_stride_aux_s
              + y_coord_diff_mod * y_coord_stride_aux_s + z_coord_diff_mod;
          }
          if ( src_index == next_nf_matrix_src_idx ) {
            matrix * nf_matrix
              = spat_adm_nf_matrix_pairs[ next_nf_matrix_idx ].second;
            // check if the compression was successful or not
            if ( nf_matrix->get_rank( ) < std::min(
                   nf_matrix->get_n_rows( ), nf_matrix->get_n_columns( ) ) ) {
              if ( !st_target_auxiliary && tar_s_level == src_s_level ) {
                if ( t_tar_idx == t_src_idx ) {
                  comp_blocks_per_time_level[ 2 * t_level ][ pos ] += 1;
                } else if ( t_tar_idx == t_src_idx + 1 ) {
                  comp_blocks_per_time_level[ 2 * t_level + 1 ][ pos ] += 1;
                } else {
                  other_comp_blocks++;
                }
              } else if ( tar_s_level == src_s_level ) {
                // for rel_aux_level = 0 no information has to be stored, so we
                // start counting at 1 in the levelwise vector.
                if ( t_tar_idx == t_src_idx ) {
                  comp_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) ]
                                                 [ pos ]
                    += 1;
                } else if ( t_tar_idx == t_src_idx + 1 ) {
                  comp_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 )
                    + 1 ][ pos ]
                    += 1;
                } else {
                  other_comp_blocks++;
                }
              } else {
                other_comp_blocks++;
              }
            } else {
              if ( !st_target_auxiliary && tar_s_level == src_s_level ) {
                if ( t_tar_idx == t_src_idx ) {
                  uncomp_blocks_per_time_level[ 2 * t_level ][ pos ] += 1;
                } else if ( t_tar_idx == t_src_idx + 1 ) {
                  uncomp_blocks_per_time_level[ 2 * t_level + 1 ][ pos ] += 1;
                } else {
                  other_uncomp_blocks++;
                }
              } else if ( tar_s_level == src_s_level ) {
                // for rel_aux_level = 0 no information has to be stored, so we
                // start counting at 1 in the levelwise vector.
                if ( t_tar_idx == t_src_idx ) {
                  uncomp_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) ]
                                                   [ pos ]
                    += 1;
                } else if ( t_tar_idx == t_src_idx + 1 ) {
                  uncomp_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 )
                    + 1 ][ pos ]
                    += 1;
                } else {
                  other_uncomp_blocks++;
                }
              } else {
                other_uncomp_blocks++;
              }
            }
            if ( (lou) next_nf_matrix_idx
              < spat_adm_nf_matrix_pairs.size( ) - 1 ) {
              next_nf_matrix_idx++;
              next_nf_matrix_src_idx
                = spat_adm_nf_matrix_pairs[ next_nf_matrix_idx ].first;
            }
          } else {
            if ( !st_target_auxiliary && tar_s_level == src_s_level ) {
              if ( t_tar_idx == t_src_idx ) {
                disc_blocks_per_time_level[ 2 * t_level ][ pos ] += 1;
              } else if ( t_tar_idx == t_src_idx + 1 ) {
                disc_blocks_per_time_level[ 2 * t_level + 1 ][ pos ] += 1;
              } else {
                other_disc_blocks++;
              }
            } else if ( tar_s_level == src_s_level ) {
              // for rel_aux_level = 0 no information has to be stored, so we
              // start counting at 1 in the levelwise vector.
              if ( t_tar_idx == t_src_idx ) {
                disc_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) ]
                                               [ pos ]
                  += 1;
              } else if ( t_tar_idx == t_src_idx + 1 ) {
                disc_blocks_per_aux_space_level[ 2 * ( rel_aux_level - 1 ) + 1 ]
                                               [ pos ]
                  += 1;
              } else {
                other_disc_blocks++;
              }
            } else {
              other_disc_blocks++;
            }
          }
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::count_fmm_operations_levelwise( std::vector< long long > &
                                                    n_s2m_operations,
  std::vector< long long > & n_m2m_operations,
  std::vector< long long > & n_m2l_operations,
  std::vector< long long > & n_l2l_operations,
  std::vector< long long > & n_l2t_operations,
  std::vector< long long > & n_s2l_operations,
  std::vector< long long > & n_m2t_operations ) const {
  lo n_max_levels = _distributed_spacetime_tree->get_n_levels( );
  // count the number of s2m operations
  n_s2m_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_s2m_operations[ i ] = 0;
  }
  for ( auto it : _m_list ) {
    if ( it->get_n_associated_leaves( ) > 0 ) {
      n_s2m_operations[ it->get_level( ) ] += it->get_n_associated_leaves( );
    }
  }

  // count the number of m2m operations
  n_m2m_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_m2m_operations[ i ] = 0;
  }
  for ( auto it : _m_list ) {
    if ( it->get_parent( )->is_active_in_upward_path( ) ) {
      n_m2m_operations[ it->get_level( ) ]
        += it->get_n_st_clusters_w_moments( );
    }
  }

  // count the number of m2l. in addition, count the number of l2t
  // operations for those clusters whose parents are not active in the
  // downward path
  n_m2l_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_m2l_operations[ i ] = 0;
  }
  n_l2t_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_l2t_operations[ i ] = 0;
  }
  for ( auto it : _m2l_list ) {
    std::vector< general_spacetime_cluster * > * associated_st_targets
      = it->get_associated_spacetime_clusters( );
    for ( auto spacetime_tar : *associated_st_targets ) {
      if ( spacetime_tar->get_interaction_list( ) != nullptr ) {
        n_m2l_operations[ it->get_level( ) ]
          += spacetime_tar->get_interaction_list( )->size( );
      }
    }
    if ( !it->get_parent( )->is_active_in_downward_path( ) ) {
      if ( it->get_n_associated_leaves( ) > 0 ) {
        n_l2t_operations[ it->get_level( ) ] += it->get_n_associated_leaves( );
      }
    }
  }

  // count the number of l2l operations
  n_l2l_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_l2l_operations[ i ] = 0;
  }
  for ( auto it : _l_list ) {
    if ( it->get_parent( )->is_active_in_downward_path( ) ) {
      n_l2l_operations[ it->get_level( ) ]
        += it->get_n_st_clusters_w_local_contributions( );
    }
  }

  // count the number of l2t operations
  for ( auto it : _l_list ) {
    if ( it->get_n_associated_leaves( ) > 0 ) {
      n_l2t_operations[ it->get_level( ) ] += it->get_n_associated_leaves( );
    }
  }

  // count the number of s2l operations
  n_s2l_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_s2l_operations[ i ] = 0;
  }
  for ( auto it : _s2l_list ) {
    std::vector< general_spacetime_cluster * > * associated_st_targets
      = it->get_associated_spacetime_clusters( );
    for ( auto spacetime_tar : *associated_st_targets ) {
      if ( spacetime_tar->get_s2l_list( ) != nullptr ) {
        n_s2l_operations[ it->get_level( ) ]
          += spacetime_tar->get_s2l_list( )->size( );
      }
    }
  }

  // count the number of m2t operations
  n_m2t_operations.resize( n_max_levels );
  for ( lo i = 0; i < n_max_levels; ++i ) {
    n_m2t_operations[ i ] = 0;
  }
  for ( auto it : _m2t_list ) {
    std::vector< general_spacetime_cluster * > * associated_st_targets
      = it->get_associated_spacetime_clusters( );
    for ( lou i = 0; i < associated_st_targets->size( ); ++i ) {
      general_spacetime_cluster * spacetime_tar
        = ( *associated_st_targets )[ i ];
      if ( spacetime_tar->get_m2t_list( ) != nullptr ) {
        n_m2t_operations[ it->get_level( ) ]
          += spacetime_tar->get_m2t_list( )->size( );
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::l_list_task( distributed_block_vector & y_pFMM,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  call_l2l_operations( current_cluster, verbose, verbose_file );
  // check if all the m2l, m2t, and s2l operations have been executed yet
  if ( ( current_cluster->get_interaction_list( ) == nullptr
         || current_cluster->get_m2l_counter( )
           == current_cluster->get_interaction_list( )->size( ) )
    && ( current_cluster->get_s2l_execution_status( ) == 2 )
    && current_cluster->get_m2t_execution_status( ) == 1 ) {
    // set status of current cluster's local contributions to completed
    current_cluster->set_downward_path_status( 2 );
    call_l2t_operations< run_count >(
      current_cluster, y_pFMM, verbose, verbose_file );
    provide_local_contributions_to_children(
      current_cluster, verbose, verbose_file );
  } else {
    current_cluster->set_downward_path_status( 1 );
  }
  reduce_nn_operations( );
  if ( _measure_tasks ) {
    _l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply( const distributed_block_vector & /*x*/,
  distributed_block_vector & /*y*/, bool /*trans*/, sc /*alpha*/,
  sc /*beta*/ ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED" << std::endl;
}

//! template specialization of @ref distributed_pFMM_matrix::apply for
//! single layer p0p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

//! template specialization of @ref distributed_pFMM_matrix::apply for
//! double layer p0p1 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::apply( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

//! template specialization of @ref distributed_pFMM_matrix::apply for
//! double layer p1p0 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::apply( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_sl_dl( x, y, trans, alpha, beta );
}

//! template specialization of @ref distributed_pFMM_matrix::apply for
//! hypersingular p1p1 matrix
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::apply( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  apply_hs( x, y, trans, alpha, beta );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_on_the_fly( const assembler_type &
                                                    matrix_assembler,
  const distributed_block_vector & x, distributed_block_vector & y, sc alpha,
  sc beta ) const {
  y.scale( beta );
  // apply the nearfield on the fly, then call the appropriate standard apply
  // routine (in which the nearfield part is disabled)
  apply_nearfield_on_the_fly( matrix_assembler, x, y, alpha );
  apply( x, y, false, alpha, 1.0 );  // the scaling by beta has already been
                                     // done, so it is set to 1.0 here.
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_sl_dl( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  // Specialization for the single and double layer operators
  _global_timer.reset( );
  // ############################################################################
  // #### multiply the global result vector by beta ####
  y.scale( beta );

  // allocate a global result vector to store the result of the pFMM
  // procedure.
  std::vector< lo > my_blocks = y.get_my_blocks( );
  distributed_block_vector y_pFMM(
    my_blocks, y.get_n_blocks( ), y.get_size_of_block( ), true, y.get_comm( ) );

  // apply pFMM procedure
  apply_pFMM_procedure< 0 >( x, y_pFMM, trans );

  y.add( y_pFMM, alpha );

  MPI_Barrier( y.get_comm( ) );
  y.synchronize_shared_parts( );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::apply_hs( const distributed_block_vector & x,
  distributed_block_vector & y, bool trans, sc alpha, sc beta ) const {
  // Specialization for the single and double layer operators
  _global_timer.reset( );
  // ############################################################################
  // #### multiply the global result vector by beta ####
  y.scale( beta );

  // allocate a global result vector to store the result of the pFMM
  // procedure.
  std::vector< lo > my_blocks = y.get_my_blocks( );
  distributed_block_vector y_pFMM(
    my_blocks, y.get_n_blocks( ), y.get_size_of_block( ), true, y.get_comm( ) );

  // apply pFMM procedure
  // We use barriers between pFMM procedures to be on the safe side. The
  // problem is that processes don't check whether their messages have
  // been sent. An unsent message would be deleted when starting a new
  // pFMM procedure first 3 runs for curl terms
  apply_pFMM_procedure< 0 >( x, y_pFMM, trans );
  MPI_Barrier( y.get_comm( ) );
  apply_pFMM_procedure< 1 >( x, y_pFMM, trans );
  MPI_Barrier( y.get_comm( ) );
  apply_pFMM_procedure< 2 >( x, y_pFMM, trans );
  MPI_Barrier( y.get_comm( ) );
  // add scaled result to y, before overwriting y_pFMM and starting next 3
  // runs
  y.add( y_pFMM, alpha );
  y_pFMM.fill( 0.0 );
  apply_pFMM_procedure< 3 >( x, y_pFMM, trans );
  MPI_Barrier( y.get_comm( ) );
  apply_pFMM_procedure< 4 >( x, y_pFMM, trans );
  MPI_Barrier( y.get_comm( ) );
  apply_pFMM_procedure< 5 >( x, y_pFMM, trans );
  MPI_Barrier( y.get_comm( ) );
  // add scaled result to y and synchronize y
  y.add( y_pFMM, -alpha );
  MPI_Barrier( y.get_comm( ) );
  y.synchronize_shared_parts( );
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_pFMM_procedure( const distributed_block_vector & x,
  distributed_block_vector & y_pFMM, bool trans ) const {
  // #### distributed pFMM ####
  // ############################################################################
  // #### setup phase ####
  //  reset the contributions of all clusters to zero
  _scheduling_tree_structure->clear_moment_contributions(
    *_scheduling_tree_structure->get_root( ) );
  _scheduling_tree_structure->clear_local_contributions(
    *_scheduling_tree_structure->get_root( ) );

  // reset the dependency data of all the clusters in the 4 lists.
  reset_scheduling_clusters_dependency_data( );

  // initialize data which is used to check for received data.
  int outcount = 0;
  std::vector< int > array_of_indices( _receive_data_information.size( ) );
  for ( lou i = 0; i < _receive_data_information.size( ); ++i ) {
    array_of_indices[ i ] = 0;
  }
  // copy the 6 FMM lists to avoid recomputing them in each application
  std::list< mesh::scheduling_time_cluster * > m_list = _m_list;
  std::list< mesh::scheduling_time_cluster * > m2l_list = _m2l_list;
  std::list< mesh::scheduling_time_cluster * > l_list = _l_list;
  std::list< mesh::scheduling_time_cluster * > m2t_list = _m2t_list;
  std::list< mesh::scheduling_time_cluster * > s2l_list = _s2l_list;
  std::list< mesh::scheduling_time_cluster * > n_list;

  // Decide whether to execute nearfield operations. In general, this depends on
  // the value of  _execute_nf_operations_in_pfmm_procedure.
  // In case of the hypersingular operator, the n_list is additionally
  // initialized only in the first run (run_count == 0, which is always the case
  // for all other operators)
  if ( _execute_nf_operations_in_pfmm_procedure && run_count == 0 ) {
    n_list = _n_list;
  }

  // std::string verbose_file = verbose_dir + "/process_";
  std::string verbose_file = "verbose/process_";
  verbose_file += std::to_string( _my_rank );
  if ( _verbose ) {
    std::filesystem::create_directory( "./verbose/" );
    // remove existing verbose file and write to new one
    remove( verbose_file.c_str( ) );
  }
  // auxiliary arrays for OpenMP dependencies
  // (must not be members in OpenMP 4.5)
  char * aux_dep_m2l = new char[ m2l_list.size( ) ];
  std::fill( aux_dep_m2l, aux_dep_m2l + m2l_list.size( ), 0 );

  char * aux_dep_m2l_send = new char[ m2l_list.size( ) ];
  std::fill( aux_dep_m2l_send, aux_dep_m2l_send + m2l_list.size( ), 0 );

  char * aux_dep_l = new char[ l_list.size( ) ];
  std::fill( aux_dep_l, aux_dep_l + l_list.size( ), 0 );

  char * aux_dep_m = new char[ m_list.size( ) ];
  std::fill( aux_dep_m, aux_dep_m + m_list.size( ), 0 );

  char * aux_dep_s2l = new char[ s2l_list.size( ) ];
  std::fill( aux_dep_s2l, aux_dep_s2l + s2l_list.size( ), 0 );

  // allocate buffers for m2l computation
  _aux_buffer_0.resize( omp_get_max_threads( ) );
  _aux_buffer_1.resize( omp_get_max_threads( ) );

  // reset the number of non-nearfield ops
  _non_nf_op_count = 0;

  // set loop timer start
  time_type::rep loop_start = 0;
  if ( _measure_tasks ) {
    loop_start = _global_timer.get_time_from_start< time_type >( );
  }

  // start the main "job scheduling" algorithm
  // the "master" thread checks for new available data, spawns tasks, and
  // removes clusters from lists
  lo scheduling_thread;
#pragma omp parallel
  {
    _aux_buffer_0[ omp_get_thread_num( ) ].resize(
      ( _temp_order + 1 ) * ( _temp_order + 1 ), _spat_contribution_size );
    _aux_buffer_1[ omp_get_thread_num( ) ].resize(
      ( _temp_order + 1 ) * ( _temp_order + 1 ), _spat_contribution_size );

    if ( run_count == 0 ) {
      _m_task_times.at( omp_get_thread_num( ) ).resize( 0 );
      _m2l_task_times.at( omp_get_thread_num( ) ).resize( 0 );
      _l_task_times.at( omp_get_thread_num( ) ).resize( 0 );
      _n_task_times.at( omp_get_thread_num( ) ).resize( 0 );
      _m_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
      _m2l_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
      _l_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
      _m2t_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
      _s2l_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
      _n_subtask_times.at( omp_get_thread_num( ) ).resize( 0 );
      _mpi_send_m2l_m2t_or_s2l.at( omp_get_thread_num( ) ).resize( 0 );
      _mpi_send_m_parent.at( omp_get_thread_num( ) ).resize( 0 );
      _mpi_send_l_children.at( omp_get_thread_num( ) ).resize( 0 );
      _mpi_recv_m2l_m2t_or_s2l.at( omp_get_thread_num( ) ).resize( 0 );
      _mpi_recv_m_parent.at( omp_get_thread_num( ) ).resize( 0 );
      _mpi_recv_l_children.at( omp_get_thread_num( ) ).resize( 0 );
    }

#pragma omp single
    {
      scheduling_thread = omp_get_thread_num( );
      if ( _verbose ) {
#pragma omp critical( verbose )
        {
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << "Scheduling thread is " << scheduling_thread
                    << std::endl;
            outfile.close( );
          }
        }
      }
      // start the receive operations
      std::vector< MPI_Request > array_of_requests(
        _receive_data_information.size( ) );
      start_receive_operations( array_of_requests );
      while ( true ) {
        if ( m_list.empty( ) && m2l_list.empty( ) && l_list.empty( )
          && n_list.empty( ) && s2l_list.empty( ) && m2t_list.empty( ) ) {
          break;
        }

        // check if data has been received since the last iteration
        if ( outcount != MPI_UNDEFINED ) {
          check_for_received_data(
            array_of_requests, array_of_indices, outcount );
        }

        // processing of received data
        // we have to do this here to spawn tasks with correct
        // dependencies
        if ( outcount != MPI_UNDEFINED && outcount > 0 ) {
          for ( lo i = 0; i < outcount; ++i ) {
            lou current_index = array_of_indices[ i ];
            scheduling_time_cluster * current_cluster
              = _receive_data_information[ current_index ].first;
            if ( _verbose ) {
#pragma omp critical( verbose )
              {
                std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
                if ( outfile.is_open( ) ) {
                  outfile << "received data of cluster "
                          << current_cluster->get_global_index( )
                          << " at level " << current_cluster->get_level( )
                          << " from process "
                          << _receive_data_information[ current_index ].second
                          << std::endl;
                  outfile.close( );
                }
              }
            }
            // distinguish which data has been received
            if ( current_index < _n_moments_to_receive_upward ) {
              if ( _measure_tasks ) {
                _mpi_recv_m_parent.at( omp_get_thread_num( ) )
                  .push_back(
                    _global_timer.get_time_from_start< time_type >( ) );
              }

              // received data are moments in the upward path. add up
              // moments and update dependencies.
              lo idx = current_cluster->get_pos_in_m_list( );

              // task depends on previously generated M-list tasks to
              // prevent collision in m2m operations
#pragma omp task depend( inout : aux_dep_m [idx:1] ) priority( 1000 )
              upward_path_task( current_index, current_cluster );
            } else if ( current_index < _n_moments_to_receive_upward
                + _n_moments_to_receive_m2l_or_m2t ) {
              if ( _measure_tasks ) {
                _mpi_recv_m2l_m2t_or_s2l.at( omp_get_thread_num( ) )
                  .push_back(
                    _global_timer.get_time_from_start< time_type >( ) );
              }
              // received data are moments for m2l or m2t. update
              // dependencies.
              std::vector< scheduling_time_cluster * > * send_list
                = current_cluster->get_send_list( );
              if ( send_list != nullptr ) {
                for ( auto it = send_list->begin( ); it != send_list->end( );
                      ++it ) {
                  lo tar_process_id = ( *it )->get_process_id( );
                  if ( tar_process_id == _my_rank ) {
                    ( *it )->update_n_ready_m2l_sources( );
                  }
                }
              }
              std::vector< scheduling_time_cluster * > * m2t_send_list
                = current_cluster->get_m2t_send_list( );
              if ( m2t_send_list != nullptr ) {
                for ( auto it = m2t_send_list->begin( );
                      it != m2t_send_list->end( ); ++it ) {
                  lo tar_process_id = ( *it )->get_process_id( );
                  if ( tar_process_id == _my_rank ) {
                    ( *it )->update_n_ready_m2t_sources( );
                  }
                }
              }
            } else if ( current_index < _n_moments_to_receive_upward
                + _n_moments_to_receive_m2l_or_m2t
                + _n_spatial_moments_to_receive ) {
              if ( _measure_tasks ) {
                _mpi_recv_m2l_m2t_or_s2l.at( omp_get_thread_num( ) )
                  .push_back(
                    _global_timer.get_time_from_start< time_type >( ) );
              }
              // received data are spatial moments for s2l operations. Update
              // the execution status of local s2l target clusters.
              std::vector< scheduling_time_cluster * > * clusters_s2l_send_list
                = current_cluster->get_s2l_send_list( );
              if ( clusters_s2l_send_list != nullptr ) {
                for ( auto s2l_target_cluster : *clusters_s2l_send_list ) {
                  lo tar_process_id = s2l_target_cluster->get_process_id( );
                  if ( tar_process_id == _my_rank ) {
                    s2l_target_cluster->set_s2l_execution_status( 1 );
                  }
                }
              }
            } else {
              if ( _measure_tasks ) {
                _mpi_recv_l_children.at( omp_get_thread_num( ) )
                  .push_back(
                    _global_timer.get_time_from_start< time_type >( ) );
              }
              // received data are local contributions. update dependencies.
              current_cluster->set_downward_path_status( 2 );
            }
          }
        }

        // check if there is a cluster in one of the 6 lists whose
        // operations are ready to be executed.
        char status = 0;
        std::list< scheduling_time_cluster * >::iterator it_current_cluster;
        find_cluster_in_m_list( m_list, it_current_cluster, status );
        if ( status == 0 ) {
          // search in l list if no cluster was found yet
          find_cluster_in_l_list( l_list, it_current_cluster, status );
          if ( status == 0 ) {
            // search in m2l and s2l list if no cluster was found yet
            find_cluster_in_m2l_list( m2l_list, it_current_cluster, status );
            if ( status == 0 ) {
              find_cluster_in_s2l_list( s2l_list, it_current_cluster, status );
              if ( status == 0 ) {
                // search in m2t list if no cluster was found yet
                find_cluster_in_m2t_list(
                  m2t_list, it_current_cluster, status );
                if ( status == 0 ) {
                  // take the first cluster from the n-list if it is not
                  // empty
                  if ( n_list.size( ) != 0 && get_nn_operations( ) < 1 ) {
                    it_current_cluster = n_list.begin( );
                    status = 6;
                  }
                } else {
                  add_nn_operations( );
                }
              } else {
                add_nn_operations( );
              }
            } else {
              add_nn_operations( );
            }
          } else {
            add_nn_operations( );
          }
        } else {
          add_nn_operations( );
        }

        // interrupt the scheduling task if there is enough work to do so
        // it can join the remaining tasks
        if ( get_nn_operations( ) > 0 && status == 0 ) {
#pragma omp taskyield
        }

        // if verbose mode is chosen, write info about next operation to
        // file
        if ( _verbose && status != 0 ) {
#pragma omp critical( verbose )
          {
            std::stringstream outss;
            outss << "scheduling ";
            switch ( status ) {
              case 1:
                outss << "m-list operations ";
                break;
              case 2:
                outss << "l-list operations ";
                break;
              case 3:
                outss << "s2l-list operations ";
                break;
              case 4:
                outss << "m2l-list operations ";
                break;
              case 5:
                outss << "m2t-list operations ";
                break;
              case 6:
                outss << "n-list operations ";
                break;
            }
            outss << "for cluster "
                  << ( *it_current_cluster )->get_global_index( )
                  << " at level " << ( *it_current_cluster )->get_level( );
            std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
            if ( outfile.is_open( ) ) {
              outfile << outss.str( ) << std::endl;
              outfile.close( );
            }
          }
        }

        scheduling_time_cluster * current_cluster = *it_current_cluster;

        // start the appropriate fmm operations according to status
        switch ( status ) {
          case 1: {
            // M-list task
            m_list.erase( it_current_cluster );

            lo idx_m_parent
              = current_cluster->get_parent( )->get_pos_in_m_list( );
            lo idx_m = current_cluster->get_pos_in_m_list( );

            if ( idx_m_parent != -1 ) {
              // m-list task depends on previously generated tasks
              // with the same parent (so the m2m operations to the parent do
              // not collide)
#pragma omp task depend( inout                                               \
                         : aux_dep_m [idx_m_parent:1], aux_dep_m [idx_m:1] ) \
  priority( 500 )
              m_list_task< run_count >(
                x, current_cluster, _verbose, verbose_file );
            } else {
              // parent is not in the m list, no dependency needed
#pragma omp task depend( inout : aux_dep_m [idx_m:1] ) priority( 500 )
              m_list_task< run_count >(
                x, current_cluster, _verbose, verbose_file );
            }
            break;
          }
          case 2: {
            // L-list task
            l_list.erase( it_current_cluster );
            lo idx_l = current_cluster->get_pos_in_l_list( );
            lo idx_s2l = current_cluster->get_pos_in_s2l_list( );
            // l-list task depends on previously generated m2l or s2l
            // tasks processing the same cluster (to prevent collisions)
            if ( idx_s2l == -1 ) {
              // cluster is not in the s2l-list, so no dependency on that
#pragma omp task depend( inout : aux_dep_l [idx_l:1] ) priority( 400 )
              l_list_task< run_count >(
                y_pFMM, current_cluster, _verbose, verbose_file );
            } else {
              // cluster depends on previous s2l-list tasks in addition
#pragma omp task depend( inout                                            \
                         : aux_dep_l [idx_l:1], aux_dep_s2l [idx_s2l:1] ) \
  priority( 400 )
              l_list_task< run_count >(
                y_pFMM, current_cluster, _verbose, verbose_file );
            }
            break;
          }

          case 3: {
            // S2L-list task
            s2l_list.erase( it_current_cluster );
            lo idx_l = current_cluster->get_pos_in_l_list( );
            lo idx_s2l = current_cluster->get_pos_in_s2l_list( );
            // s2l-list task depends on previously generated l or m2l
            // tasks processing the same cluster (to prevent collisions)
            if ( idx_l == -1 ) {
              // cluster is not in the l-list, so no dependency on that
#pragma omp task depend( inout : aux_dep_s2l [idx_s2l:1] ) priority( 300 )
              s2l_list_task< run_count >(
                x, y_pFMM, current_cluster, _verbose, verbose_file );
            } else {
#pragma omp task depend( inout                                            \
                         : aux_dep_l [idx_l:1], aux_dep_s2l [idx_s2l:1] ) \
  priority( 300 )
              s2l_list_task< run_count >(
                x, y_pFMM, current_cluster, _verbose, verbose_file );
            }
            break;
          }

          case 4: {
            // M2L-list task

            lo idx_l = current_cluster->get_pos_in_l_list( );
            lo idx_s2l = current_cluster->get_pos_in_s2l_list( );
            lou ready_int_list_size
              = current_cluster->get_n_ready_m2l_sources( );
            if ( ready_int_list_size
              == current_cluster->get_interaction_list( )->size( ) ) {
              m2l_list.erase( it_current_cluster );
            }
            // schedule the m2l-list task with the appropriate
            // dependencies
            if ( idx_l == -1 ) {
              if ( idx_s2l == -1 ) {
                // cluster is not in the s2l or l-list; no dependencies
#pragma omp task priority( 300 )
                m2l_list_task< run_count >(
                  y_pFMM, current_cluster, _verbose, verbose_file );
              } else {
                // dependent on operations in s2l-list
#pragma omp task depend( inout : aux_dep_s2l [idx_s2l:1] ) priority( 300 )
                m2l_list_task< run_count >(
                  y_pFMM, current_cluster, _verbose, verbose_file );
              }
            } else {
              if ( idx_s2l == -1 ) {
                // dependent on operations in l-list
#pragma omp task depend( inout : aux_dep_l [idx_l:1] ) priority( 300 )
                m2l_list_task< run_count >(
                  y_pFMM, current_cluster, _verbose, verbose_file );
              } else {
                // dependent on operations in l- and s2l-list
#pragma omp task depend( inout                                            \
                         : aux_dep_l [idx_l:1], aux_dep_s2l [idx_s2l:1] ) \
  priority( 300 )
                m2l_list_task< run_count >(
                  y_pFMM, current_cluster, _verbose, verbose_file );
              }
            }
            break;
          }

          case 5: {
            // M2T task
            m2t_list.erase( it_current_cluster );
            // no dependencies, possible collision are treated by atomic
            // operations
#pragma omp task priority( 200 )
            m2t_list_task< run_count >(
              y_pFMM, current_cluster, _verbose, verbose_file );
            break;
          }
          case 6: {
            // nearfield task
            n_list.erase( it_current_cluster );
            // no dependencies, possible collisions are treated by atomic
            // operations
#pragma omp task priority( 200 )
            apply_nearfield_operations(
              current_cluster, x, trans, y_pFMM, _verbose, verbose_file );
            break;
          }
        }
      }
    }
  }
  // set loop timer end
  time_type::rep loop_end = 0;
  if ( _measure_tasks ) {
    loop_end = _global_timer.get_time_from_start< time_type >( );
  }
  delete[] aux_dep_m;
  delete[] aux_dep_l;
  delete[] aux_dep_m2l;
  delete[] aux_dep_m2l_send;
  delete[] aux_dep_s2l;

  // print out task timing
  if ( _measure_tasks ) {
    save_times< run_count >( loop_end - loop_start,
      _global_timer.get_time_from_start< time_type >( ), scheduling_thread );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_nearfield_on_the_fly( const assembler_type &
                                                matrix_assembler,
  const distributed_block_vector & x, distributed_block_vector & y,
  sc alpha ) const {
  // Nearfield matrices are assembled and directly applied for all clusters
  // which are leaves or have non-empty spatially admissible nearfield lists.
  // All these clusters are sorted by the size of the matrices in their
  // nearfield lists (including spatially admissible) first.
  std::vector< lo > total_sizes(
    _clusters_with_nearfield_operations.size( ), 0 );
  for ( std::vector< general_spacetime_cluster * >::size_type cluster_index = 0;
        cluster_index < _clusters_with_nearfield_operations.size( );
        ++cluster_index ) {
    mesh::general_spacetime_cluster * current_cluster
      = _clusters_with_nearfield_operations[ cluster_index ];
    lo n_dofs_target = current_cluster->get_n_dofs< target_space >( );
    if ( current_cluster->get_n_children( ) == 0 ) {
      std::vector< general_spacetime_cluster * > * nearfield_list
        = current_cluster->get_nearfield_list( );
      for ( std::vector< general_spacetime_cluster * >::size_type src_index = 0;
            src_index < nearfield_list->size( ); ++src_index ) {
        general_spacetime_cluster * nearfield_cluster
          = ( *nearfield_list )[ src_index ];
        lo n_dofs_source = nearfield_cluster->get_n_dofs< source_space >( );
        total_sizes[ cluster_index ] += n_dofs_source * n_dofs_target;
      }
    }
    auto spat_adm_nf_list
      = current_cluster->get_spatially_admissible_nearfield_list( );
    if ( spat_adm_nf_list != nullptr ) {
      for ( auto nf_cluster : *spat_adm_nf_list ) {
        lo n_dofs_source = nf_cluster->get_n_dofs< source_space >( );
        total_sizes[ cluster_index ] += n_dofs_source * n_dofs_target;
      }
    }
  }
  std::vector< lo > permutation_index(
    _clusters_with_nearfield_operations.size( ), 0 );
  for ( lo i = 0; i != lo( permutation_index.size( ) ); i++ ) {
    permutation_index[ i ] = i;
  }
  sort( permutation_index.begin( ), permutation_index.end( ),
    [ & ]( const lo & a, const lo & b ) {
      return ( total_sizes[ a ] > total_sizes[ b ] );
    } );

  // parallel on the fly assembly and application of the nearfield matrices:
#pragma omp parallel for schedule( dynamic, 1 )
  for ( lou cluster_index = 0;
        cluster_index < _clusters_with_nearfield_operations.size( );
        ++cluster_index ) {
    mesh::general_spacetime_cluster * current_cluster
      = _clusters_with_nearfield_operations
        [ permutation_index[ cluster_index ] ];
    lo n_dofs_target = current_cluster->get_n_dofs< target_space >( );
    // allocate a local target and locala source vector
    vector_type local_target_vector( n_dofs_target, true );
    vector_type local_source_vector;
    // allocate a full_matrix which will be resized and filled anew for each
    // new nearfield cluster.
    full_matrix current_block( n_dofs_target, n_dofs_target, false );
    if ( current_cluster->get_n_children( ) == 0 ) {
      std::vector< general_spacetime_cluster * > * nearfield_list
        = current_cluster->get_nearfield_list( );
      for ( std::vector< general_spacetime_cluster * >::size_type src_index = 0;
            src_index < nearfield_list->size( ); ++src_index ) {
        mesh::general_spacetime_cluster * nearfield_cluster
          = ( *nearfield_list )[ src_index ];
        lo n_dofs_source = nearfield_cluster->get_n_dofs< source_space >( );
        // resize the full matrix current_block, and assemble it appropriately
        current_block.resize( n_dofs_target, n_dofs_source, false );
        matrix_assembler.assemble_nearfield_block(
          current_cluster, nearfield_cluster, current_block );
        // resize and fill the local source vector appropriately
        local_source_vector.resize( n_dofs_source );
        x.get_local_part< source_space >(
          nearfield_cluster, local_source_vector );
        // apply the current block to the local source vector appropriately.
        current_block.apply(
          local_source_vector, local_target_vector, false, alpha, 1.0 );
      }
    }

    std::vector< general_spacetime_cluster * > * spat_adm_nearfield_list
      = current_cluster->get_spatially_admissible_nearfield_list( );
    if ( spat_adm_nearfield_list != nullptr ) {
      for ( std::vector< general_spacetime_cluster * >::size_type src_index = 0;
            src_index < spat_adm_nearfield_list->size( ); ++src_index ) {
        general_spacetime_cluster * nearfield_cluster
          = ( *spat_adm_nearfield_list )[ src_index ];
        lo n_dofs_source = nearfield_cluster->get_n_dofs< source_space >( );
        // resize the full matrix current_block, and assemble it appropriately
        current_block.resize( n_dofs_target, n_dofs_source, false );
        matrix_assembler.assemble_nearfield_block(
          current_cluster, nearfield_cluster, current_block );
        // resize and fill the local source vector appropriately
        local_source_vector.resize( n_dofs_source );
        x.get_local_part< source_space >(
          nearfield_cluster, local_source_vector );
        // apply the current block to the local source vector appropriately.
        current_block.apply(
          local_source_vector, local_target_vector, false, alpha, 1.0 );
      }
    }
    y.add_local_part< target_space >( current_cluster, local_target_vector );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::upward_path_task( lou current_index,
  mesh::scheduling_time_cluster * current_cluster ) const {
  lo source_id = _receive_data_information[ current_index ].second;
  sc * current_moments = current_cluster->get_associated_moments( );
  sc * received_moments
    = current_cluster->get_extraneous_moment_pointer( source_id );
  lou n_assoc_clusters_with_moments
    = current_cluster->get_n_st_clusters_w_moments( );
  for ( lou j = 0; j < n_assoc_clusters_with_moments * _contribution_size;
        ++j ) {
    current_moments[ j ] += received_moments[ j ];
  }
  current_cluster->reduce_upward_path_counter( );
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::m_list_task( const distributed_block_vector & x,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _m_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  call_s2m_operations< run_count >( x, current_cluster, verbose, verbose_file );
  provide_spatial_moments_for_hybrid_s2l(
    current_cluster, verbose, verbose_file );
  provide_moments_for_m2l_or_m2t( current_cluster, verbose, verbose_file );
  call_m2m_operations( current_cluster, verbose, verbose_file );

  provide_moments_to_parents( current_cluster, verbose, verbose_file );
  reduce_nn_operations( );
  if ( _measure_tasks ) {
    _m_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::call_s2m_operations( const distributed_block_vector & sources,
  besthea::mesh::scheduling_time_cluster * t_cluster, bool verbose,
  const std::string & verbose_file ) const {
  // execute only for associated spacetime leaves
  if ( t_cluster->get_n_associated_leaves( ) > 0 ) {
    std::vector< general_spacetime_cluster * > * associated_spacetime_clusters
      = t_cluster->get_associated_spacetime_clusters( );
    lo max_relative_space_level_s2ms
      = t_cluster->get_max_relative_space_level_s2ms( );
    // if max_relative_space_level_s2ms is negative (i.e. it was not
    // initialized explicitly) there are no spatially refined clusters for
    // which spatial moments have to be computed, so we can compute moments of
    // original leaf clusters directly.
    if ( max_relative_space_level_s2ms < 0 ) {
      if ( verbose ) {
#pragma omp critical( verbose )
        {
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << "call direct S2M for cluster "
                    << t_cluster->get_global_index( ) << " at level "
                    << t_cluster->get_level( ) << std::endl;
            outfile.close( );
          }
        }
      }
      // there is an implicit taskgroup after this taskloop
#pragma omp taskloop shared( sources )
      for ( lou i = 0; i < t_cluster->get_n_associated_leaves( ); ++i ) {
        if ( _measure_tasks ) {
          _m_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        general_spacetime_cluster * current_cluster
          = ( *associated_spacetime_clusters )[ i ];

        apply_s2m_operation< run_count >( sources, current_cluster );
        if ( _measure_tasks ) {
          _m_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
      }
    } else {
      if ( verbose ) {
#pragma omp critical( verbose )
        {
          std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
          if ( outfile.is_open( ) ) {
            outfile << "call S2M for cluster " << t_cluster->get_global_index( )
                    << " at level " << t_cluster->get_level( )
                    << " via spatially refined clusters" << std::endl;
            outfile.close( );
          }
        }
      }
      // preparatory step to access associated auxiliary space-time clusters
      // in a levelwise manner.
      const std::vector< lo > * n_leaves_and_aux_cluster_per_level
        = t_cluster->get_n_associated_leaves_and_aux_clusters_per_level( );
      lo offset = 0;
      for ( lo i = 0; i < max_relative_space_level_s2ms; ++i ) {
        offset += ( *n_leaves_and_aux_cluster_per_level )[ i ];
      }
      // first, apply auxiliary s2ms operations for the clusters in the list
      // of associated auxiliary s2ms clusters.
      const std::vector< std::pair< lo, general_spacetime_cluster * > > *
        assoc_aux_s2ms_cluster_pairs
        = t_cluster->get_assoc_aux_s2ms_cluster_pairs( );
      if ( assoc_aux_s2ms_cluster_pairs != nullptr ) {
        // auxiliary s2ms operations
#pragma omp taskloop shared( sources, assoc_aux_s2ms_cluster_pairs )
        for ( lou i = 0; i < assoc_aux_s2ms_cluster_pairs->size( ); ++i ) {
          if ( _measure_tasks ) {
            _m_subtask_times.at( omp_get_thread_num( ) )
              .push_back( _global_timer.get_time_from_start< time_type >( ) );
          }

          lo current_source_index
            = ( *assoc_aux_s2ms_cluster_pairs )[ i ].first;
          general_spacetime_cluster * current_coarse_source_cluster
            = ( *assoc_aux_s2ms_cluster_pairs )[ i ].second;
          general_spacetime_cluster * current_source_cluster
            = ( *associated_spacetime_clusters )[ current_source_index ];
          apply_s2ms_operation< run_count >(
            sources, current_source_cluster, current_coarse_source_cluster );

          if ( _measure_tasks ) {
            _m_subtask_times.at( omp_get_thread_num( ) )
              .push_back( _global_timer.get_time_from_start< time_type >( ) );
          }
        }
      }

      // add up the spatial moments obtained by the auxiliary s2ms operations
      // (for all non-leaf clusters) or generate the spatial moments for
      // clusters at max_relative_space_level_s2ms appropriately.
#pragma omp taskloop shared( n_leaves_and_aux_cluster_per_level )
      for ( lo i = 0; i < ( *n_leaves_and_aux_cluster_per_level )
                        [ max_relative_space_level_s2ms ];
            ++i ) {
        if ( _measure_tasks ) {
          _m_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        general_spacetime_cluster * current_source_cluster
          = ( *associated_spacetime_clusters )[ offset + i ];
        if ( current_source_cluster->get_n_children( ) > 0 ) {
          sum_up_auxiliary_spatial_moments( current_source_cluster );
        } else {
          apply_s2ms_operation< run_count >( sources, current_source_cluster );
        }
        if ( _measure_tasks ) {
          _m_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
      }

      // pass spatial moments upwards by ms2ms operations (spatial moment to
      // spatial moment)
      for ( lo rel_space_level = max_relative_space_level_s2ms - 1;
            rel_space_level >= 0; --rel_space_level ) {
        offset -= ( *n_leaves_and_aux_cluster_per_level )[ rel_space_level ];
#pragma omp taskloop shared( sources, n_leaves_and_aux_cluster_per_level )
        for ( lo i = 0;
              i < ( *n_leaves_and_aux_cluster_per_level )[ rel_space_level ];
              ++i ) {
          if ( _measure_tasks ) {
            _m_subtask_times.at( omp_get_thread_num( ) )
              .push_back( _global_timer.get_time_from_start< time_type >( ) );
          }
          general_spacetime_cluster * current_cluster
            = ( *associated_spacetime_clusters )[ offset + i ];
          // if current_cluster is not a leaf, apply an ms2ms operation for
          // each of its children; otherwise s2ms operations.
          if ( current_cluster->get_n_children( ) > 0 ) {
            for ( auto child : *current_cluster->get_children( ) ) {
              apply_ms2ms_operation( child, current_cluster );
            }
          } else {
            apply_s2ms_operation< run_count >( sources, current_cluster );
          }
          if ( _measure_tasks ) {
            _m_subtask_times.at( omp_get_thread_num( ) )
              .push_back( _global_timer.get_time_from_start< time_type >( ) );
          }
        }
      }
      // transform the spatial moments of the regular clusters (relative space
      // level 0) into space-time moments
#pragma omp taskloop shared( sources, n_leaves_and_aux_cluster_per_level )
      for ( lo i = 0; i < ( *n_leaves_and_aux_cluster_per_level )[ 0 ]; ++i ) {
        if ( _measure_tasks ) {
          _m_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
        general_spacetime_cluster * current_cluster
          = ( *associated_spacetime_clusters )[ i ];
        apply_ms2m_operation< run_count >( current_cluster );
        if ( _measure_tasks ) {
          _m_subtask_times.at( omp_get_thread_num( ) )
            .push_back( _global_timer.get_time_from_start< time_type >( ) );
        }
      }
    }
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2m_operation( const distributed_block_vector &
  /*source_vector*/,
  general_spacetime_cluster * /*source_cluster*/ ) const {
  std::cout << "General S2M operation not implemented " << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_s2m_operation< 0 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p0( source_vector, source_cluster );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2m_operation< 0 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operations_p1_normal_drv( source_vector, source_cluster );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_s2m_operation< 0 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p0( source_vector, source_cluster );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 0
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2m_operation< 0 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_curl_p1_hs< 0 >( source_vector, source_cluster );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 1
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2m_operation< 1 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_curl_p1_hs< 1 >( source_vector, source_cluster );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 2
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2m_operation< 2 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_curl_p1_hs< 2 >( source_vector, source_cluster );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 3
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2m_operation< 3 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p1_normal_hs( source_vector, source_cluster, 0 );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 4
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2m_operation< 4 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p1_normal_hs( source_vector, source_cluster, 1 );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 5
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_s2m_operation< 5 >( const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  apply_s2m_operation_p1_normal_hs( source_vector, source_cluster, 2 );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_s2m_operation_p0( const distributed_block_vector &
                                            source_vector,
  general_spacetime_cluster * source_cluster ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  full_matrix sources;
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references of current moment and all required matrices
  sc * moment = source_cluster->get_pointer_to_moment( );

  full_matrix T;
  compute_chebyshev_quadrature_p0( T, source_cluster );
  full_matrix L;
  compute_lagrange_quadrature( L, source_cluster );

  // get the relevant entries of the source vector and store them in
  // sources
  source_vector.get_local_part< source_space >( source_cluster, sources );

  // compute D = Q * T and then the moment mu = L * D
  aux_matrix.multiply( sources, T );
  // mu = L * D with explicit cblas routine call
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_cols_aux_matrix = aux_matrix.get_n_columns( );
  lo n_rows_aux_matrix = aux_matrix.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_aux_matrix;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange,
    n_cols_aux_matrix, n_rows_aux_matrix, alpha, L.data( ), lda,
    aux_matrix.data( ), ldb, beta, moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_s2m_operations_p1_normal_drv(
    const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  full_matrix sources;
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references of current moment and all required matrices
  sc * moment = source_cluster->get_pointer_to_moment( );
  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1( T_drv, source_cluster );
  full_matrix L;
  compute_lagrange_quadrature( L, source_cluster );

  // get the relevant entries of the source vector and store them in
  // sources
  source_vector.get_local_part< source_space >( source_cluster, sources );

  // compute D = Q * T_drv and then the moment mu = L * D
  aux_matrix.multiply( sources, T_drv );
  // mu = L * D with explicit cblas routine call
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_cols_aux_matrix = aux_matrix.get_n_columns( );
  lo n_rows_aux_matrix = aux_matrix.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_aux_matrix;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange,
    n_cols_aux_matrix, n_rows_aux_matrix, alpha, L.data( ), lda,
    aux_matrix.data( ), ldb, beta, moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
template< slou dim >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_s2m_operation_curl_p1_hs(
    const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  full_matrix sources;
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );
  // get the relevant entries of the source vector and store them in
  // sources
  source_vector.get_local_part< source_space >( source_cluster, sources );

  // get references of current moment and all required matrices
  sc * moment = source_cluster->get_pointer_to_moment( );

  full_matrix L;
  compute_lagrange_quadrature( L, source_cluster );
  full_matrix T_curl_along_dim;

  compute_chebyshev_times_p1_surface_curls_along_dimension< dim >(
    T_curl_along_dim, source_cluster );

  // compute D = Q * T_curl_along_dim and then the moment mu = _alpha * L
  // * D
  aux_matrix.multiply( sources, T_curl_along_dim );
  // mu = L * D with explicit cblas routine call
  lo n_rows_lagrange = L.get_n_rows( );
  lo n_cols_aux_matrix = aux_matrix.get_n_columns( );
  lo n_rows_aux_matrix = aux_matrix.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_aux_matrix;
  sc alpha = _alpha;  // heat capacity constant;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange,
    n_cols_aux_matrix, n_rows_aux_matrix, alpha, L.data( ), lda,
    aux_matrix.data( ), ldb, beta, moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_s2m_operation_p1_normal_hs(
    const distributed_block_vector & source_vector,
    general_spacetime_cluster * source_cluster, const slou dimension ) const {
  lo n_time_elements = source_cluster->get_n_time_elements( );
  full_matrix sources;
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );
  // get the relevant entries of the source vector and store them in
  // sources
  source_vector.get_local_part< source_space >( source_cluster, sources );

  // get references of current moment and all required matrices
  sc * moment = source_cluster->get_pointer_to_moment( );

  full_matrix L_drv;
  compute_lagrange_drv_quadrature( L_drv, source_cluster );
  full_matrix T_normal_dim;
  compute_chebyshev_times_normal_quadrature_p1_along_dimension(
    T_normal_dim, dimension, source_cluster );

  // compute D = Q * T_normal_dim and then the moment mu = _alpha * L_drv
  // * D
  aux_matrix.multiply( sources, T_normal_dim );
  // mu = _alpha * L_drv * D with explicit cblas routine call
  lo n_rows_lagrange = L_drv.get_n_rows( );
  lo n_cols_aux_matrix = aux_matrix.get_n_columns( );
  lo n_rows_aux_matrix = aux_matrix.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_aux_matrix;
  sc alpha = _alpha;  // heat capacity constant
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows_lagrange,
    n_cols_aux_matrix, n_rows_aux_matrix, alpha, L_drv.data( ), lda,
    aux_matrix.data( ), ldb, beta, moment, n_rows_lagrange );
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::apply_l2t_operation( const mesh::general_spacetime_cluster *
  /*st_cluster*/,
  distributed_block_vector & /*output_vector*/ ) const {
  std::cout << "General L2T operation not implemented!" << std::endl;
}

//! template specialization for single layer p0p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_l2t_operation< 0 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p0( st_cluster, output_vector );
}

//! template specialization for double layer p0p1 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation< 0 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p0( st_cluster, output_vector );
}

//! template specialization for adjoint double layer p1p0 matrix
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  apply_l2t_operation< 0 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p1_normal_drv( st_cluster, output_vector );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 0
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation< 0 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_curl_p1_hs< 0 >( st_cluster, output_vector );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 0
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation< 1 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_curl_p1_hs< 1 >( st_cluster, output_vector );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 0
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation< 2 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_curl_p1_hs< 2 >( st_cluster, output_vector );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 0
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation< 3 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p1_normal_hs( st_cluster, 0, output_vector );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 0
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation< 4 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p1_normal_hs( st_cluster, 1, output_vector );
}

//! template specialization for hypersingular p1p1 matrix, run_count = 0
template<>
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >::
  apply_l2t_operation< 5 >( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  apply_l2t_operation_p1_normal_hs( st_cluster, 2, output_vector );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p0( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  lo n_time_elements = st_cluster->get_n_time_elements( );
  lo n_space_elements = st_cluster->get_n_space_elements( );
  full_matrix targets( n_time_elements, n_space_elements, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution
    = st_cluster->get_pointer_to_local_contribution( );

  full_matrix T;
  compute_chebyshev_quadrature_p0( T, st_cluster );
  full_matrix L;
  compute_lagrange_quadrature( L, st_cluster );

  // compute D = trans(L) * lambda and then the result Y = D * trans(T)
  //  D = trans(L) * lambda with explicit cblas routine call:
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo n_rows_lagrange = L.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_lagrange,
    n_cols_local, n_rows_lagrange, alpha, L.data( ), lda, local_contribution,
    ldb, beta, aux_matrix.data( ), n_cols_lagrange );
  // compute Y = D * trans(T)
  targets.multiply( aux_matrix, T, false, true );

  // add the results to the correct positions of the output vector
  output_vector.add_local_part< target_space >( st_cluster, targets );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  lo n_time_elements = st_cluster->get_n_time_elements( );
  lo n_space_nodes = st_cluster->get_n_space_nodes( );
  full_matrix targets( n_time_elements, n_space_nodes, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution
    = st_cluster->get_pointer_to_local_contribution( );
  full_matrix T_drv;
  compute_normal_drv_chebyshev_quadrature_p1( T_drv, st_cluster );
  full_matrix L;
  compute_lagrange_quadrature( L, st_cluster );

  // compute D = trans(L) * lambda and then the result Y = D *
  // trans(T_drv)
  //  D = trans(L) * lambda with explicit cblas routine call:
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo n_rows_lagrange = L.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_lagrange,
    n_cols_local, n_rows_lagrange, alpha, L.data( ), lda, local_contribution,
    ldb, beta, aux_matrix.data( ), n_cols_lagrange );
  // compute Y = D * trans(T_drv)
  targets.multiply( aux_matrix, T_drv, false, true );

  // add the results to the correct positions of the output vector
  output_vector.add_local_part< target_space >( st_cluster, targets );
}

template< class kernel_type, class target_space, class source_space >
template< slou dim >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_curl_p1_hs(
    const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const {
  lo n_time_elements = st_cluster->get_n_time_elements( );
  lo n_space_nodes = st_cluster->get_n_space_nodes( );
  full_matrix targets( n_time_elements, n_space_nodes, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution
    = st_cluster->get_pointer_to_local_contribution( );

  full_matrix L;
  compute_lagrange_quadrature( L, st_cluster );
  full_matrix T_curl_along_dim;
  compute_chebyshev_times_p1_surface_curls_along_dimension< dim >(
    T_curl_along_dim, st_cluster );

  // compute D = _alpha * trans(L) * lambda and then the result
  // Y = D * trans(T_curl_along_dim)
  //  D = _alpha * trans(L) * lambda with explicit cblas routine call:
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo n_rows_lagrange = L.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = _alpha;  // heat capacity constant
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_lagrange,
    n_cols_local, n_rows_lagrange, alpha, L.data( ), lda, local_contribution,
    ldb, beta, aux_matrix.data( ), n_cols_lagrange );
  // compute Y = D * trans(T_curl_along_dim)
  targets.multiply( aux_matrix, T_curl_along_dim, false, true );

  // add the results to the correct positions of the output vector
  output_vector.add_local_part< target_space >( st_cluster, targets );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_l2t_operation_p1_normal_hs(
    const mesh::general_spacetime_cluster * st_cluster, const slou dimension,
    distributed_block_vector & output_vector ) const {
  lo n_time_elements = st_cluster->get_n_time_elements( );
  lo n_space_nodes = st_cluster->get_n_space_nodes( );
  full_matrix targets( n_time_elements, n_space_nodes, false );
  full_matrix aux_matrix( n_time_elements, _spat_contribution_size, false );

  // get references local contribution and all required matrices
  const sc * local_contribution
    = st_cluster->get_pointer_to_local_contribution( );

  full_matrix L;
  compute_lagrange_quadrature( L, st_cluster );
  full_matrix T_normal_dim;
  compute_chebyshev_times_normal_quadrature_p1_along_dimension(
    T_normal_dim, dimension, st_cluster );

  // compute D = trans(L) * lambda and then the result
  // Y = D * trans(T_normal_dim)
  //  D = trans(L) * lambda with explicit cblas routine call:
  lo n_cols_lagrange = L.get_n_columns( );
  lo n_cols_local = _spat_contribution_size;
  lo n_rows_lagrange = L.get_n_rows( );
  lo lda = n_rows_lagrange;
  lo ldb = n_rows_lagrange;
  sc alpha = 1.0;
  sc beta = 0.0;
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, n_cols_lagrange,
    n_cols_local, n_rows_lagrange, alpha, L.data( ), lda, local_contribution,
    ldb, beta, aux_matrix.data( ), n_cols_lagrange );
  // compute Y = D * trans(T_normal_dim)
  targets.multiply( aux_matrix, T_normal_dim, false, true );

  // add the results to the correct positions of the output vector
  output_vector.add_local_part< target_space >( st_cluster, targets );
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::m2l_list_task( distributed_block_vector &
                                                 y_pFMM,
  besthea::mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _m2l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  std::vector< scheduling_time_cluster * > * ready_interaction_list
    = current_cluster->get_interaction_list( );
  //  for ( slou i = current_cluster->get_m2l_counter( );
  for ( slou i = 0; i < current_cluster->get_n_ready_m2l_sources( ); ++i ) {
    call_m2l_operations( ( *ready_interaction_list )[ i ], current_cluster,
      verbose, verbose_file );
    current_cluster->set_m2l_counter( (slou) ( i + 1 ) );
  }
  // check if all the m2l operations have been executed yet
  if ( current_cluster->get_m2l_counter( )
    == current_cluster->get_interaction_list( )->size( ) ) {
    // check if all l-list, s2l-list and m2t-list operations have been
    // executed. If yes, execute l2t operations or downward send operations if
    // necessary
    if ( current_cluster->get_downward_path_status( ) == 1
      && current_cluster->get_s2l_execution_status( ) == 2
      && current_cluster->get_m2t_execution_status( ) == 1 ) {
      // set status of the current cluster's local contributions to
      // completed
      current_cluster->set_downward_path_status( 2 );
      call_l2t_operations< run_count >(
        current_cluster, y_pFMM, verbose, verbose_file );
      provide_local_contributions_to_children(
        current_cluster, verbose, verbose_file );
    }
  }
  reduce_nn_operations( );
  if ( _measure_tasks ) {
    _m2l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::m2t_list_task( distributed_block_vector &
                                                 y_pFMM,
  mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _m2t_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  call_m2t_operations< run_count >(
    current_cluster, y_pFMM, verbose, verbose_file );
  current_cluster->set_m2t_execution_status( 1 );

  if ( ( current_cluster->get_interaction_list( ) == nullptr
         || current_cluster->get_m2l_counter( )
           == current_cluster->get_interaction_list( )->size( ) )
    && current_cluster->get_downward_path_status( ) == 1
    && current_cluster->get_s2l_execution_status( ) == 2 ) {
    // set status of the current cluster's local contributions to
    // completed
    current_cluster->set_downward_path_status( 2 );
    call_l2t_operations< run_count >(
      current_cluster, y_pFMM, verbose, verbose_file );
    provide_local_contributions_to_children(
      current_cluster, verbose, verbose_file );
  }
  reduce_nn_operations( );
  if ( _measure_tasks ) {
    _m2t_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::s2l_list_task( const distributed_block_vector &
                                                 x,
  distributed_block_vector & y_pFMM,
  mesh::scheduling_time_cluster * current_cluster, bool verbose,
  const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _s2l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  call_s2l_operations< run_count >( x, current_cluster, verbose, verbose_file );
  current_cluster->set_s2l_execution_status( 2 );
  // Check if all the m2l-list, l-list and m2t-list operations have been
  // executed yet. If yes, execute l2t operations or downward send operations
  // if necessary
  if ( ( current_cluster->get_interaction_list( ) == nullptr
         || current_cluster->get_m2l_counter( )
           == current_cluster->get_interaction_list( )->size( ) )
    && current_cluster->get_downward_path_status( ) == 1
    && current_cluster->get_m2t_execution_status( ) == 1 ) {
    // set status of the current cluster's local contributions to
    // completed
    current_cluster->set_downward_path_status( 2 );
    call_l2t_operations< run_count >(
      current_cluster, y_pFMM, verbose, verbose_file );
    provide_local_contributions_to_children(
      current_cluster, verbose, verbose_file );
  }

  reduce_nn_operations( );
  if ( _measure_tasks ) {
    _s2l_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::
  apply_nearfield_operations( const mesh::scheduling_time_cluster * t_cluster,
    const distributed_block_vector & sources, bool trans,
    distributed_block_vector & output_vector, bool verbose,
    const std::string & verbose_file ) const {
  if ( _measure_tasks ) {
    _n_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
  if ( verbose ) {
#pragma omp critical( verbose )
    {
      std::ofstream outfile( verbose_file.c_str( ), std::ios::app );
      if ( outfile.is_open( ) ) {
        outfile << "apply NF operations for cluster "
                << t_cluster->get_global_index( ) << " at level "
                << t_cluster->get_level( ) << std::endl;
        outfile.close( );
      }
    }
  }
  const std::vector< general_spacetime_cluster * > *
    associated_spacetime_targets
    = t_cluster->get_associated_spacetime_clusters( );
  const std::vector< lo > * assoc_nearfield_targets
    = t_cluster->get_assoc_nearfield_targets( );

  // there is an implicit taskgroup associated with this taskloop
#pragma omp taskloop shared( output_vector, _clusterwise_nf_matrices, \
  _clusterwise_spat_adm_nf_matrix_pairs, assoc_nearfield_targets )
  for ( lou i = 0; i < assoc_nearfield_targets->size( ); ++i ) {
    if ( _measure_tasks ) {
      _n_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
    vector_type local_sources;
    general_spacetime_cluster * current_spacetime_target
      = ( *associated_spacetime_targets )[ ( *assoc_nearfield_targets )[ i ] ];
    // construct a local result_vector
    vector_type local_result(
      current_spacetime_target->get_n_dofs< target_space >( ), true );
    // get the spatially admissible nearfield list of the current spacetime
    // target cluster. if it is not empty, apply the nearfield operations for
    // all the clusters in this list.
    std::vector< general_spacetime_cluster * > * st_spat_adm_nearfield_list
      = current_spacetime_target->get_spatially_admissible_nearfield_list( );
    if ( st_spat_adm_nearfield_list != nullptr ) {
      const std::vector< std::pair< lo, matrix * > > & spat_adm_nf_matrix_pairs
        = _clusterwise_spat_adm_nf_matrix_pairs.at( current_spacetime_target );
      for ( lou pair_index = 0; pair_index < spat_adm_nf_matrix_pairs.size( );
            ++pair_index ) {
        lo src_index = spat_adm_nf_matrix_pairs[ pair_index ].first;
        general_spacetime_cluster * current_spacetime_source
          = ( *st_spat_adm_nearfield_list )[ src_index ];
        local_sources.resize(
          current_spacetime_source->get_n_dofs< source_space >( ) );
        // get the sources corresponding to the current spacetime source
        // cluster
        sources.get_local_part< source_space >(
          current_spacetime_source, local_sources );

        matrix * current_block = spat_adm_nf_matrix_pairs[ pair_index ].second;
        // apply the approximated nearfield matrix (or the full, if
        // compression was not successful) and add the result to local_result
        current_block->apply( local_sources, local_result, trans, 1.0, 1.0 );
      }
    }
    if ( current_spacetime_target->get_n_children( ) == 0 ) {
      // get the nearfield list of the current spacetime target cluster and
      // apply the nearfield operations for all the clusters in this list.
      std::vector< general_spacetime_cluster * > * spacetime_nearfield_list
        = current_spacetime_target->get_nearfield_list( );

      for ( lou src_index = 0; src_index < spacetime_nearfield_list->size( );
            ++src_index ) {
        general_spacetime_cluster * current_spacetime_source
          = ( *spacetime_nearfield_list )[ src_index ];
        local_sources.resize(
          current_spacetime_source->get_n_dofs< source_space >( ) );
        // get the sources corresponding to the current spacetime source
        // cluster
        sources.get_local_part< source_space >(
          current_spacetime_source, local_sources );

        full_matrix * current_block = _clusterwise_nf_matrices.at(
          current_spacetime_target )[ src_index ];
        // apply the nearfield matrix and add the result to local_result
        current_block->apply( local_sources, local_result, trans, 1.0, 1.0 );
      }
    }
    // add the local result to the output vector
    output_vector.add_local_part< target_space >(
      current_spacetime_target, local_result );
    if ( _measure_tasks ) {
      _n_subtask_times.at( omp_get_thread_num( ) )
        .push_back( _global_timer.get_time_from_start< time_type >( ) );
    }
  }
  if ( _measure_tasks ) {
    _n_task_times.at( omp_get_thread_num( ) )
      .push_back( _global_timer.get_time_from_start< time_type >( ) );
  }
}

template< class kernel_type, class target_space, class source_space >
template< slou run_count >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::save_times( time_type::rep total_loop_duration,
  time_type::rep total_apply_duration, lo scheduling_thread ) const {
  std::filesystem::create_directory( "./task_timer/" );

  std::string scheduling_thread_file = "task_timer/scheduling_threads_process_";
  scheduling_thread_file += std::to_string( _my_rank );
  scheduling_thread_file += ".txt";
  if ( run_count == 0 ) {
    remove( scheduling_thread_file.c_str( ) );
  }

  std::ofstream outfile_txt( scheduling_thread_file.c_str( ), std::ios::app );

  if ( outfile_txt.is_open( ) ) {
    outfile_txt << "Scheduling thread in run " << run_count << ": "
                << scheduling_thread << std::endl;
  }

  std::string timer_file = "task_timer/process_";
  timer_file += std::to_string( _my_rank );
  timer_file += ".m";
  remove( timer_file.c_str( ) );

  std::ofstream outfile( timer_file.c_str( ), std::ios::app );

  if ( outfile.is_open( ) ) {
    outfile << "% Total apply duration [us]: " << std::endl;
    outfile << "T = " << total_apply_duration << ";" << std::endl;

    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      // compute thread total execution time in individual tasks
      time_type::rep us_m_sub = 0;
      time_type::rep us_m2l_sub = 0;
      time_type::rep us_l_sub = 0;
      time_type::rep us_s2l_sub = 0;
      time_type::rep us_m2t_sub = 0;
      time_type::rep us_n_sub = 0;
      time_type::rep total_time = 0;
      for ( std::size_t j = 0; j < _m_subtask_times.at( i ).size( ) / 2; ++j ) {
        us_m_sub += _m_subtask_times.at( i ).at( 2 * j + 1 )
          - _m_subtask_times.at( i ).at( 2 * j );
      }
      for ( std::size_t j = 0; j < _m2l_subtask_times.at( i ).size( ) / 2;
            ++j ) {
        us_m2l_sub += _m2l_subtask_times.at( i ).at( 2 * j + 1 )
          - _m2l_subtask_times.at( i ).at( 2 * j );
      }
      for ( std::size_t j = 0; j < _l_subtask_times.at( i ).size( ) / 2; ++j ) {
        us_l_sub += _l_subtask_times.at( i ).at( 2 * j + 1 )
          - _l_subtask_times.at( i ).at( 2 * j );
      }
      for ( std::size_t j = 0; j < _s2l_subtask_times.at( i ).size( ) / 2;
            ++j ) {
        us_s2l_sub += _s2l_subtask_times.at( i ).at( 2 * j + 1 )
          - _s2l_subtask_times.at( i ).at( 2 * j );
      }
      for ( std::size_t j = 0; j < _m2t_subtask_times.at( i ).size( ) / 2;
            ++j ) {
        us_m2t_sub += _m2t_subtask_times.at( i ).at( 2 * j + 1 )
          - _m2t_subtask_times.at( i ).at( 2 * j );
      }
      for ( std::size_t j = 0; j < _n_subtask_times.at( i ).size( ) / 2; ++j ) {
        us_n_sub += _n_subtask_times.at( i ).at( 2 * j + 1 )
          - _n_subtask_times.at( i ).at( 2 * j );
      }

      total_time
        = us_m_sub + us_m2l_sub + us_l_sub + us_s2l_sub + us_m2t_sub + us_n_sub;
      double perc_subtasks_apply
        = (double) total_time / (double) total_apply_duration;
      double perc_subtasks_loop
        = (double) total_time / (double) total_loop_duration;

      outfile << "% Thread " << i << ": " << std::endl;

      outfile << "% M subtasks duration: " << us_m_sub << " us" << std::endl;
      outfile << "% M2L subtasks duration: " << us_m2l_sub << " us"
              << std::endl;
      outfile << "% L subtasks duration: " << us_l_sub << " us" << std::endl;
      outfile << "% S2L subtasks duration: " << us_s2l_sub << " us"
              << std::endl;
      outfile << "% M2T subtasks duration: " << us_m2t_sub << " us"
              << std::endl;
      outfile << "% N subtasks duration: " << us_n_sub << " us" << std::endl;
      outfile << "% Sum: " << total_time << " us ("
              << perc_subtasks_loop * 100.0 << " % [loop], "
              << perc_subtasks_apply * 100.0 << " % [total])\n\n";

      // output main tasks
      outfile << "% M tasks: " << std::endl;
      outfile << "M" << i << " = [";
      auto it = _m_task_times.at( i ).begin( );
      for ( ; it != _m_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% M2L tasks: " << std::endl;
      outfile << "M2L" << i << " = [";
      it = _m2l_task_times.at( i ).begin( );
      for ( ; it != _m2l_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% L tasks: " << std::endl;
      outfile << "L" << i << " = [";
      it = _l_task_times.at( i ).begin( );
      for ( ; it != _l_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% S2L tasks: " << std::endl;
      outfile << "S2L" << i << " = [";
      it = _s2l_task_times.at( i ).begin( );
      for ( ; it != _s2l_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% M2T tasks: " << std::endl;
      outfile << "M2T" << i << " = [";
      it = _m2t_task_times.at( i ).begin( );
      for ( ; it != _m2t_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% N tasks: " << std::endl;
      outfile << "N" << i << " = [";
      it = _n_task_times.at( i ).begin( );
      for ( ; it != _n_task_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      // output subtasks
      outfile << "% M subtasks: " << std::endl;
      outfile << "Ms" << i << " = [";
      it = _m_subtask_times.at( i ).begin( );
      for ( ; it != _m_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% M2L subtasks: " << std::endl;
      it = _m2l_subtask_times.at( i ).begin( );
      outfile << "M2Ls" << i << " = [";
      for ( ; it != _m2l_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% L subtasks: " << std::endl;
      it = _l_subtask_times.at( i ).begin( );
      outfile << "Ls" << i << " = [";
      for ( ; it != _l_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% S2L subtasks: " << std::endl;
      it = _s2l_subtask_times.at( i ).begin( );
      outfile << "S2Ls" << i << " = [";
      for ( ; it != _s2l_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% M2T subtasks: " << std::endl;
      it = _m2t_subtask_times.at( i ).begin( );
      outfile << "M2Ts" << i << " = [";
      for ( ; it != _m2t_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% N subtasks: " << std::endl;
      it = _n_subtask_times.at( i ).begin( );
      outfile << "Ns" << i << " = [";
      for ( ; it != _n_subtask_times.at( i ).end( ); ++it ) {
        outfile << *it << ", " << *( ++it ) << "; ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      // output MPI communication
      outfile << "% M2L or M2T send: " << std::endl;
      it = _mpi_send_m2l_m2t_or_s2l.at( i ).begin( );
      outfile << "M2L_or_M2T_send" << i << " = [";
      for ( ; it != _mpi_send_m2l_m2t_or_s2l.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% parent send: " << std::endl;
      it = _mpi_send_m_parent.at( i ).begin( );
      outfile << "parent_send" << i << " = [";
      for ( ; it != _mpi_send_m_parent.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% local send: " << std::endl;
      it = _mpi_send_l_children.at( i ).begin( );
      outfile << "local_send" << i << " = [";
      for ( ; it != _mpi_send_l_children.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% M2L or M2T receive: " << std::endl;
      it = _mpi_recv_m2l_m2t_or_s2l.at( i ).begin( );
      outfile << "M2L_or_M2T_recv" << i << " = [";
      for ( ; it != _mpi_recv_m2l_m2t_or_s2l.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% parent receive: " << std::endl;
      it = _mpi_recv_m_parent.at( i ).begin( );
      outfile << "parent_recv" << i << " = [";
      for ( ; it != _mpi_recv_m_parent.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << std::endl;

      outfile << "% local receive: " << std::endl;
      it = _mpi_recv_l_children.at( i ).begin( );
      outfile << "local_recv" << i << " = [";
      for ( ; it != _mpi_recv_l_children.at( i ).end( ); ++it ) {
        outfile << *it << ", ";
      }
      outfile << " ];";
      outfile << std::endl << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    }
    outfile << "M = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2L = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2L" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "L = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "L" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "S2L = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "S2L" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2T = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2T" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "N = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "N" << i << ", ";
    }
    outfile << "};" << std::endl;

    outfile << "Ms = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "Ms" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2Ls = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2Ls" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "Ls = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "Ls" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "S2Ls = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "S2Ls" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2Ts = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2Ts" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "Ns = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "Ns" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2L_or_M2T_send = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2L_or_M2T_send" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "parent_send = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "parent_send" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "local_send = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "local_send" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "M2L_or_M2T_recv = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "M2L_or_M2T_recv" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "parent_recv = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "parent_recv" << i << ", ";
    }
    outfile << "};" << std::endl;
    outfile << "local_recv = { ";
    for ( lo i = 0; i < omp_get_max_threads( ); ++i ) {
      outfile << "local_recv" << i << ", ";
    }
    outfile << "};" << std::endl;

    outfile.close( );
  }
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space,
  source_space >::sort_clusters_in_n_list_and_associated_st_targets( ) {
  std::unordered_map< mesh::scheduling_time_cluster *, long long int >
    total_nf_sizes_in_n_list;
  for ( auto t_cluster : _n_list ) {
    long long int total_nf_size_current_t_cluster = 0;
    std::vector< general_spacetime_cluster * > * associated_spacetime_targets
      = t_cluster->get_associated_spacetime_clusters( );
    std::vector< long long int > nf_sizes_assoc_st_targets(
      associated_spacetime_targets->size( ), 0 );
    // determine the associated space-time clusters for which nearfield
    // operations have to be executed (currently: all leaves in the extended
    // tree and clusters with non-empty spatially admissible nearfield lists)
    // and count the sizes of the corresponding matrices.
    for ( lou i = 0; i < associated_spacetime_targets->size( ); ++i ) {
      mesh::general_spacetime_cluster * current_spacetime_target
        = ( *associated_spacetime_targets )[ i ];
      if ( current_spacetime_target->get_n_children( ) == 0 ) {
        std::vector< general_spacetime_cluster * > * spacetime_nearfield_list
          = current_spacetime_target->get_nearfield_list( );
        for ( lou src_index = 0; src_index < spacetime_nearfield_list->size( );
              ++src_index ) {
          full_matrix * current_block = _clusterwise_nf_matrices.at(
            current_spacetime_target )[ src_index ];
          lo n_rows = current_block->get_n_rows( );
          lo n_cols = current_block->get_n_columns( );
          nf_sizes_assoc_st_targets[ i ] += n_rows * n_cols;
        }
      }
      const std::vector< general_spacetime_cluster * > *
        current_spat_adm_nearfield
        = current_spacetime_target->get_spatially_admissible_nearfield_list( );
      if ( current_spat_adm_nearfield != nullptr ) {
        const std::vector< std::pair< lo, matrix * > > &
          spat_adm_nf_matrix_pairs
          = _clusterwise_spat_adm_nf_matrix_pairs.at(
            current_spacetime_target );
        for ( lou pair_index = 0; pair_index < spat_adm_nf_matrix_pairs.size( );
              ++pair_index ) {
          matrix * current_block
            = spat_adm_nf_matrix_pairs[ pair_index ].second;
          nf_sizes_assoc_st_targets[ i ]
            += current_block->get_n_stored_entries( );
        }
      }
      total_nf_size_current_t_cluster += nf_sizes_assoc_st_targets[ i ];
    }
    total_nf_sizes_in_n_list.insert(
      { t_cluster, total_nf_size_current_t_cluster } );
    t_cluster->sort_assoc_nearfield_targets( nf_sizes_assoc_st_targets );
  }

  // sort nearfield clusters
  _n_list.sort(
    [ &total_nf_sizes_in_n_list ]( mesh::scheduling_time_cluster * lhs,
      mesh::scheduling_time_cluster * rhs ) {
      return (
        total_nf_sizes_in_n_list[ lhs ] > total_nf_sizes_in_n_list[ rhs ] );
    } );
}

template< class kernel_type, class target_space, class source_space >
void besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
  target_space, source_space >::get_inverse_diagonal( distributed_block_vector &
  /* inverse_diagonal */ ) const {
  std::cout << "ERROR: get_diagonal not implemented for general pFMM matrices"
            << std::endl;
}

/** \cond doxygen should skip the following functions */
template<>
void besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >::
  get_inverse_diagonal( distributed_block_vector & inverse_diagonal ) const {
  const mesh::distributed_spacetime_tensor_mesh & distributed_mesh
    = _distributed_spacetime_tree->get_mesh( );
  const mesh::spacetime_tensor_mesh * local_mesh
    = distributed_mesh.get_local_mesh( );
  lo local_start_idx = distributed_mesh.get_local_start_idx( );
  // resize the distributed vector inverse_diagonal appropriately
  std::vector< lo > local_blocks = distributed_mesh.get_my_timesteps( );
  inverse_diagonal.resize(
    local_blocks, distributed_mesh.get_n_temporal_elements( ) );
  inverse_diagonal.resize_blocks(
    local_mesh->get_n_spatial_elements( ), false );
  for ( auto nearfield_matrix_list_pair : _clusterwise_nf_matrices ) {
    mesh::general_spacetime_cluster * leaf_cluster
      = nearfield_matrix_list_pair.first;
    std::vector< mesh::general_spacetime_cluster * > * nearfield_list
      = leaf_cluster->get_nearfield_list( );
    auto nf_list_it = nearfield_list->begin( );
    // find the position of the leaf cluster in its own nearfield list
    lo position = 0;
    while (
      nf_list_it != nearfield_list->end( ) && *nf_list_it != leaf_cluster ) {
      ++nf_list_it;
      ++position;
    }
    full_matrix * current_nf_matrix
      = nearfield_matrix_list_pair.second[ position ];

    std::vector< lo > all_elements_in_leaf_cluster
      = leaf_cluster->get_all_elements( );

    lo n_time_elements = leaf_cluster->get_n_time_elements( );
    lo n_space_elements = leaf_cluster->get_n_space_elements( );
    const std::vector< lo > & spacetime_elements
      = leaf_cluster->get_all_elements( );
    for ( lo i_time = 0; i_time < n_time_elements; ++i_time ) {
      // use that the spacetime elements are sorted in time, i.e. a
      // consecutive group of n_space_elements elements has the same
      // temporal component, to determine the local time index only once
      lo local_time_index
        = local_mesh->get_time_element( distributed_mesh.global_2_local(
          local_start_idx, spacetime_elements[ i_time * n_space_elements ] ) );
      lo global_time_index = distributed_mesh.local_2_global_time(
        local_start_idx, local_time_index );
      for ( lo i_space = 0; i_space < n_space_elements; ++i_space ) {
        lo current_spacetime_element_index
          = i_time * n_space_elements + i_space;
        // for the spatial mesh no transformation from local 2 global is
        // necessary since there is just one global space mesh at the
        // moment.
        lo global_space_index = local_mesh->get_space_element_index(
          distributed_mesh.global_2_local( local_start_idx,
            spacetime_elements[ current_spacetime_element_index ] ) );
        // get the diagonal entries of the current nearfield matrix
        inverse_diagonal.set( global_time_index, global_space_index,
          1.0
            / current_nf_matrix->get( current_spacetime_element_index,
              current_spacetime_element_index ) );
      }
    }
  }
  inverse_diagonal.synchronize_shared_parts( );
}
/** \endcond (end doxygen skip this)*/

template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >;

template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;

template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >;

template class besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >;
