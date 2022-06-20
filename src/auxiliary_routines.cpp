#include "besthea/auxiliary_routines.h"

#include <vector>

void sum_up_refined_mesh_vector_entries( const std::vector< sc > & fine_vector,
  const lo n_total_coarse_time_elems,
  const std::vector< std::vector< lo > > & local_time_ref_map,
  const lo local_start_idx_time,
  const std::vector< std::vector< lo > > & space_ref_map,
  std::vector< sc > & coarse_output_vector ) {
  lou n_coarse_space_elems = space_ref_map.size( );
  lou n_local_coarse_time_elems = local_time_ref_map.size( );
  lou n_child_space = space_ref_map[ 0 ].size( );
  lou n_child_time = local_time_ref_map[ 0 ].size( );
  lou n_fine_space_elems = n_coarse_space_elems * n_child_space;
  coarse_output_vector.resize(
    n_total_coarse_time_elems * n_coarse_space_elems, 0.0 );
  for ( lou i_t = 0; i_t < n_local_coarse_time_elems; ++i_t ) {
    for ( lou i_x = 0; i_x < n_coarse_space_elems; ++i_x ) {
      lou i_st_coarse
        = ( i_t + local_start_idx_time ) * n_coarse_space_elems + i_x;
      for ( lou i_t_child = 0; i_t_child < n_child_time; ++i_t_child ) {
        for ( lou i_x_child = 0; i_x_child < n_child_space; ++i_x_child ) {
          lou i_st_child
            = local_time_ref_map[ i_t ][ i_t_child ] * n_fine_space_elems
            + space_ref_map[ i_x ][ i_x_child ];
          coarse_output_vector[ i_st_coarse ] += fine_vector[ i_st_child ];
        }
      }
    }
  }
}

void scale_vector_by_inv_elem_size(
  const besthea::mesh::distributed_spacetime_tensor_mesh & st_mesh,
  besthea::linear_algebra::distributed_block_vector & vec_to_scale ) {
  // the number of space elements is the same in the local and nearfield mesh
  lo n_space_elems = st_mesh.get_local_mesh( )->get_n_spatial_elements( );
  // first, scale the entries of the vector corresponding to elements in the
  // nearfield mesh
  auto nf_mesh = st_mesh.get_nearfield_mesh( );
  if ( nf_mesh != nullptr ) {
    lo nf_start_idx = st_mesh.get_nearfield_start_idx( );
    for ( lo t_nf_idx = 0; t_nf_idx < nf_mesh->get_n_temporal_elements( );
          ++t_nf_idx ) {
      lo gl_time_idx = st_mesh.local_2_global_time( nf_start_idx, t_nf_idx );
      sc t_start, t_end, t_size;
      nf_mesh->get_temporal_nodes( t_nf_idx, &t_start, &t_end );
      t_size = t_end - t_start;
      for ( lo i_x = 0; i_x < n_space_elems; ++i_x ) {
        sc s_area = nf_mesh->get_spatial_area_using_spatial_index( i_x );
        vec_to_scale.set( gl_time_idx, i_x,
          vec_to_scale.get( gl_time_idx, i_x ) / ( t_size * s_area ) );
      }
    }
  }
  // same for local mesh
  auto local_mesh = st_mesh.get_local_mesh( );
  lo local_start_idx = st_mesh.get_local_start_idx( );
  for ( lo t_loc_idx = 0; t_loc_idx < local_mesh->get_n_temporal_elements( );
        ++t_loc_idx ) {
    lo gl_time_idx = st_mesh.local_2_global_time( local_start_idx, t_loc_idx );
    sc t_start, t_end, t_size;
    local_mesh->get_temporal_nodes( t_loc_idx, &t_start, &t_end );
    t_size = t_end - t_start;
    for ( lo i_x = 0; i_x < n_space_elems; ++i_x ) {
      sc s_area = local_mesh->get_spatial_area_using_spatial_index( i_x );
      vec_to_scale.set( gl_time_idx, i_x,
        vec_to_scale.get( gl_time_idx, i_x ) / ( t_size * s_area ) );
    }
  }
}

bool print_integers_in_cubic_grid( std::vector< long long > values,
  lo edge_length, std::string & output_file_base ) {
  lo y_stride = edge_length;
  lo x_stride = edge_length * edge_length;
  lo n_tot_entries = x_stride * edge_length;
  // determine the number of zeros in the grid
  lo n_zeros = 0;
  for ( lo i = 0; i < n_tot_entries; ++i ) {
    if ( values[ i ] == 0 ) {
      n_zeros++;
    }
  }
  // print the values in a grid only if they are not all zero.
  for ( lo z_idx = 0; z_idx < edge_length; ++z_idx ) {
    std::string output_file
      = output_file_base + "_z_idx_" + std::to_string( z_idx ) + ".txt";
    std::ofstream outfile( output_file.c_str( ), std::ios::app );
    if ( outfile.is_open( ) ) {
      for ( lo y_idx = edge_length - 1; y_idx >= 0; --y_idx ) {
        for ( lo x_idx = 0; x_idx < edge_length; ++x_idx ) {
          long long val = values[ x_idx * x_stride + y_idx * y_stride + z_idx ];
          // lo digits = 1;
          // if ( val > 0 ) {
          //   digits = (lo) ceil( log10( (double) ( val + 1 ) ) );
          // }
          outfile << val;
          if ( x_idx != edge_length - 1 ) {
            outfile << " ";  //<< std::string( max_n_digits + 1 - digits, ' ' );
          }
        }
        outfile << std::endl;
      }
    } else {
      std::cout << "failed to create output file " << output_file << std::endl;
    }
  }
  bool all_zeros = ( n_zeros == n_tot_entries );
  return all_zeros;
}
