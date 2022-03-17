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
