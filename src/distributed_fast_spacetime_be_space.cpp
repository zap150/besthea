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

#include "besthea/distributed_fast_spacetime_be_space.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/distributed_spacetime_be_identity.h"
#include "besthea/sparse_matrix.h"

template< class basis_type >
besthea::bem::distributed_fast_spacetime_be_space< basis_type >::
  distributed_fast_spacetime_be_space(
    mesh::distributed_spacetime_cluster_tree & tree )
  : spacetime_be_space< basis_type,
    besthea::linear_algebra::distributed_block_vector >( tree.get_mesh( ) ),
    _tree( &tree ) {
}

template< class basis_type >
besthea::bem::distributed_fast_spacetime_be_space<
  basis_type >::~distributed_fast_spacetime_be_space( ) {
}

// template<>
// void besthea::bem::distributed_fast_spacetime_be_space<
//   besthea::bem::basis_tri_p0 >::interpolation(
//     sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
//     distributed_block_vector_type & interpolation ) const {
//   const s_mesh_type & s_mesh = _tree->get_spatial_mesh( );
//   const t_mesh_type & t_mesh = _tree->get_temporal_mesh( );
//   lo n_timesteps = t_mesh.get_n_elements( );
//   lo n_elements = s_mesh.get_n_elements( );

//   interpolation.resize( n_timesteps );
//   interpolation.resize_blocks( n_elements );
//   linear_algebra::coordinates< 3 > centroid, n;

//   for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
//     s_mesh.get_centroid( i_elem, centroid );
//     s_mesh.get_normal( i_elem, n );
//     for ( lo d = 0; d < n_timesteps; ++d ) {
//       interpolation.set( d, i_elem,
//         f( centroid[ 0 ], centroid[ 1 ], centroid[ 2 ], n,
//           t_mesh.get_centroid( d ) ) );
//     }
//   }
// }

// template<>
// void besthea::bem::distributed_fast_spacetime_be_space<
// besthea::bem::basis_tri_p1 >::
//   interpolation(
//     sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
//     distributed_block_vector_type & interpolation ) const {
//   const s_mesh_type & s_mesh = _tree->get_spatial_mesh( );
//   const t_mesh_type & t_mesh = _tree->get_temporal_mesh( );
//   lo n_timesteps = t_mesh.get_n_elements( );
//   lo n_nodes = s_mesh.get_n_nodes( );

//   interpolation.resize( n_timesteps );
//   interpolation.resize_blocks( n_nodes );
//   linear_algebra::coordinates< 3 > x, n;

//   for ( lo i_node = 0; i_node < n_nodes; ++i_node ) {
//     s_mesh.get_node( i_node, x );
//     s_mesh.get_nodal_normal( i_node, n );
//     for ( lo d = 0; d < n_timesteps; ++d ) {
//       interpolation.set(
//         d, i_node, f( x[ 0 ], x[ 1 ], x[ 2 ], n, t_mesh.get_centroid( d ) )
//         );
//     }
//   }
// }

template< class basis_type >
void besthea::bem::distributed_fast_spacetime_be_space<
  basis_type >::L2_projection( sc ( *f )( sc, sc, sc,
                                 const linear_algebra::coordinates< 3 > &, sc ),
  linear_algebra::distributed_block_vector & projection, int order_matrix,
  int order_rhs_spatial, int order_rhs_temporal ) const {
  besthea::linear_algebra::sparse_matrix M;
  besthea::bem::distributed_spacetime_be_identity identity(
    *this, *this, order_matrix );
  identity.assemble( M );

  lo global_dim = this->_basis.dimension_global( );
  besthea::linear_algebra::vector rhs( global_dim, true );

  lo n_timesteps = get_mesh( ).get_n_temporal_elements( );
  std::vector< lo > my_blocks = get_mesh( ).get_my_timesteps( );
  const besthea::mesh::spacetime_tensor_mesh * local_mesh
    = get_mesh( ).get_local_mesh( );
  lo local_start_idx = get_mesh( ).get_local_start_idx( );
  lo n_local_timesteps = local_mesh->get_n_temporal_elements( );
  lo n_space_elements = local_mesh->get_n_spatial_elements( );

  projection.resize( my_blocks, n_timesteps );
  projection.resize_blocks( global_dim );

  lo local_dim = this->_basis.dimension_local( );
  std::vector< lo > l2g( local_dim );

  linear_algebra::coordinates< 3 > x1, x2, x3, n;
  sc area_x, basis_val, fun_val;
  sc cg_eps;
  lo n_iter;

  typename spacetime_be_space< basis_type,
    linear_algebra::distributed_block_vector >::quadrature_wrapper
    my_quadrature;
  this->init_quadrature( order_rhs_spatial, order_rhs_temporal, my_quadrature );
  lo size_t = my_quadrature._wt.size( );
  lo size_x = my_quadrature._wx.size( );
  sc * x1_ref = my_quadrature._x1_ref.data( );
  sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * wx = my_quadrature._wx.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * wt = my_quadrature._wt.data( );
  sc * t_mapped = my_quadrature._t.data( );
  sc * rhs_data = rhs.data( );
  lo * l2g_data = l2g.data( );

  for ( lo d = 0; d < n_local_timesteps; ++d ) {
    sc t_start, t_end;
    local_mesh->get_temporal_nodes( d, &t_start, &t_end );
    this->line_to_time( t_start, t_end, my_quadrature );
    for ( lo i_elem = 0; i_elem < n_space_elements; ++i_elem ) {
      local_mesh->get_spatial_nodes_using_spatial_element_index(
        i_elem, x1, x2, x3 );
      this->triangle_to_geometry( x1, x2, x3, my_quadrature );
      this->_basis.local_to_global( i_elem, l2g );
      local_mesh->get_spatial_normal_using_spatial_element_index( i_elem, n );
      // not necessary to multiply by timestep, M does not include this either
      area_x = local_mesh->get_spatial_area_using_spatial_index( i_elem );

      for ( lo i_t = 0; i_t < size_t; ++i_t ) {
        for ( lo i_x = 0; i_x < size_x; ++i_x ) {
          fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ], x3_mapped[ i_x ], n,
                      t_mapped[ i_t ] )
            * wx[ i_x ] * wt[ i_t ] * area_x;
          for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
            basis_val = this->_basis.evaluate(
              i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], n.data( ) );
            rhs_data[ l2g_data[ i_loc ] ] += basis_val * fun_val;
          }
        }
      }
    }
    cg_eps = 1e-6;
    n_iter = 200;
    M.eigen_cg_solve(
      rhs, projection.get_block( local_start_idx + d ), cg_eps, n_iter );
    rhs.fill( 0.0 );
  }
  projection.synchronize_shared_parts( );
}

template< class basis_type >
void besthea::bem::distributed_fast_spacetime_be_space< basis_type >::
  weighted_L2_projection(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    linear_algebra::distributed_block_vector & projection, int order_matrix,
    int order_rhs_spatial, int order_rhs_temporal ) const {
  besthea::bem::distributed_spacetime_be_identity identity(
    *this, *this, order_matrix );

  lo global_dim = this->_basis.dimension_global( );
  besthea::linear_algebra::vector rhs( global_dim, true );

  lo n_timesteps = get_mesh( ).get_n_temporal_elements( );
  std::vector< lo > my_blocks = get_mesh( ).get_my_timesteps( );
  const besthea::mesh::spacetime_tensor_mesh * local_mesh
    = get_mesh( ).get_local_mesh( );
  lo local_start_idx = get_mesh( ).get_local_start_idx( );
  lo n_local_timesteps = local_mesh->get_n_temporal_elements( );
  lo n_space_elements = local_mesh->get_n_spatial_elements( );

  projection.resize( my_blocks, n_timesteps );
  projection.resize_blocks( global_dim );

  lo local_dim = this->_basis.dimension_local( );
  std::vector< lo > l2g( local_dim );

  linear_algebra::coordinates< 3 > x1, x2, x3, n;
  sc area_x, basis_val, fun_val;
  sc cg_eps;
  lo n_iter;

  typename spacetime_be_space< basis_type,
    linear_algebra::distributed_block_vector >::quadrature_wrapper
    my_quadrature;
  this->init_quadrature( order_rhs_spatial, order_rhs_temporal, my_quadrature );
  lo size_t = my_quadrature._wt.size( );
  lo size_x = my_quadrature._wx.size( );
  sc * x1_ref = my_quadrature._x1_ref.data( );
  sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * wx = my_quadrature._wx.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * wt = my_quadrature._wt.data( );
  sc * t_mapped = my_quadrature._t.data( );
  sc * rhs_data = rhs.data( );
  lo * l2g_data = l2g.data( );

  for ( lo d = 0; d < n_local_timesteps; ++d ) {
    sc t_start, t_end;
    local_mesh->get_temporal_nodes( d, &t_start, &t_end );
    this->line_to_time( t_start, t_end, my_quadrature );
    besthea::linear_algebra::sparse_matrix M;
    sc temp_length = t_end - t_start;
    identity.assemble( M, true, temp_length );

    sc h_t_weight = std::pow( temp_length, 0.25 );
    for ( lo i_elem = 0; i_elem < n_space_elements; ++i_elem ) {
      local_mesh->get_spatial_nodes_using_spatial_element_index(
        i_elem, x1, x2, x3 );
      this->triangle_to_geometry( x1, x2, x3, my_quadrature );
      this->_basis.local_to_global( i_elem, l2g );
      local_mesh->get_spatial_normal_using_spatial_element_index( i_elem, n );

      area_x = local_mesh->get_spatial_area_using_spatial_index( i_elem );
      sc area_xt = area_x * temp_length;
      sc h_x_weight = std::pow( area_x, 0.25 );
      sc combined_weight = h_t_weight + h_x_weight;
      combined_weight *= combined_weight;

      for ( lo i_t = 0; i_t < size_t; ++i_t ) {
        for ( lo i_x = 0; i_x < size_x; ++i_x ) {
          fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ], x3_mapped[ i_x ], n,
                      t_mapped[ i_t ] )
            * wx[ i_x ] * wt[ i_t ] * area_xt * combined_weight;
          for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
            basis_val = this->_basis.evaluate(
              i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], n.data( ) );
            rhs_data[ l2g_data[ i_loc ] ] += basis_val * fun_val;
          }
        }
      }
    }
    cg_eps = 1e-6;
    n_iter = 200;
    M.eigen_cg_solve(
      rhs, projection.get_block( local_start_idx + d ), cg_eps, n_iter );
    rhs.fill( 0.0 );
  }
  projection.synchronize_shared_parts( );
}

template< class basis_type >
sc besthea::bem::distributed_fast_spacetime_be_space<
  basis_type >::L2_relative_error( sc ( *f )( sc, sc, sc,
                                     const linear_algebra::coordinates< 3 > &,
                                     sc ),
  const linear_algebra::distributed_block_vector & approximation,
  int order_rhs_spatial, int order_rhs_temporal ) const {
  lo local_dim = this->_basis.dimension_local( );

  sc l2_err = 0.0;
  sc l2_norm = 0.0;

  const besthea::mesh::spacetime_tensor_mesh * local_mesh
    = get_mesh( ).get_local_mesh( );
  lo local_start_idx = get_mesh( ).get_local_start_idx( );
  lo n_local_timesteps = local_mesh->get_n_temporal_elements( );
  lo n_space_elements = local_mesh->get_n_spatial_elements( );

  typename spacetime_be_space< basis_type,
    linear_algebra::distributed_block_vector >::quadrature_wrapper
    time_quadrature;
  this->init_quadrature(
    order_rhs_spatial, order_rhs_temporal, time_quadrature );
  lo size_t = time_quadrature._wt.size( );

  sc * wt = time_quadrature._wt.data( );
  sc * t_mapped = time_quadrature._t.data( );

  const sc * approximation_data = nullptr;

  // compute the local part of the l2 norm and l2 error
  for ( lo d = 0; d < n_local_timesteps; ++d ) {
    sc t_start, t_end;
    sc l2_err_timestep = 0.0;
    sc l2_norm_timestep = 0.0;
    local_mesh->get_temporal_nodes( d, &t_start, &t_end );
    this->line_to_time( t_start, t_end, time_quadrature );
    approximation_data = approximation.get_block( d + local_start_idx ).data( );

#pragma omp parallel reduction( + : l2_err_timestep, l2_norm_timestep )
    {
      typename spacetime_be_space< basis_type,
        linear_algebra::distributed_block_vector >::quadrature_wrapper
        space_quadrature;

      // init per-thread quadrature struct and aux data
      this->init_quadrature(
        order_rhs_spatial, order_rhs_temporal, space_quadrature );
      lo size_x = space_quadrature._wx.size( );
      sc * x1_ref = space_quadrature._x1_ref.data( );
      sc * x2_ref = space_quadrature._x2_ref.data( );
      sc * wx = space_quadrature._wx.data( );
      sc * x1_mapped = space_quadrature._x1.data( );
      sc * x2_mapped = space_quadrature._x2.data( );
      sc * x3_mapped = space_quadrature._x3.data( );
      std::vector< lo > l2g( local_dim );
      lo * l2g_data = l2g.data( );
      linear_algebra::coordinates< 3 > x1, x2, x3, n;

#pragma omp for
      for ( lo i_elem = 0; i_elem < n_space_elements; ++i_elem ) {
        local_mesh->get_spatial_nodes_using_spatial_element_index(
          i_elem, x1, x2, x3 );
        this->triangle_to_geometry( x1, x2, x3, space_quadrature );
        this->_basis.local_to_global( i_elem, l2g );
        local_mesh->get_spatial_normal_using_spatial_element_index( i_elem, n );
        sc area_xt = local_mesh->get_spatial_area_using_spatial_index( i_elem )
          * ( t_end - t_start );
        for ( lo i_x = 0; i_x < size_x; ++i_x ) {
          sc local_value = 0.0;
          for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
            sc basis_val = this->_basis.evaluate(
              i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], n.data( ) );
            local_value += approximation_data[ l2g_data[ i_loc ] ] * basis_val;
          }
          for ( lo i_t = 0; i_t < size_t; ++i_t ) {
            sc fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ],
              x3_mapped[ i_x ], n, t_mapped[ i_t ] );
            sc absdiff = std::abs( fun_val - local_value );
            sc absf = std::abs( fun_val );
            l2_err_timestep
              += absdiff * absdiff * wx[ i_x ] * wt[ i_t ] * area_xt;
            l2_norm_timestep += absf * absf * wx[ i_x ] * wt[ i_t ] * area_xt;
          }
        }
      }
    }
    // std::cout << "timestep " << d
    //           << ": rel. l2 err = " << l2_err_timestep /
    // l2_norm_timestep
    //           << ", l2 norm = " << l2_norm_timestep << std::endl;
    l2_err += l2_err_timestep;
    l2_norm += l2_norm_timestep;
  }

  // sum up the l2_errors and l2_norms communicatively by MPI_allreduce
  MPI_Allreduce( MPI_IN_PLACE, &l2_err, 1, get_scalar_type< sc >::MPI_SC( ),
    MPI_SUM, *_tree->get_MPI_comm( ) );
  MPI_Allreduce( MPI_IN_PLACE, &l2_norm, 1, get_scalar_type< sc >::MPI_SC( ),
    MPI_SUM, *_tree->get_MPI_comm( ) );

  sc result = std::sqrt( l2_err / l2_norm );
  return result;
}

template< class basis_type >
sc besthea::bem::distributed_fast_spacetime_be_space< basis_type >::
  weighted_L2_absolute_error(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    const linear_algebra::distributed_block_vector & approximation,
    int order_rhs_spatial, int order_rhs_temporal ) const {
  lo local_dim = this->_basis.dimension_local( );

  sc weight_l2_err = 0.0;
  sc weight_l2_norm = 0.0;

  const besthea::mesh::spacetime_tensor_mesh * local_mesh
    = get_mesh( ).get_local_mesh( );
  lo local_start_idx = get_mesh( ).get_local_start_idx( );
  lo n_local_timesteps = local_mesh->get_n_temporal_elements( );
  lo n_space_elements = local_mesh->get_n_spatial_elements( );

  typename spacetime_be_space< basis_type,
    linear_algebra::distributed_block_vector >::quadrature_wrapper
    time_quadrature;
  this->init_quadrature(
    order_rhs_spatial, order_rhs_temporal, time_quadrature );
  lo size_t = time_quadrature._wt.size( );
  sc * wt = time_quadrature._wt.data( );
  sc * t_mapped = time_quadrature._t.data( );

  const sc * approximation_data = nullptr;

  // compute the local part of the l2 norm and l2 error
  for ( lo d = 0; d < n_local_timesteps; ++d ) {
    sc t_start, t_end;
    sc weight_l2_err_timestep = 0.0;
    sc weight_l2_norm_timestep = 0.0;
    local_mesh->get_temporal_nodes( d, &t_start, &t_end );
    this->line_to_time( t_start, t_end, time_quadrature );
    approximation_data = approximation.get_block( d + local_start_idx ).data( );
    sc h_t_weight = std::pow( ( t_end - t_start ), 0.25 );

#pragma omp parallel reduction( + : weight_l2_err_timestep, weight_l2_norm_timestep )
    {
      typename spacetime_be_space< basis_type,
        linear_algebra::distributed_block_vector >::quadrature_wrapper
        space_quadrature;
      this->init_quadrature(
        order_rhs_spatial, order_rhs_temporal, space_quadrature );
      lo size_x = space_quadrature._wx.size( );
      sc * x1_ref = space_quadrature._x1_ref.data( );
      sc * x2_ref = space_quadrature._x2_ref.data( );
      sc * wx = space_quadrature._wx.data( );
      sc * x1_mapped = space_quadrature._x1.data( );
      sc * x2_mapped = space_quadrature._x2.data( );
      sc * x3_mapped = space_quadrature._x3.data( );

      std::vector< lo > l2g( local_dim );
      lo * l2g_data = l2g.data( );
      linear_algebra::coordinates< 3 > x1, x2, x3, n;

#pragma omp for
      for ( lo i_elem = 0; i_elem < n_space_elements; ++i_elem ) {
        local_mesh->get_spatial_nodes_using_spatial_element_index(
          i_elem, x1, x2, x3 );
        this->triangle_to_geometry( x1, x2, x3, space_quadrature );
        this->_basis.local_to_global( i_elem, l2g );
        local_mesh->get_spatial_normal_using_spatial_element_index( i_elem, n );
        sc spatial_area
          = local_mesh->get_spatial_area_using_spatial_index( i_elem );
        sc area_xt = spatial_area * ( t_end - t_start );
        // compute the spatial weight h_x^{1/2} with h_x ~
        // spatial_area ^ { 1 / 2 }
        sc h_x_weight = std::pow( spatial_area, 0.25 );
        sc combined_weight = h_t_weight + h_x_weight;
        combined_weight *= combined_weight;
        for ( lo i_x = 0; i_x < size_x; ++i_x ) {
          sc local_value = 0.0;
          for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
            sc basis_val = this->_basis.evaluate(
              i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], n.data( ) );
            local_value += approximation_data[ l2g_data[ i_loc ] ] * basis_val;
          }
          for ( lo i_t = 0; i_t < size_t; ++i_t ) {
            sc fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ],
              x3_mapped[ i_x ], n, t_mapped[ i_t ] );
            sc absdiff = std::abs( fun_val - local_value );
            sc absf = std::abs( fun_val );
            weight_l2_err_timestep += absdiff * absdiff * wx[ i_x ] * wt[ i_t ]
              * area_xt * combined_weight;
            weight_l2_norm_timestep += absf * absf * wx[ i_x ] * wt[ i_t ]
              * area_xt * combined_weight;
          }
        }
      }
    }
    // std::cout << "timestep " << d << ": rel. weighted l2 err = "
    //           << weight_l2_err_timestep / weight_l2_norm_timestep
    //           << ", rel. weighted l2 norm = " << weight_l2_norm_timestep
    //           << std::endl;
    weight_l2_err += weight_l2_err_timestep;
    weight_l2_norm += weight_l2_norm_timestep;
  }

  // lo local_dim = this->_basis.dimension_local( );
  // std::vector< lo > l2g( local_dim );
  // linear_algebra::coordinates< 3 > x1, x2, x3, n;
  // sc area_xt, basis_val, fun_val;
  // sc weight_l2_err = 0.0;
  // sc weight_l2_norm = 0.0;
  // sc local_value;
  // sc absdiff, absf;

  // const besthea::mesh::spacetime_tensor_mesh * local_mesh
  //   = get_mesh( ).get_local_mesh( );
  // lo local_start_idx = get_mesh( ).get_local_start_idx( );
  // lo n_local_timesteps = local_mesh->get_n_temporal_elements( );
  // lo n_space_elements = local_mesh->get_n_spatial_elements( );

  // typename spacetime_be_space< basis_type,
  //   linear_algebra::distributed_block_vector >::quadrature_wrapper
  //   my_quadrature;
  // this->init_quadrature( order_rhs_spatial, order_rhs_temporal, my_quadrature
  // ); lo size_t = my_quadrature._wt.size( ); lo size_x =
  // my_quadrature._wx.size( ); sc * x1_ref = my_quadrature._x1_ref.data( ); sc
  // * x2_ref = my_quadrature._x2_ref.data( ); sc * wx = my_quadrature._wx.data(
  // ); sc * x1_mapped = my_quadrature._x1.data( ); sc * x2_mapped =
  // my_quadrature._x2.data( ); sc * x3_mapped = my_quadrature._x3.data( ); sc *
  // wt = my_quadrature._wt.data( ); sc * t_mapped = my_quadrature._t.data( );
  // lo * l2g_data = l2g.data( );
  // const sc * approximation_data = nullptr;

  // // compute the local part of the l2 norm and l2 error
  // for ( lo d = 0; d < n_local_timesteps; ++d ) {
  //   sc t_start, t_end;
  //   sc weight_l2_err_timestep = 0.0;
  //   sc weight_l2_norm_timestep = 0.0;
  //   local_mesh->get_temporal_nodes( d, &t_start, &t_end );
  //   this->line_to_time( t_start, t_end, my_quadrature );
  //   approximation_data = approximation.get_block( d + local_start_idx ).data(
  //   ); sc h_t_weight = std::pow( ( t_end - t_start ), 0.25 ); for ( lo i_elem
  //   = 0; i_elem < n_space_elements; ++i_elem ) {
  //     local_mesh->get_spatial_nodes_using_spatial_element_index( i_elem, x1,
  //     x2, x3 ); this->triangle_to_geometry( x1, x2, x3, my_quadrature );
  //     this->_basis.local_to_global( i_elem, l2g );
  //     local_mesh->get_spatial_normal_using_spatial_element_index( i_elem, n
  //     ); sc spatial_area =
  //     local_mesh->get_spatial_area_using_spatial_index( i_elem );
  //     area_xt = spatial_area * ( t_end - t_start );
  //     // compute the spatial weight h_x^{1/2} with h_x ~ spatial_area^{1/2}
  //     sc h_x_weight = std::pow( spatial_area, 0.25 );
  //     sc combined_weight = h_t_weight + h_x_weight;
  //     combined_weight *= combined_weight;
  //     for ( lo i_x = 0; i_x < size_x; ++i_x ) {
  //       local_value = 0.0;
  //       for ( lo i_loc = 0; i_loc < local_dim; ++i_loc ) {
  //         basis_val = this->_basis.evaluate(
  //           i_elem, i_loc, x1_ref[ i_x ], x2_ref[ i_x ], n.data( ) );
  //         local_value += approximation_data[ l2g_data[ i_loc ] ] * basis_val;
  //       }
  //       for ( lo i_t = 0; i_t < size_t; ++i_t ) {
  //         fun_val = f( x1_mapped[ i_x ], x2_mapped[ i_x ], x3_mapped[ i_x ],
  //         n,
  //           t_mapped[ i_t ] );
  //         absdiff = std::abs( fun_val - local_value );
  //         absf = std::abs( fun_val );
  //         weight_l2_err_timestep += absdiff * absdiff * wx[ i_x ] * wt[ i_t ]
  //           * area_xt * combined_weight;
  //         weight_l2_norm_timestep
  //           += absf * absf * wx[ i_x ] * wt[ i_t ] * area_xt *
  //           combined_weight;
  //       }
  //     }
  //   }
  //   // std::cout << "timestep " << d << ": rel. weighted l2 err = "
  //   //           << weight_l2_err_timestep / weight_l2_norm_timestep
  //   //           << ", rel. weighted l2 norm = " << weight_l2_norm_timestep
  //   //           << std::endl;
  //   weight_l2_err += weight_l2_err_timestep;
  //   weight_l2_norm += weight_l2_norm_timestep;
  // }

  // sum up the weighted l2 errors and norms communicatively by MPI_allreduce
  MPI_Allreduce( MPI_IN_PLACE, &weight_l2_err, 1,
    get_scalar_type< sc >::MPI_SC( ), MPI_SUM, *_tree->get_MPI_comm( ) );
  MPI_Allreduce( MPI_IN_PLACE, &weight_l2_norm, 1,
    get_scalar_type< sc >::MPI_SC( ), MPI_SUM, *_tree->get_MPI_comm( ) );

  // sc result = std::sqrt( weight_l2_err / weight_l2_norm );
  sc result = std::sqrt( weight_l2_err );
  return result;
}

template class besthea::bem::distributed_fast_spacetime_be_space<
  besthea::bem::basis_tri_p0 >;
template class besthea::bem::distributed_fast_spacetime_be_space<
  besthea::bem::basis_tri_p1 >;
