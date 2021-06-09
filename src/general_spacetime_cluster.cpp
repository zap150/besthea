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

#include "besthea/general_spacetime_cluster.h"

/** Specialization of
 * @ref general_spacetime_cluster::compute_surface_curls_p1_along_dim for
 * dimension 0 */
template<>
void besthea::mesh::general_spacetime_cluster::
  compute_surface_curls_p1_along_dim< 0 >(
    std::vector< sc > & surface_curls_along_dim ) const {
  surface_curls_along_dim.resize( 3 * _n_space_elements );

  const spacetime_tensor_mesh * local_mesh = _mesh.get_local_mesh( );
  linear_algebra::indices< 3 > space_element;
  linear_algebra::coordinates< 3 > x0, x1, x2;
  linear_algebra::coordinates< 3 > normal;
  for ( lo i_el = 0; i_el < _n_space_elements; ++i_el ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component,
    // to get the spatial indices of elements in the source cluster
    lo global_space_index = local_mesh->get_space_element( _elements[ i_el ] );
    local_mesh->get_spatial_normal( global_space_index, normal );
    local_mesh->get_spatial_element( global_space_index, space_element );
    local_mesh->get_spatial_node( space_element[ 0 ], x0 );
    local_mesh->get_spatial_node( space_element[ 1 ], x1 );
    local_mesh->get_spatial_node( space_element[ 2 ], x2 );
    // same computation as in evaluate curl in basis_tri_p1; difference:
    // only the required component is computed (dim 0)
    // ######################################
    // first two rows of R^\trans, third is n
    sc a11 = x1[ 0 ] - x0[ 0 ];
    sc a12 = x1[ 1 ] - x0[ 1 ];
    sc a13 = x1[ 2 ] - x0[ 2 ];
    sc a21 = x2[ 0 ] - x0[ 0 ];
    sc a22 = x2[ 1 ] - x0[ 1 ];
    sc a23 = x2[ 2 ] - x0[ 2 ];
    // determinant to invert the matrix is the triple product. in particular it
    // is 2 * area of the triangle
    sc det = 2 * local_mesh->spatial_area( global_space_index );
    // compute only the entries of R^{-\trans} * [1;0;0] and
    // R^{-\trans} * [0;1;0] which are required
    sc g22 = ( -normal[ 2 ] * a21 + normal[ 0 ] * a23 ) / det;
    sc g23 = ( normal[ 1 ] * a21 - normal[ 0 ] * a22 ) / det;
    sc g32 = ( normal[ 2 ] * a11 - normal[ 0 ] * a13 ) / det;
    sc g33 = ( -normal[ 1 ] * a11 + normal[ 0 ] * a12 ) / det;
    // compute the 0th entry of the surface curl of each basis function and
    // store it in the vector
    surface_curls_along_dim[ 3 * i_el + 1 ]
      = normal[ 1 ] * g23 - normal[ 2 ] * g22;
    surface_curls_along_dim[ 3 * i_el + 2 ]
      = normal[ 1 ] * g33 - normal[ 2 ] * g32;
    surface_curls_along_dim[ 3 * i_el ]
      = -surface_curls_along_dim[ 3 * i_el + 1 ]
      - surface_curls_along_dim[ 3 * i_el + 2 ];
  }
}

/** Specialization of
 * @ref general_spacetime_cluster::compute_surface_curls_p1_along_dim for
 * dimension 1 */
template<>
void besthea::mesh::general_spacetime_cluster::
  compute_surface_curls_p1_along_dim< 1 >(
    std::vector< sc > & surface_curls_along_dim ) const {
  surface_curls_along_dim.resize( 3 * _n_space_elements );

  const spacetime_tensor_mesh * local_mesh = _mesh.get_local_mesh( );
  linear_algebra::indices< 3 > space_element;
  linear_algebra::coordinates< 3 > x0, x1, x2;
  linear_algebra::coordinates< 3 > normal;
  for ( lo i_el = 0; i_el < _n_space_elements; ++i_el ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component,
    // to get the spatial indices of elements in the source cluster
    lo global_space_index = local_mesh->get_space_element( _elements[ i_el ] );
    local_mesh->get_spatial_normal( global_space_index, normal );
    local_mesh->get_spatial_element( global_space_index, space_element );
    local_mesh->get_spatial_node( space_element[ 0 ], x0 );
    local_mesh->get_spatial_node( space_element[ 1 ], x1 );
    local_mesh->get_spatial_node( space_element[ 2 ], x2 );
    // same computation as in evaluate curl in basis_tri_p1; difference:
    // only the required component is computed (dim 1)
    // ######################################
    // first two rows of R^\trans, third is n
    sc a11 = x1[ 0 ] - x0[ 0 ];
    sc a12 = x1[ 1 ] - x0[ 1 ];
    sc a13 = x1[ 2 ] - x0[ 2 ];
    sc a21 = x2[ 0 ] - x0[ 0 ];
    sc a22 = x2[ 1 ] - x0[ 1 ];
    sc a23 = x2[ 2 ] - x0[ 2 ];
    // determinant to invert the matrix is the triple product. in particular it
    // is 2 * area of the triangle
    sc det = 2 * local_mesh->spatial_area( global_space_index );
    // compute only the entries of R^{-\trans} * [1;0;0] and
    // R^{-\trans} * [0;1;0] which are required
    sc g21 = ( normal[ 2 ] * a22 - normal[ 1 ] * a23 ) / det;
    sc g23 = ( normal[ 1 ] * a21 - normal[ 0 ] * a22 ) / det;
    sc g31 = ( -normal[ 2 ] * a12 + normal[ 1 ] * a13 ) / det;
    sc g33 = ( -normal[ 1 ] * a11 + normal[ 0 ] * a12 ) / det;
    // compute the 1st entry of the surface curl of each basis function and
    // store it in the vector
    surface_curls_along_dim[ 3 * i_el + 1 ]
      = normal[ 2 ] * g21 - normal[ 0 ] * g23;
    surface_curls_along_dim[ 3 * i_el + 2 ]
      = normal[ 2 ] * g31 - normal[ 0 ] * g33;
    surface_curls_along_dim[ 3 * i_el ]
      = -surface_curls_along_dim[ 3 * i_el + 1 ]
      - surface_curls_along_dim[ 3 * i_el + 2 ];
  }
}

/** Specialization of
 * @ref general_spacetime_cluster::compute_surface_curls_p1_along_dim for
 * dimension 2 */
template<>
void besthea::mesh::general_spacetime_cluster::
  compute_surface_curls_p1_along_dim< 2 >(
    std::vector< sc > & surface_curls_along_dim ) const {
  surface_curls_along_dim.resize( 3 * _n_space_elements );

  const spacetime_tensor_mesh * local_mesh = _mesh.get_local_mesh( );
  linear_algebra::indices< 3 > space_element;
  linear_algebra::coordinates< 3 > x0, x1, x2;
  linear_algebra::coordinates< 3 > normal;
  for ( lo i_el = 0; i_el < _n_space_elements; ++i_el ) {
    // use that the spacetime elements are sorted in time, i.e. a consecutive
    // group of n_space_elements elements has the same temporal component,
    // to get the spatial indices of elements in the source cluster
    lo global_space_index = local_mesh->get_space_element( _elements[ i_el ] );
    local_mesh->get_spatial_normal( global_space_index, normal );
    local_mesh->get_spatial_element( global_space_index, space_element );
    local_mesh->get_spatial_node( space_element[ 0 ], x0 );
    local_mesh->get_spatial_node( space_element[ 1 ], x1 );
    local_mesh->get_spatial_node( space_element[ 2 ], x2 );
    // same computation as in evaluate curl in basis_tri_p1; difference:
    // only the required component is computed (dim 2)
    // ######################################
    // first two rows of R^\trans, third is n
    sc a11 = x1[ 0 ] - x0[ 0 ];
    sc a12 = x1[ 1 ] - x0[ 1 ];
    sc a13 = x1[ 2 ] - x0[ 2 ];
    sc a21 = x2[ 0 ] - x0[ 0 ];
    sc a22 = x2[ 1 ] - x0[ 1 ];
    sc a23 = x2[ 2 ] - x0[ 2 ];
    // determinant to invert the matrix is the triple product. in particular it
    // is 2 * area of the triangle
    sc det = 2 * local_mesh->spatial_area( global_space_index );
    // compute only the entries of R^{-\trans} * [1;0;0] and
    // R^{-\trans} * [0;1;0] which are required
    sc g21 = ( normal[ 2 ] * a22 - normal[ 1 ] * a23 ) / det;
    sc g22 = ( -normal[ 2 ] * a21 + normal[ 0 ] * a23 ) / det;
    sc g31 = ( -normal[ 2 ] * a12 + normal[ 1 ] * a13 ) / det;
    sc g32 = ( normal[ 2 ] * a11 - normal[ 0 ] * a13 ) / det;
    // compute the 1st entry of the surface curl of each basis function and
    // store it in the vector
    surface_curls_along_dim[ 3 * i_el + 1 ]
      = normal[ 0 ] * g22 - normal[ 1 ] * g21;
    surface_curls_along_dim[ 3 * i_el + 2 ]
      = normal[ 0 ] * g32 - normal[ 1 ] * g31;
    surface_curls_along_dim[ 3 * i_el ]
      = -surface_curls_along_dim[ 3 * i_el + 1 ]
      - surface_curls_along_dim[ 3 * i_el + 2 ];
  }
}

/** template specialization for dimension 0 */
template<>
void besthea::mesh::general_spacetime_cluster::
  compute_surface_curls_p1_along_dim< 0 >( std::vector< sc > & ) const;
/** template specialization for dimension 1 */
template<>
void besthea::mesh::general_spacetime_cluster::
  compute_surface_curls_p1_along_dim< 1 >( std::vector< sc > & ) const;
/** template specialization for dimension 2 */
template<>
void besthea::mesh::general_spacetime_cluster::
  compute_surface_curls_p1_along_dim< 2 >( std::vector< sc > & ) const;
