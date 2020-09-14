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

#include "besthea/uniform_spacetime_initial_assembler.h"

#include "besthea/basis_tetra_p1.h"
#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_initial_m0_kernel_antiderivative.h"
#include "besthea/spacetime_heat_initial_m1_kernel_antiderivative.h"

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::uniform_spacetime_initial_assembler< kernel_type, test_space_type,
  trial_space_type >::uniform_spacetime_initial_assembler( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_regular_tri, int order_regular_tetra )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_regular_tri( order_regular_tri ),
    _order_regular_tetra( order_regular_tetra ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::uniform_spacetime_initial_assembler< kernel_type, test_space_type,
  trial_space_type >::~uniform_spacetime_initial_assembler( ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_initial_assembler< kernel_type,
  test_space_type,
  trial_space_type >::assemble( besthea::linear_algebra::block_row_matrix &
    global_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  lo n_timesteps = test_mesh->get_n_temporal_elements( );
  sc timestep = test_mesh->get_timestep( );
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );
  global_matrix.resize( n_timesteps );
  global_matrix.resize_blocks( n_rows, n_columns );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_elements( );
  sc t;

#pragma omp parallel
  {
    std::vector< lo > test_l2g( n_loc_rows );
    std::vector< lo > trial_l2g( n_loc_columns );

    sc test, trial, value, test_area, trial_area;

    linear_algebra::coordinates< 3 > x1, x2, x3;
    linear_algebra::coordinates< 3 > y1, y2, y3, y4;
    linear_algebra::coordinates< 3 > nx;

    sc * nx_data = nx.data( );

    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * x1_ref = my_quadrature._x1_ref.data( );
    ;
    sc * x2_ref = my_quadrature._x2_ref.data( );
    ;
    sc * y1_ref = my_quadrature._y1_ref.data( );
    sc * y2_ref = my_quadrature._y2_ref.data( );
    sc * y3_ref = my_quadrature._y3_ref.data( );
    sc * w = my_quadrature._w.data( );
    sc * x1_mapped = my_quadrature._x1.data( );
    sc * x2_mapped = my_quadrature._x2.data( );
    sc * x3_mapped = my_quadrature._x3.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );
    sc * kernel_data = my_quadrature._kernel_values.data( );
    lo size = my_quadrature._w.size( );

    for ( lo i_t = 0; i_t <= n_timesteps; ++i_t ) {
#pragma omp single
      t = timestep * i_t;

#pragma omp for schedule( dynamic )
      for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
        test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
        test_mesh->get_spatial_normal( i_test, nx );
        test_area = test_mesh->spatial_area( i_test );
        for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
          trial_mesh->get_nodes( i_trial, y1, y2, y3, y4 );
          trial_area = trial_mesh->area( i_trial );

          test_basis.local_to_global( i_test, test_l2g );
          trial_basis.local_to_global( i_trial, trial_l2g );

          triangle_and_tetrahedron_to_geometry(
            x1, x2, x3, y1, y2, y3, y4, my_quadrature );

          if ( i_t == 0 ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel_data[ i_quad ]
                = _kernel->anti_t_limit(
                    x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                    x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                    x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data )
                * w[ i_quad ];
            }
          } else {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel_data[ i_quad ]
                = _kernel->anti_t_regular(
                    x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                    x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                    x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, t )
                * w[ i_quad ];
            }
          }

          for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
            for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                  ++i_loc_trial ) {
              value = 0.0;
#pragma omp simd \
    	aligned( x1_ref, x2_ref, y1_ref, y2_ref, y3_ref, kernel_data : DATA_ALIGN ) \
    	private( test, trial ) reduction( + : value ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                test = test_basis.evaluate( i_test, i_loc_test,
                  x1_ref[ i_quad ], x2_ref[ i_quad ], nx_data );
                trial = trial_basis.evaluate( i_trial, i_loc_trial,
                  y1_ref[ i_quad ], y2_ref[ i_quad ], y3_ref[ i_quad ] );

                value += kernel_data[ i_quad ] * test * trial;
              }
              value *= test_area * trial_area;
              if ( i_t > 0 ) {
                global_matrix.add_atomic( i_t - 1, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ], value );
                if ( i_t < n_timesteps ) {
                  global_matrix.add_atomic( i_t, test_l2g[ i_loc_test ],
                    trial_l2g[ i_loc_trial ], -value );
                }
              } else {
                global_matrix.add_atomic(
                  0, test_l2g[ i_loc_test ], trial_l2g[ i_loc_trial ], -value );
              }
            }
          }
        }
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_initial_assembler< kernel_type,
  test_space_type, trial_space_type >::init_quadrature( quadrature_wrapper &
    my_quadrature ) const {
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x1
    = quadrature::triangle_x1( _order_regular_tri );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x2
    = quadrature::triangle_x2( _order_regular_tri );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_w
    = quadrature::triangle_w( _order_regular_tri );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_y1
    = quadrature::tetrahedron_x1( _order_regular_tetra );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_y2
    = quadrature::tetrahedron_x2( _order_regular_tetra );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_y3
    = quadrature::tetrahedron_x3( _order_regular_tetra );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_w
    = quadrature::tetrahedron_w( _order_regular_tetra );

  lo tri_size = tri_w.size( );
  lo tetra_size = tetra_w.size( );
  lo size = tri_size * tetra_size;

  my_quadrature._x1_ref.resize( size );
  my_quadrature._x2_ref.resize( size );
  my_quadrature._y1_ref.resize( size );
  my_quadrature._y2_ref.resize( size );
  my_quadrature._y3_ref.resize( size );
  my_quadrature._w.resize( size );

  lo counter = 0;
  for ( lo i_x = 0; i_x < tri_size; ++i_x ) {
    for ( lo i_y = 0; i_y < tetra_size; ++i_y ) {
      my_quadrature._x1_ref[ counter ] = tri_x1[ i_x ];
      my_quadrature._x2_ref[ counter ] = tri_x2[ i_x ];
      my_quadrature._y1_ref[ counter ] = tetra_y1[ i_y ];
      my_quadrature._y2_ref[ counter ] = tetra_y2[ i_y ];
      my_quadrature._y3_ref[ counter ] = tetra_y3[ i_y ];
      my_quadrature._w[ counter ] = tri_w[ i_x ] * tetra_w[ i_y ];
      ++counter;
    }
  }

  my_quadrature._x1.resize( size );
  my_quadrature._x2.resize( size );
  my_quadrature._x3.resize( size );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );
  my_quadrature._kernel_values.resize( size );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_initial_assembler< kernel_type,
  test_space_type, trial_space_type >::
  triangle_and_tetrahedron_to_geometry(
    const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3,
    const linear_algebra::coordinates< 3 > & y4,
    quadrature_wrapper & my_quadrature ) const {
  const sc * y1_ref = my_quadrature._y1_ref.data( );
  const sc * y2_ref = my_quadrature._y2_ref.data( );
  const sc * y3_ref = my_quadrature._y3_ref.data( );
  const sc * x1_ref = my_quadrature._x1_ref.data( );
  const sc * x2_ref = my_quadrature._x2_ref.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );
  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );

  lo size = my_quadrature._w.size( );

#pragma omp simd aligned(                                 \
  y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, y3_ref \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = y1[ 0 ] + ( y2[ 0 ] - y1[ 0 ] ) * y1_ref[ i ]
      + ( y3[ 0 ] - y1[ 0 ] ) * y2_ref[ i ]
      + ( y4[ 0 ] - y1[ 0 ] ) * y3_ref[ i ];
    y2_mapped[ i ] = y1[ 1 ] + ( y2[ 1 ] - y1[ 1 ] ) * y1_ref[ i ]
      + ( y3[ 1 ] - y1[ 1 ] ) * y2_ref[ i ]
      + ( y4[ 1 ] - y1[ 1 ] ) * y3_ref[ i ];
    y3_mapped[ i ] = y1[ 2 ] + ( y2[ 2 ] - y1[ 2 ] ) * y1_ref[ i ]
      + ( y3[ 2 ] - y1[ 2 ] ) * y2_ref[ i ]
      + ( y4[ 2 ] - y1[ 2 ] ) * y3_ref[ i ];
  }

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * x1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * x2_ref[ i ];
    x2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * x1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * x2_ref[ i ];
    x3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * x1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * x2_ref[ i ];
  }
}

template class besthea::bem::uniform_spacetime_initial_assembler<
  besthea::bem::spacetime_heat_initial_m0_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >;

template class besthea::bem::uniform_spacetime_initial_assembler<
  besthea::bem::spacetime_heat_initial_m1_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >;
