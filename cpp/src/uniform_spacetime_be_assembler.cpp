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

#include "besthea/uniform_spacetime_be_assembler.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/uniform_spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_heat_sl_kernel_antiderivative.h"

#include <algorithm>

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::uniform_spacetime_be_assembler( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::~uniform_spacetime_be_assembler( ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble( besthea::linear_algebra::
    block_lower_triangular_toeplitz_matrix & global_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  // number of temporal elements and timestep should be the same for test and
  // trial meshes
  lo n_timesteps = test_mesh->get_n_temporal_elements( );
  sc timestep = test_mesh->get_timestep( );
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );
  global_matrix.resize( n_timesteps );
  global_matrix.resize_blocks( n_rows, n_columns );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_spatial_elements( );
  sc scaled_delta;

#pragma omp parallel
  {
    std::vector< lo > test_l2g( n_loc_rows );
    std::vector< lo > trial_l2g( n_loc_columns );

    sc test, trial, value, test_area, trial_area;
    lo size;
    int n_shared_vertices = 0;
    int rot_test = 0;
    int rot_trial = 0;

    linear_algebra::coordinates< 3 > x1, x2, x3;
    linear_algebra::coordinates< 3 > y1, y2, y3;
    linear_algebra::coordinates< 3 > nx, ny;

    sc * nx_data = nx.data( );
    sc * ny_data = ny.data( );

    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * x1_ref = nullptr;
    sc * x2_ref = nullptr;
    sc * y1_ref = nullptr;
    sc * y2_ref = nullptr;
    sc * w = nullptr;
    sc * x1_mapped = my_quadrature._x1.data( );
    sc * x2_mapped = my_quadrature._x2.data( );
    sc * x3_mapped = my_quadrature._x3.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );
    sc * kernel_data = my_quadrature._kernel_values.data( );

    for ( lo delta = 0; delta <= n_timesteps; ++delta ) {
#pragma omp single
      scaled_delta = timestep * delta;

#pragma omp for schedule( dynamic )
      for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
        test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
        test_mesh->get_spatial_normal( i_test, nx );
        test_area = test_mesh->spatial_area( i_test );
        for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
          if ( delta == 0 ) {
            get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );
          } else {
            n_shared_vertices = 0;
            rot_test = 0;
            rot_trial = 0;
          }
          trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
          trial_mesh->get_spatial_normal( i_trial, ny );
          trial_area = trial_mesh->spatial_area( i_trial );

          test_basis.local_to_global(
            i_test, n_shared_vertices, rot_test, false, test_l2g );
          trial_basis.local_to_global(
            i_trial, n_shared_vertices, rot_trial, true, trial_l2g );

          triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices,
            rot_test, rot_trial, my_quadrature );
          x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
          x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
          y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
          y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
          w = my_quadrature._w[ n_shared_vertices ].data( );

          size = my_quadrature._w[ n_shared_vertices ].size( );

          if ( delta == 0 ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel_data[ i_quad ]
                = _kernel->anti_tau_limit(
                    x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                    x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                    x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data )
                * w[ i_quad ];
            }

            for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
              for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                    ++i_loc_trial ) {
                value = 0.0;
#pragma omp simd \
	aligned( x1_ref, x2_ref, y1_ref, y2_ref, kernel_data : DATA_ALIGN ) \
	private( test, trial ) reduction( + : value ) simdlen( DATA_WIDTH )
                for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                  test = test_basis.evaluate( i_test, i_loc_test,
                    x1_ref[ i_quad ], x2_ref[ i_quad ], nx_data,
                    n_shared_vertices, rot_test, false );
                  trial = trial_basis.evaluate( i_trial, i_loc_trial,
                    y1_ref[ i_quad ], y2_ref[ i_quad ], ny_data,
                    n_shared_vertices, rot_trial, true );

                  value += kernel_data[ i_quad ] * test * trial;
                }
                global_matrix.add_atomic( 0, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ],
                  timestep * value * test_area * trial_area );
              }
            }
          }

          if ( delta == 0 ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel_data[ i_quad ]
                = _kernel->anti_tau_anti_t_limit_in_time_regular_in_space(
                    x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                    x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                    x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data )
                * w[ i_quad ];
            }
          } else {
            if ( delta == 0 || i_test != i_trial ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                kernel_data[ i_quad ]
                  = _kernel->anti_tau_anti_t_regular_in_time_regular_in_space(
                      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                      x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data,
                      scaled_delta )
                  * w[ i_quad ];
              }
            } else {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                kernel_data[ i_quad ]
                  = _kernel->anti_tau_anti_t_regular_in_time(
                      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                      x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data,
                      scaled_delta )
                  * w[ i_quad ];
              }
            }
          }

          for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
            for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                  ++i_loc_trial ) {
              value = 0.0;
#pragma omp simd \
	aligned( x1_ref, x2_ref, y1_ref, y2_ref, kernel_data : DATA_ALIGN ) \
	private( test, trial ) reduction( + : value ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                test = test_basis.evaluate( i_test, i_loc_test,
                  x1_ref[ i_quad ], x2_ref[ i_quad ], nx_data,
                  n_shared_vertices, rot_test, false );
                trial = trial_basis.evaluate( i_trial, i_loc_trial,
                  y1_ref[ i_quad ], y2_ref[ i_quad ], ny_data,
                  n_shared_vertices, rot_trial, true );

                value += kernel_data[ i_quad ] * test * trial;
              }
              value *= test_area * trial_area;
              if ( delta > 0 ) {
                global_matrix.add_atomic( delta - 1, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ], -value );
                if ( delta < n_timesteps ) {
                  global_matrix.add_atomic( delta, test_l2g[ i_loc_test ],
                    trial_l2g[ i_loc_trial ], 2.0 * value );
                }
              } else {
                global_matrix.add_atomic(
                  0, test_l2g[ i_loc_test ], trial_l2g[ i_loc_trial ], value );
              }
              if ( delta < n_timesteps - 1 ) {
                global_matrix.add_atomic( delta + 1, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ], -value );
              }
            }
          }
        }
      }
    }
  }
}

/**
 * Specialization for single-layer operator with piecewise constant functions.
 * @param[out] global_matrix Block lower triangular Toeplitz matrix.
 */
template<>
void besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  assemble( besthea::linear_algebra::block_lower_triangular_toeplitz_matrix &
      global_matrix ) const {
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_spatial_elements( );

  // number of temporal elements and timestep should be the same for test and
  // trial meshes
  lo n_timesteps = test_mesh->get_n_temporal_elements( );
  sc timestep = test_mesh->get_timestep( );
  lo n_rows = n_test_elements;
  lo n_columns = n_trial_elements;
  global_matrix.resize( n_timesteps );
  global_matrix.resize_blocks( n_rows, n_columns );

  sc scaled_delta;

#pragma omp parallel
  {
    sc value, test_area, trial_area;
    lo size;
    int n_shared_vertices = 0;
    int rot_test = 0;
    int rot_trial = 0;

    linear_algebra::coordinates< 3 > x1, x2, x3;
    linear_algebra::coordinates< 3 > y1, y2, y3;

    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * w = nullptr;
    sc * x1_mapped = my_quadrature._x1.data( );
    sc * x2_mapped = my_quadrature._x2.data( );
    sc * x3_mapped = my_quadrature._x3.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );

    for ( lo delta = 0; delta <= n_timesteps; ++delta ) {
#pragma omp single
      scaled_delta = timestep * delta;

#pragma omp for schedule( dynamic )
      for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
        test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
        test_area = test_mesh->spatial_area( i_test );
        for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
          if ( delta == 0 ) {
            get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );
          } else {
            n_shared_vertices = 0;
            rot_test = 0;
            rot_trial = 0;
          }
          trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
          trial_area = trial_mesh->spatial_area( i_trial );

          triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices,
            rot_test, rot_trial, my_quadrature );
          w = my_quadrature._w[ n_shared_vertices ].data( );

          size = my_quadrature._w[ n_shared_vertices ].size( );

          if ( delta == 0 ) {
            value = 0.0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, w : DATA_ALIGN ) \
						  reduction( + : value ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              value += _kernel->anti_tau_limit(
                         x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                         x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                         x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr )
                * w[ i_quad ];
            }

            global_matrix.add(
              0, i_test, i_trial, timestep * value * test_area * trial_area );
          }

          value = 0.0;
          if ( delta == 0 ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, w : DATA_ALIGN ) \
						  reduction( + : value ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              value += _kernel->anti_tau_anti_t_limit_in_time_regular_in_space(
                         x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                         x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                         x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr )
                * w[ i_quad ];
            }
          } else {
            if ( delta == 0 || i_test != i_trial ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, w : DATA_ALIGN ) \
						  reduction( + : value ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                value
                  += _kernel->anti_tau_anti_t_regular_in_time_regular_in_space(
                       x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                       x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                       x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
                       scaled_delta )
                  * w[ i_quad ];
              }
            } else {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, w : DATA_ALIGN ) \
						  reduction( + : value ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                value += _kernel->anti_tau_anti_t_regular_in_time(
                           x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                           x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                           x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
                           scaled_delta )
                  * w[ i_quad ];
              }
            }
          }

          value *= test_area * trial_area;
          if ( delta > 0 ) {
            global_matrix.add( delta - 1, i_test, i_trial, -value );
            if ( delta < n_timesteps ) {
              global_matrix.add( delta, i_test, i_trial, 2.0 * value );
            }
          } else {
            global_matrix.add( 0, i_test, i_trial, value );
          }
          if ( delta < n_timesteps - 1 ) {
            global_matrix.add( delta + 1, i_test, i_trial, -value );
          }
        }
      }
    }
  }
}

/**
 * Specialization for double-layer operator with piecewise constant test
 * functions and piecewise linear trial functions.
 * @param[out] global_matrix Block lower triangular Toeplitz matrix.
 */
template<>
void besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  assemble( besthea::linear_algebra::block_lower_triangular_toeplitz_matrix &
      global_matrix ) const {
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_spatial_elements( );
  lo n_trial_nodes = trial_mesh->get_n_spatial_nodes( );

  // number of temporal elements and timestep should be the same for test and
  // trial meshes
  lo n_timesteps = test_mesh->get_n_temporal_elements( );
  sc timestep = test_mesh->get_timestep( );
  lo n_rows = n_test_elements;
  lo n_columns = n_trial_nodes;
  global_matrix.resize( n_timesteps );
  global_matrix.resize_blocks( n_rows, n_columns );

  sc scaled_delta;

#pragma omp parallel
  {
    sc value1, value2, value3;
    sc kernel, test_area, trial_area;
    lo size;
    int n_shared_vertices = 0;
    int rot_test = 0;
    int rot_trial = 0;

    linear_algebra::coordinates< 3 > x1, x2, x3;
    linear_algebra::coordinates< 3 > y1, y2, y3;
    linear_algebra::coordinates< 3 > ny;
    linear_algebra::indices< 3 > trial_element;

    sc * ny_data = ny.data( );

    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * y1_ref = nullptr;
    sc * y2_ref = nullptr;
    sc * w = nullptr;
    sc * x1_mapped = my_quadrature._x1.data( );
    sc * x2_mapped = my_quadrature._x2.data( );
    sc * x3_mapped = my_quadrature._x3.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );

    for ( lo delta = 0; delta <= n_timesteps; ++delta ) {
#pragma omp single
      scaled_delta = timestep * delta;

#pragma omp for schedule( dynamic )
      for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
        test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
        test_area = test_mesh->spatial_area( i_test );
        for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
          if ( delta == 0 ) {
            get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );
          } else {
            n_shared_vertices = 0;
            rot_test = 0;
            rot_trial = 0;
          }
          trial_mesh->get_spatial_element( i_trial, trial_element );
          trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
          trial_mesh->get_spatial_normal( i_trial, ny );
          trial_area = trial_mesh->spatial_area( i_trial );

          triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices,
            rot_test, rot_trial, my_quadrature );
          y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
          y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
          w = my_quadrature._w[ n_shared_vertices ].data( );

          size = my_quadrature._w[ n_shared_vertices ].size( );

          if ( delta == 0 ) {
            value1 = value2 = value3 = 0.0;
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, w : DATA_ALIGN ) \
						  private( kernel ) reduction( + : value1, value2, value3 ) \
					      simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel = _kernel->anti_tau_limit(
                         x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                         x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                         x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data )
                * w[ i_quad ];
              value1
                += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
              value2 += kernel * y1_ref[ i_quad ];
              value3 += kernel * y2_ref[ i_quad ];
            }

            global_matrix.add( 0, i_test, trial_element[ 0 ],
              timestep * value1 * test_area * trial_area );
            global_matrix.add( 0, i_test, trial_element[ 1 ],
              timestep * value2 * test_area * trial_area );
            global_matrix.add( 0, i_test, trial_element[ 2 ],
              timestep * value3 * test_area * trial_area );
          }

          value1 = value2 = value3 = 0.0;
          if ( delta == 0 ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, w : DATA_ALIGN ) \
						  private( kernel ) reduction( + : value1, value2, value3 ) \
					      simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel = _kernel->anti_tau_anti_t_limit_in_time_regular_in_space(
                         x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                         x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                         x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data )
                * w[ i_quad ];
              value1
                += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
              value2 += kernel * y1_ref[ i_quad ];
              value3 += kernel * y2_ref[ i_quad ];
            }
          } else {
            if ( delta == 0 || i_test != i_trial ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, w : DATA_ALIGN ) \
						  private( kernel ) reduction( + : value1, value2, value3 ) \
					      simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                kernel
                  = _kernel->anti_tau_anti_t_regular_in_time_regular_in_space(
                      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                      x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data,
                      scaled_delta )
                  * w[ i_quad ];
                value1 += kernel
                  * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
                value2 += kernel * y1_ref[ i_quad ];
                value3 += kernel * y2_ref[ i_quad ];
              }
            } else {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, w : DATA_ALIGN ) \
						  private( kernel ) reduction( + : value1, value2, value3 ) \
					      simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                kernel = _kernel->anti_tau_anti_t_regular_in_time(
                           x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                           x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                           x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data,
                           scaled_delta )
                  * w[ i_quad ];
                value1 += kernel
                  * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
                value2 += kernel * y1_ref[ i_quad ];
                value3 += kernel * y2_ref[ i_quad ];
              }
            }
          }

          value1 *= test_area * trial_area;
          value2 *= test_area * trial_area;
          value3 *= test_area * trial_area;
          if ( delta > 0 ) {
            global_matrix.add( delta - 1, i_test, trial_element[ 0 ], -value1 );
            global_matrix.add( delta - 1, i_test, trial_element[ 1 ], -value2 );
            global_matrix.add( delta - 1, i_test, trial_element[ 2 ], -value3 );
            if ( delta < n_timesteps ) {
              global_matrix.add(
                delta, i_test, trial_element[ 0 ], 2.0 * value1 );
              global_matrix.add(
                delta, i_test, trial_element[ 1 ], 2.0 * value2 );
              global_matrix.add(
                delta, i_test, trial_element[ 2 ], 2.0 * value3 );
            }
          } else {
            global_matrix.add( 0, i_test, trial_element[ 0 ], value1 );
            global_matrix.add( 0, i_test, trial_element[ 1 ], value2 );
            global_matrix.add( 0, i_test, trial_element[ 2 ], value3 );
          }
          if ( delta < n_timesteps - 1 ) {
            global_matrix.add( delta + 1, i_test, trial_element[ 0 ], -value1 );
            global_matrix.add( delta + 1, i_test, trial_element[ 1 ], -value2 );
            global_matrix.add( delta + 1, i_test, trial_element[ 2 ], -value3 );
          }
        }
      }
    }
  }
}

/**
 * Specialization for hypersingular operator with piecewise linear functions.
 * @param[out] global_matrix Block lower triangular Toeplitz matrix.
 */
template<>
void besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  assemble( besthea::linear_algebra::block_lower_triangular_toeplitz_matrix &
      global_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  // number of temporal elements and timestep should be the same for test and
  // trial meshes
  lo n_timesteps = test_mesh->get_n_temporal_elements( );
  sc timestep = test_mesh->get_timestep( );
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );
  global_matrix.resize( n_timesteps );
  global_matrix.resize_blocks( n_rows, n_columns );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_spatial_elements( );
  sc scaled_delta;

#pragma omp parallel
  {
    std::vector< lo > test_l2g( n_loc_rows );
    std::vector< lo > trial_l2g( n_loc_columns );

    sc test, trial, value, test_area, trial_area;
    lo size;
    int n_shared_vertices = 0;
    int rot_test = 0;
    int rot_trial = 0;

    linear_algebra::coordinates< 3 > x1, x2, x3;
    linear_algebra::coordinates< 3 > y1, y2, y3;
    linear_algebra::coordinates< 3 > nx, ny;

    sc * nx_data = nx.data( );
    sc * ny_data = ny.data( );

    sc test_curls[ 9 ], trial_curls[ 9 ];
    lo test_curl_offset, trial_curl_offset;
    sc curl_dot;

    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * x1_ref = nullptr;
    sc * x2_ref = nullptr;
    sc * y1_ref = nullptr;
    sc * y2_ref = nullptr;
    sc * w = nullptr;
    sc * x1_mapped = my_quadrature._x1.data( );
    sc * x2_mapped = my_quadrature._x2.data( );
    sc * x3_mapped = my_quadrature._x3.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );
    sc * kernel_data = my_quadrature._kernel_values.data( );
    sc * kernel_data_2 = my_quadrature._kernel_values_2.data( );

    for ( lo delta = 0; delta <= n_timesteps; ++delta ) {
#pragma omp single
      scaled_delta = timestep * delta;

#pragma omp for schedule( dynamic )
      for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
        test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
        test_mesh->get_spatial_normal( i_test, nx );
        test_area = test_mesh->spatial_area( i_test );
        for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
          if ( delta == 0 ) {
            get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );
          } else {
            n_shared_vertices = 0;
            rot_test = 0;
            rot_trial = 0;
          }

          trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
          trial_mesh->get_spatial_normal( i_trial, ny );
          trial_area = trial_mesh->spatial_area( i_trial );

          test_basis.local_to_global(
            i_test, n_shared_vertices, rot_test, false, test_l2g );
          trial_basis.local_to_global(
            i_trial, n_shared_vertices, rot_trial, true, trial_l2g );

          triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices,
            rot_test, rot_trial, my_quadrature );
          x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
          x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
          y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
          y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
          w = my_quadrature._w[ n_shared_vertices ].data( );

          size = my_quadrature._w[ n_shared_vertices ].size( );

          if ( delta == 0 ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                          y2_mapped, y3_mapped, kernel_data, w        \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel_data[ i_quad ]
                = _kernel->anti_tau_limit(
                    x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                    x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                    x3_mapped[ i_quad ] - y3_mapped[ i_quad ], ny_data )
                * w[ i_quad ];
            }

            test_basis.evaluate_curl(
              i_test, nx_data, n_shared_vertices, rot_test, false, test_curls );
            trial_basis.evaluate_curl(
              i_trial, ny, n_shared_vertices, rot_trial, true, trial_curls );

            for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
              for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                    ++i_loc_trial ) {
                value = 0.0;
                test_curl_offset = 3 * i_loc_test;
                trial_curl_offset = 3 * i_loc_trial;
                curl_dot = test_curls[ test_curl_offset ]
                    * trial_curls[ trial_curl_offset ]
                  + test_curls[ test_curl_offset + 1 ]
                    * trial_curls[ trial_curl_offset + 1 ]
                  + test_curls[ test_curl_offset + 2 ]
                    * trial_curls[ trial_curl_offset + 2 ];

#pragma omp simd aligned( kernel_data : DATA_ALIGN ) reduction( + : value ) \
	simdlen( DATA_WIDTH )
                for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                  value += kernel_data[ i_quad ] * curl_dot;
                }
                global_matrix.add_atomic( 0, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ],
                  timestep * value * test_area * trial_area );
              }
            }
          }

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped,         \
                          y2_mapped, y3_mapped, kernel_data, kernel_data_2, w \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
          for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
            _kernel->anti_tau_anti_t_and_anti_t(
              x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
              x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
              x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, ny_data,
              scaled_delta, kernel_data[ i_quad ], kernel_data_2[ i_quad ] );

            kernel_data[ i_quad ] *= w[ i_quad ];
            kernel_data_2[ i_quad ] *= w[ i_quad ];
          }

          test_basis.evaluate_curl(
            i_test, nx_data, n_shared_vertices, rot_test, false, test_curls );
          trial_basis.evaluate_curl(
            i_trial, ny_data, n_shared_vertices, rot_trial, true, trial_curls );

          for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
            for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                  ++i_loc_trial ) {
              value = 0.0;
              test_curl_offset = 3 * i_loc_test;
              trial_curl_offset = 3 * i_loc_trial;
              curl_dot = test_curls[ test_curl_offset ]
                  * trial_curls[ trial_curl_offset ]
                + test_curls[ test_curl_offset + 1 ]
                  * trial_curls[ trial_curl_offset + 1 ]
                + test_curls[ test_curl_offset + 2 ]
                  * trial_curls[ trial_curl_offset + 2 ];

#pragma omp simd \
	aligned( x1_ref, x2_ref, y1_ref, y2_ref, kernel_data, kernel_data_2 : DATA_ALIGN ) \
	private( test, trial ) reduction( + : value ) simdlen( DATA_WIDTH )
              for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
                test = test_basis.evaluate( i_test, i_loc_test,
                  x1_ref[ i_quad ], x2_ref[ i_quad ], nx_data,
                  n_shared_vertices, rot_test, false );
                trial = trial_basis.evaluate( i_trial, i_loc_trial,
                  y1_ref[ i_quad ], y2_ref[ i_quad ], ny_data,
                  n_shared_vertices, rot_trial, true );

                value += kernel_data_2[ i_quad ] * test * trial;
                value += kernel_data[ i_quad ] * curl_dot;
              }
              value *= test_area * trial_area;
              if ( delta > 0 ) {
                global_matrix.add_atomic( delta - 1, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ], -value );
                if ( delta < n_timesteps ) {
                  global_matrix.add_atomic( delta, test_l2g[ i_loc_test ],
                    trial_l2g[ i_loc_trial ], 2.0 * value );
                }
              } else {
                global_matrix.add_atomic(
                  0, test_l2g[ i_loc_test ], trial_l2g[ i_loc_trial ], value );
              }
              if ( delta < n_timesteps - 1 ) {
                global_matrix.add_atomic( delta + 1, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ], -value );
              }
            }
          }
        }
      }
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::init_quadrature( quadrature_wrapper & my_quadrature )
  const {
  // Use triangle rules for disjoint elements
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x1
    = quadrature::triangle_x1( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x2
    = quadrature::triangle_x2( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_w
    = quadrature::triangle_w( _order_regular );
  lo tri_size = tri_w.size( );
  lo tri_size2 = tri_size * tri_size;

  int n_shared_vertices = 0;
  my_quadrature._x1_ref[ n_shared_vertices ].resize( tri_size2 );
  my_quadrature._x2_ref[ n_shared_vertices ].resize( tri_size2 );
  my_quadrature._y1_ref[ n_shared_vertices ].resize( tri_size2 );
  my_quadrature._y2_ref[ n_shared_vertices ].resize( tri_size2 );
  my_quadrature._w[ n_shared_vertices ].resize( tri_size2 );

  lo counter = 0;
  for ( lo i_x = 0; i_x < tri_size; ++i_x ) {
    for ( lo i_y = 0; i_y < tri_size; ++i_y ) {
      my_quadrature._x1_ref[ n_shared_vertices ][ counter ] = tri_x1[ i_x ];
      my_quadrature._x2_ref[ n_shared_vertices ][ counter ] = tri_x2[ i_x ];
      my_quadrature._y1_ref[ n_shared_vertices ][ counter ] = tri_x1[ i_y ];
      my_quadrature._y2_ref[ n_shared_vertices ][ counter ] = tri_x2[ i_y ];
      my_quadrature._w[ n_shared_vertices ][ counter ]
        = tri_w[ i_x ] * tri_w[ i_y ];
      ++counter;
    }
  }

  // Use tensor Gauss rules for singular configurations
  const std::vector< sc, besthea::allocator_type< sc > > & line_x
    = quadrature::line_x( _order_singular );
  const std::vector< sc, besthea::allocator_type< sc > > & line_w
    = quadrature::line_w( _order_singular );
  lo line_size = line_x.size( );
  lo line_size4 = line_size * line_size * line_size * line_size;
  sc jacobian = 0.0;

  for ( n_shared_vertices = 1; n_shared_vertices <= 3; ++n_shared_vertices ) {
    my_quadrature._x1_ref[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );
    my_quadrature._x2_ref[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );
    my_quadrature._y1_ref[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );
    my_quadrature._y2_ref[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );
    my_quadrature._w[ n_shared_vertices ].resize(
      line_size4 * n_simplices[ n_shared_vertices ] );

    counter = 0;
    for ( int i_simplex = 0; i_simplex < n_simplices[ n_shared_vertices ];
          ++i_simplex ) {
      for ( lo i_ksi = 0; i_ksi < line_size; ++i_ksi ) {
        for ( lo i_eta1 = 0; i_eta1 < line_size; ++i_eta1 ) {
          for ( lo i_eta2 = 0; i_eta2 < line_size; ++i_eta2 ) {
            for ( lo i_eta3 = 0; i_eta3 < line_size; ++i_eta3 ) {
              hypercube_to_triangles( line_x[ i_ksi ], line_x[ i_eta1 ],
                line_x[ i_eta2 ], line_x[ i_eta3 ], n_shared_vertices,
                i_simplex,
                my_quadrature._x1_ref[ n_shared_vertices ][ counter ],
                my_quadrature._x2_ref[ n_shared_vertices ][ counter ],
                my_quadrature._y1_ref[ n_shared_vertices ][ counter ],
                my_quadrature._y2_ref[ n_shared_vertices ][ counter ],
                jacobian );
              my_quadrature._w[ n_shared_vertices ][ counter ] = 4.0 * jacobian
                * line_w[ i_ksi ] * line_w[ i_eta1 ] * line_w[ i_eta2 ]
                * line_w[ i_eta3 ];
              ++counter;
            }
          }
        }
      }
    }
  }

  lo size = std::max( tri_size2,
    *std::max_element( n_simplices.begin( ), n_simplices.end( ) )
      * line_size4 );
  my_quadrature._x1.resize( size );
  my_quadrature._x2.resize( size );
  my_quadrature._x3.resize( size );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );
  my_quadrature._kernel_values.resize( size );
  my_quadrature._kernel_values_2.resize( size );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::
  triangles_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3, int n_shared_vertices,
    int rot_test, int rot_trial, quadrature_wrapper & my_quadrature ) const {
  const sc * x1rot = nullptr;
  const sc * x2rot = nullptr;
  const sc * x3rot = nullptr;
  const sc * y1rot = nullptr;
  const sc * y2rot = nullptr;
  const sc * y3rot = nullptr;

  switch ( rot_test ) {
    case 0:
      x1rot = x1.data( );
      x2rot = x2.data( );
      x3rot = x3.data( );
      break;
    case 1:
      x1rot = x2.data( );
      x2rot = x3.data( );
      x3rot = x1.data( );
      break;
    case 2:
      x1rot = x3.data( );
      x2rot = x1.data( );
      x3rot = x2.data( );
      break;
  }

  switch ( rot_trial ) {
    case 0:
      if ( n_shared_vertices == 2 ) {
        y1rot = y2.data( );
        y2rot = y1.data( );
        y3rot = y3.data( );
      } else {
        y1rot = y1.data( );
        y2rot = y2.data( );
        y3rot = y3.data( );
      }
      break;
    case 1:
      if ( n_shared_vertices == 2 ) {
        y1rot = y3.data( );
        y2rot = y2.data( );
        y3rot = y1.data( );
      } else {
        y1rot = y2.data( );
        y2rot = y3.data( );
        y3rot = y1.data( );
      }
      break;
    case 2:
      if ( n_shared_vertices == 2 ) {
        y1rot = y1.data( );
        y2rot = y3.data( );
        y3rot = y2.data( );
      } else {
        y1rot = y3.data( );
        y2rot = y1.data( );
        y3rot = y2.data( );
      }
      break;
  }

  const sc * x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
  const sc * x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
  const sc * y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );

  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo size = my_quadrature._w[ n_shared_vertices ].size( );

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped        \
                          : DATA_ALIGN ) aligned( x1_ref, x2_ref \
                                                  : DATA_ALIGN ) \
  simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1rot[ 0 ] + ( x2rot[ 0 ] - x1rot[ 0 ] ) * x1_ref[ i ]
      + ( x3rot[ 0 ] - x1rot[ 0 ] ) * x2_ref[ i ];
    x2_mapped[ i ] = x1rot[ 1 ] + ( x2rot[ 1 ] - x1rot[ 1 ] ) * x1_ref[ i ]
      + ( x3rot[ 1 ] - x1rot[ 1 ] ) * x2_ref[ i ];
    x3_mapped[ i ] = x1rot[ 2 ] + ( x2rot[ 2 ] - x1rot[ 2 ] ) * x1_ref[ i ]
      + ( x3rot[ 2 ] - x1rot[ 2 ] ) * x2_ref[ i ];
  }

#pragma omp simd aligned( y1_mapped, y2_mapped, y3_mapped        \
                          : DATA_ALIGN ) aligned( y1_ref, y2_ref \
                                                  : DATA_ALIGN ) \
  simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = y1rot[ 0 ] + ( y2rot[ 0 ] - y1rot[ 0 ] ) * y1_ref[ i ]
      + ( y3rot[ 0 ] - y1rot[ 0 ] ) * y2_ref[ i ];
    y2_mapped[ i ] = y1rot[ 1 ] + ( y2rot[ 1 ] - y1rot[ 1 ] ) * y1_ref[ i ]
      + ( y3rot[ 1 ] - y1rot[ 1 ] ) * y2_ref[ i ];
    y3_mapped[ i ] = y1rot[ 2 ] + ( y2rot[ 2 ] - y1rot[ 2 ] ) * y1_ref[ i ]
      + ( y3rot[ 2 ] - y1rot[ 2 ] ) * y2_ref[ i ];
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::hypercube_to_triangles_vertex( sc ksi, sc eta1, sc eta2,
  sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
  sc & jacobian ) const {
  jacobian = ksi * ksi * ksi * eta2;

  switch ( simplex ) {
    case 0:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1;
      y1_ref = ksi * eta2 * ( 1 - eta3 );
      y2_ref = ksi * eta2 * eta3;
      break;
    case 1:
      x1_ref = ksi * eta2 * ( 1 - eta3 );
      x2_ref = ksi * eta2 * eta3;
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1;
      break;
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::hypercube_to_triangles_edge( sc ksi, sc eta1, sc eta2,
  sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
  sc & jacobian ) const {
  jacobian = ksi * ksi * ksi * eta1 * eta1;

  switch ( simplex ) {
    case 0:
      x1_ref = ksi * ( 1 - eta1 * eta3 );
      x2_ref = ksi * ( eta1 * eta3 );
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1 * ( 1 - eta2 );
      break;
    case 1:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1;
      y1_ref = ksi * ( 1 - eta1 * eta2 );
      y2_ref = ksi * eta1 * eta2 * ( 1 - eta3 );
      jacobian *= eta2;
      break;
    case 2:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1 * ( 1 - eta2 );
      y1_ref = ksi * ( 1 - eta1 * eta2 * eta3 );
      y2_ref = ksi * eta1 * eta2 * eta3;
      jacobian *= eta2;
      break;
    case 3:
      x1_ref = ksi * ( 1 - eta1 * eta2 );
      x2_ref = ksi * eta1 * eta2 * ( 1 - eta3 );
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1;
      jacobian *= eta2;
      break;
    case 4:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      y1_ref = ksi * ( 1 - eta1 * eta2 );
      y2_ref = ksi * eta1 * eta2;
      jacobian *= eta2;
      break;
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::get_type( lo i_test, lo i_trial, int & n_shared_vertices,
  int & rot_test, int & rot_trial ) const {
  // check for identical
  if ( i_test == i_trial ) {
    n_shared_vertices = 3;
    rot_test = 0;
    rot_trial = 0;
    return;
  }

  linear_algebra::indices< 3 > test_elem;
  linear_algebra::indices< 3 > trial_elem;

  _test_space->get_mesh( )->get_spatial_element( i_test, test_elem );
  _trial_space->get_mesh( )->get_spatial_element( i_trial, trial_elem );

  // check for shared edge
  for ( int i_rot_test = 0; i_rot_test < 3; ++i_rot_test ) {
    for ( int i_rot_trial = 0; i_rot_trial < 3; ++i_rot_trial ) {
      if ( ( trial_elem[ i_rot_trial ]
             == test_elem[ map[ ( i_rot_test + 1 ) ] ] )
        && ( trial_elem[ map[ ( i_rot_trial + 1 ) ] ]
          == test_elem[ i_rot_test ] ) ) {
        n_shared_vertices = 2;
        rot_test = i_rot_test;
        rot_trial = i_rot_trial;
        return;
      }
    }
  }

  // check for shared vertex
  for ( int i_rot_test = 0; i_rot_test < 3; ++i_rot_test ) {
    for ( int i_rot_trial = 0; i_rot_trial < 3; ++i_rot_trial ) {
      if ( test_elem[ i_rot_test ] == trial_elem[ i_rot_trial ] ) {
        n_shared_vertices = 1;
        rot_test = i_rot_test;
        rot_trial = i_rot_trial;
        return;
      }
    }
  }

  // disjoint
  n_shared_vertices = 0;
  rot_test = 0;
  rot_trial = 0;
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::hypercube_to_triangles_identical( sc ksi, sc eta1,
  sc eta2, sc eta3, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref,
  sc & y2_ref, sc & jacobian ) const {
  jacobian = ksi * ksi * ksi * eta1 * eta1 * eta2;

  switch ( simplex ) {
    case 0:
      x1_ref = ksi * eta1 * ( 1 - eta2 );
      x2_ref = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
      y1_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      y2_ref = ksi * ( 1 - eta1 );
      break;
    case 1:
      x1_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      x2_ref = ksi * ( 1 - eta1 );
      y1_ref = ksi * eta1 * ( 1 - eta2 );
      y2_ref = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
      break;
    case 2:
      x1_ref = ksi * ( 1 - eta1 * ( 1 - eta2 * ( 1 - eta3 ) ) );
      x2_ref = ksi * eta1 * ( 1 - eta2 * ( 1 - eta3 ) );
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1 * ( 1 - eta2 );
      break;
    case 3:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1 * ( 1 - eta2 );
      y1_ref = ksi * ( 1 - eta1 * ( 1 - eta2 * ( 1 - eta3 ) ) );
      y2_ref = ksi * eta1 * ( 1 - eta2 * ( 1 - eta3 ) );
      break;
    case 4:
      x1_ref = ksi * ( 1 - eta1 );
      x2_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      y1_ref = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
      y2_ref = ksi * eta1 * ( 1 - eta2 );
      break;
    case 5:
      x1_ref = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
      x2_ref = ksi * eta1 * ( 1 - eta2 );
      y1_ref = ksi * ( 1 - eta1 );
      y2_ref = ksi * eta1 * ( 1 - eta2 * eta3 );
      break;
  }
}

template class besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
template class besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
