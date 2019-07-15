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

#include "besthea/quadrature.h"

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

///*
template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble( besthea::linear_algebra::
    block_lower_triangular_toeplitz_matrix & global_matrix ) {
  lo block_dim = _test_space->get_mesh( )->get_n_temporal_elements( );
  global_matrix.resize( block_dim );

  lo n_rows = _test_space->get_basis( ).dimension_global( );
  lo n_columns = _trial_space->get_basis( ).dimension_global( );
  global_matrix.resize_blocks( n_rows, n_columns );

  lo n_loc_rows = _test_space->get_basis( ).dimension_local( );
  lo n_loc_columns = _trial_space->get_basis( ).dimension_local( );
  full_matrix_type local_matrix( n_loc_rows, n_loc_columns );

  lo n_elements = _test_space->get_mesh( )->get_n_spatial_elements( );
  lo n_timesteps = _test_space->get_mesh( )->get_n_temporal_elements( );
  sc ht = _test_space->get_mesh( )->get_timestep( );
  sc scaled_delta;

#pragma omp parallel shared( global_matrix )
  {
    sc kernel2, kernel1;
    sc test_area, trial_area;
    lo size;
    int type_int = 0;
    int rot_test = 0;
    int rot_trial = 0;

    sc x1[ 3 ], x2[ 3 ], x3[ 3 ];
    sc y1[ 3 ], y2[ 3 ], y3[ 3 ];
    sc nx[ 3 ], ny[ 3 ];

    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * x1_mapped = my_quadrature._x1.data( );
    sc * x2_mapped = my_quadrature._x2.data( );
    sc * x3_mapped = my_quadrature._x3.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );
    sc * w = nullptr;

    for ( lo delta = 0; delta <= n_timesteps; ++delta ) {
      scaled_delta = ht * delta;

#pragma omp for schedule( dynamic, 16 )
      for ( lo i_test = 0; i_test < n_elements; ++i_test ) {
        _test_space->get_mesh( )->get_spatial_nodes( i_test, x1, x2, x3 );
        _test_space->get_mesh( )->get_spatial_normal( i_test, nx );
        test_area = _test_space->get_mesh( )->spatial_area( i_test );
        for ( lo i_trial = 0; i_trial < n_elements; ++i_trial ) {
          if ( delta == 0 ) {
            get_type( i_test, i_trial, type_int, rot_test, rot_trial );
          } else {
            type_int = 0;
            rot_test = 0;
            rot_trial = 0;
          }
          _trial_space->get_mesh( )->get_spatial_nodes( i_trial, y1, y2, y3 );
          _trial_space->get_mesh( )->get_spatial_normal( i_trial, ny );
          trial_area = _trial_space->get_mesh( )->spatial_area( i_trial );
          triangles_to_geometry( x1, x2, x3, y1, y2, y3, type_int, rot_test,
            rot_trial, my_quadrature );

          w = my_quadrature._w[ type_int ].data( );

          size = my_quadrature._w[ type_int ].size( );

          if ( delta == 0 ) {
            kernel1 = 0.0;
#pragma omp simd reduction( + : kernel1 ) simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel1 += _kernel->anti_tau_limit(
                           x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                           x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                           x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx, ny )
                * w[ i_quad ];
            }
            global_matrix.add(
              0, i_test, i_trial, ht * kernel1 * test_area * trial_area );
          }

          kernel2 = 0.0;
#pragma omp simd reduction( + : kernel2 ) simdlen( DATA_WIDTH )
          for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
            kernel2 += _kernel->anti_tau_anti_t(
                         x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                         x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                         x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx, ny,
                         scaled_delta )
              * w[ i_quad ];
          }
          kernel2 *= test_area * trial_area;
          if ( delta > 0 ) {
            global_matrix.add( delta - 1, i_test, i_trial, -kernel2 );
            if ( delta < n_timesteps ) {
              global_matrix.add( delta, i_test, i_trial, 2.0 * kernel2 );
            }
          } else {
            global_matrix.add( 0, i_test, i_trial, kernel2 );
          }
          if ( delta < n_timesteps - 1 ) {
            global_matrix.add( delta + 1, i_test, i_trial, -kernel2 );
          }
        }
      }
    }
  }
}
//*/
/*
template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::assemble( besthea::linear_algebra::
    block_lower_triangular_toeplitz_matrix & global_matrix ) {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  // number of temporal elements and timestep should be the same for test and
  // trial meshes
  lo n_timesteps = test_mesh->get_n_temporal_elements( );
  sc ht = test_mesh->get_timestep( );
  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );
  global_matrix.resize( n_timesteps );
  global_matrix.resize_blocks( n_rows, n_columns );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  lo n_test_elements = test_mesh->get_n_spatial_elements( );
  lo n_trial_elements = trial_mesh->get_n_spatial_elements( );
  sc scaled_delta;

#pragma omp parallel shared( global_matrix )
  {
    full_matrix_type local_matrix( n_loc_rows, n_loc_columns, true );
    sc * local_matrix_data = local_matrix.data( );

    std::vector< lo > test_l2g( n_loc_rows );
    std::vector< lo > trial_l2g( n_loc_columns );
    std::vector< sc > test_values( n_loc_rows );
    std::vector< sc > trial_values( n_loc_columns );
    sc * test_data = test_values.data( );
    sc * trial_data = trial_values.data( );

    sc kernel1 = 0.0;
    sc kernel2 = 0.0;
    sc test_area, trial_area;
    lo size;
    int type_int = 0;
    int rot_test = 0;
    int rot_trial = 0;

    sc x1[ 3 ], x2[ 3 ], x3[ 3 ];
    sc y1[ 3 ], y2[ 3 ], y3[ 3 ];
    sc nx[ 3 ], ny[ 3 ];

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
    sc * test_data = my_quadrature._test_values.data( );
    sc * trial_data = my_quadrature._trial_values.data( );

    for ( lo delta = 0; delta <= n_timesteps; ++delta ) {
      scaled_delta = ht * delta;

#pragma omp for schedule( dynamic, 16 )
      for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
        test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
        test_mesh->get_spatial_normal( i_test, nx );
        test_area = test_mesh->spatial_area( i_test );
        for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
          if ( delta == 0 ) {
            get_type( i_test, i_trial, type_int, rot_test, rot_trial );
          } else {
            type_int = 0;
            rot_test = 0;
            rot_trial = 0;
          }
          trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
          trial_mesh->get_spatial_normal( i_trial, ny );
          trial_area = trial_mesh->spatial_area( i_trial );

          test_basis.local_to_global( i_test,
            static_cast< besthea::bem::adjacency >( type_int ), rot_test, false,
            test_l2g );
          trial_basis.local_to_global( i_trial,
            static_cast< besthea::bem::adjacency >( type_int ), rot_trial, true,
            trial_l2g );

          triangles_to_geometry( x1, x2, x3, y1, y2, y3, type_int, rot_test,
            rot_trial, my_quadrature );
          x1_ref = my_quadrature._x1_ref[ type_int ].data( );
          x2_ref = my_quadrature._x2_ref[ type_int ].data( );
          y1_ref = my_quadrature._y1_ref[ type_int ].data( );
          y2_ref = my_quadrature._y2_ref[ type_int ].data( );
          w = my_quadrature._w[ type_int ].data( );

          size = my_quadrature._w[ type_int ].size( );

          if ( delta == 0 ) {
            local_matrix.fill( 0.0 );
#pragma omp simd simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              kernel_data[ i_quad ] = _kernel->anti_tau_limit(
                x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx, ny );
              test_basis.evaluate( i_test, x1_ref[ i_quad ], x2_ref[ i_quad ],
                nx, static_cast< besthea::bem::adjacency >( type_int ),
                rot_test, false, test_values );
              trial_basis.evaluate( i_trial, y1_ref[ i_quad ], y2_ref[ i_quad ],
                ny, static_cast< besthea::bem::adjacency >( type_int ),
                rot_trial, true, trial_values );
              for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
                for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                      ++i_loc_trial ) {
                  local_matrix_data[ i_loc_test + i_loc_trial * n_loc_rows ]
                    += w[ i_quad ] * kernel1 * test_data[ i_loc_test ]
                    * trial_data[ i_loc_trial ];
                }
              }
            }
            for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
              for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                    ++i_loc_trial ) {
                global_matrix.add_atomic( 0, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ],
                  ht
                    * local_matrix_data[ i_loc_test + i_loc_trial * n_loc_rows ]
                    * test_area * trial_area );
              }
            }
          }

          local_matrix.fill( 0.0 );
#pragma omp simd \
        private( kernel2 ) \
        reduction( + : local_matrix_data [ 0 : n_loc_rows * n_loc_columns ] ) \
        simdlen( DATA_WIDTH )
          for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
            kernel2 = _kernel->anti_tau_anti_t(
              x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
              x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
              x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx, ny, scaled_delta );
            test_basis.evaluate( i_test, x1_ref[ i_quad ], x2_ref[ i_quad ], nx,
              static_cast< besthea::bem::adjacency >( type_int ), rot_test,
              false, test_values );
            trial_basis.evaluate( i_trial, y1_ref[ i_quad ], y2_ref[ i_quad ],
              ny, static_cast< besthea::bem::adjacency >( type_int ), rot_trial,
              true, trial_values );
            for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
              for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                    ++i_loc_trial ) {
                local_matrix_data[ i_loc_test + i_loc_trial * n_loc_rows ]
                  += w[ i_quad ] * kernel2 * test_data[ i_loc_test ]
                  * trial_data[ i_loc_trial ];
              }
            }
          }

          kernel2 *= test_area * trial_area;
          for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
            for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                  ++i_loc_trial ) {
              if ( delta > 0 ) {
                global_matrix.add_atomic( delta - 1, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ],
                  -local_matrix_data[ i_loc_test + i_loc_trial * n_loc_rows ]
                    * test_area * trial_area );
                if ( delta < n_timesteps ) {
                  global_matrix.add_atomic( delta, test_l2g[ i_loc_test ],
                    trial_l2g[ i_loc_trial ],
                    2.0
                      * local_matrix_data[ i_loc_test
                        + i_loc_trial * n_loc_rows ]
                      * test_area * trial_area );
                }
              } else {
                global_matrix.add_atomic( 0, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ],
                  local_matrix_data[ i_loc_test + i_loc_trial * n_loc_rows ]
                    * test_area * trial_area );
              }
              if ( delta < n_timesteps - 1 ) {
                global_matrix.add_atomic( delta + 1, test_l2g[ i_loc_test ],
                  trial_l2g[ i_loc_trial ],
                  -local_matrix_data[ i_loc_test + i_loc_trial * n_loc_rows ]
                    * test_area * trial_area );
              }
            }
          }
        }
      }
    }
  }
}
*/
template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::init_quadrature( quadrature_wrapper & my_quadrature ) {
  // Use triangle rules for disjoint elements
  const std::vector< sc > & tri_x1 = quadrature::triangle_x1( _order_regular );
  const std::vector< sc > & tri_x2 = quadrature::triangle_x2( _order_regular );
  const std::vector< sc > & tri_w = quadrature::triangle_w( _order_regular );
  lo tri_size = tri_x1.size( );
  lo tri_size2 = tri_size * tri_size;

  int type_int = 0;
  my_quadrature._x1_ref[ type_int ].resize( tri_size2 );
  my_quadrature._x2_ref[ type_int ].resize( tri_size2 );
  my_quadrature._y1_ref[ type_int ].resize( tri_size2 );
  my_quadrature._y2_ref[ type_int ].resize( tri_size2 );
  my_quadrature._w[ type_int ].resize( tri_size2 );

  lo counter = 0;
  for ( lo i_x = 0; i_x < tri_size; ++i_x ) {
    for ( lo i_y = 0; i_y < tri_size; ++i_y ) {
      my_quadrature._x1_ref[ type_int ][ counter ] = tri_x1[ i_x ];
      my_quadrature._x2_ref[ type_int ][ counter ] = tri_x2[ i_x ];
      my_quadrature._y1_ref[ type_int ][ counter ] = tri_x1[ i_y ];
      my_quadrature._y2_ref[ type_int ][ counter ] = tri_x2[ i_y ];
      my_quadrature._w[ type_int ][ counter ] = tri_w[ i_x ] * tri_w[ i_y ];
      ++counter;
    }
  }

  // Use tensor Gauss rules for singular configurations
  const std::vector< sc > & line_x = quadrature::line_x( _order_singular );
  const std::vector< sc > & line_w = quadrature::line_w( _order_singular );
  lo line_size = line_x.size( );
  lo line_size4 = line_size * line_size * line_size * line_size;
  sc jacobian = 0.0;

  for ( type_int = 1; type_int <= 3; ++type_int ) {
    my_quadrature._x1_ref[ type_int ].resize(
      line_size4 * n_simplices[ type_int ] );
    my_quadrature._x2_ref[ type_int ].resize(
      line_size4 * n_simplices[ type_int ] );
    my_quadrature._y1_ref[ type_int ].resize(
      line_size4 * n_simplices[ type_int ] );
    my_quadrature._y2_ref[ type_int ].resize(
      line_size4 * n_simplices[ type_int ] );
    my_quadrature._w[ type_int ].resize( line_size4 * n_simplices[ type_int ] );

    counter = 0;
    for ( int i_simplex = 0; i_simplex < n_simplices[ type_int ];
          ++i_simplex ) {
      for ( lo i_ksi = 0; i_ksi < line_size; ++i_ksi ) {
        for ( lo i_eta1 = 0; i_eta1 < line_size; ++i_eta1 ) {
          for ( lo i_eta2 = 0; i_eta2 < line_size; ++i_eta2 ) {
            for ( lo i_eta3 = 0; i_eta3 < line_size; ++i_eta3 ) {
              hypercube_to_triangles( line_x[ i_ksi ], line_x[ i_eta1 ],
                line_x[ i_eta2 ], line_x[ i_eta3 ],
                static_cast< besthea::bem::adjacency >( type_int ), i_simplex,
                my_quadrature._x1_ref[ type_int ][ counter ],
                my_quadrature._x2_ref[ type_int ][ counter ],
                my_quadrature._y1_ref[ type_int ][ counter ],
                my_quadrature._y2_ref[ type_int ][ counter ], jacobian );
              my_quadrature._w[ type_int ][ counter ] = 4.0 * jacobian
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
  my_quadrature._test_values.resize( size );
  my_quadrature._trial_values.resize( size );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::uniform_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::triangles_to_geometry( const sc * x1, const sc * x2,
  const sc * x3, const sc * y1, const sc * y2, const sc * y3, int type_int,
  int rot_test, int rot_trial, quadrature_wrapper & my_quadrature ) {
  const sc * x1rot = nullptr;
  const sc * x2rot = nullptr;
  const sc * x3rot = nullptr;
  const sc * y1rot = nullptr;
  const sc * y2rot = nullptr;
  const sc * y3rot = nullptr;

  switch ( rot_test ) {
    case 0:
      x1rot = x1;
      x2rot = x2;
      x3rot = x3;
      break;
    case 1:
      x1rot = x2;
      x2rot = x3;
      x3rot = x1;
      break;
    case 2:
      x1rot = x3;
      x2rot = x1;
      x3rot = x2;
      break;
  }

  switch ( rot_trial ) {
    case 0:
      if ( type_int == 2 ) {
        y1rot = y2;
        y2rot = y1;
        y3rot = y3;
      } else {
        y1rot = y1;
        y2rot = y2;
        y3rot = y3;
      }
      break;
    case 1:
      if ( type_int == 2 ) {
        y1rot = y3;
        y2rot = y2;
        y3rot = y1;
      } else {
        y1rot = y2;
        y2rot = y3;
        y3rot = y1;
      }
      break;
    case 2:
      if ( type_int == 2 ) {
        y1rot = y1;
        y2rot = y3;
        y3rot = y2;
      } else {
        y1rot = y3;
        y2rot = y1;
        y3rot = y2;
      }
      break;
  }

  const sc * x1_ref = my_quadrature._x1_ref[ type_int ].data( );
  const sc * x2_ref = my_quadrature._x2_ref[ type_int ].data( );
  const sc * y1_ref = my_quadrature._y1_ref[ type_int ].data( );
  const sc * y2_ref = my_quadrature._y2_ref[ type_int ].data( );

  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo size = my_quadrature._w[ type_int ].size( );

#pragma omp simd simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1rot[ 0 ] + ( x2rot[ 0 ] - x1rot[ 0 ] ) * x1_ref[ i ]
      + ( x3rot[ 0 ] - x1rot[ 0 ] ) * x2_ref[ i ];
    x2_mapped[ i ] = x1rot[ 1 ] + ( x2rot[ 1 ] - x1rot[ 1 ] ) * x1_ref[ i ]
      + ( x3rot[ 1 ] - x1rot[ 1 ] ) * x2_ref[ i ];
    x3_mapped[ i ] = x1rot[ 2 ] + ( x2rot[ 2 ] - x1rot[ 2 ] ) * x1_ref[ i ]
      + ( x3rot[ 2 ] - x1rot[ 2 ] ) * x2_ref[ i ];
  }

#pragma omp simd simdlen( DATA_WIDTH )
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
  trial_space_type >::get_type( lo i_test, lo i_trial, int & type_int,
  int & rot_test, int & rot_trial ) const {
  // check for identical
  if ( i_test == i_trial ) {
    type_int = 3;
    rot_test = 0;
    rot_trial = 0;
    return;
  }

  lo test_elem[ 3 ];
  lo trial_elem[ 3 ];

  _test_space->get_mesh( )->get_spatial_element( i_test, test_elem );
  _trial_space->get_mesh( )->get_spatial_element( i_trial, trial_elem );

  // check for shared edge
  for ( int i_rot_test = 0; i_rot_test < 3; ++i_rot_test ) {
    for ( int i_rot_trial = 0; i_rot_trial < 3; ++i_rot_trial ) {
      if ( ( trial_elem[ i_rot_trial ]
             == test_elem[ map[ ( i_rot_test + 1 ) ] ] )
        && ( trial_elem[ map[ ( i_rot_trial + 1 ) ] ]
          == test_elem[ i_rot_test ] ) ) {
        type_int = 2;
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
        type_int = 1;
        rot_test = i_rot_test;
        rot_trial = i_rot_trial;
        return;
      }
    }
  }

  // disjoint
  type_int = 0;
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
