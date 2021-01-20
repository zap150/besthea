
#include "besthea/uniform_spacetime_be_onthefly_matrix.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"

#include <iostream>


template< class kernel_type, class test_space_type, class trial_space_type >
besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::uniform_spacetime_be_onthefly_matrix( kernel_type & kernel,
 test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular )
  : block_matrix(test_space.get_mesh()->get_n_temporal_elements(), test_space.get_mesh()->get_n_spatial_elements(), trial_space.get_mesh()->get_n_spatial_elements()),
    _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ) {

  init_quadrature();

}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::~uniform_spacetime_be_onthefly_matrix( ) {
}






template<class kernel_type, class test_space_type, class trial_space_type>
sc besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::
  get_value(lo delta, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing, bool special) const {
  
  return 0;

}

template<>
sc besthea::uniform_spacetime_be_onthefly_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_value(lo delta, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing, bool special) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );

  sc ttau;

  sc value, test_area, trial_area;
  lo size;
  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;

  const sc * w = nullptr;
  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );

  ttau = timestep * delta;

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  test_area = test_mesh->spatial_area( i_test );

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
    rot_test, rot_trial, quadr_changing );
  w = my_quadrature._w[ n_shared_vertices ].data( );

  size = my_quadrature._w[ n_shared_vertices ].size( );

  if ( delta == 0 && special ) {
    value = 0.0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                  y2_mapped, y3_mapped, w : DATA_ALIGN ) \
      reduction( + : value ) simdlen( DATA_WIDTH )
    for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
      value += _kernel->anti_tau_limit(
                  x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                  x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                  x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
                  nullptr )
        * w[ i_quad ];
    }

    value *= timestep * test_area * trial_area;
    return value;
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
                  x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
                  nullptr )
        * w[ i_quad ];
    }
  } else {
    if ( i_test != i_trial ) {
#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                  y2_mapped, y3_mapped, w : DATA_ALIGN ) \
      reduction( + : value ) simdlen( DATA_WIDTH )
      for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
        value
          += _kernel->anti_tau_anti_t_regular_in_time_regular_in_space(
                x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
                nullptr, ttau )
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
                    nullptr, ttau )
          * w[ i_quad ];
      }
    }
  }

  value *= test_area * trial_area;

  return value;
}






template<class kernel_type, class test_space_type, class trial_space_type>
sc besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::
  get(lo d, lo i, lo j, quadrature_wrapper_changing & quadr_changing ) const {
  
  return 0;

}

template<>
sc besthea::uniform_spacetime_be_onthefly_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get( lo d, lo i, lo j, quadrature_wrapper_changing & quadr_changing ) const {
    
  // if ( delta > 0 ) {
  //   global_matrix.add( delta - 1, i_test, i_trial, -value );
  //   if ( delta < n_timesteps ) {
  //     global_matrix.add( delta, i_test, i_trial, 2.0 * value );
  //   }
  // } else {
  //   global_matrix.add( 0, i_test, i_trial, value );
  // }
  //
  // if ( delta < n_timesteps - 1 ) {
  //   global_matrix.add( delta + 1, i_test, i_trial, -value );
  // }

  // pro n_timesteps = 4:
  // V0:  val0  -val1
  // V1: -val0  2val1  -val2
  // V2:        -val1  2val2  -val3
  // V3:               -val2  2val3  -val4

  sc result = 0;

  if ( d > 0 ) {
    result -=     get_value(d-1, i, j, quadr_changing);
    result += 2 * get_value(d,   i, j, quadr_changing);
    result -=     get_value(d+1, i, j, quadr_changing);
  } else if (d == 0) {
    result +=     get_value(0,   i, j, quadr_changing, true);
    result +=     get_value(0,   i, j, quadr_changing);
    result -=     get_value(1,   i, j, quadr_changing);
  }

  return result;
}



template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::
  apply( const block_vector_type & x, block_vector_type & y,
  bool trans, sc alpha, sc beta ) const {
  
  // y = alpha*A*x + beta*y;
  // basic matrix-vector multiplication for now
  if(trans) {
    std::cerr << "I dont support trans matrices\n";
    return;
  }

  lo rows_in_block = _n_rows;
  lo cols_in_block = _n_columns;
  lo blocks = _block_dim;

#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(quadr_size);

#pragma omp for collapse(2)
    for (lo inner_row = 0; inner_row < rows_in_block; inner_row++) {
      for (lo timestep = 0; timestep < blocks; timestep++) {
        sc y_val = 0;

        for (lo inner_col = 0; inner_col < cols_in_block; inner_col++) {
          sc matrix_val = get(timestep, inner_row, inner_col, quadr_changing);

          lo max_block = blocks - timestep;
          for (lo block = 0; block < max_block; block++) {
            sc x_val = x.get(block, inner_col);

            y_val += alpha * matrix_val * x_val;
          }
        }

        y_val += beta * y.get(timestep, inner_row);
        y.set(timestep, inner_row, y_val);
      }
    }
    
  }


  // for (lo block_row = 0; block_row < blocks; block_row++) {
  //   vector_type& y_block_data = y.get_block(block_row);

  //   for (lo inner_row = 0; inner_row < rows_in_block; inner_row++) {
  //     y_block_data[inner_row] *= beta;

  //     for (lo block_col = 0; block_col < blocks; block_col++) {
  //       const vector_type& x_block_data = x.get_block(block_col);

  //       for (lo inner_col = 0; inner_col < cols_in_block; inner_col++) {
          
  //         sc matrix_val;
  //         if(trans)
  //           matrix_val = get(block_col - block_row, inner_col, inner_row, quadr_changing);
  //         else
  //           matrix_val = get(block_row - block_col, inner_row, inner_col, quadr_changing);

  //         y_block_data[inner_row] += alpha * matrix_val * x_block_data[inner_col];

  //       }
  //     }
  //   }
  // }

}





template< class kernel_type, class test_space_type, class trial_space_type >
bool besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::
  check_equal(besthea::linear_algebra::block_lower_triangular_toeplitz_matrix & assembled, sc epsilon) const {

  lo n_timesteps = _test_space->get_mesh()->get_n_temporal_elements();
  lo n_rows = _n_rows;
  lo n_cols = _n_columns;

  quadrature_wrapper_changing quadr_changing(quadr_size);

  if(n_timesteps != assembled.get_block_dim())
  {
    std::cerr << "Not matching blockdim\n";
    return false;
  }

  if(n_rows != assembled.get_n_rows())
  {
    std::cerr << "Not mathing row count\n";
    return false;
  }

  if(n_cols != assembled.get_n_columns())
  {
    std::cerr << "not matching col count\n";
    return false;
  }

  bool result = true;
  for(int d = 0; d < n_timesteps; d++)
  {
    for(int r = 0; r < n_rows; r++)
    {
      for(int c = 0; c < n_cols; c++)
      {
        if(std::abs(assembled.get(d, r, c) - this->get(d, r, c, quadr_changing)) > epsilon)
        {
          std::cerr << "Not mathing value d=" << d << " r=" << r << " c=" << c << ": " << assembled.get(d, r, c) << " X " << this->get(d, r, c, quadr_changing) << "\n";
          result = false;
        }
      }
    }
  }

  return result;

}





template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::init_quadrature( ) {
  // Use triangle rules for disjoint elements
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x1
    = besthea::bem::quadrature::triangle_x1( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x2
    = besthea::bem::quadrature::triangle_x2( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_w
    = besthea::bem::quadrature::triangle_w( _order_regular );
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
    = besthea::bem::quadrature::line_x( _order_singular );
  const std::vector< sc, besthea::allocator_type< sc > > & line_w
    = besthea::bem::quadrature::line_w( _order_singular );
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
  // my_quadrature._x1.resize( size );
  // my_quadrature._x2.resize( size );
  // my_quadrature._x3.resize( size );
  // my_quadrature._y1.resize( size );
  // my_quadrature._y2.resize( size );
  // my_quadrature._y3.resize( size );
  // my_quadrature._kernel_values.resize( size );
  // my_quadrature._kernel_values_2.resize( size );
  this->quadr_size = size;
}



template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::triangles_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3, int n_shared_vertices,
    int rot_test, int rot_trial,
    quadrature_wrapper_changing & quadr_changing ) const {
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

  sc * x1_mapped = quadr_changing._x1.data( );
  sc * x2_mapped = quadr_changing._x2.data( );
  sc * x3_mapped = quadr_changing._x3.data( );
  sc * y1_mapped = quadr_changing._y1.data( );
  sc * y2_mapped = quadr_changing._y2.data( );
  sc * y3_mapped = quadr_changing._y3.data( );

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


template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::uniform_spacetime_be_onthefly_matrix<kernel_type, test_space_type, trial_space_type>::get_type( lo i_test, lo i_trial, int & n_shared_vertices,
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
void besthea::uniform_spacetime_be_onthefly_matrix< kernel_type, test_space_type,
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

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::uniform_spacetime_be_onthefly_matrix< kernel_type, test_space_type,
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
void besthea::uniform_spacetime_be_onthefly_matrix< kernel_type, test_space_type,
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




template class besthea::uniform_spacetime_be_onthefly_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;

