
#include "besthea/uniform_spacetime_be_onthefly_matrix_cpu.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"

#include <iostream>


template< class kernel_type, class test_space_type, class trial_space_type >
besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>::uniform_spacetime_be_onthefly_matrix_cpu( kernel_type & kernel,
 test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular )
  : block_matrix(test_space.get_mesh()->get_n_temporal_elements(), test_space.get_basis().dimension_global(), trial_space.get_basis().dimension_global()),
    _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ) {

  init_quadrature();

}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>::~uniform_spacetime_be_onthefly_matrix_cpu( ) {

}

















template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_values_delta0special(sc * values_out, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );

  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );

  const sc test_area = test_mesh->spatial_area( i_test );
  const sc trial_area = trial_mesh->spatial_area( i_trial );

  get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );

  triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices, rot_test, rot_trial, quadr_changing );

  const sc * w = my_quadrature._w[ n_shared_vertices ].data( );
  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );

  lo size = my_quadrature._w[ n_shared_vertices ].size( );


  sc value = 0;

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
  *values_out = value;
  return;

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_values_delta0(sc * values_out, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );

  const sc test_area = test_mesh->spatial_area( i_test );
  const sc trial_area = trial_mesh->spatial_area( i_trial );

  get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );

  triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices, rot_test, rot_trial, quadr_changing );

  const sc * w = my_quadrature._w[ n_shared_vertices ].data( );
  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );

  lo size = my_quadrature._w[ n_shared_vertices ].size( );


  sc value = 0;

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

  value *= test_area * trial_area;
  *values_out = value;
  return;

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_values_regular(sc * values_out, lo delta, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing) const {

  // assuming i_test != i_trial

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );

  sc ttau = timestep * delta;

  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );

  const sc test_area = test_mesh->spatial_area( i_test );
  const sc trial_area = trial_mesh->spatial_area( i_trial );

  triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices, rot_test, rot_trial, quadr_changing );

  const sc * w = my_quadrature._w[ n_shared_vertices ].data( );
  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );

  lo size = my_quadrature._w[ n_shared_vertices ].size( );


  sc value = 0;
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

  value *= test_area * trial_area;
  *values_out = value;
  return;

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_values_singular(sc * values_out, lo delta, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing) const {

  // assuming i_test = i_trial

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );

  sc ttau = timestep * delta;

  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );

  const sc test_area = test_mesh->spatial_area( i_test );
  const sc trial_area = trial_mesh->spatial_area( i_trial );

  triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices, rot_test, rot_trial, quadr_changing );

  const sc * w = my_quadrature._w[ n_shared_vertices ].data( );
  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );

  lo size = my_quadrature._w[ n_shared_vertices ].size( );


  sc value = 0;

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

  value *= test_area * trial_area;
  *values_out = value;
  return;

}

















template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_delta0special(sc * values_out, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );

  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;
  linear_algebra::coordinates< 3 > ny;
  sc * ny_data = ny.data( );

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
  trial_mesh->get_spatial_normal( i_trial, ny );

  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );

  triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices, rot_test, rot_trial, quadr_changing );

  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );
  
  const sc * y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
  const sc * w = my_quadrature._w[ n_shared_vertices ].data( );

  lo size = my_quadrature._sizes[ n_shared_vertices ];


  sc kernel;
  sc value1, value2, value3;

  value1 = value2 = value3 = 0.0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                        y2_mapped, y3_mapped, w : DATA_ALIGN ) \
            private( kernel ) reduction( + : value1, value2, value3 ) \
              simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel = _kernel->anti_tau_limit(
                x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
                ny_data )
      * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc factor = timestep * test_area * trial_area;
  values_out[0] = value1 * factor;
  values_out[1] = value2 * factor;
  values_out[2] = value3 * factor;
  return;

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_delta0(sc * values_out, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;
  linear_algebra::coordinates< 3 > ny;
  sc * ny_data = ny.data( );

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
  trial_mesh->get_spatial_normal( i_trial, ny );

  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  get_type( i_test, i_trial, n_shared_vertices, rot_test, rot_trial );

  triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices, rot_test, rot_trial, quadr_changing );

  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );
  
  const sc * y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
  const sc * w = my_quadrature._w[ n_shared_vertices ].data( );

  lo size = my_quadrature._sizes[ n_shared_vertices ];


  sc kernel;
  sc value1, value2, value3;

  value1 = value2 = value3 = 0.0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
                y2_mapped, y3_mapped, w : DATA_ALIGN ) \
    private( kernel ) reduction( + : value1, value2, value3 ) \
      simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel = _kernel->anti_tau_anti_t_limit_in_time_regular_in_space(
                x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
                ny_data )
      * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc factor = test_area * trial_area;
  values_out[0] = value1 * factor;
  values_out[1] = value2 * factor;
  values_out[2] = value3 * factor;
  return;

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_regular(sc * values_out, lo delta, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );

  sc ttau = timestep * delta;

  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;
  linear_algebra::coordinates< 3 > ny;
  sc * ny_data = ny.data( );

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
  trial_mesh->get_spatial_normal( i_trial, ny );

  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices, rot_test, rot_trial, quadr_changing );

  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );
  
  const sc * y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
  const sc * w = my_quadrature._w[ n_shared_vertices ].data( );

  lo size = my_quadrature._sizes[ n_shared_vertices ];


  sc kernel;
  sc value1, value2, value3;

  value1 = value2 = value3 = 0.0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
              y2_mapped, y3_mapped, w : DATA_ALIGN ) \
  private( kernel ) reduction( + : value1, value2, value3 ) \
    simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel
      = _kernel->anti_tau_anti_t_regular_in_time_regular_in_space(
          x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
          x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
          x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
          ny_data, ttau )
      * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc factor = test_area * trial_area;
  values_out[0] = value1 * factor;
  values_out[1] = value2 * factor;
  values_out[2] = value3 * factor;
  return;

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_singular(sc * values_out, lo delta, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );

  sc ttau = timestep * delta;

  int n_shared_vertices = 0;
  int rot_test = 0;
  int rot_trial = 0;

  linear_algebra::coordinates< 3 > x1, x2, x3;
  linear_algebra::coordinates< 3 > y1, y2, y3;
  linear_algebra::coordinates< 3 > ny;
  sc * ny_data = ny.data( );

  test_mesh->get_spatial_nodes( i_test, x1, x2, x3 );
  trial_mesh->get_spatial_nodes( i_trial, y1, y2, y3 );
  trial_mesh->get_spatial_normal( i_trial, ny );

  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  triangles_to_geometry( x1, x2, x3, y1, y2, y3, n_shared_vertices, rot_test, rot_trial, quadr_changing );

  const sc * x1_mapped = quadr_changing._x1.data( );
  const sc * x2_mapped = quadr_changing._x2.data( );
  const sc * x3_mapped = quadr_changing._x3.data( );
  const sc * y1_mapped = quadr_changing._y1.data( );
  const sc * y2_mapped = quadr_changing._y2.data( );
  const sc * y3_mapped = quadr_changing._y3.data( );
  
  const sc * y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
  const sc * w = my_quadrature._w[ n_shared_vertices ].data( );

  lo size = my_quadrature._sizes[ n_shared_vertices ];


  sc kernel;
  sc value1, value2, value3;

  value1 = value2 = value3 = 0.0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, y1_mapped, \
              y2_mapped, y3_mapped, w : DATA_ALIGN ) \
  private( kernel ) reduction( + : value1, value2, value3 ) \
    simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel = _kernel->anti_tau_anti_t_regular_in_time(
                x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nullptr,
                ny_data, ttau )
      * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc factor = test_area * trial_area;
  values_out[0] = value1 * factor;
  values_out[1] = value2 * factor;
  values_out[2] = value3 * factor;
  return;

}
















template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>::
  get_values(sc * values_out, lo delta, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing, bool special) const {

  if(delta == 0 && special) {
    get_values_delta0special(values_out, i_test, i_trial, quadr_changing);
    return;
  }

  if ( delta == 0 ) {
    get_values_delta0(values_out, i_test, i_trial, quadr_changing);
  } else {
    if ( i_test != i_trial ) {
      get_values_regular(values_out, delta, i_test, i_trial, quadr_changing);
    } else {
      get_values_singular(values_out, delta, i_test, i_trial, quadr_changing);
    }
  }

  return;

}















template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_cpu( const block_vector_type & x, block_vector_type & y,
  bool trans, sc alpha, sc beta ) const {
  
  // y = alpha*A*x + beta*y;
  if(trans) {
    std::cerr << "I dont support trans matrices\n";
    return;
  }

  lo rows_in_block = _n_rows;
  lo cols_in_block = _n_columns;
  lo blocks = _block_dim;


  // permuting the vector y should (in theory) prevent false sharing and improve data locality
  // in practice it does really work
  block_vector_type y_perm;
  y_perm.copy_permute(y, beta);

#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(my_quadrature._max_size);
    sc val_prev;
    sc val_curr;
    sc val_next;

#pragma omp for
    for (lo inner_row = 0; inner_row < rows_in_block; inner_row++) {
      for (lo inner_col = 0; inner_col < cols_in_block; inner_col++) {

        get_values(&val_prev, 0, inner_row, inner_col, quadr_changing, true);
        get_values(&val_curr, 0, inner_row, inner_col, quadr_changing);
        get_values(&val_next, 1, inner_row, inner_col, quadr_changing);

        sc matrix_val = val_prev + val_curr - val_next;
        lo max_block = blocks;
        for (lo block = 0; block < max_block; block++) {
          sc y_val = alpha * matrix_val * x.get(block, inner_col);
          y_perm.add(inner_row, block, y_val);
        }

        for (lo diag = 1; diag < blocks; diag++) {
          val_prev = val_curr;
          val_curr = val_next;
          get_values(&val_next, diag+1, inner_row, inner_col, quadr_changing);

          matrix_val = -val_prev + 2*val_curr - val_next;
          
          lo max_block = blocks - diag;
          for (lo block = 0; block < max_block; block++) {
            lo block_row = diag + block;
            lo block_col = block;
            sc x_val = x.get(block_col, inner_col);
            sc y_val = alpha * matrix_val * x_val;
            y_perm.add(inner_row, block_row, y_val);
          }

        }
      }
    }

  }


  y.copy_permute(y_perm);

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_cpu( const block_vector_type & x, block_vector_type & y,
  bool trans, sc alpha, sc beta ) const {
  
  // y = alpha*A*x + beta*y;
  if(trans) {
    std::cerr << "I dont support trans matrices\n";
    return;
  }

  auto & trial_basis = _trial_space->get_basis( );

  lo tst_spelems_count = _test_space->get_mesh()->get_n_spatial_elements();
  lo trl_spelems_count = _trial_space->get_mesh()->get_n_spatial_elements();
  lo blocks = _block_dim;


  block_vector_type y_perm;
  y_perm.copy_permute(y, beta);

#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(my_quadrature._max_size);
    sc vals_prev[3];
    sc vals_curr[3];
    sc vals_next[3];
    lo row;
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols_0(3);
    std::vector< lo > cols(3);
    int n_shared_vertices = 0;
    int rot_test = 0;
    int rot_trial = 0;
    sc matrix_vals[3];

#pragma omp for
    for (lo i_tst = 0; i_tst < tst_spelems_count; i_tst++) {
      for (lo i_trl = 0; i_trl < trl_spelems_count; i_trl++) {

        row = i_tst;
        get_type( i_tst, i_trl, n_shared_vertices, rot_test, rot_trial );
        trial_basis.local_to_global(i_trl, n_shared_vertices, rot_trial, true, cols_0 );
        n_shared_vertices = 0;
        rot_test = 0;
        rot_trial = 0;
        trial_basis.local_to_global(i_trl, n_shared_vertices, rot_trial, true, cols );

        get_values(vals_prev, 0, i_tst, i_trl, quadr_changing, true);
        get_values(vals_curr, 0, i_tst, i_trl, quadr_changing);
        get_values(vals_next, 1, i_tst, i_trl, quadr_changing);

        matrix_vals[0] = vals_prev[0] + vals_curr[0];
        matrix_vals[1] = vals_prev[1] + vals_curr[1];
        matrix_vals[2] = vals_prev[2] + vals_curr[2];
        max_block = blocks;
        for (lo block = 0; block < max_block; block++) {
          block_row = 0 + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x.get(block_col, cols_0[0]);
          y_val += matrix_vals[1] * x.get(block_col, cols_0[1]);
          y_val += matrix_vals[2] * x.get(block_col, cols_0[2]);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

        matrix_vals[0] = -vals_next[0];
        matrix_vals[1] = -vals_next[1];
        matrix_vals[2] = -vals_next[2];
        max_block = blocks;
        for (lo block = 0; block < max_block; block++) {
          block_row = 0 + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x.get(block_col, cols[0]);
          y_val += matrix_vals[1] * x.get(block_col, cols[1]);
          y_val += matrix_vals[2] * x.get(block_col, cols[2]);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

        matrix_vals[0] = -vals_curr[0];
        matrix_vals[1] = -vals_curr[1];
        matrix_vals[2] = -vals_curr[2];
        max_block = blocks - 1;
        for (lo block = 0; block < max_block; block++) {
          block_row = 1 + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x.get(block_col, cols_0[0]);
          y_val += matrix_vals[1] * x.get(block_col, cols_0[1]);
          y_val += matrix_vals[2] * x.get(block_col, cols_0[2]);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

        vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;


        for (lo diag = 1; diag < blocks; diag++) {
          vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
          vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
          get_values(vals_next, diag+1, i_tst, i_trl, quadr_changing);

          matrix_vals[0] = -vals_prev[0] + 2*vals_curr[0] - vals_next[0];
          matrix_vals[1] = -vals_prev[1] + 2*vals_curr[1] - vals_next[1];
          matrix_vals[2] = -vals_prev[2] + 2*vals_curr[2] - vals_next[2];

          max_block = blocks - diag;
          for (lo block = 0; block < max_block; block++) {
            block_row = diag + block;
            block_col = block;
            sc y_val = 0;
            y_val += matrix_vals[0] * x.get(block_col, cols[0]);
            y_val += matrix_vals[1] * x.get(block_col, cols[1]);
            y_val += matrix_vals[2] * x.get(block_col, cols[2]);
            y_val *= alpha;
            y_perm.add(row, block_row, y_val);
          }

        }
      }
    }

  }


  y.copy_permute(y_perm);

}















template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_regular(  const block_vector_type & x, block_vector_type & y_perm, sc alpha ) const {

  lo rows_in_block = _n_rows;
  lo cols_in_block = _n_columns;
  lo blocks = _block_dim;

#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(my_quadrature._max_size);
    sc val_prev;
    sc val_curr;
    sc val_next;

#pragma omp for
    for (lo inner_row = 0; inner_row < rows_in_block; inner_row++) {
      for (lo inner_col = 0; inner_col < cols_in_block; inner_col++) {

        if (inner_row == inner_col) // singular is done elsewhere
          continue;

        val_prev = 0; // delta0 is done elsewhere
        val_curr = 0;
        get_values_regular(&val_next, 1, inner_row, inner_col, quadr_changing);

        sc matrix_val = val_prev + val_curr - val_next;
        lo max_block = blocks;
        for (lo block = 0; block < max_block; block++) {
          sc y_val = alpha * matrix_val * x.get(block, inner_col);
          y_perm.add(inner_row, block, y_val);
        }

        for (lo diag = 1; diag < blocks; diag++) {
          val_prev = val_curr;
          val_curr = val_next;
          get_values_regular(&val_next, diag+1, inner_row, inner_col, quadr_changing);

          matrix_val = -val_prev + 2*val_curr - val_next;
          
          lo max_block = blocks - diag;
          for (lo block = 0; block < max_block; block++) {
            lo block_row = diag + block;
            lo block_col = block;
            sc x_val = x.get(block_col, inner_col);
            sc y_val = alpha * matrix_val * x_val;
            y_perm.add(inner_row, block_row, y_val);
          }

        }
      }
    }

  }

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_singular( const block_vector_type & x, block_vector_type & y_perm, sc alpha) const {

  lo rows_in_block = _n_rows;
  lo blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(my_quadrature._max_size);
    sc val_prev;
    sc val_curr;
    sc val_next;
    
#pragma omp for
    for (lo i = 0; i < rows_in_block; i++) {
      lo inner_row = i;
      lo inner_col = i;

      val_prev = 0; // delta0 is done elsewhere
      val_curr = 0;
      get_values_singular(&val_next, 1, inner_row, inner_col, quadr_changing);

      sc matrix_val = val_prev + val_curr - val_next;
      lo max_block = blocks;
      for (lo block = 0; block < max_block; block++) {
        sc y_val = alpha * matrix_val * x.get(block, inner_col);
        y_perm.add(inner_row, block, y_val);
      }

      for (lo diag = 1; diag < blocks; diag++) {
        val_prev = val_curr;
        val_curr = val_next;
        get_values_singular(&val_next, diag+1, inner_row, inner_col, quadr_changing);

        matrix_val = -val_prev + 2*val_curr - val_next;

        lo max_block = blocks - diag;
        for (lo block = 0; block < max_block; block++) {
          lo block_row = diag + block;
          lo block_col = block;
          sc x_val = x.get(block_col, inner_col);
          sc y_val = alpha * matrix_val * x_val;
          y_perm.add(inner_row, block_row, y_val);
        }

      }
    }

  }

}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_delta0(   const block_vector_type & x, block_vector_type & y_perm, sc alpha ) const {

  lo rows_in_block = _n_rows;
  lo cols_in_block = _n_columns;
  lo blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(my_quadrature._max_size);
    sc val_prev;
    sc val_curr;

#pragma omp for
    for (lo inner_row = 0; inner_row < rows_in_block; inner_row++) {
      for (lo inner_col = 0; inner_col < cols_in_block; inner_col++) {

        get_values_delta0special(&val_prev, inner_row, inner_col, quadr_changing);
        get_values_delta0(&val_curr, inner_row, inner_col, quadr_changing);

        sc matrix_val = val_prev + val_curr;
        lo max_block = blocks;
        for (lo block = 0; block < max_block; block++) {
          sc y_val = alpha * matrix_val * x.get(block, inner_col);
          y_perm.add(inner_row, block, y_val);
        }

        constexpr lo diag = 1;
        matrix_val = -val_curr;
        max_block = blocks - diag;
        for (lo block = 0; block < max_block; block++) {
          lo block_row = diag + block;
          lo block_col = block;
          sc x_val = x.get(block_col, inner_col);
          sc y_val = alpha * matrix_val * x_val;
          y_perm.add(inner_row, block_row, y_val);
        }

      }
    }

  }

}















template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_regular(  const block_vector_type & x, block_vector_type & y_perm, sc alpha ) const {

  auto & trial_basis = _trial_space->get_basis( );

  lo tst_spelems_count = _test_space->get_mesh()->get_n_spatial_elements();
  lo trl_spelems_count = _trial_space->get_mesh()->get_n_spatial_elements();
  lo blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(my_quadrature._max_size);
    sc vals_prev[3];
    sc vals_curr[3];
    sc vals_next[3];
    lo row;
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols(3);
    constexpr int n_shared_vertices = 0;
    constexpr int rot_trial = 0;
    sc matrix_vals[3];

#pragma omp for
    for (lo i_tst = 0; i_tst < tst_spelems_count; i_tst++) {
      for (lo i_trl = 0; i_trl < trl_spelems_count; i_trl++) {

        if (i_tst == i_trl) // singular is done elsewhere
          continue;

        row = i_tst;

        trial_basis.local_to_global(i_trl, n_shared_vertices, rot_trial, true, cols );

        vals_prev[0] = 0;   vals_prev[1] = 0;   vals_prev[2] = 0; // delta0 is done elsewhere
        vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0; 
        get_values_regular(vals_next, 1, i_tst, i_trl, quadr_changing);

        matrix_vals[0] = -vals_next[0];
        matrix_vals[1] = -vals_next[1];
        matrix_vals[2] = -vals_next[2];
        max_block = blocks;
        for (lo block = 0; block < max_block; block++) {
          block_row = 0 + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x.get(block_col, cols[0]);
          y_val += matrix_vals[1] * x.get(block_col, cols[1]);
          y_val += matrix_vals[2] * x.get(block_col, cols[2]);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

        for (lo diag = 1; diag < blocks; diag++) {
          vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
          vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
          get_values_regular(vals_next, diag+1, i_tst, i_trl, quadr_changing);

          matrix_vals[0] = -vals_prev[0] + 2*vals_curr[0] - vals_next[0];
          matrix_vals[1] = -vals_prev[1] + 2*vals_curr[1] - vals_next[1];
          matrix_vals[2] = -vals_prev[2] + 2*vals_curr[2] - vals_next[2];

          max_block = blocks - diag;
          for (lo block = 0; block < max_block; block++) {
            block_row = diag + block;
            block_col = block;
            sc y_val = 0;
            y_val += matrix_vals[0] * x.get(block_col, cols[0]);
            y_val += matrix_vals[1] * x.get(block_col, cols[1]);
            y_val += matrix_vals[2] * x.get(block_col, cols[2]);
            y_val *= alpha;
            y_perm.add(row, block_row, y_val);
          }

        }
      }
    }

  }


}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_singular( const block_vector_type & x, block_vector_type & y_perm, sc alpha) const {

  auto & trial_basis = _trial_space->get_basis( );

  lo tst_spelems_count = _test_space->get_mesh()->get_n_spatial_elements();
  lo blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(my_quadrature._max_size);
    sc vals_prev[3];
    sc vals_curr[3];
    sc vals_next[3];
    lo row;
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols(3);
    constexpr int n_shared_vertices = 0;
    constexpr int rot_trial = 0;
    sc matrix_vals[3];

#pragma omp for
    for (lo i = 0; i < tst_spelems_count; i++) {
      lo i_tst = i;
      lo i_trl = i;

      row = i_tst;

      trial_basis.local_to_global(i_trl, n_shared_vertices, rot_trial, true, cols );

      vals_prev[0] = 0;   vals_prev[1] = 0;   vals_prev[2] = 0; // delta0 is done elsewhere
      vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
      get_values_singular(vals_next, 1, i_tst, i_trl, quadr_changing);

      matrix_vals[0] = -vals_next[0];
      matrix_vals[1] = -vals_next[1];
      matrix_vals[2] = -vals_next[2];
      max_block = blocks;
      for (lo block = 0; block < max_block; block++) {
        block_row = 0 + block;
        block_col = block;
        sc y_val = 0;
        y_val += matrix_vals[0] * x.get(block_col, cols[0]);
        y_val += matrix_vals[1] * x.get(block_col, cols[1]);
        y_val += matrix_vals[2] * x.get(block_col, cols[2]);
        y_val *= alpha;
        y_perm.add(row, block_row, y_val);
      }

      for (lo diag = 1; diag < blocks; diag++) {
        vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
        vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
        get_values_singular(vals_next, diag+1, i_tst, i_trl, quadr_changing);

        matrix_vals[0] = -vals_prev[0] + 2*vals_curr[0] - vals_next[0];
        matrix_vals[1] = -vals_prev[1] + 2*vals_curr[1] - vals_next[1];
        matrix_vals[2] = -vals_prev[2] + 2*vals_curr[2] - vals_next[2];

        max_block = blocks - diag;
        for (lo block = 0; block < max_block; block++) {
          block_row = diag + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x.get(block_col, cols[0]);
          y_val += matrix_vals[1] * x.get(block_col, cols[1]);
          y_val += matrix_vals[2] * x.get(block_col, cols[2]);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

      }
    }

  }


}



template<>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_delta0(   const block_vector_type & x, block_vector_type & y_perm, sc alpha ) const {

  auto & trial_basis = _trial_space->get_basis( );

  lo tst_spelems_count = _test_space->get_mesh()->get_n_spatial_elements();
  lo trl_spelems_count = _trial_space->get_mesh()->get_n_spatial_elements();
  lo blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_wrapper_changing quadr_changing(my_quadrature._max_size);
    sc vals_prev[3];
    sc vals_curr[3];
    lo row;
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols_0(3);
    int n_shared_vertices = 0;
    int rot_test = 0;
    int rot_trial = 0;
    sc matrix_vals[3];

#pragma omp for
    for (lo i_tst = 0; i_tst < tst_spelems_count; i_tst++) {
      for (lo i_trl = 0; i_trl < trl_spelems_count; i_trl++) {

        row = i_tst;
        get_type( i_tst, i_trl, n_shared_vertices, rot_test, rot_trial );
        trial_basis.local_to_global(i_trl, n_shared_vertices, rot_trial, true, cols_0 );

        get_values_delta0special(vals_prev, i_tst, i_trl, quadr_changing);
        get_values_delta0(vals_curr, i_tst, i_trl, quadr_changing);

        matrix_vals[0] = vals_prev[0] + vals_curr[0];
        matrix_vals[1] = vals_prev[1] + vals_curr[1];
        matrix_vals[2] = vals_prev[2] + vals_curr[2];
        max_block = blocks;
        for (lo block = 0; block < max_block; block++) {
          block_row = 0 + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x.get(block_col, cols_0[0]);
          y_val += matrix_vals[1] * x.get(block_col, cols_0[1]);
          y_val += matrix_vals[2] * x.get(block_col, cols_0[2]);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

        matrix_vals[0] = -vals_curr[0];
        matrix_vals[1] = -vals_curr[1];
        matrix_vals[2] = -vals_curr[2];
        constexpr lo diag = 1;
        max_block = blocks - diag;
        for (lo block = 0; block < max_block; block++) {
          block_row = diag + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x.get(block_col, cols_0[0]);
          y_val += matrix_vals[1] * x.get(block_col, cols_0[1]);
          y_val += matrix_vals[2] * x.get(block_col, cols_0[2]);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

      }
    }

  }


}















template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>::
  apply( const block_vector_type & x, block_vector_type & y, bool trans, sc alpha, sc beta ) const {

  if(trans) {
    std::cerr << "I dont support trans matrices\n";
    return;
  }
  
  // permuting the vector y should (in theory) prevent false sharing and improve data locality
  // in practice it does really work
  block_vector_type y_perm;
  y_perm.copy_permute(y, beta);


  this->apply_regular(x, y_perm, alpha);
  this->apply_singular(x, y_perm, alpha);
  this->apply_delta0(x, y_perm, alpha);


  y.copy_permute(y_perm);

}
















template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>::init_quadrature( ) {
  // Use triangle rules for disjoint elements
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x1
    = besthea::bem::quadrature::triangle_x1( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_x2
    = besthea::bem::quadrature::triangle_x2( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tri_w
    = besthea::bem::quadrature::triangle_w( _order_regular );
  lo tri_size = tri_w.size( );
  lo tri_size2 = tri_size * tri_size;

  lo size = tri_size2;
  int n_shared_vertices = 0;
  my_quadrature._sizes[ n_shared_vertices ] = size;
  my_quadrature._x1_ref[ n_shared_vertices ].resize( size );
  my_quadrature._x2_ref[ n_shared_vertices ].resize( size );
  my_quadrature._y1_ref[ n_shared_vertices ].resize( size );
  my_quadrature._y2_ref[ n_shared_vertices ].resize( size );
  my_quadrature._w[ n_shared_vertices ].resize( size );

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
    size = line_size4 * n_simplices[ n_shared_vertices ];
    my_quadrature._sizes[ n_shared_vertices ] = size;
    my_quadrature._x1_ref[ n_shared_vertices ].resize( size );
    my_quadrature._x2_ref[ n_shared_vertices ].resize( size );
    my_quadrature._y1_ref[ n_shared_vertices ].resize( size );
    my_quadrature._y2_ref[ n_shared_vertices ].resize( size );
    my_quadrature._w[ n_shared_vertices ].resize( size );

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

  my_quadrature._max_size = *std::max_element(my_quadrature._sizes.begin(), my_quadrature._sizes.end());

}



template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>::triangles_to_geometry( const linear_algebra::coordinates< 3 > & x1,
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
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<kernel_type, test_space_type, trial_space_type>::get_type( lo i_test, lo i_trial, int & n_shared_vertices,
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
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu< kernel_type, test_space_type,
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
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu< kernel_type, test_space_type,
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
void besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu< kernel_type, test_space_type,
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




template class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
  

template class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

// TODO: add the rest of them

