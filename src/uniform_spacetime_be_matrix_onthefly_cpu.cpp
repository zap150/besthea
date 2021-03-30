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

#include "besthea/uniform_spacetime_be_matrix_onthefly_cpu.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_be_space.h"
#include "besthea/time_measurer.h"

#include <iostream>
#include <exception>



template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  < kernel_type, test_space_type, trial_space_type >::
  uniform_spacetime_be_matrix_onthefly_cpu( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular, int order_regular )
  : block_matrix(
      test_space.get_mesh()->get_n_temporal_elements(),
      test_space.get_basis().dimension_global(),
      trial_space.get_basis().dimension_global()
    ),
    _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ) {

  switch(_order_singular) {
    case 1:
    case 2:
    case 4:
    case 5:
      break;
    default:
      if(besthea::settings::output_verbosity.warnings >= 1) {
        std::cerr << "BESTHEA Warning: quadrature_order_singular "
          << _order_singular
          << " not available. Using default quadrature_order_singular=1.\n";
      }
      _order_singular = 1;
      break;
  }

  switch(_order_regular) {
    case 1:
    case 2:
    case 4:
    case 5:
      break;
    default:
      if(besthea::settings::output_verbosity.warnings >= 1) {
        std::cerr << "BESTHEA Warning: quadrature_order_regular "
          << _order_regular
          << " not available. Using default quadrature_order_regular=1.\n";
      }
      _order_regular = 1;
      break;
  }

  init_quadrature();

}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  < kernel_type, test_space_type, trial_space_type >::
  ~uniform_spacetime_be_matrix_onthefly_cpu( ) {

}















template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_values_regular(sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  // assuming delta > 0 && i_test != i_trial

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );
  sc ttau = timestep * delta;
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * w = quadr_reference._w[0].data( );
  lo size = quadr_reference._sizes[0];
  sc value = 0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, \
                          y1_mapped, y2_mapped, y3_mapped, \
                          w : DATA_ALIGN ) \
            reduction( + : value ) simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    value += _kernel->anti_tau_anti_t_regular_in_time_regular_in_space(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, nullptr, ttau ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  *values_out = value * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_values_singular(sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {
  
  // assuming delta > 0 && i_test == i_trial

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );
  sc ttau = timestep * delta;
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * w = quadr_reference._w[0].data( );
  lo size = quadr_reference._sizes[0];
  sc value = 0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, \
                          y1_mapped, y2_mapped, y3_mapped, \
                          w : DATA_ALIGN ) \
            reduction( + : value ) simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    value += _kernel->anti_tau_anti_t_regular_in_time(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, nullptr, ttau ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  *values_out = value * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_values_delta0(sc * values_out, lo i_test, lo i_trial,
    int n_shared_vertices, [[maybe_unused]] int rot_test,
    [[maybe_unused]] int rot_trial, const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * w = quadr_reference._w[ n_shared_vertices ].data( );
  lo size = quadr_reference._sizes[ n_shared_vertices ];
  sc value = 0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, \
                          y1_mapped, y2_mapped, y3_mapped, \
                          w : DATA_ALIGN ) \
            reduction( + : value ) simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    value += _kernel->anti_tau_anti_t_limit_in_time_regular_in_space(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, nullptr ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  *values_out = value * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  get_values_delta0special(sc * values_out, lo i_test, lo i_trial,
    int n_shared_vertices, [[maybe_unused]] int rot_test,
    [[maybe_unused]] int rot_trial, const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {
  
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * w = quadr_reference._w[ n_shared_vertices ].data( );
  lo size = quadr_reference._sizes[ n_shared_vertices ];
  sc value = 0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, \
                          y1_mapped, y2_mapped, y3_mapped, \
                          w : DATA_ALIGN ) \
            reduction( + : value ) simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    value += _kernel->anti_tau_limit(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, nullptr ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area * timestep;
  *values_out = value * multiplier;
  return;
}

















template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_regular(sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  // assuming delta > 0 && i_test != i_trial

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );
  sc ttau = timestep * delta;
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  linear_algebra::coordinates< 3 > ny;
  trial_mesh->get_spatial_normal( i_trial, ny );
  sc * ny_data = ny.data( );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * y1_ref = quadr_reference._y1_ref[0].data( );
  const sc * y2_ref = quadr_reference._y2_ref[0].data( );
  const sc * w = quadr_reference._w[0].data( );
  lo size = quadr_reference._sizes[0];
  sc kernel;
  sc value1 = 0, value2 = 0, value3 = 0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, \
                          y1_mapped, y2_mapped, y3_mapped, \
                          w : DATA_ALIGN ) \
            private( kernel ) \
            reduction( + : value1, value2, value3 ) \
            simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel = _kernel->anti_tau_anti_t_regular_in_time_regular_in_space(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, ny_data, ttau ) * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  values_out[0] = value1 * multiplier;
  values_out[1] = value2 * multiplier;
  values_out[2] = value3 * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_singular(sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  // assuming delta > 0 && i_test == i_trial

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );
  sc ttau = timestep * delta;
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  linear_algebra::coordinates< 3 > ny;
  trial_mesh->get_spatial_normal( i_trial, ny );
  sc * ny_data = ny.data( );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * y1_ref = quadr_reference._y1_ref[0].data( );
  const sc * y2_ref = quadr_reference._y2_ref[0].data( );
  const sc * w = quadr_reference._w[0].data( );
  lo size = quadr_reference._sizes[0];
  sc kernel;
  sc value1 = 0, value2 = 0, value3 = 0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, \
                          y1_mapped, y2_mapped, y3_mapped, \
                          w : DATA_ALIGN ) \
            private( kernel ) \
            reduction( + : value1, value2, value3 ) \
            simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel = _kernel->anti_tau_anti_t_regular_in_time(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, ny_data, ttau ) * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  values_out[0] = value1 * multiplier;
  values_out[1] = value2 * multiplier;
  values_out[2] = value3 * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_delta0(sc * values_out, lo i_test, lo i_trial,
    int n_shared_vertices, [[maybe_unused]] int rot_test,
    [[maybe_unused]] int rot_trial, const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  linear_algebra::coordinates< 3 > ny;
  trial_mesh->get_spatial_normal( i_trial, ny );
  sc * ny_data = ny.data( );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * y1_ref = quadr_reference._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = quadr_reference._y2_ref[ n_shared_vertices ].data( );
  const sc * w = quadr_reference._w[ n_shared_vertices ].data( );
  lo size = quadr_reference._sizes[ n_shared_vertices ];
  sc kernel;
  sc value1 = 0, value2 = 0, value3 = 0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, \
                          y1_mapped, y2_mapped, y3_mapped, \
                          w : DATA_ALIGN ) \
            private( kernel ) \
            reduction( + : value1, value2, value3 ) \
            simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel = _kernel->anti_tau_anti_t_limit_in_time_regular_in_space(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, ny_data ) * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  values_out[0] = value1 * multiplier;
  values_out[1] = value2 * multiplier;
  values_out[2] = value3 * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_delta0special(sc * values_out, lo i_test, lo i_trial,
    int n_shared_vertices, [[maybe_unused]] int rot_test,
    [[maybe_unused]] int rot_trial, const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );
  
  sc timestep = test_mesh->get_timestep( );
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  linear_algebra::coordinates< 3 > ny;
  trial_mesh->get_spatial_normal( i_trial, ny );
  sc * ny_data = ny.data( );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * y1_ref = quadr_reference._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = quadr_reference._y2_ref[ n_shared_vertices ].data( );
  const sc * w = quadr_reference._w[ n_shared_vertices ].data( );
  lo size = quadr_reference._sizes[ n_shared_vertices ];
  sc kernel;
  sc value1 = 0, value2 = 0, value3 = 0;

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, \
                          y1_mapped, y2_mapped, y3_mapped, \
                          w : DATA_ALIGN ) \
            private( kernel ) \
            reduction( + : value1, value2, value3 ) \
            simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    kernel = _kernel->anti_tau_limit(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, ny_data ) * w[ i_quad ];
    value1 += kernel * ( (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ] );
    value2 += kernel * y1_ref[ i_quad ];
    value3 += kernel * y2_ref[ i_quad ];
  }

  sc multiplier = test_area * trial_area * timestep;
  values_out[0] = value1 * multiplier;
  values_out[1] = value2 * multiplier;
  values_out[2] = value3 * multiplier;
  return;
}















template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_regular(sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  // assuming delta > 0 && i_test != i_trial

  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  sc timestep = test_mesh->get_timestep( );
  sc ttau = timestep * delta;
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  linear_algebra::coordinates< 3 > nx, ny;
  test_mesh->get_spatial_normal( i_test, nx );
  trial_mesh->get_spatial_normal( i_trial, ny );
  sc * nx_data = nx.data( );
  sc * ny_data = ny.data( );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * x1_ref = quadr_reference._x1_ref[0].data( );
  const sc * x2_ref = quadr_reference._x2_ref[0].data( );
  const sc * y1_ref = quadr_reference._y1_ref[0].data( );
  const sc * y2_ref = quadr_reference._y2_ref[0].data( );
  const sc * w = quadr_reference._w[0].data( );

  sc test_curls[ 9 ], trial_curls[ 9 ];
  sc curl_dot[ 9 ];
  lo test_curl_offset, trial_curl_offset;
  test_basis.evaluate_curl( i_test, nx, 0, 0, false, test_curls );
  trial_basis.evaluate_curl( i_trial, ny, 0, 0, true, trial_curls );
  for ( lo i_loc_test = 0; i_loc_test < 3; ++i_loc_test ) {
    test_curl_offset = 3 * i_loc_test;
    for ( lo i_loc_trial = 0; i_loc_trial < 3; ++i_loc_trial ) {
      trial_curl_offset = 3 * i_loc_trial;
      curl_dot[ i_loc_trial * 3 + i_loc_test ]
        = test_curls[ test_curl_offset     ] * trial_curls[ trial_curl_offset ]
        + test_curls[ test_curl_offset + 1 ] * trial_curls[ trial_curl_offset + 1 ]
        + test_curls[ test_curl_offset + 2 ] * trial_curls[ trial_curl_offset + 2 ];
    }
  }

  lo size = quadr_reference._sizes[0];
  sc phi1x, phi1y;
  sc kernel1, kernel2;
  sc value11 = 0, value12 = 0, value13 = 0;
  sc value21 = 0, value22 = 0, value23 = 0;
  sc value31 = 0, value32 = 0, value33 = 0;

#pragma omp simd \
        aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref : DATA_ALIGN ) \
        aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w : DATA_ALIGN ) \
        private( kernel1, kernel2, phi1x, phi1y ) \
        reduction( + : value11, value12, value13 ) \
        reduction( + : value21, value22, value23 ) \
        reduction( + : value31, value32, value33 ) \
        simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    _kernel->anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space(
        x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
        x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
        x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
        nx_data, ny_data, ttau, &kernel1, &kernel2 );

    phi1x = (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ];
    phi1y = (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ];

    value11 += ( kernel1 * curl_dot[ 0 ] + kernel2 * phi1x            * phi1y )            * w[ i_quad ];
    value21 += ( kernel1 * curl_dot[ 1 ] + kernel2 * x1_ref[ i_quad ] * phi1y )            * w[ i_quad ];
    value31 += ( kernel1 * curl_dot[ 2 ] + kernel2 * x2_ref[ i_quad ] * phi1y )            * w[ i_quad ];
    value12 += ( kernel1 * curl_dot[ 3 ] + kernel2 * phi1x            * y1_ref[ i_quad ] ) * w[ i_quad ];
    value22 += ( kernel1 * curl_dot[ 4 ] + kernel2 * x1_ref[ i_quad ] * y1_ref[ i_quad ] ) * w[ i_quad ];
    value32 += ( kernel1 * curl_dot[ 5 ] + kernel2 * x2_ref[ i_quad ] * y1_ref[ i_quad ] ) * w[ i_quad ];
    value13 += ( kernel1 * curl_dot[ 6 ] + kernel2 * phi1x            * y2_ref[ i_quad ] ) * w[ i_quad ];
    value23 += ( kernel1 * curl_dot[ 7 ] + kernel2 * x1_ref[ i_quad ] * y2_ref[ i_quad ] ) * w[ i_quad ];
    value33 += ( kernel1 * curl_dot[ 8 ] + kernel2 * x2_ref[ i_quad ] * y2_ref[ i_quad ] ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  values_out[0] = value11 * multiplier;
  values_out[1] = value12 * multiplier;
  values_out[2] = value13 * multiplier;
  values_out[3] = value21 * multiplier;
  values_out[4] = value22 * multiplier;
  values_out[5] = value23 * multiplier;
  values_out[6] = value31 * multiplier;
  values_out[7] = value32 * multiplier;
  values_out[8] = value33 * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_singular(sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  // assuming delta > 0 && i_test == i_trial

  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  sc timestep = test_mesh->get_timestep( );
  sc ttau = timestep * delta;
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  linear_algebra::coordinates< 3 > nx, ny;
  test_mesh->get_spatial_normal( i_test, nx );
  trial_mesh->get_spatial_normal( i_trial, ny );
  sc * nx_data = nx.data( );
  sc * ny_data = ny.data( );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * x1_ref = quadr_reference._x1_ref[0].data( );
  const sc * x2_ref = quadr_reference._x2_ref[0].data( );
  const sc * y1_ref = quadr_reference._y1_ref[0].data( );
  const sc * y2_ref = quadr_reference._y2_ref[0].data( );
  const sc * w = quadr_reference._w[0].data( );

  sc test_curls[ 9 ], trial_curls[ 9 ];
  sc curl_dot[ 9 ];
  lo test_curl_offset, trial_curl_offset;
  test_basis.evaluate_curl( i_test, nx, 0, 0, false, test_curls );
  trial_basis.evaluate_curl( i_trial, ny, 0, 0, true, trial_curls );
  for ( lo i_loc_test = 0; i_loc_test < 3; ++i_loc_test ) {
    test_curl_offset = 3 * i_loc_test;
    for ( lo i_loc_trial = 0; i_loc_trial < 3; ++i_loc_trial ) {
      trial_curl_offset = 3 * i_loc_trial;
      curl_dot[ i_loc_trial * 3 + i_loc_test ]
        = test_curls[ test_curl_offset     ] * trial_curls[ trial_curl_offset     ]
        + test_curls[ test_curl_offset + 1 ] * trial_curls[ trial_curl_offset + 1 ]
        + test_curls[ test_curl_offset + 2 ] * trial_curls[ trial_curl_offset + 2 ];
    }
  }

  lo size = quadr_reference._sizes[0];
  sc phi1x, phi1y;
  sc kernel1, kernel2;
  sc value11 = 0, value12 = 0, value13 = 0;
  sc value21 = 0, value22 = 0, value23 = 0;
  sc value31 = 0, value32 = 0, value33 = 0;

#pragma omp simd \
        aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref : DATA_ALIGN ) \
        aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w : DATA_ALIGN ) \
        private( kernel1, kernel2, phi1x, phi1y ) \
        reduction( + : value11, value12, value13 ) \
        reduction( + : value21, value22, value23 ) \
        reduction( + : value31, value32, value33 ) \
        simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    _kernel->anti_tau_anti_t_and_anti_t_regular_in_time(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nx_data, ny_data, ttau, &kernel1, &kernel2 );

    phi1x = (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ];
    phi1y = (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ];

    value11 += ( kernel1 * curl_dot[ 0 ] + kernel2 * phi1x            * phi1y )            * w[ i_quad ];
    value21 += ( kernel1 * curl_dot[ 1 ] + kernel2 * x1_ref[ i_quad ] * phi1y )            * w[ i_quad ];
    value31 += ( kernel1 * curl_dot[ 2 ] + kernel2 * x2_ref[ i_quad ] * phi1y )            * w[ i_quad ];
    value12 += ( kernel1 * curl_dot[ 3 ] + kernel2 * phi1x            * y1_ref[ i_quad ] ) * w[ i_quad ];
    value22 += ( kernel1 * curl_dot[ 4 ] + kernel2 * x1_ref[ i_quad ] * y1_ref[ i_quad ] ) * w[ i_quad ];
    value32 += ( kernel1 * curl_dot[ 5 ] + kernel2 * x2_ref[ i_quad ] * y1_ref[ i_quad ] ) * w[ i_quad ];
    value13 += ( kernel1 * curl_dot[ 6 ] + kernel2 * phi1x            * y2_ref[ i_quad ] ) * w[ i_quad ];
    value23 += ( kernel1 * curl_dot[ 7 ] + kernel2 * x1_ref[ i_quad ] * y2_ref[ i_quad ] ) * w[ i_quad ];
    value33 += ( kernel1 * curl_dot[ 8 ] + kernel2 * x2_ref[ i_quad ] * y2_ref[ i_quad ] ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  values_out[0] = value11 * multiplier;
  values_out[1] = value12 * multiplier;
  values_out[2] = value13 * multiplier;
  values_out[3] = value21 * multiplier;
  values_out[4] = value22 * multiplier;
  values_out[5] = value23 * multiplier;
  values_out[6] = value31 * multiplier;
  values_out[7] = value32 * multiplier;
  values_out[8] = value33 * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_delta0(sc * values_out, lo i_test, lo i_trial,
    int n_shared_vertices, int rot_test, int rot_trial,
    const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  linear_algebra::coordinates< 3 > nx, ny;
  test_mesh->get_spatial_normal( i_test, nx );
  trial_mesh->get_spatial_normal( i_trial, ny );
  sc * nx_data = nx.data( );
  sc * ny_data = ny.data( );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * x1_ref = quadr_reference._x1_ref[ n_shared_vertices ].data( );
  const sc * x2_ref = quadr_reference._x2_ref[ n_shared_vertices ].data( );
  const sc * y1_ref = quadr_reference._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = quadr_reference._y2_ref[ n_shared_vertices ].data( );
  const sc * w = quadr_reference._w[ n_shared_vertices ].data( );

  sc test_curls[ 9 ], trial_curls[ 9 ];
  sc curl_dot[ 9 ];
  lo test_curl_offset, trial_curl_offset;
  test_basis.evaluate_curl( i_test, nx, n_shared_vertices, rot_test, false, test_curls );
  trial_basis.evaluate_curl( i_trial, ny, n_shared_vertices, rot_trial, true, trial_curls );
  for ( lo i_loc_test = 0; i_loc_test < 3; ++i_loc_test ) {
    test_curl_offset = 3 * i_loc_test;
    for ( lo i_loc_trial = 0; i_loc_trial < 3; ++i_loc_trial ) {
      trial_curl_offset = 3 * i_loc_trial;
      curl_dot[ i_loc_trial * 3 + i_loc_test ]
        = test_curls[ test_curl_offset     ] * trial_curls[ trial_curl_offset ]
        + test_curls[ test_curl_offset + 1 ] * trial_curls[ trial_curl_offset + 1 ]
        + test_curls[ test_curl_offset + 2 ] * trial_curls[ trial_curl_offset + 2 ];
    }
  }

  lo size = quadr_reference._sizes[ n_shared_vertices ];
  sc phi1x, phi1y;
  sc kernel1, kernel2;
  sc value11 = 0, value12 = 0, value13 = 0;
  sc value21 = 0, value22 = 0, value23 = 0;
  sc value31 = 0, value32 = 0, value33 = 0;

#pragma omp simd \
        aligned( x1_mapped, x2_mapped, x3_mapped, x1_ref, x2_ref : DATA_ALIGN ) \
        aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref, w : DATA_ALIGN ) \
        private( kernel1, kernel2, phi1x, phi1y ) \
        reduction( + : value11, value12, value13 ) \
        reduction( + : value21, value22, value23 ) \
        reduction( + : value31, value32, value33 ) \
        simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    _kernel->anti_tau_anti_t_and_anti_t_limit_in_time_regular_in_space(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nx_data, ny_data, &kernel1, &kernel2 );

    phi1x = (sc) 1.0 - x1_ref[ i_quad ] - x2_ref[ i_quad ];
    phi1y = (sc) 1.0 - y1_ref[ i_quad ] - y2_ref[ i_quad ];

    value11 += ( kernel1 * curl_dot[ 0 ] + kernel2 * phi1x            * phi1y )            * w[ i_quad ];
    value21 += ( kernel1 * curl_dot[ 1 ] + kernel2 * x1_ref[ i_quad ] * phi1y )            * w[ i_quad ];
    value31 += ( kernel1 * curl_dot[ 2 ] + kernel2 * x2_ref[ i_quad ] * phi1y )            * w[ i_quad ];
    value12 += ( kernel1 * curl_dot[ 3 ] + kernel2 * phi1x            * y1_ref[ i_quad ] ) * w[ i_quad ];
    value22 += ( kernel1 * curl_dot[ 4 ] + kernel2 * x1_ref[ i_quad ] * y1_ref[ i_quad ] ) * w[ i_quad ];
    value32 += ( kernel1 * curl_dot[ 5 ] + kernel2 * x2_ref[ i_quad ] * y1_ref[ i_quad ] ) * w[ i_quad ];
    value13 += ( kernel1 * curl_dot[ 6 ] + kernel2 * phi1x            * y2_ref[ i_quad ] ) * w[ i_quad ];
    value23 += ( kernel1 * curl_dot[ 7 ] + kernel2 * x1_ref[ i_quad ] * y2_ref[ i_quad ] ) * w[ i_quad ];
    value33 += ( kernel1 * curl_dot[ 8 ] + kernel2 * x2_ref[ i_quad ] * y2_ref[ i_quad ] ) * w[ i_quad ];
  }

  sc multiplier = test_area * trial_area;
  values_out[0] = value11 * multiplier;
  values_out[1] = value12 * multiplier;
  values_out[2] = value13 * multiplier;
  values_out[3] = value21 * multiplier;
  values_out[4] = value22 * multiplier;
  values_out[5] = value23 * multiplier;
  values_out[6] = value31 * multiplier;
  values_out[7] = value32 * multiplier;
  values_out[8] = value33 * multiplier;
  return;
}





template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  get_values_delta0special(sc * values_out, lo i_test, lo i_trial,
    int n_shared_vertices, int rot_test, int rot_trial,
    const quadrature_nodes & quadr_nodes_tst,
    const quadrature_nodes & quadr_nodes_trl) const {

  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  sc timestep = test_mesh->get_timestep( );
  sc test_area = test_mesh->spatial_area( i_test );
  sc trial_area = trial_mesh->spatial_area( i_trial );

  linear_algebra::coordinates< 3 > nx, ny;
  test_mesh->get_spatial_normal( i_test, nx );
  trial_mesh->get_spatial_normal( i_trial, ny );
  sc * ny_data = ny.data( );

  const sc * x1_mapped = quadr_nodes_tst._xs.data( );
  const sc * x2_mapped = quadr_nodes_tst._ys.data( );
  const sc * x3_mapped = quadr_nodes_tst._zs.data( );
  const sc * y1_mapped = quadr_nodes_trl._xs.data( );
  const sc * y2_mapped = quadr_nodes_trl._ys.data( );
  const sc * y3_mapped = quadr_nodes_trl._zs.data( );
  const sc * w = quadr_reference._w[ n_shared_vertices ].data( );

  sc test_curls[ 9 ], trial_curls[ 9 ];
  sc curl_dot[ 9 ];
  lo test_curl_offset, trial_curl_offset;
  test_basis.evaluate_curl( i_test, nx, n_shared_vertices, rot_test, false, test_curls );
  trial_basis.evaluate_curl( i_trial, ny, n_shared_vertices, rot_trial, true, trial_curls );
  for ( lo i_loc_test = 0; i_loc_test < 3; ++i_loc_test ) {
    test_curl_offset = 3 * i_loc_test;
    for ( lo i_loc_trial = 0; i_loc_trial < 3; ++i_loc_trial ) {
      trial_curl_offset = 3 * i_loc_trial;
      curl_dot[ i_loc_trial * 3 + i_loc_test ]
        = test_curls[ test_curl_offset     ] * trial_curls[ trial_curl_offset ]
        + test_curls[ test_curl_offset + 1 ] * trial_curls[ trial_curl_offset + 1 ]
        + test_curls[ test_curl_offset + 2 ] * trial_curls[ trial_curl_offset + 2 ];
    }
  }

  lo size = quadr_reference._sizes[ n_shared_vertices ];
  sc value11 = 0, value12 = 0, value13 = 0;
  sc value21 = 0, value22 = 0, value23 = 0;
  sc value31 = 0, value32 = 0, value33 = 0;
  sc value = 0;

#pragma omp simd \
        aligned( x1_mapped, x2_mapped, x3_mapped : DATA_ALIGN ) \
        aligned( y1_mapped, y2_mapped, y3_mapped, w : DATA_ALIGN ) \
        reduction( + : value11 ) simdlen( DATA_WIDTH )
  for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
    value += _kernel->anti_tau_limit(
      x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
      x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
      x3_mapped[ i_quad ] - y3_mapped[ i_quad ],
      nullptr, ny_data ) * w[ i_quad ];
  }
  value11 = curl_dot[0];   value12 = curl_dot[3];   value13 = curl_dot[6];
  value21 = curl_dot[1];   value22 = curl_dot[4];   value23 = curl_dot[7];
  value31 = curl_dot[2];   value32 = curl_dot[5];   value33 = curl_dot[8];

  sc multiplier = test_area * trial_area * timestep * value;
  values_out[0] = value11 * multiplier;
  values_out[1] = value12 * multiplier;
  values_out[2] = value13 * multiplier;
  values_out[3] = value21 * multiplier;
  values_out[4] = value22 * multiplier;
  values_out[5] = value23 * multiplier;
  values_out[6] = value31 * multiplier;
  values_out[7] = value32 * multiplier;
  values_out[8] = value33 * multiplier;
  return;
}
















template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  <kernel_type, test_space_type, trial_space_type>::
  apply_regular_cpu( const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha ) const {

  lo n_tst_elems = _test_space->get_mesh()->get_n_spatial_elements();

  apply_regular_cpu(x_perm, y_perm, alpha, 0, n_tst_elems);

}















template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_regular_cpu( const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha,
    lo tst_elem_start, lo tst_elem_end ) const {

  lo trl_spelems_count = _trial_space->get_mesh()->get_n_spatial_elements();
  lo n_blocks = _block_dim;

#pragma omp parallel
  {
    quadrature_nodes quadr_nodes_tst(quadr_reference._sizes[0]);
    quadrature_nodes quadr_nodes_trl(quadr_reference._sizes[0]);
    sc val_prev;
    sc val_curr;
    sc val_next;
    sc matrix_val;

#pragma omp for
    for (lo i_tst = tst_elem_start; i_tst < tst_elem_end; i_tst++) {
      triangles_to_geometry_tst(i_tst, 0, 0, quadr_nodes_tst);
      for (lo i_trl = 0; i_trl < trl_spelems_count; i_trl++) {
        if (i_tst != i_trl) {
          triangles_to_geometry_trl(i_trl, 0, 0, quadr_nodes_trl);

          const lo &row = i_tst;
          const lo &col = i_trl;

          val_curr = 0;
          val_next = 0;

          for (lo diag = 0; diag < n_blocks; diag++) {
            val_prev = val_curr;
            val_curr = val_next;
            get_values_regular(&val_next, diag+1, i_tst, i_trl,
              quadr_nodes_tst, quadr_nodes_trl);

            matrix_val = -val_prev + 2*val_curr - val_next;
            
            lo max_block = n_blocks - diag;
            for (lo block = 0; block < max_block; block++) {
              lo block_row = diag + block;
              lo block_col = block;
              sc x_val = x_perm.get(col, block_col);
              sc y_val = alpha * matrix_val * x_val;
              y_perm.add(row, block_row, y_val);
            }
          }

        }
      }
    }

  }

}



template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_singular_cpu( const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha) const {

  lo n_elems = _test_space->get_mesh()->get_n_spatial_elements();
  lo n_blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_nodes quadr_nodes_tst(quadr_reference._sizes[0]);
    quadrature_nodes quadr_nodes_trl(quadr_reference._sizes[0]);
    sc val_prev;
    sc val_curr;
    sc val_next;
    sc matrix_val;
    
#pragma omp for
    for (lo i_elem = 0; i_elem < n_elems; i_elem++) {
      triangles_to_geometry_tst(i_elem, 0, 0, quadr_nodes_tst);
      triangles_to_geometry_trl(i_elem, 0, 0, quadr_nodes_trl);

      const lo &row = i_elem;
      const lo &col = i_elem;

      // delta0 is done elsewhere
      val_curr = 0;
      val_next = 0;

      for (lo diag = 0; diag < n_blocks; diag++) {
        val_prev = val_curr;
        val_curr = val_next;
        get_values_singular(&val_next, diag+1, i_elem, i_elem,
          quadr_nodes_tst, quadr_nodes_trl);

        matrix_val = -val_prev + 2*val_curr - val_next;

        lo max_block = n_blocks - diag;
        for (lo block = 0; block < max_block; block++) {
          lo block_row = diag + block;
          lo block_col = block;
          sc x_val = x_perm.get(col, block_col);
          sc y_val = alpha * matrix_val * x_val;
          y_perm.add(row, block_row, y_val);
        }

      }
    }

  }

}



template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >::
  apply_delta0_cpu( const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha ) const {

  lo rows_in_block = _n_rows;
  lo cols_in_block = _n_columns;
  lo blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_nodes quadr_nodes_tst(quadr_reference._max_size);
    quadrature_nodes quadr_nodes_trl(quadr_reference._max_size);
    sc val_prev;
    sc val_curr;
    int n_shared_vertices, rot_tst, rot_trl;

#pragma omp for
    for (lo i_tst = 0; i_tst < rows_in_block; i_tst++) {
      for (lo i_trl = 0; i_trl < cols_in_block; i_trl++) {
        get_type(i_tst, i_trl, n_shared_vertices, rot_tst, rot_trl);
        triangles_to_geometry_tst(i_tst, n_shared_vertices, rot_tst, quadr_nodes_tst);
        triangles_to_geometry_trl(i_trl, n_shared_vertices, rot_trl, quadr_nodes_trl);

        const lo &row = i_tst;
        const lo &col = i_trl;

        get_values_delta0special(&val_prev, i_tst, i_trl, n_shared_vertices,
          rot_tst, rot_trl, quadr_nodes_tst, quadr_nodes_trl);
        get_values_delta0(&val_curr, i_tst, i_trl, n_shared_vertices,
          rot_tst, rot_trl, quadr_nodes_tst, quadr_nodes_trl);

        sc matrix_val = val_prev + val_curr;
        lo max_block = blocks;
        for (lo block = 0; block < max_block; block++) {
          sc y_val = alpha * matrix_val * x_perm.get(col, block);
          y_perm.add(row, block, y_val);
        }

        matrix_val = -val_curr;
        max_block = blocks - 1;
        for (lo block = 0; block < max_block; block++) {
          lo block_row = 1 + block;
          lo block_col = block;
          sc x_val = x_perm.get(col, block_col);
          sc y_val = alpha * matrix_val * x_val;
          y_perm.add(row, block_row, y_val);
        }

      }
    }

  }

}















template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_regular_cpu(  const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha,
    lo tst_elem_start, lo tst_elem_end ) const {

  auto & trial_basis = _trial_space->get_basis( );

  lo trl_spelems_count = _trial_space->get_mesh()->get_n_spatial_elements();
  lo n_blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_nodes quadr_nodes_tst(quadr_reference._sizes[0]);
    quadrature_nodes quadr_nodes_trl(quadr_reference._sizes[0]);
    sc vals_prev[3];
    sc vals_curr[3];
    sc vals_next[3];
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols(3);
    sc matrix_vals[3];

#pragma omp for
    for (lo i_tst = tst_elem_start; i_tst < tst_elem_end; i_tst++) {
      triangles_to_geometry_tst(i_tst, 0, 0, quadr_nodes_tst);
      for (lo i_trl = 0; i_trl < trl_spelems_count; i_trl++) {
        if (i_tst != i_trl) {
          triangles_to_geometry_trl(i_trl, 0, 0, quadr_nodes_trl);

          const lo &row = i_tst;
          trial_basis.local_to_global(i_trl, 0, 0, true, cols );

          vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0; 
          vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

          for (lo diag = 0; diag < n_blocks; diag++) {
            vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
            vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
            get_values_regular(vals_next, diag+1, i_tst, i_trl, quadr_nodes_tst, quadr_nodes_trl);

            matrix_vals[0] = -vals_prev[0] + 2*vals_curr[0] - vals_next[0];
            matrix_vals[1] = -vals_prev[1] + 2*vals_curr[1] - vals_next[1];
            matrix_vals[2] = -vals_prev[2] + 2*vals_curr[2] - vals_next[2];

            max_block = n_blocks - diag;
            for (lo block = 0; block < max_block; block++) {
              block_row = diag + block;
              block_col = block;
              sc y_val = 0;
              y_val += matrix_vals[0] * x_perm.get(cols[0], block_col);
              y_val += matrix_vals[1] * x_perm.get(cols[1], block_col);
              y_val += matrix_vals[2] * x_perm.get(cols[2], block_col);
              y_val *= alpha;
              y_perm.add(row, block_row, y_val);
            }
          }

        }
      }
    }

  }


}



template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_singular_cpu( const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha) const {

  auto & trial_basis = _trial_space->get_basis( );

  lo tst_spelems_count = _test_space->get_mesh()->get_n_spatial_elements();
  lo n_blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_nodes quadr_nodes_tst(quadr_reference._sizes[0]);
    quadrature_nodes quadr_nodes_trl(quadr_reference._sizes[0]);
    sc vals_prev[3];
    sc vals_curr[3];
    sc vals_next[3];
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols(3);
    sc matrix_vals[3];

#pragma omp for
    for (lo i_elem = 0; i_elem < tst_spelems_count; i_elem++) {
      triangles_to_geometry_tst(i_elem, 0, 0, quadr_nodes_tst);
      triangles_to_geometry_trl(i_elem, 0, 0, quadr_nodes_trl);

      const lo &row = i_elem;
      trial_basis.local_to_global(i_elem, 0, 0, true, cols );

      vals_curr[0] = 0;   vals_curr[1] = 0;   vals_curr[2] = 0;
      vals_next[0] = 0;   vals_next[1] = 0;   vals_next[2] = 0;

      for (lo diag = 0; diag < n_blocks; diag++) {
        vals_prev[0] = vals_curr[0];   vals_prev[1] = vals_curr[1];   vals_prev[2] = vals_curr[2];
        vals_curr[0] = vals_next[0];   vals_curr[1] = vals_next[1];   vals_curr[2] = vals_next[2];
        get_values_singular(vals_next, diag+1, i_elem, i_elem, quadr_nodes_tst, quadr_nodes_trl);

        matrix_vals[0] = -vals_prev[0] + 2*vals_curr[0] - vals_next[0];
        matrix_vals[1] = -vals_prev[1] + 2*vals_curr[1] - vals_next[1];
        matrix_vals[2] = -vals_prev[2] + 2*vals_curr[2] - vals_next[2];

        max_block = n_blocks - diag;
        for (lo block = 0; block < max_block; block++) {
          block_row = diag + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x_perm.get(cols[0], block_col);
          y_val += matrix_vals[1] * x_perm.get(cols[1], block_col);
          y_val += matrix_vals[2] * x_perm.get(cols[2], block_col);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

      }
    }

  }


}



template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_delta0_cpu( const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha ) const {

  auto & trial_basis = _trial_space->get_basis( );

  lo tst_spelems_count = _test_space->get_mesh()->get_n_spatial_elements();
  lo trl_spelems_count = _trial_space->get_mesh()->get_n_spatial_elements();
  lo n_blocks = _block_dim;


#pragma omp parallel
  {
    quadrature_nodes quadr_nodes_tst(quadr_reference._max_size);
    quadrature_nodes quadr_nodes_trl(quadr_reference._max_size);
    sc vals_prev[3];
    sc vals_curr[3];
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols_0(3);
    int n_shared_vertices, rot_tst, rot_trl;
    sc matrix_vals[3];

#pragma omp for
    for (lo i_tst = 0; i_tst < tst_spelems_count; i_tst++) {
      for (lo i_trl = 0; i_trl < trl_spelems_count; i_trl++) {
        get_type( i_tst, i_trl, n_shared_vertices, rot_tst, rot_trl );
        triangles_to_geometry_tst(i_tst, n_shared_vertices, rot_tst, quadr_nodes_tst);
        triangles_to_geometry_trl(i_trl, n_shared_vertices, rot_trl, quadr_nodes_trl);

        const lo &row = i_tst;
        trial_basis.local_to_global(i_trl, n_shared_vertices, rot_trl, true, cols_0 );

        get_values_delta0special(vals_prev, i_tst, i_trl, n_shared_vertices,
          rot_tst, rot_trl, quadr_nodes_tst, quadr_nodes_trl);
        get_values_delta0(vals_curr, i_tst, i_trl, n_shared_vertices,
          rot_tst, rot_trl, quadr_nodes_tst, quadr_nodes_trl);

        matrix_vals[0] = vals_prev[0] + vals_curr[0];
        matrix_vals[1] = vals_prev[1] + vals_curr[1];
        matrix_vals[2] = vals_prev[2] + vals_curr[2];
        for (lo block = 0; block < n_blocks; block++) {
          block_row = 0 + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x_perm.get(cols_0[0], block_col);
          y_val += matrix_vals[1] * x_perm.get(cols_0[1], block_col);
          y_val += matrix_vals[2] * x_perm.get(cols_0[2], block_col);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

        matrix_vals[0] = -vals_curr[0];
        matrix_vals[1] = -vals_curr[1];
        matrix_vals[2] = -vals_curr[2];
        constexpr lo diag = 1;
        max_block = n_blocks - diag;
        for (lo block = 0; block < max_block; block++) {
          block_row = diag + block;
          block_col = block;
          sc y_val = 0;
          y_val += matrix_vals[0] * x_perm.get(cols_0[0], block_col);
          y_val += matrix_vals[1] * x_perm.get(cols_0[1], block_col);
          y_val += matrix_vals[2] * x_perm.get(cols_0[2], block_col);
          y_val *= alpha;
          y_perm.add(row, block_row, y_val);
        }

      }
    }

  }


}















template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_regular_cpu(  const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha,
    lo tst_elem_start, lo tst_elem_end ) const {

  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );

  lo trl_spelems_count = _trial_space->get_mesh()->get_n_spatial_elements();
  lo n_blocks = _block_dim;

  std::vector<block_vector_type> y_perm_thread_privates;

#pragma omp parallel
  {
#pragma omp single
    y_perm_thread_privates.resize(omp_get_num_threads());

    block_vector_type &y_perm_private = y_perm_thread_privates[omp_get_thread_num()];
    y_perm_private.resize(y_perm.get_block_size());
    y_perm_private.resize_blocks(y_perm.get_size_of_block());
    y_perm_private.fill(0);

    quadrature_nodes quadr_nodes_tst(quadr_reference._sizes[0]);
    quadrature_nodes quadr_nodes_trl(quadr_reference._sizes[0]);
    sc vals_prev[9];
    sc vals_curr[9];
    sc vals_next[9];
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols(3);
    std::vector< lo > rows(3);
    sc matrix_vals[9];
    sc x_vals[3];
    sc y_vals[3];

#pragma omp for
    for (lo i_tst = tst_elem_start; i_tst < tst_elem_end; i_tst++) {
      triangles_to_geometry_tst(i_tst, 0, 0, quadr_nodes_tst);
      for (lo i_trl = 0; i_trl < trl_spelems_count; i_trl++) {
        if(i_tst != i_trl) {
          triangles_to_geometry_trl(i_trl, 0, 0, quadr_nodes_trl);

          test_basis.local_to_global(i_tst, 0, 0, false, rows );
          trial_basis.local_to_global(i_trl, 0, 0, true, cols );

          for(lo j = 0; j < 9; j++) vals_curr[j] = 0;
          for(lo j = 0; j < 9; j++) vals_next[j] = 0;

          for (lo diag = 0; diag < n_blocks; diag++) {
            for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
            for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
            get_values_regular(vals_next, diag+1, i_tst, i_trl,
              quadr_nodes_tst, quadr_nodes_trl);

            for(lo j = 0; j < 9; j++)
              matrix_vals[j] = -vals_prev[j] + 2*vals_curr[j] - vals_next[j];

            max_block = n_blocks - diag;
            for (lo block = 0; block < max_block; block++) {
              block_row = diag + block;
              block_col = block;
              x_vals[0] = alpha * x_perm.get(cols[0], block_col);
              x_vals[1] = alpha * x_perm.get(cols[1], block_col);
              x_vals[2] = alpha * x_perm.get(cols[2], block_col);
              y_vals[0] = y_vals[1] = y_vals[2] = 0;
              for(lo r = 0; r < 3; r++)
                for(lo c = 0; c < 3; c++)
                  y_vals[r] += matrix_vals[3*r+c] * x_vals[c];
              y_perm_private.add(rows[0], block_row, y_vals[0]);
              y_perm_private.add(rows[1], block_row, y_vals[1]);
              y_perm_private.add(rows[2], block_row, y_vals[2]);
            }
          }

        }
      }
    }

    // add the thread-private y_perm vectors together,
    // each thread is now assigned several blocks of the vectors
#pragma omp for
    for(lo b = 0; b < y_perm.get_block_size(); b++) {
      vector_type &dst_block = y_perm.get_block(b);
      for(size_t k = 0; k < y_perm_thread_privates.size(); k++) {
        vector_type &src_block = y_perm_thread_privates[k].get_block(b);
        dst_block.add(src_block);
      }
    }

  }


}



template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_singular_cpu( const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha) const {

  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );

  lo tst_spelems_count = _test_space->get_mesh()->get_n_spatial_elements();
  lo n_blocks = _block_dim;

  std::vector<block_vector_type> y_perm_thread_privates;

#pragma omp parallel
  {
#pragma omp single
    y_perm_thread_privates.resize(omp_get_num_threads());

    block_vector_type &y_perm_private = y_perm_thread_privates[omp_get_thread_num()];
    y_perm_private.resize(y_perm.get_block_size());
    y_perm_private.resize_blocks(y_perm.get_size_of_block());
    y_perm_private.fill(0);

    quadrature_nodes quadr_nodes_tst(quadr_reference._sizes[0]);
    quadrature_nodes quadr_nodes_trl(quadr_reference._sizes[0]);
    sc vals_prev[9];
    sc vals_curr[9];
    sc vals_next[9];
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols(3);
    std::vector< lo > rows(3);
    sc matrix_vals[9];
    sc x_vals[3];
    sc y_vals[3];

#pragma omp for
    for (lo i_elem = 0; i_elem < tst_spelems_count; i_elem++) {
      triangles_to_geometry_tst(i_elem, 0, 0, quadr_nodes_tst);
      triangles_to_geometry_trl(i_elem, 0, 0, quadr_nodes_trl);

      test_basis.local_to_global(i_elem, 0, 0, false, rows );
      trial_basis.local_to_global(i_elem, 0, 0, true, cols );

      for(lo j = 0; j < 9; j++) vals_curr[j] = 0;
      for(lo j = 0; j < 9; j++) vals_next[j] = 0;

      for (lo diag = 0; diag < n_blocks; diag++) {
        for(lo j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
        for(lo j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
        get_values_singular(vals_next, diag+1, i_elem, i_elem,
          quadr_nodes_tst, quadr_nodes_trl);

        for(lo j = 0; j < 9; j++)
          matrix_vals[j] = -vals_prev[j] + 2*vals_curr[j] - vals_next[j];

        max_block = n_blocks - diag;
        for (lo block = 0; block < max_block; block++) {
          block_row = diag + block;
          block_col = block;
          x_vals[0] = alpha * x_perm.get(cols[0], block_col);
          x_vals[1] = alpha * x_perm.get(cols[1], block_col);
          x_vals[2] = alpha * x_perm.get(cols[2], block_col);
          y_vals[0] = y_vals[1] = y_vals[2] = 0;
          for(lo r = 0; r < 3; r++)
            for(lo c = 0; c < 3; c++)
              y_vals[r] += matrix_vals[3*r+c] * x_vals[c];
          y_perm_private.add(rows[0], block_row, y_vals[0]);
          y_perm_private.add(rows[1], block_row, y_vals[1]);
          y_perm_private.add(rows[2], block_row, y_vals[2]);
        }

      }

    }

    // add the thread-private y_perm vectors together,
    // each thread is now assigned several blocks of the vectors
#pragma omp for
    for(lo b = 0; b < y_perm.get_block_size(); b++) {
      vector_type &dst_block = y_perm.get_block(b);
      for(size_t k = 0; k < y_perm_thread_privates.size(); k++) {
        vector_type &src_block = y_perm_thread_privates[k].get_block(b);
        dst_block.add(src_block);
      }
    }

  }


}



template<>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >::
  apply_delta0_cpu( const block_vector_type & x_perm,
    block_vector_type & y_perm, sc alpha ) const {

  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );

  lo tst_spelems_count = _test_space->get_mesh()->get_n_spatial_elements();
  lo trl_spelems_count = _trial_space->get_mesh()->get_n_spatial_elements();
  lo n_blocks = _block_dim;

  std::vector<block_vector_type> y_perm_thread_privates;

#pragma omp parallel
  {
#pragma omp single
    y_perm_thread_privates.resize(omp_get_num_threads());

    block_vector_type &y_perm_private = y_perm_thread_privates[omp_get_thread_num()];
    y_perm_private.resize(y_perm.get_block_size());
    y_perm_private.resize_blocks(y_perm.get_size_of_block());
    y_perm_private.fill(0);

    quadrature_nodes quadr_nodes_tst(quadr_reference._max_size);
    quadrature_nodes quadr_nodes_trl(quadr_reference._max_size);
    sc vals_prev[9];
    sc vals_curr[9];
    lo max_block;
    lo block_row, block_col;
    std::vector< lo > cols_0(3);
    std::vector< lo > rows_0(3);
    int n_shared_vertices, rot_tst, rot_trl;
    sc matrix_vals[9];
    sc x_vals[3];
    sc y_vals[3];

#pragma omp for
    for (lo i_tst = 0; i_tst < tst_spelems_count; i_tst++) {
      for (lo i_trl = 0; i_trl < trl_spelems_count; i_trl++) {
        get_type( i_tst, i_trl, n_shared_vertices, rot_tst, rot_trl );
        triangles_to_geometry_tst(i_tst, n_shared_vertices, rot_tst, quadr_nodes_tst);
        triangles_to_geometry_trl(i_trl, n_shared_vertices, rot_trl, quadr_nodes_trl);

        test_basis.local_to_global(i_tst, n_shared_vertices, rot_tst, false, rows_0 );
        trial_basis.local_to_global(i_trl, n_shared_vertices, rot_trl, true, cols_0 );

        get_values_delta0special(vals_prev, i_tst, i_trl, n_shared_vertices,
          rot_tst, rot_trl, quadr_nodes_tst, quadr_nodes_trl);
        get_values_delta0(vals_curr, i_tst, i_trl, n_shared_vertices,
          rot_tst, rot_trl, quadr_nodes_tst, quadr_nodes_trl);

        for(lo j = 0; j < 9; j++) matrix_vals[j] = vals_prev[j] + vals_curr[j];
        max_block = n_blocks;
        for (lo block = 0; block < max_block; block++) {
          block_row = 0 + block;
          block_col = block;
          x_vals[0] = alpha * x_perm.get(cols_0[0], block_col);
          x_vals[1] = alpha * x_perm.get(cols_0[1], block_col);
          x_vals[2] = alpha * x_perm.get(cols_0[2], block_col);
          y_vals[0] = y_vals[1] = y_vals[2] = 0;
          for(lo r = 0; r < 3; r++)
            for(lo c = 0; c < 3; c++)
              y_vals[r] += matrix_vals[3*r+c] * x_vals[c];
          y_perm_private.add(rows_0[0], block_row, y_vals[0]);
          y_perm_private.add(rows_0[1], block_row, y_vals[1]);
          y_perm_private.add(rows_0[2], block_row, y_vals[2]);
        }

        for(lo j = 0; j < 9; j++) matrix_vals[j] = -vals_curr[j];
        max_block = n_blocks - 1;
        for (lo block = 0; block < max_block; block++) {
          block_row = 1 + block;
          block_col = block;
          x_vals[0] = alpha * x_perm.get(cols_0[0], block_col);
          x_vals[1] = alpha * x_perm.get(cols_0[1], block_col);
          x_vals[2] = alpha * x_perm.get(cols_0[2], block_col);
          y_vals[0] = y_vals[1] = y_vals[2] = 0;
          for(lo r = 0; r < 3; r++)
            for(lo c = 0; c < 3; c++)
              y_vals[r] += matrix_vals[3*r+c] * x_vals[c];
          y_perm_private.add(rows_0[0], block_row, y_vals[0]);
          y_perm_private.add(rows_0[1], block_row, y_vals[1]);
          y_perm_private.add(rows_0[2], block_row, y_vals[2]);
        }

      }
    }

    // add the thread-private y_perm vectors together,
    // each thread is now assigned several blocks of the vectors
#pragma omp for
    for(lo b = 0; b < y_perm.get_block_size(); b++) {
      vector_type &dst_block = y_perm.get_block(b);
      for(size_t k = 0; k < y_perm_thread_privates.size(); k++) {
        vector_type &src_block = y_perm_thread_privates[k].get_block(b);
        dst_block.add(src_block);
      }
    }

  }


}















template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  <kernel_type, test_space_type, trial_space_type>::
  apply( const block_vector_type & x, block_vector_type & y,
    bool trans, sc alpha, sc beta ) const {

  if(trans) {
    if(besthea::settings::output_verbosity.warnings >= 1) {
      std::cerr << "BESTHEA Warning: transposed matrices are not supported. "
        "Apply operation will not be performed.\n";
    }
    return;
  }

  if(x.get_block_size() != this->get_block_dim() ||
     x.get_size_of_block() != this->get_dim_domain()) {
    std::cerr << "BESTHEA Error: x block vector dimension ("
      << x.get_block_size() << "*" << x.get_size_of_block()
      << ") does not match block matrix domain dimension ("
      << this->get_block_dim() << "*" << this->get_dim_domain()
      << ").\n";
    throw std::runtime_error("BESTHEA Exception: incompatible matrix and vector dimensions");
  }

  if(y.get_block_size() != this->get_block_dim() ||
     y.get_size_of_block() != this->get_dim_range()) {
    std::cerr << "BESTHEA Error: y block vector dimension ("
      << y.get_block_size() << "*" << y.get_size_of_block()
      << ") does not match block matrix range dimension ("
      << this->get_block_dim() << "*" << this->get_dim_range()
      << "). Apply will not be performed.\n";
    throw std::runtime_error("BESTHEA Exception: incompatible matrix and vector dimensions");
  }

  besthea::tools::time_measurer tm_combined, tm_perm, tm_apply;
  tm_combined.start();
  
  // permuting the vector x should improve data locality
  // permuting the vector y should prevent false sharing and improve data locality
  block_vector_type x_perm;
  block_vector_type y_perm;

  tm_perm.start();
  x_perm.copy_permute(x, alpha);

  if(beta == 0.0) {
    y_perm.resize_match_perm(y, true);
  } else if(beta == 1.0) {
    y_perm.copy_permute(y);
  } else {
    y_perm.copy_permute(y, beta);
  }
  tm_perm.stop();

  tm_apply.start();
  apply_cpu(x_perm, y_perm, 1.0, 1.0);
  tm_apply.stop();


  tm_perm.start();
  y.copy_permute(y_perm);
  tm_perm.stop();
  
  tm_combined.stop();

  if(besthea::settings::output_verbosity.timers >= 1) {
    if(besthea::settings::output_verbosity.timers >= 2) {
      std::cout << "BESTHEA Info: apply, vector permutation elapsed time = "
        << tm_perm.get_time() << " seconds\n";
      std::cout << "BESTHEA Info: apply, apply itself elapsed time = "
        << tm_apply.get_time() << " seconds\n";
    }
    std::cout << "BESTHEA Info: apply elapsed time = "
      << tm_combined.get_time() << " seconds\n";
  }

}



template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  <kernel_type, test_space_type, trial_space_type>::
  apply_cpu( const block_vector_type & x_perm, block_vector_type & y_perm,
    sc alpha, sc beta ) const {

  besthea::tools::time_measurer tm_scale, tm_reg, tm_sing, tm_d0;

  tm_scale.start();
  if(beta == 0.0) {
    y_perm.fill(0);
  } else if(beta != 1.0) {
    y_perm.scale(beta);
  }
  tm_scale.stop();

  tm_reg.start();
  this->apply_regular_cpu(x_perm, y_perm, alpha);
  tm_reg.stop();
  tm_sing.start();
  this->apply_singular_cpu(x_perm, y_perm, alpha);
  tm_sing.stop();
  tm_d0.start();
  this->apply_delta0_cpu(x_perm, y_perm, alpha);
  tm_d0.stop();
  


  if(besthea::settings::output_verbosity.timers >= 2) {
    std::cout << "BESTHEA Info: apply, scalein elapsed time = "
      << tm_scale.get_time() << " seconds\n";
    std::cout << "BESTHEA Info: apply, regular elapsed time = "
      << tm_reg.get_time() << " seconds\n";
    std::cout << "BESTHEA Info: apply, singular elapsed time = "
      << tm_sing.get_time() << " seconds\n";
    std::cout << "BESTHEA Info: apply, delta0 elapsed time = "
      << tm_d0.get_time() << " seconds\n";
  }

}
















template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  <kernel_type, test_space_type, trial_space_type>::init_quadrature( ) {
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
  quadr_reference._sizes[ n_shared_vertices ] = size;
  quadr_reference._x1_ref[ n_shared_vertices ].resize( size );
  quadr_reference._x2_ref[ n_shared_vertices ].resize( size );
  quadr_reference._y1_ref[ n_shared_vertices ].resize( size );
  quadr_reference._y2_ref[ n_shared_vertices ].resize( size );
  quadr_reference._w[ n_shared_vertices ].resize( size );

  lo counter = 0;
  for ( lo i_x = 0; i_x < tri_size; ++i_x ) {
    for ( lo i_y = 0; i_y < tri_size; ++i_y ) {
      quadr_reference._x1_ref[ n_shared_vertices ][ counter ] = tri_x1[ i_x ];
      quadr_reference._x2_ref[ n_shared_vertices ][ counter ] = tri_x2[ i_x ];
      quadr_reference._y1_ref[ n_shared_vertices ][ counter ] = tri_x1[ i_y ];
      quadr_reference._y2_ref[ n_shared_vertices ][ counter ] = tri_x2[ i_y ];
      quadr_reference._w[ n_shared_vertices ][ counter ]
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
    quadr_reference._sizes[ n_shared_vertices ] = size;
    quadr_reference._x1_ref[ n_shared_vertices ].resize( size );
    quadr_reference._x2_ref[ n_shared_vertices ].resize( size );
    quadr_reference._y1_ref[ n_shared_vertices ].resize( size );
    quadr_reference._y2_ref[ n_shared_vertices ].resize( size );
    quadr_reference._w[ n_shared_vertices ].resize( size );

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
                quadr_reference._x1_ref[ n_shared_vertices ][ counter ],
                quadr_reference._x2_ref[ n_shared_vertices ][ counter ],
                quadr_reference._y1_ref[ n_shared_vertices ][ counter ],
                quadr_reference._y2_ref[ n_shared_vertices ][ counter ],
                jacobian );
              quadr_reference._w[ n_shared_vertices ][ counter ] = 4.0 * jacobian
                * line_w[ i_ksi ] * line_w[ i_eta1 ] * line_w[ i_eta2 ]
                * line_w[ i_eta3 ];
              ++counter;
            }
          }
        }
      }
    }
  }

  quadr_reference._max_size = *std::max_element(quadr_reference._sizes.begin(),
    quadr_reference._sizes.end());

}



template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  <kernel_type, test_space_type, trial_space_type>::
  triangles_to_geometry_tst( lo i_tst, int n_shared_vertices, int rot_test,
    quadrature_nodes & quadr_nodes_tst ) const {
      
  linear_algebra::coordinates< 3 > x1, x2, x3;
  _test_space->get_mesh()->get_spatial_nodes( i_tst, x1, x2, x3 );

  const sc * x1rot = nullptr;
  const sc * x2rot = nullptr;
  const sc * x3rot = nullptr;

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

  const sc * x1_ref = quadr_reference._x1_ref[ n_shared_vertices ].data( );
  const sc * x2_ref = quadr_reference._x2_ref[ n_shared_vertices ].data( );

  sc * x1_mapped = quadr_nodes_tst._xs.data( );
  sc * x2_mapped = quadr_nodes_tst._ys.data( );
  sc * x3_mapped = quadr_nodes_tst._zs.data( );

  lo size = quadr_reference._sizes[ n_shared_vertices ];

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

}



template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  <kernel_type, test_space_type, trial_space_type>::
  triangles_to_geometry_trl( lo i_trl, int n_shared_vertices, int rot_trial,
    quadrature_nodes & quadr_nodes_trl ) const {
      
  linear_algebra::coordinates< 3 > y1, y2, y3;
  _trial_space->get_mesh()->get_spatial_nodes( i_trl, y1, y2, y3 );

  const sc * y1rot = nullptr;
  const sc * y2rot = nullptr;
  const sc * y3rot = nullptr;

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
  const sc * y1_ref = quadr_reference._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = quadr_reference._y2_ref[ n_shared_vertices ].data( );

  sc * y1_mapped = quadr_nodes_trl._xs.data( );
  sc * y2_mapped = quadr_nodes_trl._ys.data( );
  sc * y3_mapped = quadr_nodes_trl._zs.data( );

  lo size = quadr_reference._sizes[ n_shared_vertices ];

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
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  <kernel_type, test_space_type, trial_space_type>::
  get_type( lo i_test, lo i_trial, int & n_shared_vertices,
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
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  < kernel_type, test_space_type, trial_space_type >::
  hypercube_to_triangles_identical( sc ksi, sc eta1,
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
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  < kernel_type, test_space_type, trial_space_type >::
  hypercube_to_triangles_vertex( sc ksi, sc eta1, sc eta2,
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
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu
  < kernel_type, test_space_type, trial_space_type >::
  hypercube_to_triangles_edge( sc ksi, sc eta1, sc eta2,
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













template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
    besthea::bem::spacetime_heat_sl_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
    besthea::bem::spacetime_heat_dl_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_cpu<
    besthea::bem::spacetime_heat_hs_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;



