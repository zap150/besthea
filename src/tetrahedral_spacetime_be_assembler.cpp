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

#include "besthea/tetrahedral_spacetime_be_assembler.h"

#include "besthea/quadrature.h"
#include "besthea/spacetime_basis_tetra_p1.h"
#include "besthea/spacetime_constant_kernel.h"
#include "besthea/spacetime_heat_kernel.h"

#include <algorithm>

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::tetrahedral_spacetime_be_assembler( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_singular, int order_regular )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _order_singular( order_singular ),
    _order_regular( order_regular ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::~tetrahedral_spacetime_be_assembler( ) {
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::assemble( besthea::linear_algebra::full_matrix &
    global_matrix ) const {
  auto & test_basis = _test_space->get_basis( );
  auto & trial_basis = _trial_space->get_basis( );
  auto test_mesh = _test_space->get_mesh( );
  auto trial_mesh = _trial_space->get_mesh( );

  lo n_rows = test_basis.dimension_global( );
  lo n_columns = trial_basis.dimension_global( );
  global_matrix.resize( n_rows, n_columns );

  lo n_loc_rows = test_basis.dimension_local( );
  lo n_loc_columns = trial_basis.dimension_local( );

  lo n_test_elements = test_mesh->get_n_elements( );
  lo n_trial_elements = trial_mesh->get_n_elements( );

#pragma omp parallel
  {
    std::vector< lo > test_l2g( n_loc_rows );
    std::vector< lo > trial_l2g( n_loc_columns );

    linear_algebra::coordinates< 4 > x1, x2, x3, x4;
    linear_algebra::coordinates< 4 > y1, y2, y3, y4;
    linear_algebra::indices< 4 > perm_test, perm_trial;
    int n_shared_vertices = 0;
    lo size;
    sc test_area, trial_area, test, trial, value;

    lo * perm_test_data = perm_test.data( );
    lo * perm_trial_data = perm_trial.data( );

    sc * nx_data = nullptr;
    sc * ny_data = nullptr;

    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * x1_ref = nullptr;
    sc * x2_ref = nullptr;
    sc * x3_ref = nullptr;
    sc * y1_ref = nullptr;
    sc * y2_ref = nullptr;
    sc * y3_ref = nullptr;
    sc * w = nullptr;
    sc * x1_mapped = my_quadrature._x1.data( );
    sc * x2_mapped = my_quadrature._x2.data( );
    sc * x3_mapped = my_quadrature._x3.data( );
    sc * t_mapped = my_quadrature._t.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );
    sc * tau_mapped = my_quadrature._tau.data( );
    sc * kernel_data = my_quadrature._kernel_values.data( );

#pragma omp for schedule( dynamic )
    for ( lo i_test = 0; i_test < n_test_elements; ++i_test ) {
      test_mesh->get_nodes( i_test, x1, x2, x3, x4 );
      test_area = test_mesh->area( i_test );
      for ( lo i_trial = 0; i_trial < n_trial_elements; ++i_trial ) {
        trial_mesh->get_nodes( i_trial, y1, y2, y3, y4 );
        trial_area = trial_mesh->area( i_trial );

        get_type( i_test, i_trial, n_shared_vertices, perm_test, perm_trial );

        test_basis.local_to_global( i_test, test_l2g );
        trial_basis.local_to_global( i_trial, trial_l2g );

        tetrahedra_to_geometry( x1, x2, x3, x4, y1, y2, y3, y4,
          n_shared_vertices, perm_test, perm_trial, my_quadrature );
        x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
        x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
        x3_ref = my_quadrature._x3_ref[ n_shared_vertices ].data( );
        y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
        y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
        y3_ref = my_quadrature._y3_ref[ n_shared_vertices ].data( );
        w = my_quadrature._w[ n_shared_vertices ].data( );
        size = my_quadrature._w[ n_shared_vertices ].size( );

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, t_mapped,   \
                          y1_mapped, y2_mapped, y3_mapped, tau_mapped, \
                          kernel_data, w                               \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
        for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
          kernel_data[ i_quad ]
            = _kernel->evaluate( x1_mapped[ i_quad ] - y1_mapped[ i_quad ],
                x2_mapped[ i_quad ] - y2_mapped[ i_quad ],
                x3_mapped[ i_quad ] - y3_mapped[ i_quad ], nx_data, ny_data,
                t_mapped[ i_quad ] - tau_mapped[ i_quad ] )
            * w[ i_quad ];
        }

        for ( lo i_loc_test = 0; i_loc_test < n_loc_rows; ++i_loc_test ) {
          for ( lo i_loc_trial = 0; i_loc_trial < n_loc_columns;
                ++i_loc_trial ) {
            value = 0.0;
#pragma omp simd aligned( x1_ref, x2_ref, x3_ref, y1_ref, y2_ref, y3_ref, \
kernel_data : DATA_ALIGN ) private( test, trial ) reduction( + : value ) \
simdlen( DATA_WIDTH )
            for ( lo i_quad = 0; i_quad < size; ++i_quad ) {
              test = test_basis.evaluate( i_test, i_loc_test, x1_ref[ i_quad ],
                x2_ref[ i_quad ], x3_ref[ i_quad ], nx_data, perm_test_data );
              trial = trial_basis.evaluate( i_trial, i_loc_trial,
                y1_ref[ i_quad ], y2_ref[ i_quad ], y3_ref[ i_quad ], ny_data,
                perm_trial_data );

              value += kernel_data[ i_quad ] * test * trial;
            }
            global_matrix.add_atomic( test_l2g[ i_loc_test ],
              trial_l2g[ i_loc_trial ], value * test_area * trial_area );
          }
        }

      }  // i_trial
    }    // i_test
  }      // omp parallel
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::init_quadrature( quadrature_wrapper &
    my_quadrature ) const {
  // Use triangle rules for disjoint elements
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_x1
    = quadrature::tetrahedron_x1( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_x2
    = quadrature::tetrahedron_x2( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_x3
    = quadrature::tetrahedron_x3( _order_regular );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_w
    = quadrature::tetrahedron_w( _order_regular );
  lo tetra_size = tetra_w.size( );
  lo tetra_size2 = tetra_size * tetra_size;

  int n_shared_vertices = 0;
  my_quadrature._x1_ref[ n_shared_vertices ].resize( tetra_size2 );
  my_quadrature._x2_ref[ n_shared_vertices ].resize( tetra_size2 );
  my_quadrature._x3_ref[ n_shared_vertices ].resize( tetra_size2 );
  my_quadrature._y1_ref[ n_shared_vertices ].resize( tetra_size2 );
  my_quadrature._y2_ref[ n_shared_vertices ].resize( tetra_size2 );
  my_quadrature._y3_ref[ n_shared_vertices ].resize( tetra_size2 );
  my_quadrature._w[ n_shared_vertices ].resize( tetra_size2 );

  lo counter = 0;
  for ( lo i_x = 0; i_x < tetra_size; ++i_x ) {
    for ( lo i_y = 0; i_y < tetra_size; ++i_y ) {
      my_quadrature._x1_ref[ n_shared_vertices ][ counter ] = tetra_x1[ i_x ];
      my_quadrature._x2_ref[ n_shared_vertices ][ counter ] = tetra_x2[ i_x ];
      my_quadrature._x3_ref[ n_shared_vertices ][ counter ] = tetra_x3[ i_x ];
      my_quadrature._y1_ref[ n_shared_vertices ][ counter ] = tetra_x1[ i_y ];
      my_quadrature._y2_ref[ n_shared_vertices ][ counter ] = tetra_x2[ i_y ];
      my_quadrature._y3_ref[ n_shared_vertices ][ counter ] = tetra_x3[ i_y ];
      my_quadrature._w[ n_shared_vertices ][ counter ]
        = tetra_w[ i_x ] * tetra_w[ i_y ];
      ++counter;
    }
  }

  lo size = tetra_size2;

  my_quadrature._x1.resize( size );
  my_quadrature._x2.resize( size );
  my_quadrature._x3.resize( size );
  my_quadrature._t.resize( size );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );
  my_quadrature._tau.resize( size );
  my_quadrature._kernel_values.resize( size );
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::
  tetrahedra_to_geometry( const linear_algebra::coordinates< 4 > & x1,
    const linear_algebra::coordinates< 4 > & x2,
    const linear_algebra::coordinates< 4 > & x3,
    const linear_algebra::coordinates< 4 > & x4,
    const linear_algebra::coordinates< 4 > & y1,
    const linear_algebra::coordinates< 4 > & y2,
    const linear_algebra::coordinates< 4 > & y3,
    const linear_algebra::coordinates< 4 > & y4, int n_shared_vertices,
    const besthea::linear_algebra::indices< 4 > & perm_test,
    const besthea::linear_algebra::indices< 4 > & perm_trial,
    quadrature_wrapper & my_quadrature ) const {
  const sc * x1rot = nullptr;
  const sc * x2rot = nullptr;
  const sc * x3rot = nullptr;
  const sc * x4rot = nullptr;
  const sc * y1rot = nullptr;
  const sc * y2rot = nullptr;
  const sc * y3rot = nullptr;
  const sc * y4rot = nullptr;

  // TODO with perm_test, perm_trial
  x1rot = x1.data( );
  x2rot = x2.data( );
  x3rot = x3.data( );
  x4rot = x4.data( );

  y1rot = y1.data( );
  y2rot = y2.data( );
  y3rot = y3.data( );
  y4rot = y4.data( );

  const sc * x1_ref = my_quadrature._x1_ref[ n_shared_vertices ].data( );
  const sc * x2_ref = my_quadrature._x2_ref[ n_shared_vertices ].data( );
  const sc * x3_ref = my_quadrature._x3_ref[ n_shared_vertices ].data( );
  const sc * y1_ref = my_quadrature._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = my_quadrature._y2_ref[ n_shared_vertices ].data( );
  const sc * y3_ref = my_quadrature._y3_ref[ n_shared_vertices ].data( );

  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * t_mapped = my_quadrature._t.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );
  sc * tau_mapped = my_quadrature._tau.data( );

  lo size = my_quadrature._w[ n_shared_vertices ].size( );

#pragma omp simd aligned( x1_mapped, x2_mapped, x3_mapped, t_mapped      \
                          : DATA_ALIGN ) aligned( x1_ref, x2_ref, x3_ref \
                                                  : DATA_ALIGN )         \
  simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    x1_mapped[ i ] = x1rot[ 0 ] + ( x2rot[ 0 ] - x1rot[ 0 ] ) * x1_ref[ i ]
      + ( x3rot[ 0 ] - x1rot[ 0 ] ) * x2_ref[ i ]
      + ( x4rot[ 0 ] - x1rot[ 0 ] ) * x3_ref[ i ];
    x2_mapped[ i ] = x1rot[ 1 ] + ( x2rot[ 1 ] - x1rot[ 1 ] ) * x1_ref[ i ]
      + ( x3rot[ 1 ] - x1rot[ 1 ] ) * x2_ref[ i ]
      + ( x4rot[ 1 ] - x1rot[ 1 ] ) * x3_ref[ i ];
    x3_mapped[ i ] = x1rot[ 2 ] + ( x2rot[ 2 ] - x1rot[ 2 ] ) * x1_ref[ i ]
      + ( x3rot[ 2 ] - x1rot[ 2 ] ) * x2_ref[ i ]
      + ( x4rot[ 2 ] - x1rot[ 2 ] ) * x3_ref[ i ];
    t_mapped[ i ] = x1rot[ 3 ] + ( x2rot[ 3 ] - x1rot[ 3 ] ) * x1_ref[ i ]
      + ( x3rot[ 3 ] - x1rot[ 3 ] ) * x2_ref[ i ]
      + ( x4rot[ 3 ] - x1rot[ 3 ] ) * x3_ref[ i ];
  }

#pragma omp simd aligned( y1_mapped, y2_mapped, y3_mapped, tau_mapped \
                          : DATA_ALIGN ) aligned( y1_ref, y2_ref      \
                                                  : DATA_ALIGN )      \
  simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = y1rot[ 0 ] + ( y2rot[ 0 ] - y1rot[ 0 ] ) * y1_ref[ i ]
      + ( y3rot[ 0 ] - y1rot[ 0 ] ) * y2_ref[ i ]
      + ( y4rot[ 0 ] - y1rot[ 0 ] ) * y3_ref[ i ];
    y2_mapped[ i ] = y1rot[ 1 ] + ( y2rot[ 1 ] - y1rot[ 1 ] ) * y1_ref[ i ]
      + ( y3rot[ 1 ] - y1rot[ 1 ] ) * y2_ref[ i ]
      + ( y4rot[ 1 ] - y1rot[ 1 ] ) * y3_ref[ i ];
    y3_mapped[ i ] = y1rot[ 2 ] + ( y2rot[ 2 ] - y1rot[ 2 ] ) * y1_ref[ i ]
      + ( y3rot[ 2 ] - y1rot[ 2 ] ) * y2_ref[ i ]
      + ( y4rot[ 2 ] - y1rot[ 2 ] ) * y3_ref[ i ];
    tau_mapped[ i ] = y1rot[ 3 ] + ( y2rot[ 3 ] - y1rot[ 3 ] ) * y1_ref[ i ]
      + ( y3rot[ 3 ] - y1rot[ 3 ] ) * y2_ref[ i ]
      + ( y4rot[ 3 ] - y1rot[ 3 ] ) * y3_ref[ i ];
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type, trial_space_type >::get_type( lo i_test, lo i_trial,
  int & n_shared_vertices, besthea::linear_algebra::indices< 4 > & perm_test,
  besthea::linear_algebra::indices< 4 > & perm_trial ) const {
  // TO REMOVE
  n_shared_vertices = 0;
  for ( int i = 0; i < 4; ++i ) {
    perm_test[ i ] = i;
    perm_trial[ i ] = i;
  }
  return;

  // check for identical
  if ( i_test == i_trial ) {
    n_shared_vertices = 4;
    for ( int i = 0; i < 4; ++i ) {
      perm_test[ i ] = i;
      perm_trial[ i ] = i;
    }
    return;
  }

  linear_algebra::indices< 4 > test_elem;
  linear_algebra::indices< 4 > trial_elem;

  _test_space->get_mesh( )->get_element( i_test, test_elem );
  _trial_space->get_mesh( )->get_element( i_trial, trial_elem );

  // disjoint
  n_shared_vertices = 0;
  for ( int i = 0; i < 4; ++i ) {
    perm_test[ i ] = i;
    perm_trial[ i ] = i;
  }
}

template class besthea::bem::tetrahedral_spacetime_be_assembler<
  besthea::bem::spacetime_heat_kernel,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p1 >,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p1 > >;

template class besthea::bem::tetrahedral_spacetime_be_assembler<
  besthea::bem::spacetime_constant_kernel,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p1 >,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p1 > >;
