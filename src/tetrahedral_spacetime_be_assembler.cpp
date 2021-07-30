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
#include "besthea/spacetime_heat_kernel_normal_derivative.h"

#include <algorithm>

template< class kernel_type, class test_space_type, class trial_space_type >
besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type, test_space_type,
  trial_space_type >::tetrahedral_spacetime_be_assembler( kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  int order_regular, int order_singular, int singular_refinements )
  : _kernel( &kernel ),
    _test_space( &test_space ),
    _trial_space( &trial_space ),
    _singular_refinements( singular_refinements ),
    _order_regular( order_regular ),
    _order_singular( order_singular ) {
  _admissibles.resize( singular_refinements );
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

  quadrature_wrapper_ref ref_quadrature;
  init_quadrature_ref( ref_quadrature );

#pragma omp parallel
  {
    std::vector< lo > test_l2g( n_loc_rows );
    std::vector< lo > trial_l2g( n_loc_columns );

    linear_algebra::coordinates< 4 > x1, x2, x3, x4;
    linear_algebra::coordinates< 4 > y1, y2, y3, y4;
    linear_algebra::coordinates< 3 > ny;
    linear_algebra::indices< 4 > perm_test, perm_trial;
    int n_shared_vertices = 0;
    lo size;
    sc test_area, trial_area, test, trial, value;

    lo * perm_test_data = perm_test.data( );
    lo * perm_trial_data = perm_trial.data( );

    sc * nx_data = nullptr;
    sc * ny_data = ny.data( );  // nullptr;

    quadrature_wrapper my_quadrature;
    init_quadrature( ref_quadrature, my_quadrature );
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
        trial_mesh->get_spatial_normal( i_trial, ny );
        trial_area = trial_mesh->area( i_trial );

        get_type( i_test, i_trial, n_shared_vertices, perm_test, perm_trial );

        test_basis.local_to_global( i_test, test_l2g );
        trial_basis.local_to_global( i_trial, trial_l2g );

        tetrahedra_to_geometry( x1, x2, x3, x4, y1, y2, y3, y4,
          n_shared_vertices, perm_test, perm_trial, ref_quadrature,
          my_quadrature );
        // n_shared_vertices = 0;
        x1_ref = ref_quadrature._x1_ref[ n_shared_vertices ].data( );
        x2_ref = ref_quadrature._x2_ref[ n_shared_vertices ].data( );
        x3_ref = ref_quadrature._x3_ref[ n_shared_vertices ].data( );
        y1_ref = ref_quadrature._y1_ref[ n_shared_vertices ].data( );
        y2_ref = ref_quadrature._y2_ref[ n_shared_vertices ].data( );
        y3_ref = ref_quadrature._y3_ref[ n_shared_vertices ].data( );

        w = ref_quadrature._w[ n_shared_vertices ].data( );
        size = ref_quadrature._w[ n_shared_vertices ].size( );

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
            // std::cout << value * test_area * trial_area << " ";
            global_matrix.add_atomic( test_l2g[ i_loc_test ],
              trial_l2g[ i_loc_trial ], value * test_area * trial_area );
          }

          // std::cout << std::endl;
        }
        // exit( 0 );
      }  // i_trial
    }    // i_test
  }      // omp parallel
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::init_quadrature( const quadrature_wrapper_ref &
                                         ref_quadrature,
  quadrature_wrapper & my_quadrature ) const {
  // maximum size
  auto size = ref_quadrature._w[ 0 ].size( );
  for ( int i_shared = 1; i_shared <= 4; ++i_shared ) {
    size = std::max( size, ref_quadrature._w[ i_shared ].size( ) );
  }
  // std::cout << size << std::endl;
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
  test_space_type,
  trial_space_type >::init_quadrature_ref( quadrature_wrapper_ref &
    ref_quadrature ) const {
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

  // disjoint elements
  int n_shared_vertices = 0;
  ref_quadrature._x1_ref[ n_shared_vertices ].resize( tetra_size2 );
  ref_quadrature._x2_ref[ n_shared_vertices ].resize( tetra_size2 );
  ref_quadrature._x3_ref[ n_shared_vertices ].resize( tetra_size2 );
  ref_quadrature._y1_ref[ n_shared_vertices ].resize( tetra_size2 );
  ref_quadrature._y2_ref[ n_shared_vertices ].resize( tetra_size2 );
  ref_quadrature._y3_ref[ n_shared_vertices ].resize( tetra_size2 );
  ref_quadrature._w[ n_shared_vertices ].resize( tetra_size2 );

  lo counter = 0;
  for ( lo i_x = 0; i_x < tetra_size; ++i_x ) {
    for ( lo i_y = 0; i_y < tetra_size; ++i_y ) {
      ref_quadrature._x1_ref[ n_shared_vertices ][ counter ] = tetra_x1[ i_x ];
      ref_quadrature._x2_ref[ n_shared_vertices ][ counter ] = tetra_x2[ i_x ];
      ref_quadrature._x3_ref[ n_shared_vertices ][ counter ] = tetra_x3[ i_x ];
      ref_quadrature._y1_ref[ n_shared_vertices ][ counter ] = tetra_x1[ i_y ];
      ref_quadrature._y2_ref[ n_shared_vertices ][ counter ] = tetra_x2[ i_y ];
      ref_quadrature._y3_ref[ n_shared_vertices ][ counter ] = tetra_x3[ i_y ];
      ref_quadrature._w[ n_shared_vertices ][ counter ]
        = tetra_w[ i_x ] * tetra_w[ i_y ];
      ++counter;
    }
  }

  init_quadrature_shared_4( ref_quadrature );
  init_quadrature_shared_3( ref_quadrature );
  init_quadrature_shared_2( ref_quadrature );
  init_quadrature_shared_1( ref_quadrature );

  std::cout << "Quadrature over tensor product of tetrahedra:" << std::endl;
  std::cout << "  disjoint:  " << ref_quadrature._w[ 0 ].size( ) << std::endl;
  std::cout << "  vertex:    " << ref_quadrature._w[ 1 ].size( ) << std::endl;
  std::cout << "  edge:      " << ref_quadrature._w[ 2 ].size( ) << std::endl;
  std::cout << "  face:      " << ref_quadrature._w[ 3 ].size( ) << std::endl;
  std::cout << "  identical: " << ref_quadrature._w[ 4 ].size( ) << std::endl;
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::init_quadrature_shared_4( quadrature_wrapper_ref &
    ref_quadrature ) const {
  element tetra( { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 } );

  element_pair initial_pair = { tetra, tetra, 0 };

  ref_quadrature._ready_elems.resize( 0 );
  refine_reference_recursively( initial_pair, ref_quadrature._ready_elems );

  create_quadrature_points( ref_quadrature, 4 );

  // for ( auto it : _admissibles ) {
  //   std::cout << it << std::endl;
  // }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::init_quadrature_shared_3( quadrature_wrapper_ref &
    ref_quadrature ) const {
  element tetra1( { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 } );
  element tetra2( { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, -1.0 } );

  element_pair initial_pair = { tetra1, tetra2, 0 };

  ref_quadrature._ready_elems.resize( 0 );
  refine_reference_recursively( initial_pair, ref_quadrature._ready_elems );

  create_quadrature_points( ref_quadrature, 3 );

  // for ( auto it : _admissibles ) {
  //   std::cout << it << std::endl;
  // }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::init_quadrature_shared_2( quadrature_wrapper_ref &
    ref_quadrature ) const {
  element tetra1( { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 } );
  element tetra2( { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 },
    { 0.0, 0.0, -1.0 } );

  element_pair initial_pair = { tetra1, tetra2, 0 };

  ref_quadrature._ready_elems.resize( 0 );
  refine_reference_recursively( initial_pair, ref_quadrature._ready_elems );

  create_quadrature_points( ref_quadrature, 2 );

  // for ( auto it : _admissibles ) {
  //   std::cout << it << std::endl;
  // }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::init_quadrature_shared_1( quadrature_wrapper_ref &
    ref_quadrature ) const {
  element tetra1( { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 } );
  element tetra2( { 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 },
    { 0.0, 0.0, -1.0 } );

  element_pair initial_pair = { tetra1, tetra2, 0 };

  ref_quadrature._ready_elems.resize( 0 );
  refine_reference_recursively( initial_pair, ref_quadrature._ready_elems );

  create_quadrature_points( ref_quadrature, 1 );

  // for ( auto it : _admissibles ) {
  //   std::cout << it << std::endl;
  // }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::refine_reference_recursively( element_pair el,
  std::list< element_pair > & ready_elems ) const {
  int level = std::get< 2 >( el );
  if ( level < _singular_refinements ) {
    // std::cout << level << std::endl;
    std::array< element, 8 > new_elements_1, new_elements_2;
    new_elements_1 = std::get< 0 >( el ).refine( );
    new_elements_2 = std::get< 1 >( el ).refine( );

    lo n_shared_nodes = 0;

    for ( lo i = 0; i < 8; ++i ) {
      for ( lo j = 0; j < 8; ++j ) {
        n_shared_nodes = new_elements_1[ i ].admissible( new_elements_2[ j ] );
        bool adm = ( n_shared_nodes < 1 );

        if ( adm ) {
          // push element pair into the ready_elems list
          ready_elems.push_back( std::make_tuple(
            new_elements_1[ i ], new_elements_2[ j ], level + 1 ) );
          //_admissibles.at( level ) += 1;
        } else if ( ( !adm ) && ( level + 1 < _singular_refinements ) ) {
          // further refine the pair of nonadmissible elements
          refine_reference_recursively( std::make_tuple( new_elements_1[ i ],
                                          new_elements_2[ j ], level + 1 ),
            ready_elems );
        } else {
          // push the pair of admissible elements into ready_elems list, do
          // nothing with nonadmissible ones
          // only for debugging purposes, can be delete later

          // can be later used to include also pairs with 1, 2, 3 shared
          // vertices
          if ( n_shared_nodes < 4 ) {
            // push into the ready_elems list (for now)
            ready_elems.push_back( std::make_tuple(
              new_elements_1[ i ], new_elements_2[ j ], level + 1 ) );
            _admissibles.at( level ) += 1;
          }
        }
      }
    }
  }

  // std::array< element, 8 > new_elements;
  // new_elements = el.refine( );

  // for ( lo i = 0; i < 8; ++i ) {
  //   for ( lo j = 0; j < 8; ++j ) {
  //     if ( ( i == j ) && ( level < _singular_refinements ) ) {
  //       refine_reference_recursively( new_elements[ i ], level + 1 );
  //     } else if ( ( i == j ) && ( level == _singular_refinements ) ) {
  //       continue;
  //     } else {
  //       // generate quadrature points & weights for both el_i, el_j and
  //       store
  //       // them at the end of four separate unrolled vectors

  //       // modified version of tetrahedral_to_geometry to map from
  //       reference to
  //       // current
  //     }
  //   }
  // }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::create_quadrature_points( quadrature_wrapper_ref &
                                                  ref_quadrature,
  int n_shared_vertices ) const {
  std::list< element_pair > & ready_elems = ref_quadrature._ready_elems;

  const std::vector< sc, besthea::allocator_type< sc > > & tetra_w
    = quadrature::tetrahedron_w( _order_singular );

  for ( auto & it : ready_elems ) {
    element & el1 = std::get< 0 >( it );
    element & el2 = std::get< 1 >( it );

    sc mult = el1.area( ) * el2.area( ) * 36.0;
    std::vector< sc > x1, x2, x3, y1, y2, y3;
    reference_to_subreference(
      el1, el2, n_shared_vertices, x1, x2, x3, y1, y2, y3 );

    for ( lou i_x = 0; i_x < x1.size( ); ++i_x ) {
      for ( lou i_y = 0; i_y < y1.size( ); ++i_y ) {
        ref_quadrature._x1_ref[ n_shared_vertices ].push_back( x1[ i_x ] );
        ref_quadrature._x2_ref[ n_shared_vertices ].push_back( x2[ i_x ] );
        ref_quadrature._x3_ref[ n_shared_vertices ].push_back( x3[ i_x ] );
        ref_quadrature._w[ n_shared_vertices ].push_back(
          tetra_w[ i_x ] * tetra_w[ i_y ] * mult );
      }
    }

    switch ( n_shared_vertices ) {
      case 4:
        for ( lou i_x = 0; i_x < x1.size( ); ++i_x ) {
          for ( lou i_y = 0; i_y < y1.size( ); ++i_y ) {
            ref_quadrature._y1_ref[ n_shared_vertices ].push_back( y1[ i_y ] );
            ref_quadrature._y2_ref[ n_shared_vertices ].push_back( y2[ i_y ] );
            ref_quadrature._y3_ref[ n_shared_vertices ].push_back( y3[ i_y ] );
          }
        }
        break;
      case 3:
        for ( lou i_x = 0; i_x < x1.size( ); ++i_x ) {
          for ( lou i_y = 0; i_y < y1.size( ); ++i_y ) {
            ref_quadrature._y1_ref[ n_shared_vertices ].push_back( y1[ i_y ] );
            ref_quadrature._y2_ref[ n_shared_vertices ].push_back( y2[ i_y ] );
            ref_quadrature._y3_ref[ n_shared_vertices ].push_back( -y3[ i_y ] );
          }
        }
        break;
      case 2:
        for ( lou i_x = 0; i_x < x1.size( ); ++i_x ) {
          for ( lou i_y = 0; i_y < y1.size( ); ++i_y ) {
            ref_quadrature._y1_ref[ n_shared_vertices ].push_back( y1[ i_y ] );
            ref_quadrature._y2_ref[ n_shared_vertices ].push_back( -y2[ i_y ] );
            ref_quadrature._y3_ref[ n_shared_vertices ].push_back( -y3[ i_y ] );
          }
        }
        break;
      case 1:
        for ( lou i_x = 0; i_x < x1.size( ); ++i_x ) {
          for ( lou i_y = 0; i_y < y1.size( ); ++i_y ) {
            ref_quadrature._y1_ref[ n_shared_vertices ].push_back( -y1[ i_y ] );
            ref_quadrature._y2_ref[ n_shared_vertices ].push_back( -y2[ i_y ] );
            ref_quadrature._y3_ref[ n_shared_vertices ].push_back( -y3[ i_y ] );
          }
        }
        break;
    }
  }
}

template< class kernel_type, class test_space_type, class trial_space_type >
void besthea::bem::tetrahedral_spacetime_be_assembler< kernel_type,
  test_space_type,
  trial_space_type >::reference_to_subreference( const element & el1,
  const element & el2, int type_int, std::vector< sc > & x1,
  std::vector< sc > & x2, std::vector< sc > & x3, std::vector< sc > & y1,
  std::vector< sc > & y2, std::vector< sc > & y3 ) const {
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_x1
    = quadrature::tetrahedron_x1( _order_singular );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_x2
    = quadrature::tetrahedron_x2( _order_singular );
  const std::vector< sc, besthea::allocator_type< sc > > & tetra_x3
    = quadrature::tetrahedron_x3( _order_singular );

  lo size = tetra_x1.size( );

  x1.resize( size );
  x2.resize( size );
  x3.resize( size );
  y1.resize( size );
  y2.resize( size );
  y3.resize( size );

  auto & nodes1 = el1._nodes;
  const besthea::linear_algebra::coordinates< 3 > & sub_x1 = nodes1[ 0 ];
  const besthea::linear_algebra::coordinates< 3 > & sub_x2 = nodes1[ 1 ];
  const besthea::linear_algebra::coordinates< 3 > & sub_x3 = nodes1[ 2 ];
  const besthea::linear_algebra::coordinates< 3 > & sub_x4 = nodes1[ 3 ];

  auto & nodes2 = el2._nodes;
  const besthea::linear_algebra::coordinates< 3 > & sub_y1 = nodes2[ 0 ];
  const besthea::linear_algebra::coordinates< 3 > & sub_y2 = nodes2[ 1 ];
  const besthea::linear_algebra::coordinates< 3 > & sub_y3 = nodes2[ 2 ];
  const besthea::linear_algebra::coordinates< 3 > & sub_y4 = nodes2[ 3 ];

  for ( lo i = 0; i < size; ++i ) {
    x1[ i ] = sub_x1[ 0 ] + ( sub_x2[ 0 ] - sub_x1[ 0 ] ) * tetra_x1[ i ]
      + ( sub_x3[ 0 ] - sub_x1[ 0 ] ) * tetra_x2[ i ]
      + ( sub_x4[ 0 ] - sub_x1[ 0 ] ) * tetra_x3[ i ];
    x2[ i ] = sub_x1[ 1 ] + ( sub_x2[ 1 ] - sub_x1[ 1 ] ) * tetra_x1[ i ]
      + ( sub_x3[ 1 ] - sub_x1[ 1 ] ) * tetra_x2[ i ]
      + ( sub_x4[ 1 ] - sub_x1[ 1 ] ) * tetra_x3[ i ];
    x3[ i ] = sub_x1[ 2 ] + ( sub_x2[ 2 ] - sub_x1[ 2 ] ) * tetra_x1[ i ]
      + ( sub_x3[ 2 ] - sub_x1[ 2 ] ) * tetra_x2[ i ]
      + ( sub_x4[ 2 ] - sub_x1[ 2 ] ) * tetra_x3[ i ];

    y1[ i ] = sub_y1[ 0 ] + ( sub_y2[ 0 ] - sub_y1[ 0 ] ) * tetra_x1[ i ]
      + ( sub_y3[ 0 ] - sub_y1[ 0 ] ) * tetra_x2[ i ]
      + ( sub_y4[ 0 ] - sub_y1[ 0 ] ) * tetra_x3[ i ];
    y2[ i ] = sub_y1[ 1 ] + ( sub_y2[ 1 ] - sub_y1[ 1 ] ) * tetra_x1[ i ]
      + ( sub_y3[ 1 ] - sub_y1[ 1 ] ) * tetra_x2[ i ]
      + ( sub_y4[ 1 ] - sub_y1[ 1 ] ) * tetra_x3[ i ];
    y3[ i ] = sub_y1[ 2 ] + ( sub_y2[ 2 ] - sub_y1[ 2 ] ) * tetra_x1[ i ]
      + ( sub_y3[ 2 ] - sub_y1[ 2 ] ) * tetra_x2[ i ]
      + ( sub_y4[ 2 ] - sub_y1[ 2 ] ) * tetra_x3[ i ];
  }
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
    const quadrature_wrapper_ref & ref_quadrature,
    quadrature_wrapper & my_quadrature ) const {
  const sc * x1rot = nullptr;
  const sc * x2rot = nullptr;
  const sc * x3rot = nullptr;
  const sc * x4rot = nullptr;
  const sc * y1rot = nullptr;
  const sc * y2rot = nullptr;
  const sc * y3rot = nullptr;
  const sc * y4rot = nullptr;

  std::vector< sc * > x_data = { (sc *) x1.data( ), (sc *) x2.data( ),
    (sc *) x3.data( ), (sc *) x4.data( ) };
  x1rot = x_data[ perm_test[ 0 ] ];  // = x1.data( );
  x2rot = x_data[ perm_test[ 1 ] ];  // = x2.data( );
  x3rot = x_data[ perm_test[ 2 ] ];  // = x3.data( );
  x4rot = x_data[ perm_test[ 3 ] ];  // = x4.data( );

  std::vector< sc * > y_data = { (sc *) y1.data( ), (sc *) y2.data( ),
    (sc *) y3.data( ), (sc *) y4.data( ) };
  y1rot = y_data[ perm_trial[ 0 ] ];  // = y1.data( );
  y2rot = y_data[ perm_trial[ 1 ] ];  // = y2.data( );
  y3rot = y_data[ perm_trial[ 2 ] ];  // = y3.data( );
  y4rot = y_data[ perm_trial[ 3 ] ];  // = y4.data( );

  const sc * x1_ref = ref_quadrature._x1_ref[ n_shared_vertices ].data( );
  const sc * x2_ref = ref_quadrature._x2_ref[ n_shared_vertices ].data( );
  const sc * x3_ref = ref_quadrature._x3_ref[ n_shared_vertices ].data( );
  const sc * y1_ref = ref_quadrature._y1_ref[ n_shared_vertices ].data( );
  const sc * y2_ref = ref_quadrature._y2_ref[ n_shared_vertices ].data( );
  const sc * y3_ref = ref_quadrature._y3_ref[ n_shared_vertices ].data( );

  sc * x1_mapped = my_quadrature._x1.data( );
  sc * x2_mapped = my_quadrature._x2.data( );
  sc * x3_mapped = my_quadrature._x3.data( );
  sc * t_mapped = my_quadrature._t.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );
  sc * tau_mapped = my_quadrature._tau.data( );

  lo size = ref_quadrature._w[ n_shared_vertices ].size( );

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
  for ( int i = 0; i < 4; ++i ) {
    perm_test[ i ] = i;
    perm_trial[ i ] = i;
  }
  n_shared_vertices = 0;

  if ( i_test == i_trial ) {
    n_shared_vertices = 4;
    return;
  }

  linear_algebra::indices< 4 > test_elem;
  linear_algebra::indices< 4 > trial_elem;

  _test_space->get_mesh( )->get_element( i_test, test_elem );
  _trial_space->get_mesh( )->get_element( i_trial, trial_elem );

  for ( lo i = 0; i < 4; ++i ) {
    for ( lo j = 0; j < 4; ++j ) {
      if ( test_elem[ i ] == trial_elem[ j ] ) {
        ++n_shared_vertices;
      }
    }
  }

  if ( n_shared_vertices == 0 ) {
    return;
  }

  std::array< int, 4 > idx_test;
  std::array< int, 4 > idx_trial;
  std::array< int, 4 > non_shared_test = { 1, 1, 1, 1 };
  std::array< int, 4 > non_shared_trial = { 1, 1, 1, 1 };

  int counter = 0;

  for ( lo i = 0; i < 4; ++i ) {
    for ( lo j = 0; j < 4; ++j ) {
      if ( test_elem[ i ] == trial_elem[ j ] ) {
        idx_test[ counter ] = i;
        idx_trial[ counter ] = j;
        non_shared_test[ i ] = 0;
        non_shared_trial[ j ] = 0;
        ++counter;
      }
    }
  }

  for ( lo i = 0; i < n_shared_vertices; ++i ) {
    perm_test[ i ] = idx_test[ i ];
    perm_trial[ i ] = idx_trial[ i ];
  }

  counter = n_shared_vertices;
  for ( int i = 0; i < 4; ++i ) {
    if ( non_shared_test[ i ] == 1 ) {
      perm_test[ counter ] = i;
      ++counter;
    }
  }

  counter = n_shared_vertices;
  for ( int i = 0; i < 4; ++i ) {
    if ( non_shared_trial[ i ] == 1 ) {
      perm_trial[ counter ] = i;
      ++counter;
    }
  }

  // END TO REMOVE
}

template class besthea::bem::tetrahedral_spacetime_be_assembler<
  besthea::bem::spacetime_heat_kernel,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p1 >,
  besthea::bem::tetrahedral_spacetime_be_space<
    besthea::bem::spacetime_basis_tetra_p1 > >;

template class besthea::bem::tetrahedral_spacetime_be_assembler<
  besthea::bem::spacetime_heat_kernel_normal_derivative,
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
