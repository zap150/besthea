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

#include "besthea/uniform_spacetime_be_evaluator.h"

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/quadrature.h"
#include "besthea/uniform_spacetime_be_space.h"
#include "besthea/uniform_spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_heat_sl_kernel_antiderivative.h"
#include "omp.h"

template< class kernel_type, class space_type >
besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::uniform_spacetime_be_evaluator( kernel_type & kernel,
  space_type & space, int order_spatial )
  : _kernel( &kernel ), _space( &space ), _order_spatial( order_spatial ) {
}

template< class kernel_type, class space_type >
besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::~uniform_spacetime_be_evaluator( ) {
}
/*
template< class kernel_type, class space_type >
void besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::evaluate( const std::vector< sc > & x,
  const block_vector_type & density, block_vector_type & result ) const {
  auto & basis = _space->get_basis( );
  auto mesh = _space->get_mesh( );

  lo n_timesteps = mesh->get_n_temporal_elements( );
  sc timestep = mesh->get_timestep( );
  lo n_points = x.size( ) / 3;
  lo n_elements = mesh->get_n_spatial_elements( );
  lo loc_dim = basis.dimension_local( );

  result.resize( n_timesteps );
  result.resize_blocks( n_points, true );

  int max_threads = omp_get_max_threads( );

#pragma omp parallel
  {
    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * y1_ref = my_quadrature._y1_ref.data( );
    sc * y2_ref = my_quadrature._y2_ref.data( );
    sc * wy = my_quadrature._wy.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );
    lo size_quad = my_quadrature._wy.size( );

    sc * kernel_data = my_quadrature._kernel_values.data( );
    sc * x1 = my_quadrature._x1.data( );
    sc * x2 = my_quadrature._x2.data( );
    sc * x3 = my_quadrature._x3.data( );

    lo my_n_points = n_points / max_threads;
    int thread_num = omp_get_thread_num( );
    lo thread_offset = thread_num * my_n_points;
    const sc * my_x_data = x.data( ) + 3 * thread_offset;
    sc * my_result_data = nullptr;
    if ( thread_num == max_threads - 1 ) {
      my_n_points += n_points % max_threads;
    }

    lo size_chunk = my_quadrature._kernel_values.size( );
    lo my_n_chunks = my_n_points / size_chunk;
    lo chunk_offset;

    sc scaled_delta, area, basis_value, density_value;
    sc y1[ 3 ], y2[ 3 ], y3[ 3 ], ny[ 3 ];
    std::vector< lo > l2g( loc_dim );

    for ( lo i_chunk = 0; i_chunk <= my_n_chunks; ++i_chunk ) {
      chunk_offset = i_chunk * size_chunk;

      // modifying size of the chunk for the last iteration
      if ( i_chunk == my_n_chunks ) {
        size_chunk = my_n_points % size_chunk;
        if ( size_chunk == 0 ) {
          break;
        }
      }

      // copying points into local buffers
#pragma omp simd aligned( x1, x2, x3 : DATA_ALIGN ) simdlen( DATA_WIDTH )
      for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
        x1[ i_point ] = my_x_data[ 3 * ( chunk_offset + i_point ) ];
        x2[ i_point ] = my_x_data[ 3 * ( chunk_offset + i_point ) + 1 ];
        x3[ i_point ] = my_x_data[ 3 * ( chunk_offset + i_point ) + 2 ];
      }

      for ( lo delta = 0; delta <= n_timesteps - 1; ++delta ) {
        scaled_delta = ( delta + 0.5 ) * timestep;

        for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
          mesh->get_spatial_nodes( i_elem, y1, y2, y3 );
          mesh->get_spatial_normal( i_elem, ny );
          area = mesh->spatial_area( i_elem );
          basis.local_to_global( i_elem, l2g );
          triangle_to_geometry( y1, y2, y3, my_quadrature );

          if ( delta == 0 ) {
            for ( lo i_quad = 0; i_quad < size_quad; ++i_quad ) {
#pragma omp simd aligned(                                      \
  x1, x2, x3, y1_mapped, y2_mapped, y3_mapped, kernel_data, wy \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
              for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
                kernel_data[ i_point ]
                  = _kernel->anti_tau_limit(
                      x1[ i_point ] - y1_mapped[ i_quad ],
                      x2[ i_point ] - y2_mapped[ i_quad ],
                      x3[ i_point ] - y3_mapped[ i_quad ], ny )
                  * wy[ i_quad ];
              }  // i_point

              for ( lo i_loc = 0; i_loc < loc_dim; ++i_loc ) {
                basis_value = basis.evaluate(
                  i_elem, i_loc, y1_ref[ i_quad ], y2_ref[ i_quad ], ny );
                basis_value *= area;

                // adding + values
                for ( lo k = 0; k <= n_timesteps - 1; ++k ) {
                  my_result_data = result.get_block( k ).data( ) + thread_offset
                    + chunk_offset;
                  density_value = density.get( k, l2g[ i_loc ] ) * basis_value;

#pragma omp simd aligned( kernel_data : DATA_ALIGN ) simdlen( DATA_WIDTH )
                  for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
                    my_result_data[ i_point ]
                      += kernel_data[ i_point ] * density_value;
                  }  // i_point
                }    // k
              }      // i_loc
            }        // i_quad
          }          // delta == 0

          for ( lo i_quad = 0; i_quad < size_quad; ++i_quad ) {
#pragma omp simd aligned(                                      \
  x1, x2, x3, y1_mapped, y2_mapped, y3_mapped, kernel_data, wy \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
              kernel_data[ i_point ]
                = _kernel->anti_tau_regular(
                    x1[ i_point ] - y1_mapped[ i_quad ],
                    x2[ i_point ] - y2_mapped[ i_quad ],
                    x3[ i_point ] - y3_mapped[ i_quad ], ny, scaled_delta )
                * wy[ i_quad ];
            }  // i_point

            for ( lo i_loc = 0; i_loc < loc_dim; ++i_loc ) {
              basis_value = basis.evaluate(
                i_elem, i_loc, y1_ref[ i_quad ], y2_ref[ i_quad ], ny );
              basis_value *= area;

              // adding + values
              for ( lo k = delta + 1; k <= n_timesteps - 1; ++k ) {
                my_result_data = result.get_block( k ).data( ) + thread_offset
                  + chunk_offset;
                density_value
                  = density.get( k - delta - 1, l2g[ i_loc ] ) * basis_value;

#pragma omp simd aligned( kernel_data : DATA_ALIGN ) simdlen( DATA_WIDTH )
                for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
                  my_result_data[ i_point ]
                    += kernel_data[ i_point ] * density_value;
                }  // i_point
              }    // k

              // adding - values
              for ( lo k = delta; k <= n_timesteps - 1; ++k ) {
                my_result_data = result.get_block( k ).data( ) + thread_offset
                  + chunk_offset;
                density_value
                  = density.get( k - delta, l2g[ i_loc ] ) * basis_value;

#pragma omp simd aligned( kernel_data : DATA_ALIGN ) simdlen( DATA_WIDTH )
                for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
                  my_result_data[ i_point ]
                    -= kernel_data[ i_point ] * density_value;
                }  // i_point
              }    // k

            }  // i_loc
          }    // i_quad

        }  // i_elem
      }    // delta
    }      // i_chunk
  }        // omp parallel
}
*/
///*
template< class kernel_type, class space_type >
void besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::evaluate( const std::vector< sc > & x,
  const block_vector_type & density, block_vector_type & result ) const {
  auto & basis = _space->get_basis( );
  auto mesh = _space->get_mesh( );

  lo n_timesteps = mesh->get_n_temporal_elements( );
  sc timestep = mesh->get_timestep( );
  lo n_points = x.size( ) / 3;
  lo n_elements = mesh->get_n_spatial_elements( );
  lo loc_dim = basis.dimension_local( );

  result.resize( n_timesteps );
  result.resize_blocks( n_points, true );

  int max_threads = omp_get_max_threads( );

  int sc_width = DATA_ALIGN / sizeof( sc );
  // number of whole simd vectors in total
  lo n_vectors = n_points / sc_width;
  // number of points processed in a scalar mode
  lo n_scalar = n_points % sc_width;
  lo n_vectors_per_thread = n_vectors / max_threads;
  lo n_vectors_rest = n_vectors % max_threads;

#pragma omp parallel
  {
    int thread_num = omp_get_thread_num( );

    quadrature_wrapper my_quadrature;
    init_quadrature( my_quadrature );
    sc * y1_ref = my_quadrature._y1_ref.data( );
    sc * y2_ref = my_quadrature._y2_ref.data( );
    sc * wy = my_quadrature._wy.data( );
    sc * y1_mapped = my_quadrature._y1.data( );
    sc * y2_mapped = my_quadrature._y2.data( );
    sc * y3_mapped = my_quadrature._y3.data( );
    lo size_quad = my_quadrature._wy.size( );

    sc * kernel_data = my_quadrature._kernel_values.data( );
    sc * x1 = my_quadrature._x1.data( );
    sc * x2 = my_quadrature._x2.data( );
    sc * x3 = my_quadrature._x3.data( );

    sc * my_result_data = nullptr;

    // number of points to process per thread
    lo my_n_vectors = n_vectors_per_thread;
    if ( thread_num < n_vectors_rest ) {
      ++my_n_vectors;
    }
    lo my_n_points = my_n_vectors * sc_width;

    // starting offset
    lo my_offset = thread_num
      * my_n_points;  // all preceding threads have at least as many points as
                      // this (except for the scalar part)
    // adding the additional vectors
    if ( thread_num < n_vectors_rest ) {
      my_offset += thread_num * sc_width;
    } else {
      my_offset += n_vectors_rest * sc_width;
    }

    // last threads processing the scalar portion as well
    if ( thread_num == max_threads - 1 ) {
      my_n_points += n_scalar;
    }

    const sc * my_x_data = x.data( ) + 3 * my_offset;

    lo size_chunk = my_quadrature._kernel_values.size( );
    lo my_n_chunks = my_n_points / size_chunk;
    lo chunk_offset;

    sc scaled_delta, area, basis_value, density_value;
    sc y1[ 3 ], y2[ 3 ], y3[ 3 ], ny[ 3 ];
    std::vector< lo > l2g( loc_dim );

    for ( lo i_chunk = 0; i_chunk <= my_n_chunks; ++i_chunk ) {
      chunk_offset = i_chunk * size_chunk;

      // modifying size of the chunk for the last iteration
      if ( i_chunk == my_n_chunks ) {
        size_chunk = my_n_points % size_chunk;
        if ( size_chunk == 0 ) {
          break;
        }
      }

      // copying points into local buffers
#pragma omp simd aligned( x1, x2, x3 : DATA_ALIGN ) simdlen( DATA_WIDTH )
      for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
        x1[ i_point ] = my_x_data[ 3 * ( chunk_offset + i_point ) ];
        x2[ i_point ] = my_x_data[ 3 * ( chunk_offset + i_point ) + 1 ];
        x3[ i_point ] = my_x_data[ 3 * ( chunk_offset + i_point ) + 2 ];
      }

      for ( lo delta = 0; delta <= n_timesteps - 1; ++delta ) {
        scaled_delta = ( delta + 0.5 ) * timestep;

        for ( lo i_elem = 0; i_elem < n_elements; ++i_elem ) {
          mesh->get_spatial_nodes( i_elem, y1, y2, y3 );
          mesh->get_spatial_normal( i_elem, ny );
          area = mesh->spatial_area( i_elem );
          basis.local_to_global( i_elem, l2g );
          triangle_to_geometry( y1, y2, y3, my_quadrature );

          if ( delta == 0 ) {
            for ( lo i_quad = 0; i_quad < size_quad; ++i_quad ) {
#pragma omp simd aligned(                                      \
  x1, x2, x3, y1_mapped, y2_mapped, y3_mapped, kernel_data, wy \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
              for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
                kernel_data[ i_point ]
                  = _kernel->anti_tau_limit(
                      x1[ i_point ] - y1_mapped[ i_quad ],
                      x2[ i_point ] - y2_mapped[ i_quad ],
                      x3[ i_point ] - y3_mapped[ i_quad ], ny )
                  * wy[ i_quad ];
              }  // i_point

              for ( lo i_loc = 0; i_loc < loc_dim; ++i_loc ) {
                basis_value = basis.evaluate(
                  i_elem, i_loc, y1_ref[ i_quad ], y2_ref[ i_quad ], ny );
                basis_value *= area;

                // adding + values
                for ( lo k = 0; k <= n_timesteps - 1; ++k ) {
                  my_result_data
                    = result.get_block( k ).data( ) + my_offset + chunk_offset;
                  density_value = density.get( k, l2g[ i_loc ] ) * basis_value;

#pragma omp simd aligned( kernel_data, my_result_data \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
                  for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
                    my_result_data[ i_point ]
                      += kernel_data[ i_point ] * density_value;
                  }  // i_point
                }    // k
              }      // i_loc
            }        // i_quad
          }          // delta == 0

          for ( lo i_quad = 0; i_quad < size_quad; ++i_quad ) {
#pragma omp simd aligned(                                      \
  x1, x2, x3, y1_mapped, y2_mapped, y3_mapped, kernel_data, wy \
  : DATA_ALIGN ) simdlen( DATA_WIDTH )
            for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
              kernel_data[ i_point ]
                = _kernel->anti_tau_regular(
                    x1[ i_point ] - y1_mapped[ i_quad ],
                    x2[ i_point ] - y2_mapped[ i_quad ],
                    x3[ i_point ] - y3_mapped[ i_quad ], ny, scaled_delta )
                * wy[ i_quad ];
            }  // i_point

            for ( lo i_loc = 0; i_loc < loc_dim; ++i_loc ) {
              basis_value = basis.evaluate(
                i_elem, i_loc, y1_ref[ i_quad ], y2_ref[ i_quad ], ny );
              basis_value *= area;

              // adding + values
              for ( lo k = delta + 1; k <= n_timesteps - 1; ++k ) {
                my_result_data
                  = result.get_block( k ).data( ) + my_offset + chunk_offset;
                density_value
                  = density.get( k - delta - 1, l2g[ i_loc ] ) * basis_value;

#pragma omp simd aligned( kernel_data, my_result_data \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
                for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
                  my_result_data[ i_point ]
                    += kernel_data[ i_point ] * density_value;
                }  // i_point
              }    // k

              // adding - values
              for ( lo k = delta; k <= n_timesteps - 1; ++k ) {
                my_result_data
                  = result.get_block( k ).data( ) + my_offset + chunk_offset;
                density_value
                  = density.get( k - delta, l2g[ i_loc ] ) * basis_value;

#pragma omp simd aligned( kernel_data, my_result_data \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
                for ( lo i_point = 0; i_point < size_chunk; ++i_point ) {
                  my_result_data[ i_point ]
                    -= kernel_data[ i_point ] * density_value;
                }  // i_point
              }    // k

            }  // i_loc
          }    // i_quad

        }  // i_elem
      }    // delta
    }      // i_chunk
  }        // omp parallel
}
//*/
template< class kernel_type, class space_type >
void besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::init_quadrature( quadrature_wrapper & my_quadrature ) const {
  // calling copy constructor of std::vector
  my_quadrature._y1_ref = quadrature::triangle_x1( _order_spatial );
  my_quadrature._y2_ref = quadrature::triangle_x2( _order_spatial );
  my_quadrature._wy = quadrature::triangle_w( _order_spatial );

  lo size = my_quadrature._wy.size( );
  my_quadrature._y1.resize( size );
  my_quadrature._y2.resize( size );
  my_quadrature._y3.resize( size );

  lo size_chunk = DATA_WIDTH * 32;
  my_quadrature._kernel_values.resize( size_chunk );
  my_quadrature._x1.resize( size_chunk );
  my_quadrature._x2.resize( size_chunk );
  my_quadrature._x3.resize( size_chunk );
}

template< class kernel_type, class space_type >
void besthea::bem::uniform_spacetime_be_evaluator< kernel_type,
  space_type >::triangle_to_geometry( const sc * x1, const sc * x2,
  const sc * x3, quadrature_wrapper & my_quadrature ) const {
  const sc * y1_ref = my_quadrature._y1_ref.data( );
  const sc * y2_ref = my_quadrature._y2_ref.data( );
  sc * y1_mapped = my_quadrature._y1.data( );
  sc * y2_mapped = my_quadrature._y2.data( );
  sc * y3_mapped = my_quadrature._y3.data( );

  lo size = my_quadrature._wy.size( );

#pragma omp simd aligned( y1_mapped, y2_mapped, y3_mapped, y1_ref, y2_ref \
                          : DATA_ALIGN ) simdlen( DATA_WIDTH )
  for ( lo i = 0; i < size; ++i ) {
    y1_mapped[ i ] = x1[ 0 ] + ( x2[ 0 ] - x1[ 0 ] ) * y1_ref[ i ]
      + ( x3[ 0 ] - x1[ 0 ] ) * y2_ref[ i ];
    y2_mapped[ i ] = x1[ 1 ] + ( x2[ 1 ] - x1[ 1 ] ) * y1_ref[ i ]
      + ( x3[ 1 ] - x1[ 1 ] ) * y2_ref[ i ];
    y3_mapped[ i ] = x1[ 2 ] + ( x2[ 2 ] - x1[ 2 ] ) * y1_ref[ i ]
      + ( x3[ 2 ] - x1[ 2 ] ) * y2_ref[ i ];
  }
}

template class besthea::bem::uniform_spacetime_be_evaluator<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::uniform_spacetime_be_evaluator<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class besthea::bem::uniform_spacetime_be_evaluator<
  besthea::bem::uniform_spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::uniform_spacetime_be_evaluator<
  besthea::bem::uniform_spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
