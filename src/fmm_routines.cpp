/*
Copyright (c) 2021, VSB - Technical University of Ostrava and Graz University of
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

#include "besthea/fmm_routines.h"

#include "besthea/chebyshev_evaluator.h"
#include "besthea/vector.h"

#include <math.h>

using besthea::linear_algebra::vector;

void compute_spatial_m2m_coeffs( const lo n_space_levels, const lo spat_order,
  const sc spat_half_size_bounding_box_unpadded,
  const std::vector< sc > & spatial_paddings_per_space_level,
  std::vector< besthea::linear_algebra::vector > & m2m_coeffs_s_left,
  std::vector< besthea::linear_algebra::vector > & m2m_coeffs_s_right ) {
  m2m_coeffs_s_left.resize( n_space_levels - 1 );
  m2m_coeffs_s_right.resize( n_space_levels - 1 );

  for ( lo i = 0; i < n_space_levels - 1; ++i ) {
    m2m_coeffs_s_left[ i ].resize( ( spat_order + 1 ) * ( spat_order + 1 ) );
    m2m_coeffs_s_right[ i ].resize( ( spat_order + 1 ) * ( spat_order + 1 ) );
  }

  vector nodes( spat_order + 1, false );
  for ( lo i = 0; i <= spat_order; ++i )
    nodes[ i ] = cos( ( M_PI * ( 2 * i + 1 ) ) / ( 2 * ( spat_order + 1 ) ) );
  // evaluate Chebyshev polynomials at the nodes (needed for coefficients)
  vector all_values_cheb_std_intrvl(
    ( spat_order + 1 ) * ( spat_order + 1 ), false );

  besthea::bem::chebyshev_evaluator chebyshev( spat_order );
  chebyshev.evaluate( nodes, all_values_cheb_std_intrvl );
  // vector to store values of Chebyshev polynomials for transformed intervals
  vector all_values_cheb_trf_intrvl(
    ( spat_order + 1 ) * ( spat_order + 1 ), false );
  // initialize vectors to store transformed nodes
  vector nodes_l_child( spat_order + 1, false );
  vector nodes_r_child( spat_order + 1, false );

  sc h_par_no_pad = spat_half_size_bounding_box_unpadded;
  sc h_child_no_pad;

  for ( lo curr_level = 0; curr_level < n_space_levels - 1; ++curr_level ) {
    h_child_no_pad = h_par_no_pad / 2.0;
    sc padding_par = spatial_paddings_per_space_level[ curr_level ];
    sc padding_child = spatial_paddings_per_space_level[ curr_level + 1 ];
    // transform the nodes from [-1, 1] to the child interval and then back to
    // [-1, 1] with the transformation of the parent interval:
    for ( lo j = 0; j <= spat_order; ++j ) {
      nodes_l_child[ j ] = 1.0 / ( h_par_no_pad + padding_par )
        * ( -h_child_no_pad + ( h_child_no_pad + padding_child ) * nodes[ j ] );
      nodes_r_child[ j ] = 1.0 / ( h_par_no_pad + padding_par )
        * ( h_child_no_pad + ( h_child_no_pad + padding_child ) * nodes[ j ] );
    }
    // compute m2m coefficients at current level along all dimensions
    // for i1 > i0 the coefficients are known to be zero
    chebyshev.evaluate( nodes_l_child, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= spat_order; ++i0 ) {
      for ( lo i1 = 0; i1 <= i0; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i1 * ( spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i0 * ( spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( spat_order + 1.0 );
        if ( i1 == 0 ) {
          coeff /= 2.0;
        }
        m2m_coeffs_s_left[ curr_level ][ ( spat_order + 1 ) * i0 + i1 ] = coeff;
      }
    }

    chebyshev.evaluate( nodes_r_child, all_values_cheb_trf_intrvl );
    for ( lo i0 = 0; i0 <= spat_order; ++i0 ) {
      for ( lo i1 = 0; i1 <= i0; ++i1 ) {
        sc coeff = 0;
        for ( lo n = 0; n <= spat_order; ++n ) {
          coeff += all_values_cheb_std_intrvl[ i1 * ( spat_order + 1 ) + n ]
            * all_values_cheb_trf_intrvl[ i0 * ( spat_order + 1 ) + n ];
        }
        coeff *= 2.0 / ( spat_order + 1 );
        if ( i1 == 0 ) {
          coeff /= 2.0;
        }
        m2m_coeffs_s_right[ curr_level ][ ( spat_order + 1 ) * i0 + i1 ]
          = coeff;
      }
    }
    // update for next iteration
    h_par_no_pad = h_child_no_pad;
  }
}
