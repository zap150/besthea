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

/** @file lagrange_interpolant.h
 * @brief Evaluation of Lagrange polynomials.
 */

#ifndef INCLUDE_BESTHEA_LAGRANGE_INTERPOLANT_H_
#define INCLUDE_BESTHEA_LAGRANGE_INTERPOLANT_H_

#include "besthea/settings.h"
#include "besthea/vector.h"
#include <cmath>

namespace besthea {
  namespace bem {
    class lagrange_interpolant;
  }
}

/**
 * Class for the evaluation of Lagrange polynomials for Chebyshev nodes
 */
class besthea::bem::lagrange_interpolant {
 public:
  using vector_type = besthea::linear_algebra::vector;

  /**
   * Constructor.
   * @param[in] order Order of the Lagrange polynomials
   * @note @p _nodes are initialized with Chebyshev nodes of order (order + 1)
   */
  lagrange_interpolant( const lo order )
    : _order( order ),
      _nodes( vector_type( order + 1, false ) ) {
    for ( lo i = 0; i <= _order; ++i )
      _nodes[ i ] = cos( ( _pi * ( 2 * i + 1 ) ) / ( 2 * ( _order + 1 ) ) );
  }

  lagrange_interpolant( const lagrange_interpolant & that ) = delete;
  
  /**
   * Evaluate Lagrange polynomials in [-1, 1]
   * @param[in] index Index of the Lagrange polynomial which is evaluated
   *                  (has to be in {0, ..., order})
   * @param[in] eval_points Points in [-1, 1] where polynomial is evaluated
   * @param[in,out]  values  Resulting values (at input its size should be at 
   *                         least the size of @p eval_points)
   */
  void evaluate( const lo index, const vector_type eval_points, 
                 vector_type & values) {
    // initialize values to 1;
    for ( lo i = 0; i < eval_points.size( ); ++i )
      values[ i ] = 1.0;
    for ( lo k = 0; k < index; ++ k)
      for ( lo i = 0; i < eval_points.size( ); ++i )
        values[ i ] *= ( eval_points[ i ] - _nodes[ k ] ) / 
                       ( _nodes[ index ] - _nodes[ k ] );
    for ( lo k = index + 1; k <= _order; ++ k )
      for ( lo i = 0; i < eval_points.size( ); ++i )
        values[ i ] *= ( eval_points[ i ] - _nodes[ k ] ) / 
                       ( _nodes[ index ] - _nodes[ k ] );
  }
  
  vector_type get_nodes( ) {
    return _nodes;
  };
  
 private:
  lo _order;           //!< number of interpolation nodes
  vector_type _nodes;  //!< interpolation nodes determining the Lagrange
                       //!< polynomials
  const sc _pi{ M_PI }; //!< Auxiliary variable
};

#endif /* INCLUDE_BESTHEA_LAGRANGE_INTERPOLANT_H_ */
