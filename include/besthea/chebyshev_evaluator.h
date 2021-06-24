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

/** @file chebyshev_evaluator.h
 * @brief Contains a class which allows to evaluate 1D Chebyshev polynomials and
 * their first derivatives in the standard interval [-1,1].
 * @note updated documentation
 */

#ifndef INCLUDE_BESTHEA_CHEBYSHEV_EVALUATOR_H_
#define INCLUDE_BESTHEA_CHEBYSHEV_EVALUATOR_H_

#include "besthea/settings.h"
#include "besthea/vector.h"

#include <cmath>

namespace besthea {
  namespace bem {
    class chebyshev_evaluator;
  }
}

/**
 * Class for the evaluation of 1D Chebyshev polynomials and their first
 * derivatives in the standard interval [-1,1].
 */
class besthea::bem::chebyshev_evaluator {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Constructor.
   * @param[in] order Highest order of the evaluated Chebyshev polynomials.
   */
  chebyshev_evaluator( const lo order ) : _order( order ) {
  }

  chebyshev_evaluator( const chebyshev_evaluator & that ) = delete;

  /**
   * Sets the order up to which Chebyshev polynomials and their derivatives are
   * evaluated by the corresponding evaluation routines.
   * @param[in] order Highest order of the evaluated Chebyshev polynomials.
   */
  void set_order( int order ) {
    _order = order;
  }

  /**
   * Evaluate the Chebyshev polynomials of order {0,..., @p _order } for points
   * in the interval [-1,1].
   * @param[in] eval_points Points in [-1, 1] where polynomials are evaluated.
   * @param[in,out]  all_values Resulting values. At input the size of this
   *                            vector should be at least
   *                            ( @p _order + 1) * size( @p eval_points ).
   * @note If n evaluation points are provided, @p all_values [ i * n + j ] is
   * the value of the i-th Chebyshev polynomial evaluated at the j-th point for
   * all i in {0, ..., @p _order } and j in {0, ..., n-1}.
   */
  void evaluate(
    const vector_type & eval_points, vector_type & all_values ) const {
    const lo sz = eval_points.size( );
    // Chebyshev polynomial T_0(x) = 1.
    for ( lo i = 0; i < sz; ++i ) all_values[ i ] = 1.0;
    if ( _order > 0 ) {
      // Chebyshev polynomial T_1(x) = x.
      for ( lo i = 0; i < sz; ++i ) {
        all_values[ sz + i ] = eval_points[ i ];
      }
      // Recursion formula to compute other polynomial values.
      for ( lo j = 2; j <= _order; ++j ) {
        for ( lo i = 0; i < sz; ++i ) {
          all_values[ j * sz + i ]
            = 2 * eval_points[ i ] * all_values[ ( j - 1 ) * sz + i ];
          all_values[ j * sz + i ] -= all_values[ ( j - 2 ) * sz + i ];
        }
      }
    }
  }

  /**
   * Evaluate derivatives of all Chebyshev polynomial up to the given order for
   * points in the interval [-1, 1].
   * @param[in] eval_points Points in [-1, 1] where polynomials are evaluated.
   * @param[in,out] all_values Resulting values. At input the size of this
   *                           vector should be at least
   *                           ( @p _order + 1) * size( @p eval_points ).
   * @note If n evaluation points are provided, @p all_values [ i * n + j ] is
   * the value of the i-th Chebyshev polynomial's derivative evaluated at the
   * j-th point for all i in {0, ..., @p _order } and j in {0, ..., n-1}.
   */
  void evaluate_derivative(
    const vector_type & eval_points, vector_type & all_values ) const {
    const lo sz = eval_points.size( );
    // Derivative T_0'(x) = 0.
    for ( lo i = 0; i < sz; ++i ) all_values[ i ] = 0.0;
    if ( _order > 0 ) {
      // Derivative T_1'(x) = 1.
      for ( lo i = 0; i < sz; ++i ) {
        all_values[ sz + i ] = 1.0;
      }
    }
    if ( _order > 1 ) {
      // Derivative T_2'(x) = 4x.
      for ( lo i = 0; i < sz; ++i ) {
        all_values[ 2 * sz + i ] = 4.0 * eval_points[ i ];
      }
      // Recursion formula to compute other values of polynomial derivatives.
      for ( lo j = 3; j <= _order; ++j ) {
        for ( lo i = 0; i < sz; ++i ) {
          all_values[ j * sz + i ] = ( 2.0 * j ) / ( j - 1 ) * eval_points[ i ]
            * all_values[ ( j - 1 ) * sz + i ];
          all_values[ j * sz + i ]
            -= ( j / ( j - 2.0 ) ) * all_values[ ( j - 2 ) * sz + i ];
        }
      }
    }
  }

 private:
  lo _order;  //!< Highest order of evaluated Chebyshev polynomials, i.e. all
              //!< polynomials with orders in { 0,..., @p _order } are
              //!< evaluated.
};

#endif /* INCLUDE_BESTHEA_CHEBYSHEV_EVALUATOR_H_ */
