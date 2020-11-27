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
#include "besthea/chebyshev_evaluator.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main( int, char ** ) {
  using vector_type = besthea::linear_algebra::vector;
  using lagrange_interpolant = besthea::bem::lagrange_interpolant;
  using chebyshev_evaluator = besthea::bem::chebyshev_evaluator;
  // test of lagrange_interpolant class;
  // set order of lagrange polynomials, number of evaluation points and index of
  // polynomial which is evaluated
  lo order = 5;
  lo n_eval_points = 10;
  lo poly_index = 0;
  // declare evaluation points and lagrange_values
  vector_type eval_points( n_eval_points ), lagrange_values( n_eval_points );
  // initialize evaluation points uniformly in [-1, 1]
  for ( lo i = 0; i < n_eval_points; ++i ) {
    eval_points[ i ] = -1.0 + 2.0 * i / ( n_eval_points - 1.0 );
  }
  // evaluate lagrange polynomial
  lagrange_interpolant lagrange( order );
  lagrange.evaluate( poly_index, eval_points, lagrange_values );
  std::cout << "values of lagrange polynomial are: " << std::endl;
  for ( lo i = 0; i < n_eval_points; ++i )
    std::cout << lagrange_values[ i ] << " ";
  std::cout << std::endl;
  // evaluate chebyshev polynomial
  vector_type all_cheb_values( n_eval_points * ( order + 1 ) );
  chebyshev_evaluator chebyshev( order );
  chebyshev.evaluate( eval_points, all_cheb_values );
  std::cout << "values of chebyshev polynomials (order <= " << order;
  std::cout << " are: " << std::endl;
  for ( lo i = 0; i <= order; ++i ) {
    for ( lo j = 0; j < n_eval_points; ++j )
      printf( "%.4f ", all_cheb_values[ n_eval_points * i + j ] );
    std::cout << std::endl;
  }
  chebyshev.evaluate_derivative( eval_points, all_cheb_values );
  std::cout << "values of derived chebyshev polynomials (order <= " << order;
  std::cout << " are: " << std::endl;
  for ( lo i = 0; i <= order; ++i ) {
    for ( lo j = 0; j < n_eval_points; ++j )
      printf( "%.4f ", all_cheb_values[ n_eval_points * i + j ] );
    std::cout << std::endl;
  }
}
