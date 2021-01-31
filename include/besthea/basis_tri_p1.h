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

/** @file basis_tri_p1.h
 * @brief Contains a class representing p1 (piecewise linear) basis functions on
 * a triangular surface mesh.
 * @note updated documentation
 */

#ifndef INCLUDE_BESTHEA_BASIS_TRI_P1_H_
#define INCLUDE_BESTHEA_BASIS_TRI_P1_H_

#include "besthea/basis_function.h"
#include "besthea/triangular_surface_mesh.h"

namespace besthea {
  namespace bem {
    class basis_tri_p1;
  }
}

/**
 *  Class representing a piecewise linear function on a triangular mesh.
 */
class besthea::bem::basis_tri_p1
  : public besthea::bem::basis_function< besthea::bem::basis_tri_p1 > {
 public:
  /**
   * Constructor.
   * @param[in] mesh Triangular surface mesh on which the basis functions are
   *                 defined.
   */
  basis_tri_p1( const mesh_type & mesh );

  /**
   * Destructor.
   */
  virtual ~basis_tri_p1( );

  /**
   * Returns the number of basis functions supported on a single element.
   *
   * This is always 3.
   */
  virtual lo dimension_local( ) const;

  /**
   * Returns the number of basis functions on the whole mesh.
   *
   * This is the number of all nodes in the underlying triangular surface mesh.
   */
  virtual lo dimension_global( ) const;

  /**
   * Returns the global indices of the nodes of the given element.
   * @param[in] i_elem Element index.
   * @param[out] indices Global node indices of the element.
   */
  void do_local_to_global( lo i_elem, std::vector< lo > & indices ) const;

  /**
   * Returns the global indices of the nodes of the given element. Their order
   * is modified according to the given parameters (regularized quadrature).
   * @param[in] i_elem Element index.
   * @param[in] n_shared_vertices Number of shared vertices in currect elements
   * (regularized quadrature).
   * @param[in] rotation Virtual element rotation (regularized quadrature).
   * @param[in] swap Virtual element inversion (regularized quadrature).
   * @param[out] indices Global node indices of the element in modified order.
   */
  void do_local_to_global( lo i_elem, int n_shared_vertices, int rotation,
    bool swap, std::vector< lo > & indices ) const;

  /**
   * Evaluates a basis function in a point in an element. The point is given by
   * coordinates in the reference triangle
   * (\f$ (x_1,x_2) \in (0,1)\times(0,1-x_1) \f$).
   * @param[in] i_elem Element index.
   * @param[in] i_fun Local basis function index.
   * @param[in] x1_ref First coordinate of reference quadrature point.
   * @param[in] x2_ref Second coordinate of reference quadrature point.
   * @param[in] n Outward normal vector on the element.
   * \note By the nature of the basis functions, the result does not depend on
   * the choice of the element, and in particular not on the outward normal
   * vector.
   */
#pragma omp declare simd uniform( this, i_elem, i_fun, n ) simdlen( DATA_WIDTH )
  sc do_evaluate( [[maybe_unused]] lo i_elem, lo i_fun, sc x1_ref, sc x2_ref,
    [[maybe_unused]] const sc * n ) const {
    sc value = 0.0;

    if ( i_fun == 0 ) {
      value = 1 - x1_ref - x2_ref;
    } else if ( i_fun == 1 ) {
      value = x1_ref;
    } else if ( i_fun == 2 ) {
      value = x2_ref;
    }

    return value;
  }

  /**
   * Evaluates a basis function in a point in an element. The point is given by
   * coordinates in the reference triangle
   * (\f$ (x_1,x_2) \in (0,1)\times(0,1-x_1) \f$).
   * @param[in] i_elem Element index.
   * @param[in] i_fun Local basis function index.
   * @param[in] x1_ref First coordinate of reference quadrature point.
   * @param[in] x2_ref Second coordinate of reference quadrature point.
   * @param[in] n Outward normal vector on the element
   * @param[in] n_shared_vertices Number of shared vertices in currect elements
   * (regularized quadrature).
   * @param[in] rotation Virtual element rotation (regularized quadrature).
   * @param[in] swap Virtual element inversion (regularized quadrature).
   * \note By the nature of the basis functions, the result does not depend on
   * the choice of the element, and in particular not on the outward normal
   * vector.
   * \note The regularized quadrature parameters do not influence the result
   * either.
   */
#pragma omp declare simd uniform( this, i_elem, i_fun, n, n_shared_vertices, \
  rotation, swap ) simdlen( DATA_WIDTH )
  sc do_evaluate( [[maybe_unused]] lo i_elem, lo i_fun, sc x1_ref, sc x2_ref,
    [[maybe_unused]] const sc * n, [[maybe_unused]] int n_shared_vertices,
    [[maybe_unused]] int rotation, [[maybe_unused]] bool swap ) const {
    sc value = 0.0;

    if ( i_fun == 0 ) {
      value = 1 - x1_ref - x2_ref;
    } else if ( i_fun == 1 ) {
      value = x1_ref;
    } else if ( i_fun == 2 ) {
      value = x2_ref;
    }

    return value;
  }

  /**
   * Evaluates the surface curl of all basis functions in a given element.
   * @param[in] i_elem Element index.
   * @param[in] n Outward normal vector on the element
   * @param[in] n_shared_vertices Number of shared vertices in currect elements
   * (regularized quadrature).
   * @param[in] rotation Virtual element rotation (regularized quadrature).
   * @param[in] swap Virtual element inversion (regularized quadrature).
   * @param[out] curls Surface curls of all three shape functions.
   */
  void evaluate_curl( lo i_elem, const linear_algebra::coordinates< 3 > & n,
    int n_shared_vertices, int rotation, bool swap, sc * curls ) const;
};

#endif /* INCLUDE_BESTHEA_BASIS_TRI_P1_H_ */
