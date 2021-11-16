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

/** @file basis_function.h
 * @brief Contains a parent class for all spatial finite element basis
 * functions defined on surfaces.
 * @note updated documentation
 */

#ifndef INCLUDE_BESTHEA_BASIS_FUNCTION_H_
#define INCLUDE_BESTHEA_BASIS_FUNCTION_H_

#include "besthea/coordinates.h"
#include "besthea/mesh.h"
#include "besthea/settings.h"

#include <array>
#include <vector>

namespace besthea {
  namespace bem {
    template< class derived_type >
    class basis_function;
  }
}

/**
 *  Class representing a basis function.
 */
template< class derived_type >
class besthea::bem::basis_function {
 protected:
  using mesh_type = besthea::mesh::mesh;  //!< Mesh type.

 public:
  /**
   * Default constructor.
   */
  basis_function( ) : _mesh( nullptr ) {
  }

  basis_function( const basis_function & that ) = delete;

  /**
   * Destructor.
   */
  virtual ~basis_function( ) {
  }

  /**
   * Returns this cast to the descendant's type.
   */
  derived_type * derived( ) {
    return static_cast< derived_type * >( this );
  }

  /**
   * Returns this cast to the descendant's type.
   */
  const derived_type * derived( ) const {
    return static_cast< const derived_type * >( this );
  }

  /**
   * Returns the number of basis functions supported on a single element.
   */
  virtual lo dimension_local( ) const = 0;

  /**
   * Returns the number of basis functions on the whole mesh.
   */
  virtual lo dimension_global( ) const = 0;

  /**
   * Provides global indices for local contributions.
   * @param[in] i_elem Element index.
   * @param[out] indices Global indices for local contributions.
   * @remark Example: In case of p1 basis functions the global indices of the
   * nodes of the element with index \p i_elem are returned.
   */
  void local_to_global( lo i_elem, std::vector< lo > & indices ) const {
    derived( )->do_local_to_global( i_elem, indices );
  }

  /**
   * Provides global indices for local contributions.
   * @param[in] i_elem Element index.
   * @param[in] n_shared_vertices Number of shared vertices in current elements
   * (regularized quadrature).
   * @param[in] rotation Virtual element rotation (regularized quadrature).
   * @param[in] swap Virtual element inversion (regularized quadrature).
   * @param[out] indices Global indices for local contributions.
   * @note This is an auxiliary routine to return the global indices in a
   * modified order suitable for regularized quadrature.
   */
  void local_to_global( lo i_elem, int n_shared_vertices, int rotation,
    bool swap, std::vector< lo > & indices ) const {
    derived( )->do_local_to_global(
      i_elem, n_shared_vertices, rotation, swap, indices );
  }

  /**
   * Evaluates a basis function in a point in an element. The point is given by
   * coordinates in the reference triangle
   * \f$ (x_1,x_2) \in (0,1)\times(0,1-x_1) \f$.
   * @param[in] i_elem Element index.
   * @param[in] i_fun Local basis function index.
   * @param[in] x1_ref First coordinate of reference quadrature point.
   * @param[in] x2_ref Second coordinate of reference quadrature point.
   * @param[in] n Outward normal vector on the element
   */
#pragma omp declare simd uniform( i_elem, i_fun, n ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc evaluate( lo i_elem, lo i_fun, sc x1_ref, sc x2_ref, const sc * n ) const {
    return derived( )->do_evaluate( i_elem, i_fun, x1_ref, x2_ref, n );
  }

  /**
   * Evaluates a basis function in a point in an element. The point is given
   * by coordinates in the reference triangle
   * \f$ (x_1,x_2) \in (0,1)\times(0,1-x_1) \f$.
   * @param[in] i_elem Element index.
   * @param[in] i_fun Local basis function index.
   * @param[in] x1_ref First coordinate of reference quadrature point.
   * @param[in] x2_ref Second coordinate of reference quadrature point.
   * @param[in] n Outward normal vector on the element.
   * @param[in] n_shared_vertices Number of shared vertices in current elements
   * (regularized quadrature).
   * @param[in] rotation Virtual element rotation (regularized quadrature).
   * @param[in] swap Virtual element inversion (regularized quadrature).
   * @note This is an auxiliary routine to evaluate a basis function suitable
   * for regularized quadrature.
   */
#pragma omp declare simd uniform( i_elem, i_fun, n, n_shared_vertices, \
  rotation, swap ) simdlen( BESTHEA_SIMD_WIDTH )
  sc evaluate( lo i_elem, lo i_fun, sc x1_ref, sc x2_ref, const sc * n,
    int n_shared_vertices, int rotation, bool swap ) const {
    return derived( )->do_evaluate(
      i_elem, i_fun, x1_ref, x2_ref, n, n_shared_vertices, rotation, swap );
  }

 protected:
  const mesh_type * _mesh;  //!< Pointer to the underlying mesh.

  const std::array< int, 5 > _map{ 0, 1, 2, 0,
    1 };  //!< Auxiliary array for mapping DOFs under
          // rotation (regularized quadrature).
};

#endif /* INCLUDE_BESTHEA_BASIS_FUNCTION_H_ */
