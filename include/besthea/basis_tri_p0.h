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

/** @file basis_tri_p0.h
 * @brief Contains a class representing p0 (piecewise constant) basis functions
 * on a triangular surface mesh.
 * @note updated documentation
 */

#ifndef INCLUDE_BESTHEA_BASIS_TRI_P0_H_
#define INCLUDE_BESTHEA_BASIS_TRI_P0_H_

#include "besthea/basis_function.h"
#include "besthea/triangular_surface_mesh.h"

namespace besthea {
  namespace bem {
    class basis_tri_p0;
  }
}

/**
 *  Class representing constant functions on a triangular mesh.
 */
class besthea::bem::basis_tri_p0
  : public besthea::bem::basis_function< besthea::bem::basis_tri_p0 > {
 public:
  /**
   * Constructor.
   * @param[in] mesh Triangular surface mesh on which the basis functions are
   *                 defined.
   */
  basis_tri_p0( const mesh_type & mesh );

  /**
   * Destructor.
   */
  virtual ~basis_tri_p0( );

  /**
   * Returns the number of basis functions supported on a single element.
   *
   * This is always 1.
   */
  virtual lo dimension_local( ) const;

  /**
   * Returns the number of basis functions on the whole mesh.
   *
   * This is the number of all elements in the underlying triangular surface
   * mesh.
   */
  virtual lo dimension_global( ) const;

  /**
   * Returns the global index of the element itself.
   * @param[in] i_elem Element index.
   * @param[out] indices Element index at position 0 of the vector.
   */
  void do_local_to_global( lo i_elem, std::vector< lo > & indices ) const;

  /**
   * Returns the index of the element itself.
   * @param[in] i_elem Element index.
   * @param[in] n_shared_vertices Number of shared vertices in currect elements
   *                              (regularized quadrature). This parameter is
   *                              ignored.
   * @param[in] rotation Virtual element rotation (regularized quadrature). This
   *                     parameter is ignored.
   * @param[in] swap Virtual element inversion (regularized quadrature). This
   *                 parameter is ignored.
   * @param[out] indices Element index at position 0 of the vector.
   */
  void do_local_to_global( lo i_elem, [[maybe_unused]] int n_shared_vertices,
    [[maybe_unused]] int rotation, [[maybe_unused]] bool swap,
    std::vector< lo > & indices ) const;

  /**
   * Evaluates the basis function.
   * @param[in] i_elem Element index.
   * @param[in] i_fun Local basis function index.
   * @param[in] x1_ref First coordinate of reference quadrature point.
   * @param[in] x2_ref Second coordinate of reference quadrature point.
   * @param[in] n Outward normal vector on the element
   * \note By the nature of the basis functions, the result is always 1
   * independent of the provided parameters. In particular, all parameters are
   * ignored.
   */
#pragma omp declare simd uniform( this, i_elem, i_fun, n ) simdlen( DATA_WIDTH )
  sc do_evaluate( [[maybe_unused]] lo i_elem, [[maybe_unused]] lo i_fun,
    [[maybe_unused]] sc x1_ref, [[maybe_unused]] sc x2_ref,
    [[maybe_unused]] const sc * n ) const {
    return 1.0;
  }

  /**
   * Evaluates the basis function.
   * @param[in] i_elem Element index.
   * @param[in] i_fun Local basis function index.
   * @param[in] x1_ref First coordinate of reference quadrature point.
   * @param[in] x2_ref Second coordinate of reference quadrature point.
   * @param[in] n Outward normal vector on the element
   * @param[in] n_shared_vertices Number of shared vertices in currect elements
   * (regularized quadrature).
   * @param[in] rotation Virtual element rotation (regularized quadrature).
   * @param[in] swap Virtual element inversion (regularized quadrature).
   * \note By the nature of the basis functions, the result is always 1
   * independent of the provided parameters. In particular, all parameters are
   * ignored.
   */
#pragma omp declare simd uniform( this, i_elem, i_fun, n, n_shared_vertices, \
  rotation, swap ) simdlen( DATA_WIDTH )
  sc do_evaluate( [[maybe_unused]] lo i_elem, [[maybe_unused]] lo i_fun,
    [[maybe_unused]] sc x1_ref, [[maybe_unused]] sc x2_ref,
    [[maybe_unused]] const sc * n, [[maybe_unused]] int n_shared_vertices,
    [[maybe_unused]] int rotation, [[maybe_unused]] bool swap ) const {
    return 1.0;
  }
};

#endif /* INCLUDE_BESTHEA_BASIS_TRI_P0_H_ */
