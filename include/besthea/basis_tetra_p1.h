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

/** @file basis_tetra_p1.h
 * @brief Contains a class representing piecewise linear basis functions on a
 * tetrahedral (spatial) volume mesh.
 * @note updated documentation
 */

#ifndef INCLUDE_BESTHEA_BASIS_TETRA_P1_H_
#define INCLUDE_BESTHEA_BASIS_TETRA_P1_H_

#include "besthea/tetrahedral_volume_mesh.h"
#include "besthea/volume_basis_function.h"

namespace besthea {
  namespace bem {
    class basis_tetra_p1;
  }
}

/**
 *  Class representing a piecewise linear function on a tetrahedral mesh.
 */
class besthea::bem::basis_tetra_p1
  : public besthea::bem::volume_basis_function< besthea::bem::basis_tetra_p1 > {
 public:
  /**
   * Constructor.
   * @param[in] mesh Tetrahedral volume mesh on which the basis functions are
   *                 defined.
   */
  basis_tetra_p1( const mesh_type & mesh );

  /**
   * Destructor.
   */
  virtual ~basis_tetra_p1( );

  /**
   * Returns the number of basis functions supported on a single tetrahedron.
   *
   * This is always 4.
   */
  virtual lo dimension_local( ) const;

  /**
   * Returns the number of basis functions on the whole mesh.
   *
   * This is the number of all nodes in the underlying tetrahedral volume mesh.
   */
  virtual lo dimension_global( ) const;

  /**
   * Provides the global indices of the 4 nodes of the tetrahedron with given
   * index.
   * @param[in] i_elem Index of the tetrahedron.
   * @param[out] indices Global indices of the nodes of the tetrahedron.
   */
  void do_local_to_global( lo i_elem, std::vector< lo > & indices ) const;

  /**
   * Evaluates a basis function in a point in a tetrahedron. The point is given
   * by coordinates in the reference tetrahedron
   * \f$ (x_1,x_2,x_3) \in (0,1)\times(0,1-x_1)\times(0,1-x_1-x_2) \f$.
   * @param[in] i_elem Index of the tetrahedron.
   * @param[in] i_fun Local basis function index.
   * @param[in] x1_ref First coordinate of reference quadrature point.
   * @param[in] x2_ref Second coordinate of reference quadrature point.
   * @param[in] x3_ref Third coordinate of reference quadrature point.
   */
#pragma omp declare simd uniform( this, i_elem, i_fun ) \
  simdlen( BESTHEA_SIMD_WIDTH )
  sc do_evaluate( [[maybe_unused]] lo i_elem, lo i_fun, sc x1_ref, sc x2_ref,
    sc x3_ref ) const {
    sc value = 0.0;

    if ( i_fun == 0 ) {
      value = 1 - x1_ref - x2_ref - x3_ref;
    } else if ( i_fun == 1 ) {
      value = x1_ref;
    } else if ( i_fun == 2 ) {
      value = x2_ref;
    } else if ( i_fun == 3 ) {
      value = x3_ref;
    }

    return value;
  }
};

#endif /* INCLUDE_BESTHEA_BASIS_TETRA_P1_H_ */
