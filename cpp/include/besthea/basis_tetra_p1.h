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

/** @file basis_tetra_p1.h
 * @brief
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
 *  Class representing a piecewise linear function on a triangular mesh.
 */
class besthea::bem::basis_tetra_p1
  : public besthea::bem::volume_basis_function< besthea::bem::basis_tetra_p1 > {
 public:
  /**
   * Constructor.
   * @param[in] mesh Mesh.
   */
  basis_tetra_p1( const mesh_type & mesh );

  /**
   * Destructor.
   */
  virtual ~basis_tetra_p1( );

  /**
   * Returns number of basis functions supported on i_elem.
   */
  virtual lo dimension_local( ) const;

  /**
   * Returns number of basis functions on the whole mesh.
   */
  virtual lo dimension_global( ) const;

  /**
   * Provides global indices for local contributions.
   * @param[in] i_elem Element index.
   * @param[out] indices Global indices for local contributions.
   */
  void do_local_to_global( lo i_elem, std::vector< lo > & indices ) const;

  /**
   * Evaluates the basis function.
   * @param[in] i_elem Element index.
   * @param[in] i_fun Local basis function index.
   * @param[in] x1_ref First coordinate of reference quadrature point.
   * @param[in] x2_ref Second coordinate of reference quadrature point.
   * @param[in] x3_ref Third coordinate of reference quadrature point.
   */
#pragma omp declare simd uniform( this, i_elem, i_fun ) simdlen( DATA_WIDTH )
  sc do_evaluate( lo i_elem, lo i_fun, sc x1_ref, sc x2_ref, sc x3_ref ) const {
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
