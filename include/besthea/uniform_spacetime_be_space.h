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

/** @file uniform_spacetime_be_space.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_

#include "besthea/block_vector.h"
#include "besthea/settings.h"
#include "besthea/spacetime_be_space.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"

namespace besthea {
  namespace bem {
    template< class basis_type >
    class uniform_spacetime_be_space;
    template< class kernel_type, class space_type >
    class uniform_spacetime_be_evaluator;
  }
}

/**
 *  Class representing a boundary element space.
 */
template< class basis_type >
class besthea::bem::uniform_spacetime_be_space
  : public besthea::bem::spacetime_be_space< basis_type,
      besthea::linear_algebra::block_vector > {
  template< class, class >
  friend class besthea::bem::uniform_spacetime_be_evaluator;

  using st_mesh_type
    = besthea::mesh::uniform_spacetime_tensor_mesh;  //!< Spacetime mesh type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.

 public:
  uniform_spacetime_be_space( const uniform_spacetime_be_space & that )
    = delete;

  /**
   * Destructor.
   */
  virtual ~uniform_spacetime_be_space( );

  /**
   * Constructing mesh from a file.
   * @param[in] spacetime_mesh Reference to uniform_spacetime_tensor_mesh.h.
   */
  uniform_spacetime_be_space( st_mesh_type & spacetime_mesh );

  /**
   * Returns pointer to the mesh.
   */
  st_mesh_type * get_mesh( ) {
    return _spacetime_mesh;
  }

  /**
   * Returns pointer to the mesh.
   */
  const st_mesh_type * get_mesh( ) const {
    return _spacetime_mesh;
  }

  /**
   * Projects a function to the boundary element space.
   * @param[in] f Function to be projected.
   * @param[out] projection Projection vector.
   * @param[in] order_matrix Spatial quadrature order to assemble the mass
   * matrix.
   * @param[in] order_rhs_spatial Spatial triangular quadrature order to
   * assemble the right-hand side.
   * @param[in] order_rhs_temporal Temporal line quadrature order to assemble
   * the right-hand side.
   */
  virtual void L2_projection(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    block_vector_type & projection, int order_matrix = 2,
    int order_rhs_spatial = 5, int order_rhs_temporal = 4 ) const override;

  /**
   * Returns the L2 relative error |f-approximation|/|f|.
   * @param[in] f Function in infinite dimensional space.
   * @param[in] approximation Function in finite dimensional space.
   * @param[in] order_rhs_spatial Spatial triangular quadrature order to
   * assemble the right-hand side.
   * @param[in] order_rhs_temporal Temporal line quadrature order to assemble
   * the right-hand side.
   * @warning It is not checked whether the function has L2-norm 0. If this is
   * the case a division by zero occurs.
   */
  virtual sc L2_relative_error(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    const block_vector_type & approximation, int order_rhs_spatial = 5,
    int order_rhs_temporal = 4 ) const override;

  /**
   * Projects a function to the boundary element space. ONLY USE SPECIALIZED
   * FUNCTIONS!
   * @param[in] f Function to be projected.
   * @param[out] interpolation Interpolation vector.
   */
  virtual void interpolation(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    block_vector_type & interpolation ) const override;

 protected:
  st_mesh_type * _spacetime_mesh;  //!< uniform spacetime tensor mesh
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_ */
