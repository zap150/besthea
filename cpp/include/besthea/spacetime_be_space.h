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

/** @file spacetime_be_space.h
 * @brief File contains root class of boundary element spaces.
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_BE_SPACE_H_
#define INCLUDE_BESTHEA_SPACETIME_BE_SPACE_H_

#include "besthea/block_vector.h"
#include "besthea/coordinates.h"
#include "besthea/mesh.h"
#include "besthea/settings.h"

#include <vector>

namespace besthea {
  namespace bem {
    template< class basis_type >
    class spacetime_be_space;
  }
}

/**
 * Superclass of all boundary element spaces.
 */
template< class basis_type >
class besthea::bem::spacetime_be_space {
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.
  using mesh_type = besthea::mesh::mesh;      //!< Spacetime mesh type.

 protected:
  /**
   * Wraps the mapped quadrature point so that they can be private for OpenMP
   * threads
   */
  struct quadrature_wrapper {
    std::vector< sc, besthea::allocator_type< sc > >
      _wx;  //!< Spatial quadrature weights
    std::vector< sc, besthea::allocator_type< sc > >
      _x1_ref;  //!< First coordinates of quadrature nodes in the reference
                //!< spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2_ref;  //!< Second coordinates of quadrature nodes in the reference
                //!< spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x1;  //!< First coordinates of quadrature nodes in the spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2;  //!< Second coordinates of quadrature nodes in the spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x3;  //!< Third coordinates of quadrature nodes in the spatial element

    std::vector< sc, besthea::allocator_type< sc > >
      _wt;  //!< Temporal quadrature weights
    std::vector< sc, besthea::allocator_type< sc > >
      _t_ref;  //!< Coordinates of quadrature nodes in the reference temporal
               //!< element
    std::vector< sc, besthea::allocator_type< sc > >
      _t;  //!< Coordinates of quadrature nodes in the temporal element
  };

 public:
  spacetime_be_space( const mesh_type & mesh );

  /**
   * Destructor.
   */
  virtual ~spacetime_be_space( );

  /**
   * Returns reference to the basis function.
   */
  basis_type & get_basis( ) {
    return _basis;
  }

  /**
   * Returns pointer to the basis function.
   */
  const basis_type & get_basis( ) const {
    return _basis;
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
    int order_rhs_spatial = 5, int order_rhs_temporal = 4 ) const = 0;

  /**
   * Returns the L2 relative error |f-approximation|/|f|.
   * @param[in] f Function in infinite dimensional space.
   * @param[in] approximation Function in finite dimensional space.
   * @param[in] order_rhs_spatial Spatial triangular quadrature order to
   * assemble the right-hand side.
   * @param[in] order_rhs_temporal Temporal line quadrature order to assemble
   * the right-hand side.
   */
  virtual sc L2_relative_error(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    const block_vector_type & approximation, int order_rhs_spatial = 5,
    int order_rhs_temporal = 4 ) const = 0;

  /**
   * Projects a function to the boundary element space. ONLY USE SPECIALIZED
   * FUNCTIONS!
   * @param[in] f Function to be projected.
   * @param[out] interpolation Interpolation vector.
   */
  virtual void interpolation(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    block_vector_type & interpolation ) const;

  /**
   * Returns the l2 relative error |f-approximation|/|f|.
   * @param[in] f Function in finite dimensional space.
   * @param[out] approximation Function in finite dimensional space.
   */
  sc l2_relative_error( const block_vector_type & f,
    const block_vector_type & approximation ) const;

 protected:
  /**
   * Initializes quadrature structures.
   * @param[in] order_rhs_spatial Triangle spatial quadrature order for RHS.
   * @param[in] order_rhs_temporal Line temporal quadrature order for RHS.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( int order_rhs_spatial, int order_rhs_temporal,
    quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps the quadrature nodes from the reference triangle to the actual
   * geometry.
   * @param[in] x1 Coordinates of the first node of the test element.
   * @param[in] x2 Coordinates of the second node of the test element.
   * @param[in] x3 Coordinates of the third node of the test element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void triangle_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps the quadrature nodes from reference interval to the actual time.
   * @param[in] d Index of time interval.
   * @param[in] timestep Timestep size.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void line_to_time(
    lo d, sc timestep, quadrature_wrapper & my_quadrature ) const;

  basis_type _basis;  //!< spatial basis function (temporal is constant)
};

#endif /* INCLUDE_BESTHEA_SPACETIME_BE_SPACE_H_ */
