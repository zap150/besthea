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

/** @file uniform_spacetime_be_space.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_

#include "besthea/block_vector.h"
#include "besthea/settings.h"
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
class besthea::bem::uniform_spacetime_be_space {
  template< class, class >
  friend class besthea::bem::uniform_spacetime_be_evaluator;

 private:
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
  using st_mesh_type
    = besthea::mesh::uniform_spacetime_tensor_mesh;  //!< Spacetime mesh type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.

  uniform_spacetime_be_space( const uniform_spacetime_be_space & that )
    = delete;

  /**
   * Destructor.
   */
  ~uniform_spacetime_be_space( );

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
  void l2_projection( sc ( *f )( sc, sc, sc, sc *, sc ),
    block_vector_type & projection, int order_matrix = 2,
    int order_rhs_spatial = 5, int order_rhs_temporal = 4 );

  /**
   * Returns the .
   * @param[in] f Function in infinite dimensional space.
   * @param[out] approximation Function in finite dimensional space.
   * @param[in] order_rhs_spatial Spatial triangular quadrature order to
   * assemble the right-hand side.
   * @param[in] order_rhs_temporal Temporal line quadrature order to assemble
   * the right-hand side.
   */
  sc l2_relative_error( sc ( *f )( sc, sc, sc, sc *, sc ),
    block_vector_type & approximation, int order_rhs_spatial = 5,
    int order_rhs_temporal = 4 );

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
  void triangle_to_geometry( const sc * x1, const sc * x2, const sc * x3,
    quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps the quadrature nodes from reference interval to the actual time.
   * @param[in] d Index of time interval.
   * @param[in] timestep Timestep size.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void line_to_time(
    lo d, sc timestep, quadrature_wrapper & my_quadrature ) const;

  st_mesh_type * _spacetime_mesh;  //!< uniform spacetime tensor mesh

  basis_type _basis;  //!< spatial basis function (temporal is constant)
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_ */
