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

/** @file fe_space.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_FE_SPACE_H_
#define INCLUDE_BESTHEA_FE_SPACE_H_

#include "besthea/coordinates.h"
#include "besthea/settings.h"
#include "besthea/tetrahedral_volume_mesh.h"
#include "besthea/vector.h"

#include <vector>

namespace besthea {
  namespace bem {
    template< class basis_type >
    class fe_space;
  }
}

/**
 * Superclass of all boundary element spaces.
 */
template< class basis_type >
class besthea::bem::fe_space {
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using mesh_type
    = besthea::mesh::tetrahedral_volume_mesh;  //!< Space mesh type.

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
      _x3_ref;  //!< First coordinates of quadrature nodes in the reference
                //!< spatial element

    std::vector< sc, besthea::allocator_type< sc > >
      _x1;  //!< First coordinates of quadrature nodes in the spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2;  //!< Second coordinates of quadrature nodes in the spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x3;  //!< Third coordinates of quadrature nodes in the spatial element
  };

 public:
  /**
   * Constructor
   */
  fe_space( const mesh_type & mesh );

  /**
   * Destructor.
   */
  ~fe_space( );

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
   * Returns reference of the mesh.
   */
  const mesh_type & get_mesh( ) const {
    return _mesh;
  }

  /**
   * Projects a function to the boundary element space.
   * @param[in] f Function to be projected.
   * @param[out] projection Projection vector.
   * @param[in] order_matrix Quadrature order to assemble the mass
   * matrix.
   * @param[in] order_rhs Triangular quadrature order to
   * assemble the right-hand side.
   */
  void L2_projection( sc ( *f )( sc, sc, sc ), vector_type & projection,
    int order_matrix = 2, int order_rhs = 4 ) const;

  /**
   * Returns the L2 relative error |f-approximation|/|f|.
   * @param[in] f Function in infinite dimensional space.
   * @param[in] approximation Function in finite dimensional space.
   * @param[in] order_rhs Triangular quadrature order to
   * assemble the right-hand side.
   */
  sc L2_relative_error( sc ( *f )( sc, sc, sc ),
    const vector_type & approximation, int order_rhs = 4 ) const;

  /**
   * Projects a function to the boundary element space. ONLY USE SPECIALIZED
   * FUNCTIONS!
   * @param[in] f Function to be projected.
   * @param[out] interpolation Interpolation vector.
   */
  void interpolation(
    sc ( *f )( sc, sc, sc ), vector_type & interpolation ) const;

  /**
   * Returns the l2 relative error |f-approximation|/|f|.
   * @param[in] f Function in finite dimensional space.
   * @param[out] approximation Function in finite dimensional space.
   */
  sc l2_relative_error(
    const vector_type & f, const vector_type & approximation ) const;

 protected:
  /**
   * Initializes quadrature structures.
   * @param[in] order_rhs Triangle quadrature order for RHS.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature(
    int order_rhs, quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps the quadrature nodes from the reference tetrahedron to the actual
   * geometry.
   * @param[in] x1 Coordinates of the first node of the element.
   * @param[in] x2 Coordinates of the second node of the element.
   * @param[in] x3 Coordinates of the third node of the element.
   * @param[in] x4 Coordinates of the fourth node of the element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void tetrahedron_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & x4,
    quadrature_wrapper & my_quadrature ) const;

  basis_type _basis;        //!< basis function
  const mesh_type & _mesh;  //!< tetrahedral mesh
};

#endif /* INCLUDE_BESTHEA_FE_SPACE_H_ */
