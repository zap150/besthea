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

/** @file uniform_spacetime_initial_assembler.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_INITIAL_ASSEMBLER_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_INITIAL_ASSEMBLER_H_

#include "besthea/block_row_matrix.h"
#include "besthea/fe_space.h"
#include "besthea/uniform_spacetime_be_space.h"

namespace besthea {
  namespace bem {
    template< class kernel_type, class test_space_type, class trial_space_type >
    class uniform_spacetime_initial_assembler;
  }
}

/**
 *  Class representing a boundary element matrix assembler.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::bem::uniform_spacetime_initial_assembler {
 private:
  /**
   * Wraps the mapped quadrature point so that they can be private for OpenMP
   * threads
   */
  struct quadrature_wrapper {
    std::vector< sc, besthea::allocator_type< sc > >
      _x1_ref;  //!< First coordinates of quadrature nodes in (0,1)x(0,1-x1) to
                //!< be mapped to the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2_ref;  //!< Second coordinates of quadrature nodes in (0,1)x(0,1-x1) to
                //!< be mapped to the test element

    std::vector< sc, besthea::allocator_type< sc > >
      _y1_ref;  //!< First coordinates of quadrature nodes in
                //!< conv((0,0,0),(1,0,0),(0,1,0),(0,0,1)) to be mapped to the
                //!< trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y2_ref;  //!< Second coordinates of quadrature nodes in
                //!< conv((0,0,0),(1,0,0),(0,1,0),(0,0,1)) to be mapped to the
                //!< trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y3_ref;  //!< Third coordinates of quadrature nodes in
                //!< conv((0,0,0),(1,0,0),(0,1,0),(0,0,1)) to be mapped to the
                //!< trial element

    std::vector< sc, besthea::allocator_type< sc > >
      _w;  //!< Quadrature weights

    std::vector< sc, besthea::allocator_type< sc > >
      _x1;  //!< First coordinates of quadrature nodes in the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2;  //!< Second coordinates of quadrature nodes in the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _x3;  //!< Third coordinates of quadrature nodes in the test element

    std::vector< sc, besthea::allocator_type< sc > >
      _y1;  //!< First coordinates of quadrature nodes in the trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y2;  //!< Second coordinates of quadrature nodes in the trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y3;  //!< Third coordinates of quadrature nodes in the trial element

    std::vector< sc, besthea::allocator_type< sc > >
      _kernel_values;  //!< Buffer for storing kernel values.
  };

 public:
  /**
   * Constructor.
   * @param[in] kernel Spacetime kernel antiderivative object.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   * @param[in] order_regular_tri Triangle quadrature order for regular
   * quadrature.
   * @param[in] order_regular_tetra Tetrahedron quadrature order for regular
   * quadrature.
   */
  uniform_spacetime_initial_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_regular_tri = 5, int order_regular_tetra = 4 );

  uniform_spacetime_initial_assembler(
    const uniform_spacetime_initial_assembler & that )
    = delete;

  /**
   * Destructor.
   */
  ~uniform_spacetime_initial_assembler( );

  /**
   * Assembles the spacetime matrix.
   * @param[out] global_matrix Block row matrix.
   */
  void assemble(
    besthea::linear_algebra::block_row_matrix & global_matrix ) const;

 private:
  /**
   * Initializes quadrature structures.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps the quadrature nodes from reference triangles to the actual geometry.
   * @param[in] x1 Coordinates of the first node of the test element.
   * @param[in] x2 Coordinates of the second node of the test element.
   * @param[in] x3 Coordinates of the third node of the test element.
   * @param[in] y1 Coordinates of the first node of the trial element.
   * @param[in] y2 Coordinates of the second node of the trial element.
   * @param[in] y3 Coordinates of the third node of the trial element.
   * @param[in] y3 Coordinates of the fourth node of the trial element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void triangle_and_tetrahedron_to_geometry(
    const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3,
    const linear_algebra::coordinates< 3 > & y4,
    quadrature_wrapper & my_quadrature ) const;

  kernel_type * _kernel;  //!< Initial kernel temporal antiderivative.

  test_space_type * _test_space;  //!< Boundary element test space.

  trial_space_type * _trial_space;  //!< Finite element trial space.

  int _order_regular_tri;  //!< Triangle quadrature order for the regular
                           //!< integrals.

  int _order_regular_tetra;  //!< Tetrahedron quadrature order for the regular
                             //!< integrals.
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_INITIAL_ASSEMBLER_H_ */
