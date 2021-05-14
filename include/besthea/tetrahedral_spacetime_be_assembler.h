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

/** @file tetrahedral_spacetime_be_assembler.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_BE_ASSEMBLER_H_
#define INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_BE_ASSEMBLER_H_

#include "besthea/full_matrix.h"
#include "besthea/tetrahedral_spacetime_be_space.h"

#include <array>

namespace besthea {
  namespace bem {
    template< class kernel_type, class test_space_type, class trial_space_type >
    class tetrahedral_spacetime_be_assembler;
  }
}

/**
 *  Class representing a boundary element matrix assembler.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::bem::tetrahedral_spacetime_be_assembler {
 private:
  /**
   * Wraps the mapped quadrature point so that they can be private for OpenMP
   * threads
   */
  struct quadrature_wrapper {
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _x1_ref;  //!< First coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the test element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _x2_ref;  //!< Second coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the test element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _x3_ref;  //!< Third coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the test element

    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _y1_ref;  //!< First coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the trial
                //!< element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _y2_ref;  //!< Second coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the trial
                //!< element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _y3_ref;  //!< Third coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the trial
                //!< element

    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _w;  //!< Quadrature weights including transformation Jacobians

    std::vector< sc, besthea::allocator_type< sc > >
      _x1;  //!< First coordinates of quadrature nodes in the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2;  //!< Second coordinates of quadrature nodes in the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _x3;  //!< Third coordinates of quadrature nodes in the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _t;  //!< Temporal coordinates of quadrature nodes in the test element

    std::vector< sc, besthea::allocator_type< sc > >
      _y1;  //!< First coordinates of quadrature nodes in the trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y2;  //!< Second coordinates of quadrature nodes in the trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y3;  //!< Third coordinates of quadrature nodes in the trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _tau;  //!< Temporal coordinates of quadrature nodes in the test element

    std::vector< sc, besthea::allocator_type< sc > >
      _kernel_values;  //!< Buffer for storing kernel values.
  };

 public:
  /**
   * Constructor.
   * @param[in] kernel Spacetime kernel antiderivative object.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   * @param[in] order_singular Line quadrature order for regularized quadrature.
   * @param[in] order_regular Triangle quadrature order for regular quadrature.
   */
  tetrahedral_spacetime_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular = 4, int order_regular = 4 );

  tetrahedral_spacetime_be_assembler(
    const tetrahedral_spacetime_be_assembler & that )
    = delete;

  /**
   * Destructor.
   */
  ~tetrahedral_spacetime_be_assembler( );

  /**
   * Assembles the spacetime matrix.
   * @param[out] global_matrix Full matrix.
   */
  void assemble( besthea::linear_algebra::full_matrix & global_matrix ) const;

 private:
  /**
   * Initializes quadrature structures.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( quadrature_wrapper & my_quadrature ) const;

  /**
   * Determines the configuration of two tetrahedral elements.
   * @param[in] i_test Index of the test element.
   * @param[in] i_trial Index of the trial element.
   * @param[out] type_int Type of the configuration (number of vertices shared).
   * @param[out] perm_test Permutation of the test element.
   * @param[out] perm_trial Permutation of the trial element.
   */
  void get_type( lo i_test, lo i_trial, int & type_int,
    besthea::linear_algebra::indices< 4 > & perm_test,
    besthea::linear_algebra::indices< 4 > & perm_trial ) const;

  /**
   * Maps the quadrature nodes from reference triangles to the actual geometry.
   * @param[in] x1 Coordinates of the first node of the test element.
   * @param[in] x2 Coordinates of the second node of the test element.
   * @param[in] x3 Coordinates of the third node of the test element.
   * @param[in] x4 Coordinates of the fourth node of the test element.
   * @param[in] y1 Coordinates of the first node of the trial element.
   * @param[in] y2 Coordinates of the second node of the trial element.
   * @param[in] y3 Coordinates of the third node of the trial element.
   * @param[in] y4 Coordinates of the fourth node of the trial element.
   * @param[in] type_int Type of the configuration (number of vertices shared).
   * @param[in] perm_test Permutation of the test element.
   * @param[in] perm_trial Permutation of the trial element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void tetrahedra_to_geometry( const linear_algebra::coordinates< 4 > & x1,
    const linear_algebra::coordinates< 4 > & x2,
    const linear_algebra::coordinates< 4 > & x3,
    const linear_algebra::coordinates< 4 > & x4,
    const linear_algebra::coordinates< 4 > & y1,
    const linear_algebra::coordinates< 4 > & y2,
    const linear_algebra::coordinates< 4 > & y3,
    const linear_algebra::coordinates< 4 > & y4, int type_int,
    const besthea::linear_algebra::indices< 4 > & perm_test,
    const besthea::linear_algebra::indices< 4 > & perm_trial,
    quadrature_wrapper & my_quadrature ) const;

  kernel_type * _kernel;  //!< Kernel temporal antiderivative.

  test_space_type * _test_space;  //!< Boundary element test space.

  trial_space_type * _trial_space;  //!< Boundary element trial space.

  int _order_singular;  //!< Line quadrature order for the singular integrals.

  int _order_regular;  //!< Tetrahedron quadrature order for the regular
                       //!< integrals.
};

#endif /* INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_BE_ASSEMBLER_H_ */
