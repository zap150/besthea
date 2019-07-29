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

/** @file uniform_spacetime_be_assembler.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ASSEMBLER_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ASSEMBLER_H_

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/block_lower_triangular_toeplitz_matrix.h"
#include "besthea/uniform_spacetime_be_space.h"

#include <array>

namespace besthea {
  namespace bem {
    template< class kernel_type, class test_space_type, class trial_space_type >
    class uniform_spacetime_be_assembler;
  }
}

/**
 *  Class representing a boundary element matrix assembler.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::bem::uniform_spacetime_be_assembler {
 private:
  /**
   * Wraps the mapped quadrature point so that they can be private for OpenMP
   * threads
   */
  struct quadrature_wrapper {
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _x1_ref;  //!< First coordinates of quadrature nodes in (0,1)x(0,1-x1) to
                //!< be mapped to the test element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _x2_ref;  //!< Second coordinates of quadrature nodes in (0,1)x(0,1-x1) to
                //!< be mapped to the test element

    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _y1_ref;  //!< First coordinates of quadrature nodes in (0,1)x(0,1-x1) to
                //!< be mapped to the trial element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _y2_ref;  //!< Second coordinates of quadrature nodes in (0,1)x(0,1-x1) to
                //!< be mapped to the trial element

    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _w;  //!< Quadrature weights including transformation Jacobians

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
    std::vector< sc, besthea::allocator_type< sc > >
      _kernel_values_2;  //!< Buffer for storing additional kernel values.
  };

 public:
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< Full matrix type.

  /**
   * Constructor.
   * @param[in] kernel Spacetime kernel antiderivative object.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   * @param[in] order_singular Line quadrature order for regularized quadrature.
   * @param[in] order_regular Triangle quadrature order for regular quadrature.
   */
  uniform_spacetime_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular = 4, int order_regular = 4 );

  uniform_spacetime_be_assembler( const uniform_spacetime_be_assembler & that )
    = delete;

  /**
   * Destructor.
   */
  ~uniform_spacetime_be_assembler( );

  /**
   * Assembles the spacetime matrix.
   * @param[out] global_matrix Block lower triangular Toeplitz matrix.
   */
  void assemble(
    besthea::linear_algebra::block_lower_triangular_toeplitz_matrix &
      global_matrix ) const;

 private:
  /**
   * Initializes quadrature structures.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( quadrature_wrapper & my_quadrature ) const;

  /**
   * Determines the configuration of two triangular elements.
   * @param[in] i_test Index of the test element.
   * @param[in] i_trial Index of the trial element.
   * @param[out] type_int Type of the configuration (number of vertices shared).
   * @param[out] rot_test Virtual rotation of the test element.
   * @param[out] rot_trial Virtual rotation of the trial element.
   */
  void get_type( lo i_test, lo i_trial, int & type_int, int & rot_test,
    int & rot_trial ) const;

  /**
   * Maps the quadrature nodes from reference triangles to the actual geometry.
   * @param[in] x1 Coordinates of the first node of the test element.
   * @param[in] x2 Coordinates of the second node of the test element.
   * @param[in] x3 Coordinates of the third node of the test element.
   * @param[in] y1 Coordinates of the first node of the trial element.
   * @param[in] y2 Coordinates of the second node of the trial element.
   * @param[in] y3 Coordinates of the third node of the trial element.
   * @param[in] type_int Type of the configuration (number of vertices shared).
   * @param[in] rot_test Virtual rotation of the test element.
   * @param[in] rot_trial Virtual rotation of the trial element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void triangles_to_geometry( const sc * x1, const sc * x2, const sc * x3,
    const sc * y1, const sc * y2, const sc * y3, int type_int, int rot_test,
    int rot_trial, quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps quadratures nodes from hypercube to triangles
   * @param[in] ksi ksi variable in (0,1).
   * @param[in] eta1 eta_1 variable in (0,1).
   * @param[in] eta2 eta_2 variable in (0,1).
   * @param[in] eta3 eta_3 variable in (0,1).
   * @param[in] type Type of configuration (disjoint, shared vertex, shared
   * edge, identical)
   * @param[in] simplex Simplex index.
   * @param[out] x1_ref First coordinate of quadrature node to be mapped to the
   * test element.
   * @param[out] x2_ref Second coordinate of quadrature node to be mapped to the
   * test element.
   * @param[out] y1_ref First coordinate of quadrature node to be mapped to the
   * trial element.
   * @param[out] y2_ref Second coordinate of quadrature node to be mapped to the
   * trial element.
   * @param[out] jacobian Jacobian of the transformation.
   */
  void hypercube_to_triangles( sc ksi, sc eta1, sc eta2, sc eta3,
    int n_shared_vertices, int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref,
    sc & y2_ref, sc & jacobian ) const {
    switch ( n_shared_vertices ) {
      case 1:
        hypercube_to_triangles_vertex( ksi, eta1, eta2, eta3, simplex, x1_ref,
          x2_ref, y1_ref, y2_ref, jacobian );
        break;
      case 2:
        hypercube_to_triangles_edge( ksi, eta1, eta2, eta3, simplex, x1_ref,
          x2_ref, y1_ref, y2_ref, jacobian );
        break;
      case 3:
        hypercube_to_triangles_identical( ksi, eta1, eta2, eta3, simplex,
          x1_ref, x2_ref, y1_ref, y2_ref, jacobian );
        break;
      case 0:
      default:
        return;
    }
  }

  /**
   * Maps quadratures nodes from hypercube to triangles (shared vertex case)
   * @param[in] ksi ksi variable in (0,1).
   * @param[in] eta1 eta_1 variable in (0,1).
   * @param[in] eta2 eta_2 variable in (0,1).
   * @param[in] eta3 eta_3 variable in (0,1).
   * @param[in] simplex Simplex index.
   * @param[out] x1_ref First coordinate of quadrature node to be mapped to the
   * test element.
   * @param[out] x2_ref Second coordinate of quadrature node to be mapped to the
   * test element.
   * @param[out] y1_ref First coordinate of quadrature node to be mapped to the
   * trial element.
   * @param[out] y2_ref Second coordinate of quadrature node to be mapped to the
   * trial element.
   * @param[out] jacobian Jacobian of the transformation.
   */
  void hypercube_to_triangles_vertex( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;

  /**
   * Maps quadratures nodes from hypercube to triangles (shared edge case)
   * @param[in] ksi ksi variable in (0,1).
   * @param[in] eta1 eta_1 variable in (0,1).
   * @param[in] eta2 eta_2 variable in (0,1).
   * @param[in] eta3 eta_3 variable in (0,1).
   * @param[in] simplex Simplex index.
   * @param[out] x1_ref First coordinate of quadrature node to be mapped to the
   * test element.
   * @param[out] x2_ref Second coordinate of quadrature node to be mapped to the
   * test element.
   * @param[out] y1_ref First coordinate of quadrature node to be mapped to the
   * trial element.
   * @param[out] y2_ref Second coordinate of quadrature node to be mapped to the
   * trial element.
   * @param[out]jacobian Jacobian of the transformation.
   */
  void hypercube_to_triangles_edge( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;

  /**
   * Maps quadratures nodes from hypercube to triangles (identical case)
   * @param[in] ksi ksi variable in (0,1).
   * @param[in] eta1 eta_1 variable in (0,1).
   * @param[in] eta2 eta_2 variable in (0,1).
   * @param[in] eta3 eta_3 variable in (0,1).
   * @param[in] simplex Simplex index.
   * @param[out] x1_ref First coordinate of quadrature node to be mapped to the
   * test element.
   * @param[out] x2_ref Second coordinate of quadrature node to be mapped to the
   * test element.
   * @param[out] y1_ref First coordinate of quadrature node to be mapped to the
   * trial element.
   * @param[out] y2_ref Second coordinate of quadrature node to be mapped to the
   * trial element.
   * @param[out] jacobian Jacobian of the transformation.
   */
  void hypercube_to_triangles_identical( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;

  kernel_type * _kernel;  //!< Kernel temporal antiderivative.

  test_space_type * _test_space;  //!< Boundary element test space.

  trial_space_type * _trial_space;  //!< Boundary element trial space.

  int _order_singular;  //!< Line quadrature order for the singular integrals.

  int _order_regular;  //!< Triangle quadrature order for the regular integrals.

  static const int data_align{
    DATA_ALIGN
  };  //!< Intel cannot work with DATA_ALIGN directly

  static constexpr std::array< int, 5 > map{ 0, 1, 2, 0,
    1 };  //!< Auxiliary array for mapping DOFs under
          // rotation (regularized quadrature). Performs fast modulo 3.

  static constexpr std::array< int, 4 > n_simplices{ 1, 2, 5,
    6 };  //!< Number of simplices for all configurations (disjoint, shared
          // vertex, shared edge, identical).
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ASSEMBLER_H_ */
