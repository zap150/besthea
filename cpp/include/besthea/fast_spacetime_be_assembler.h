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

/** @file fast_spacetime_be_assembler.h
 * @brief File containing a class for assembling pFMM matrices
 */

#ifndef INCLUDE_BESTHEA_FAST_SPACETIME_BE_ASSEMBLER_H_
#define INCLUDE_BESTHEA_FAST_SPACETIME_BE_ASSEMBLER_H_

#include "besthea/block_matrix.h"
#include "besthea/chebyshev_evaluator.h"
#include "besthea/fast_spacetime_be_space.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/pFMM_matrix.h"
#include "besthea/quadrature.h"
#include "besthea/settings.h"
#include "besthea/space_cluster.h"
#include "besthea/sparse_matrix.h"
#include "besthea/time_cluster.h"
#include "besthea/vector.h"

#include <omp.h>
#include <vector>

namespace besthea {
  namespace bem {
    template< class kernel_type, class test_space_type, class trial_space_type >
    class fast_spacetime_be_assembler;
  }
}

/**
 *  Class representing a boundary element fast matrix assembler.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::bem::fast_spacetime_be_assembler {
 private:
  using sparse_matrix_type = besthea::linear_algebra::sparse_matrix;
  using pfmm_matrix_type
    = besthea::linear_algebra::pFMM_matrix;               //!< pFMM matrix type.
  using time_cluster_type = besthea::mesh::time_cluster;  //!< time cluster type
  using space_cluster_type
    = besthea::mesh::space_cluster;                     //!< space cluster type
  using vector_type = besthea::linear_algebra::vector;  //!< vector type

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
  /**
   * Constructor.
   * @param[in] kernel Spacetime kernel antiderivative object.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   * @param[in] order_singular Line quadrature order for regularized quadrature.
   * @param[in] order_regular Triangle quadrature order for regular quadrature.
   * @param[in] spat_order Degree of Chebyshev polynomials for expansion in pFMM
   * matrix.
   * @param[in] temp_order degree of Lagrange interpolation polynomials in time
   * for pFMM matrix.
   * @param[in] cutoff_param Cutoff parameter for the nearfield approximation
   * (elements further than cutoff_param * diagonal of the lowest level cluster
   * will be ignored).
   * @param[in] uniform Uniform time discretization to save memory.
   */
  fast_spacetime_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular = 4, int order_regular = 4, int spat_order = 5,
    int temp_order = 5, sc cutoff_param = 3.0, bool uniform = false );

  fast_spacetime_be_assembler( const fast_spacetime_be_assembler & that )
    = delete;

  /**
   * Destructor.
   */
  ~fast_spacetime_be_assembler( );

  /**
   * Assembles the fast spacetime matrix.
   * @param[out] global_matrix Assembled pFMM matrix.
   */
  void assemble( besthea::linear_algebra::pFMM_matrix & global_matrix ) const;

 private:
  /**
   * Initializes quadrature structures.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( quadrature_wrapper & my_quadrature ) const;

  /**
   * Assembles temporal nearfield matrices.
   * @param[out] global_matrix Partially assembled pFMM matrix.
   */
  void assemble_nearfield(
    besthea::linear_algebra::pFMM_matrix & global_matrix ) const;

  /** Assembles temporal farfield nonapproximated by the pFMM.
   * @param[out] global_matrix Partially assembled pFMM matrix.
   */
  void assemble_farfield_nonapproximated(
    besthea::linear_algebra::pFMM_matrix & global_matrix ) const;

  /** Assembles temporal farfield nonapproximated by the pFMM.
   * @param[out] global_matrix Partially assembled pFMM matrix.
   */
  void assemble_nonapproximated_uniform(
    besthea::linear_algebra::pFMM_matrix & global_matrix ) const;

  /**
   * Assembles nearfield matrix
   * @param[in] t0 Initial time of the test function quadrature.
   * @param[in] t1 Final time of the test function quadrature.
   * @param[in] tau0 Initial time of the trial function quadrature.
   * @param[in] tau1 Final time of the trial function quadrature.
   * @param[out] nearfield_matrix Reference to the matrix which should be
   * assembled.
   */
  void assemble_nearfield_matrix( sc t0, sc t1, sc tau0, sc tau1,
    sparse_matrix_type & nearfield_matrix ) const;

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
  void triangles_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3, int type_int, int rot_test,
    int rot_trial, quadrature_wrapper & my_quadrature ) const;

  /**
   * Returns true if the two elements are closer than the limit given by the
   * size of the finest level spatial tree cluster and coefficient.
   * @param[in] trial_idx Index of the trial spatial mesh element.
   * @param[in] test_idx Index of the test spatial mesh element.
   */
  bool is_spatial_nearfield( lo test_idx, lo trial_idx ) const;

  /**
   * Precomputes elements contributing to the nonzero pattern of sparse
   * nearfield (and possibly farfield) matrices.
   */
  void precompute_nonzeros( );

  /**
   * Computes quadrature of the Lagrange polynomials up to a given order over an
   * element in the cluster.
   */
  void compute_lagrange_quadrature( time_cluster_type * cluster );

  /**
   * Compute quadrature of the Chebyshev polynomials.
   */
  void compute_chebyshev_quadrature( space_cluster_type * cluster );

  kernel_type * _kernel;            //!< Kernel temporal antiderivative.
  test_space_type * _test_space;    //!< Boundary element test space.
  trial_space_type * _trial_space;  //!< Boundary element trial space.
  int _order_singular;  //!< Line quadrature order for the singular integrals.
  int _order_regular;  //!< Triangle quadrature order for the regular integrals.
  static constexpr std::array< int, 5 > map{ 0, 1, 2, 0,
    1 };  //!< Auxiliary array for mapping DOFs under
          // rotation (regularized quadrature). Performs fast modulo 3.
  static constexpr std::array< int, 4 > n_simplices{ 1, 2, 5,
    6 };  //!< Number of simplices for all configurations (disjoint, shared
          // vertex, shared edge, identical).
  sc _space_cluster_size;  //!< Size of the finest level space clusters.
  sc _cutoff_param;  //!< Coefficient for determining the spatial nearfield
                     //!< (_cutoff_param * diagonal of the lowest level
                     //!< cluster).
  bool _uniform;     //!< uniform assembly
  std::vector< std::pair< lo, lo > >
    _nonzeros;  //!< indices of spatial element contributing to the nonzero
                //!< pattern of the spatial matrices

  int _spat_order;  //!< degree of Chebyshev polynomials for expansion in
                    //!< space in pFMM
  int _temp_order;  //!< degree of Lagrange interpolation polynomials in time
                    //!< for pFMM
  bem::lagrange_interpolant _lagrange;
};

#endif /* INCLUDE_BESTHEA_FAST_SPACETIME_BE_ASSEMBLER_H_ */
