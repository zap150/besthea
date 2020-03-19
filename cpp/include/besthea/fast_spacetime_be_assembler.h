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
#include "besthea/full_matrix.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/pFMM_matrix.h"
#include "besthea/quadrature.h"
#include "besthea/settings.h"
#include "besthea/space_cluster.h"
#include "besthea/spacetime_cluster.h"
#include "besthea/sparse_matrix.h"
#include "besthea/time_cluster.h"
#include "besthea/vector.h"

#include <omp.h>
#include <set>
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
  using sparse_matrix_type
    = besthea::linear_algebra::sparse_matrix;  //!< shortcut for the sparse
                                               //!< matrix type
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< shortcut for the full matrix
                                             //!< type
  using pfmm_matrix_type = besthea::linear_algebra::pFMM_matrix< kernel_type,
    test_space_type, trial_space_type >;  //!< shortcut for the pFMM matrix type
                                          //!< type.
  using time_cluster_type = besthea::mesh::time_cluster;  //!< time cluster type
  using spacetime_cluster_type
    = besthea::mesh::spacetime_cluster;  //!< time cluster type
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

    std::vector< sc, besthea::allocator_type< sc > >
      _y1_ref_cheb;  //!< First coordinates of quadrature nodes for the
                     //!< Chebyshev polynomials in (0,1)x(0,1-x1) to be mapped
                     //!< to the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _y2_ref_cheb;  //!< Second coordinates of quadrature nodes for the
                     //!< Chebyshev polynomials in (0,1)x(0,1-x1) to be mapped
                     //!< to the test element
    vector_type
      _y1_polynomial;  //!< Coordinates for evaluation of the Chebyshev
                       //!< polynomials in the interval [-1,1] in x direction
    vector_type
      _y2_polynomial;  //!< Coordinates for evaluation of the Chebyshev
                       //!< polynomials in the interval [-1,1] in y direction
    vector_type
      _y3_polynomial;  //!< Coordinates for evaluation of the Chebyshev
                       //!< polynomials in the interval [-1,1] in z direction
    std::vector< sc, besthea::allocator_type< sc > >
      _wy_cheb;  //!< Quadrature weights including
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
   * @param[in] alpha Heat conductivity paremeter.
   * @param[in] temp_order degree of Lagrange interpolation polynomials in time
   * for pFMM matrix.
   * @param[in] cutoff_param Cutoff parameter for the nearfield approximation
   * (elements further than cutoff_param * diagonal of the lowest level cluster
   * will be ignored).
   * @param[in] uniform Uniform time discretization to save memory.
   */
  fast_spacetime_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular = 4, int order_regular = 4, int temp_order = 5,
    int spat_order = 5, sc alpha = 1.0, sc cutoff_param = 3.0,
    bool uniform = false );

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
  void assemble( pfmm_matrix_type & global_matrix ) const;

 private:
  /**
   * Initializes quadrature structures.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( quadrature_wrapper & my_quadrature ) const;

  /**
   * Initializes quadrature structures.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature_polynomials( quadrature_wrapper & my_quadrature ) const;

  /**
   * Assembles temporal nearfield matrices.
   * @param[out] global_matrix Partially assembled pFMM matrix.
   * \todo If the number of levels in the spacetime cluster tree is less than
   * the number of levels in the time tree than not enough nearfield matrices
   * are computed. Change is necessary!
   */
  void assemble_nearfield( pfmm_matrix_type & global_matrix ) const;

  /** Assembles temporal farfield nonapproximated by the pFMM.
   * @param[out] global_matrix Partially assembled pFMM matrix.
   */
  void assemble_farfield_nonapproximated(
    pfmm_matrix_type & global_matrix ) const;

  /** Assembles temporal farfield nonapproximated by the pFMM.
   * @param[out] global_matrix Partially assembled pFMM matrix.
   */
  void assemble_nonapproximated_uniform(
    pfmm_matrix_type & global_matrix ) const;

  /**
   * Assembles nearfield matrix
   * @param[in] t0 Initial time of the test function quadrature.
   * @param[in] t1 Final time of the test function quadrature.
   * @param[in] tau0 Initial time of the trial function quadrature.
   * @param[in] tau1 Final time of the trial function quadrature.
   * @param[out] nearfield_matrix Reference to the matrix which should be
   * assembled.
   */
  void assemble_nearfield_matrix(
    sc t0, sc t1, sc tau0, sc tau1, full_matrix_type & nearfield_matrix ) const;

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
   * @param[in] n_shared_vertices Number of vertices shared between elements.
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
   * @param[out] jacobian Jacobian of the transformation.
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
   * Maps the quadrature nodes from the reference triangle to the actual
   * geometry.
   * @param[in] y1 Coordinates of the first node of the test element.
   * @param[in] y2 Coordinates of the second node of the test element.
   * @param[in] y3 Coordinates of the third node of the test element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void triangle_to_geometry( const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3,
    quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps from the spatial cluster to the interval [-1, 1] where the Chebyshev
   * polynomials are defined.
   * @param[out] my_quadrature Structure holding mapping from the cluster
   * to the interval [-1,1].
   * @param[in] x_start Border of the space cluster for which the Chebyshev
   * polynomials are evaluated.
   * @param[in] x_end Border of the space cluster for which the Chebyshev
   * polynomials are evaluated.
   * @param[in] y_start Border of the space cluster for which the Chebyshev
   * polynomials are evaluated.
   * @param[in] y_end Border of the space cluster for which the Chebyshev
   * polynomials are evaluated.
   * @param[in] z_start Border of the space cluster for which the Chebyshev
   * polynomials are evaluated.
   * @param[in] z_end Border of the space cluster for which the Chebyshev
   * polynomials are evaluated.
   *
   */
  void cluster_to_polynomials( quadrature_wrapper & my_quadrature, sc x_start,
    sc x_end, sc y_start, sc y_end, sc z_start, sc z_end ) const;

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
   * Computes quadrature of the Lagrange polynomials up to a given order over
   * all elements in the cluster.
   * \param[in] cluster   Time cluster for which the quadratures are computed
   *                      and where they are stored.
   */
  void compute_lagrange_quadrature( time_cluster_type * cluster ) const;

  /**
   * Computes integrals of the derivatives of the Lagrange polynomials up to a
   * given order over all elements in the cluster.
   * \param[in] cluster   Time cluster for which the integrals are computed
   *                      and where they are stored.
   */
  void compute_lagrange_drv_integrals( time_cluster_type * cluster ) const;

  /**
   * Compute the surface curls of p1 functions and store them in the
   * space_cluster.
   * \param[in] cluster   Space cluster for which the curls are computed
   *                      and where they are stored.
   * \see besthea::mesh::space_cluster for a detailed explanation on
   * how the curls area stored.
   * \warning Only the information from \p _trial_space is used for the
   * computation (Could cause problems if test and trial space differ).
   */
  void compute_surface_curls( space_cluster_type * cluster ) const;

  /**
   * Computes various data which is needed for the application of the pFMM
   * matrix. This includes quadratures of Chebyshev polynomials
   * (various variants depending on the operator), quadratures of Lagrange
   * polynomials(or their derivatives) and in case of the hypersingular operator
   * also the mappings to realize the integrals of curls of p1 functions
   * \note If the required quadratures are already computed they are not
   * recomputed. TODO: currently disactivated
   * \warning Only the number of columns of the Chebyshev quadrature (or its
   *          derivative ) of the first spatial leaf cluster is checked to see
   *          if the quadratures have been computed.
   */
  void compute_required_data(
    std::set< time_cluster_type * > & time_clusters_spacetime_leaves,
    std::set< space_cluster_type * > & space_clusters_spacetime_leaves ) const;

  /**
   * Compute quadrature of the Chebyshev polynomials and p0 basis functions for
   * a space cluster.
   * \param[in] cluster   Space cluster for which the quadratures are computed
   *                      and where they are stored.
   */
  void compute_chebyshev_quadrature_p0( space_cluster_type * cluster ) const;

  /**
   * Compute quadrature of the Chebyshev polynomials and p1 basis functions for
   * a space cluster.
   * \param[in] cluster   Space cluster for which the quadratures are computed
   *                      and where they are stored.
   */
  void compute_chebyshev_quadrature_p1( space_cluster_type * cluster ) const;

  /**
   * Compute quadrature of the normal derivatives of the Chebyshev polynomials
   * times p1 basis functions for a space cluster.
   * \param[in] cluster   Space cluster for which the quadratures are computed
   *                      and where they are stored.
   */
  void compute_normal_drv_chebyshev_quadrature(
    space_cluster_type * cluster ) const;

  /**
   * Compute quadrature of the the Chebyshev polynomials times p1 basis
   * functions times normal vector for a space cluster.
   * \param[in] cluster   Space cluster for which the quadratures are computed
   *                      and where they are stored.
   */
  void compute_chebyshev_times_normal_quadrature(
    space_cluster_type * cluster ) const;

  /**
   * \todo do documentation
   */
  void initialize_moment_and_local_contributions( ) const;

  kernel_type * _kernel;            //!< Kernel temporal antiderivative.
  test_space_type * _test_space;    //!< Boundary element test space.
  trial_space_type * _trial_space;  //!< Boundary element trial space.
  int _order_singular;  //!< Line quadrature order for the singular integrals.
  int _order_regular;  //!< Triangle quadrature order for the regular integrals.
  sc _cutoff_param;    //!< Coefficient for determining the spatial nearfield
                       //!< (_cutoff_param * diagonal of the lowest level
                       //!< cluster).
  bool _uniform;       //!< uniform assembly
  int _temp_order;     //!< degree of Lagrange interpolation polynomials in time
                       //!< for pFMM
  int _spat_order;     //!< degree of Chebyshev polynomials for expansion in
                       //!< space in pFMM
  int _m2l_integration_order;  //!< _m2l_integration_order + 1 quadrature
                               //!< points are used for the approximation of
                               //!< the m2l coefficients.
  sc _alpha;                   //!< Heat conductivity
  mutable bem::lagrange_interpolant
    _lagrange;  //!< Evaluator of the Lagrange polynomials.
  mutable bem::chebyshev_evaluator
    _chebyshev;  //!< Evaluator of the Chebyshev polynomials.
  static constexpr std::array< int, 5 > map{ 0, 1, 2, 0,
    1 };  //!< Auxiliary array for mapping DOFs under
          // rotation (regularized quadrature). Performs fast modulo 3.
  static constexpr std::array< int, 4 > n_simplices{ 1, 2, 5,
    6 };  //!< Number of simplices for all configurations (disjoint, shared
          // vertex, shared edge, identical).
  sc _space_cluster_size;  //!< Size of the finest level space clusters.

  std::vector< std::pair< lo, lo > >
    _nonzeros;  //!< indices of spatial element contributing to the nonzero
                //!< pattern of the spatial matrices
};

#endif /* INCLUDE_BESTHEA_FAST_SPACETIME_BE_ASSEMBLER_H_ */
