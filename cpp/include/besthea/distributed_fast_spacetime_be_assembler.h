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

/** @file distributed_fast_spacetime_be_assembler.h
 * @brief File containing a class for assembling distributed pFMM matrices
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_ASSEMBLER_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_ASSEMBLER_H_

#include "besthea/chebyshev_evaluator.h"
#include "besthea/distributed_pFMM_matrix.h"
#include "besthea/fast_spacetime_be_space.h"
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/pFMM_matrix.h"
#include "besthea/quadrature.h"
#include "besthea/settings.h"
#include "besthea/tree_structure.h"
#include "besthea/vector.h"

#include <mpi.h>
#include <omp.h>
#include <set>
#include <vector>

namespace besthea {
  namespace bem {
    template< class kernel_type, class test_space_type, class trial_space_type >
    class distributed_fast_spacetime_be_assembler;
  }
}

/**
 *  Class representing a boundary element fast matrix assembler.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::bem::distributed_fast_spacetime_be_assembler {
 private:
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< shortcut for the full matrix
                                             //!< type
  using pfmm_matrix_type = besthea::linear_algebra::
    distributed_pFMM_matrix< kernel_type, 
    test_space_type, trial_space_type >;  //!< shortcut for the pFMM matrix type
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
   * @param[in] comm  MPI communicator associated with the assembler.
   * @param[in] order_singular Line quadrature order for regularized quadrature.
   * @param[in] order_regular Triangle quadrature order for regular quadrature.
   * @param[in] temp_order Degree of Lagrange interpolation polynomials in time
   *                       for pFMM matrix.
   * @param[in] spat_order Degree of Chebyshev polynomials for expansion in pFMM
   *                       matrix.
   * @param[in] alpha Heat conductivity parameter.
   */
  distributed_fast_spacetime_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    MPI_Comm * comm, int order_singular = 4, int order_regular = 4, 
    int temp_order = 5, int spat_order = 5, sc alpha = 1.0);

  distributed_fast_spacetime_be_assembler( 
    const distributed_fast_spacetime_be_assembler & that ) = delete;

  /**
   * Destructor.
   */
  ~distributed_fast_spacetime_be_assembler( );

  /**
   * Assembles the fast spacetime matrix.
   * @param[out] global_matrix Assembled pFMM matrix.
   * @warning Currently there are some restrictions on the test 
   * and trial spaces (same meshes and trees)
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
   * Assembles temporal nearfield matrices clusterwise.
   * @param[in,out] global_matrix pFMM matrix which is partially assembled.
   */
  void assemble_nearfield( pfmm_matrix_type & global_matrix ) const;

  /**
   * Assembles clusterwise nearfield matrix
   * @param[in] target_cluster  Target for which the matrix is assembled.
   * @param[in] source_cluster  Source in the nearfield of the target for which 
   *                            the matrix is assembled.
   * @param[in,out] nearfield_matrix Reference to the matrix which should be
   * assembled.
   */  
  void assemble_nearfield_matrix( 
    mesh::general_spacetime_cluster * target_cluster, 
    mesh::general_spacetime_cluster * source_cluster, 
    full_matrix_type & nearfield_matrix ) const;

  /**
   * Determines the configuration of two spatial, triangular elements.
   * @param[in] i_test_space Index of the spatial test element.
   * @param[in] i_trial_space Index of the spatial trial element.
   * @param[out] type_int Type of the configuration (number of vertices shared).
   * @param[out] rot_test Virtual rotation of the test element.
   * @param[out] rot_trial Virtual rotation of the trial element.
   * @warning The routine relies on the fact, that the spatial mesh is the same
   * for the nearfield mesh and the local mesh of a distributed spacetime mesh.
   * If this is not the case the result is meaningless.
   */
  void get_type( lo i_test_space, lo i_trial_space, int & type_int, 
    int & rot_test, int & rot_trial ) const;

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
   * Initializes data storage for moment and local contributions for all 
   * clusters in the test and trial space spacetime cluster tree.
   */
  void initialize_moment_and_local_contributions( ) const;

  kernel_type * _kernel;            //!< Kernel temporal antiderivative.
  test_space_type * _test_space;    //!< Boundary element test space.
  trial_space_type * _trial_space;  //!< Boundary element trial space.
  int _order_singular;  //!< Line quadrature order for the singular integrals.
  int _order_regular;  //!< Triangle quadrature order for the regular integrals.
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
  int _my_rank;           //!< MPI rank of the current process.
  const MPI_Comm * _comm; //!< MPI communicator associated with the pFMM matrix.
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_ASSEMBLER_H_ */
