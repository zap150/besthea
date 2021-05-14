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

/** @file distributed_fast_spacetime_be_assembler.h
 * @brief Contains a class for assembling distributed pFMM matrices.
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_ASSEMBLER_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_ASSEMBLER_H_

#include "besthea/chebyshev_evaluator.h"
#include "besthea/distributed_pFMM_matrix.h"
#include "besthea/fast_spacetime_be_space.h"
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/lagrange_interpolant.h"
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
 * Class representing an assembler for fast boundary element matrices.
 * @tparam kernel_type @ref spacetime_kernel_antiderivative used for the
 *                          evaluation of antiderivatives in the integration
 *                          routines. E.g., when assembling the single layer
 *                          matrix the kernel is
 *                          @ref spacetime_heat_sl_kernel_antiderivative.
 * @tparam test_space_type @ref distributed_fast_spacetime_be_space that
 *                         specifies the basis functions of the test space.
 *                         Supported are tensor product basis functions, p0 in
 *                         time and p0 or p1 in space.
 * @tparam trial_space_type @ref distributed_fast_spacetime_be_space that
 *                          specifies the basis functions of the test space.
 *                          Supported are tensor product basis functions, p0 in
 *                          time and p0 or p1 in space.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::bem::distributed_fast_spacetime_be_assembler {
 private:
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< Shortcut for the full matrix
                                             //!< type.
  using pfmm_matrix_type
    = besthea::linear_algebra::distributed_pFMM_matrix< kernel_type,
      test_space_type,
      trial_space_type >;  //!< Shortcut for the pFMM matrix type.
  using vector_type
    = besthea::linear_algebra::vector;  //!< Shortcut for the vector type.

  /**
   * Wrapper for the quadrature data.
   *
   * @note On the triangle the quadrature nodes and weights are stored in arrays
   *       of vectors. For each spatial configuration (disjoint triangles,
   *       shared vertex, shared edge, identical triangles) a separate array is
   *       used.
   * @note The idea is to initialize quadrature nodes on reference elements once
   *       and map them to the actual geometry appropriately, whenever needed.
   * @note Allows the data to be private for OpenMP threads.
   */
  struct quadrature_wrapper {
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _x1_ref;  //!< First coordinates of quadrature nodes in test reference
                //!< triangle (0,1)x(0,1-x1).
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _x2_ref;  //!< Second coordinates of quadrature nodes in test reference
                //!< triangle (0,1)x(0,1-x1).

    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _y1_ref;  //!< First coordinates of quadrature nodes in trial reference
                //!< triangle (0,1)x(0,1-x1).
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _y2_ref;  //!< Second coordinates of quadrature nodes in trial reference
                //!< triangle (0,1)x(0,1-x1).

    std::array< std::vector< sc, besthea::allocator_type< sc > >, 4 >
      _w;  //!< Quadrature weights including the Jacobians of the
           //!< integral transformation.

    std::vector< sc, besthea::allocator_type< sc > >
      _x1;  //!< First coordinates of quadrature nodes in an actual test
            //!< element.
    std::vector< sc, besthea::allocator_type< sc > >
      _x2;  //!< Second coordinates of quadrature nodes in an actual test
            //!< element.
    std::vector< sc, besthea::allocator_type< sc > >
      _x3;  //!< Third coordinates of quadrature nodes in an actual test
            //!< element.

    std::vector< sc, besthea::allocator_type< sc > >
      _y1;  //!< First coordinates of quadrature nodes in an actual trial
            //!< element.
    std::vector< sc, besthea::allocator_type< sc > >
      _y2;  //!< Second coordinates of quadrature nodes in an actual trial
            //!< element.
    std::vector< sc, besthea::allocator_type< sc > >
      _y3;  //!< Third coordinates of quadrature nodes in an actual trial
            //!< element.

    std::vector< sc, besthea::allocator_type< sc > >
      _kernel_values;  //!< Buffer for storing kernel values.
    std::vector< sc, besthea::allocator_type< sc > >
      _kernel_values_2;  //!< Buffer for storing additional kernel values.

    std::vector< sc, besthea::allocator_type< sc > >
      _y1_ref_cheb;  //!< First coordinates of quadrature nodes in the reference
                     //!< triangle (0,1)x(0,1-x1) used for numerical integration
                     //!< of Chebyshev polynomials.
                     //!< @todo unused in this class.
    std::vector< sc, besthea::allocator_type< sc > >
      _y2_ref_cheb;  //!< Second coordinates of quadrature nodes in the
                     //!< reference triangle (0,1)x(0,1-x1) used for numerical
                     //!< integration of Chebyshev polynomials.
                     //!< @todo unused in this class.
    vector_type _y1_polynomial;  //!< Transformed coordinates for the evaluation
                                 //!< of Chebyshev polynomials in the interval
                                 //!< [-1,1] in x direction
                                 //!< @todo unused in this class.
    vector_type _y2_polynomial;  //!< Transformed coordinates for the evaluation
                                 //!< of Chebyshev polynomials in the interval
                                 //!< [-1,1] in y direction
                                 //!< @todo unused in this class.
    vector_type _y3_polynomial;  //!< Transformed coordinates for the evaluation
                                 //!< of Chebyshev polynomials in the interval
                                 //!< [-1,1] in z direction
                                 //!< @todo unused in this class.
    std::vector< sc, besthea::allocator_type< sc > >
      _wy_cheb;  //!< Quadrature weights for triangle quadrature. Does not
                 //!< include the Jacobian of the integral transformation.
                 //!< @todo unused in this class.
  };

 public:
  /**
   * Constructor.
   * @param[in] kernel Kernel used for the evaluation of the temporal
   *                   antiderivatives, see @ref _kernel
   * @param[in] test_space Boundary element test space, see @ref _test_space.
   * @param[in] trial_space Boundary element trial space, see @ref _trial_space.
   * @param[in] comm  MPI communicator associated with the assembler.
   * @param[in] order_singular Line quadrature order for regularized quadrature,
   *                           see @ref _order_singular.
   * @param[in] order_regular Triangle quadrature order for regular quadrature,
   *                          see @ref _order_regular
   * @param[in] temp_order Degree of Lagrange interpolation polynomials in time
   *                       for pFMM matrix.
   * @param[in] spat_order Largest degree of Chebyshev polynomials for expansion
   *                       in pFMM matrix.
   * @param[in] alpha Heat conductivity parameter.
   */
  distributed_fast_spacetime_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    MPI_Comm * comm, int order_singular = 4, int order_regular = 4,
    int temp_order = 5, int spat_order = 5, sc alpha = 1.0 );

  distributed_fast_spacetime_be_assembler(
    const distributed_fast_spacetime_be_assembler & that )
    = delete;

  /**
   * Destructor.
   */
  ~distributed_fast_spacetime_be_assembler( );

  /**
   * Assembles the fast spacetime matrix.
   *
   * The nearfield matrices of @p global_matrix are assembled. Furthermore all
   * structures needed for the application of the distributed pFMM matrix vector
   * multiplication are initialized.
   * @param[out] global_matrix Assembled pFMM matrix.
   * @param[in] info_mode  If true, the nearfield matrices are not assembled.
   *                       This reduces the execution time drastically and
   *                       allows to determine the memory complexity faster.
   * @warning Currently there are some restrictions on the combinations of test
   * and trial spaces. In particular, they have to be defined on the same meshes
   * and have to be associated with the same cluster trees.
   */
  void assemble(
    pfmm_matrix_type & global_matrix, bool info_mode = false ) const;

 private:
  /**
   * Initializes quadrature structures for evaluation of nearfield integrals of
   * pFMM matrix.
   *
   * The quadrature points on the test and trial reference triangle and the
   * quadrature weights for each spatial configuration (disjoint triangles,
   * shared vertex, shared edge, identical triangles) are initialized. The other
   * structures used for nearfield quadrature are resized appropriately.
   * @param[in,out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( quadrature_wrapper & my_quadrature ) const;

  /**
   * Initializes quadrature structures used to integrate Chebyshev polynomials
   * on triangles.
   *
   * The quadrature points and weights on the reference triangle are
   * initialized. The other structures used for integration of Chebyshev
   * polynomials are resized appropriately.
   *
   * @param[out] my_quadrature Wrapper holding quadrature data.
   * @todo unused in this class.
   */
  void init_quadrature_polynomials( quadrature_wrapper & my_quadrature ) const;

  /**
   * Assembles the nearfield matrices of the pFMM matrix clusterwise, i.e. by
   * considering the cluster tree associated with the test and trial space. Only
   * the nearfield matrices corresponding to leaf clusters owned by the
   * executing MPI process are assembled.
   *
   * @note The nearfield of a cluster is determined in @ref
   * besthea::mesh::distributed_spacetime_cluster_tree::fill_nearfield_and_interaction_lists
   * @param[in,out] global_matrix pFMM matrix whose local nearfield (depending
   *                              on the executing MPI process) is assembled.
   */
  void assemble_nearfield( pfmm_matrix_type & global_matrix ) const;

  /**
   * Assembles a nearfield matrix associated to a given target and source
   * cluster.
   * @param[in] target_cluster  Spacetime target cluster
   * @param[in] source_cluster  Spacetime source cluster in the nearfield of the
   *                            spacetime target cluster.
   * @param[in,out] nearfield_matrix Reference to the matrix which is assembled.
   */
  void assemble_nearfield_matrix(
    mesh::general_spacetime_cluster * target_cluster,
    mesh::general_spacetime_cluster * source_cluster,
    full_matrix_type & nearfield_matrix ) const;

  /**
   * Determines the configuration of two spatial, triangular elements
   * (triangles), i.e. whether they are disjoint, the same, or share a vertex or
   * edge.
   * @param[in] i_test_space Index of the spatial test element.
   * @param[in] i_trial_space Index of the spatial trial element.
   * @param[out] type_int Type of the configuration (number of vertices shared).
   * @param[out] rot_test Virtual rotation of the test element.
   * @param[out] rot_trial Virtual rotation of the trial element.
   *
   * @warning The routine relies on the fact, that the spatial mesh is the same
   * for the nearfield mesh and the local mesh of a distributed spacetime mesh.
   * If this is not the case the result is worng.
   */
  void get_type( lo i_test_space, lo i_trial_space, int & type_int,
    int & rot_test, int & rot_trial ) const;

  /**
   * Maps a quadratures node (ksi, eta_1, eta_2, eta_3) from the hypercube
   * (0,1)^4 to the reference triangles (the tensor product of the test and
   * trial reference triangles).
   *
   * The mapping depends on the number of shared vertices (1,2 or 3) and the
   * index of the simplex (Depending on the number of shared vertices, the
   * singular integral is split into several integrals over 4d simplices.)
   * @param[in] ksi Value of ksi in (0,1).
   * @param[in] eta1 Value of eta_1 in (0,1).
   * @param[in] eta2 Value of eta_2 in (0,1).
   * @param[in] eta3 Value of eta_3 in (0,1).
   * @param[in] n_shared_vertices Number of vertices shared between elements.
   * @param[in] simplex Simplex index.
   * @param[out] x1_ref First coordinate of the resulting quadrature node in the
   *                    test element.
   * @param[out] x2_ref Second coordinate of the resulting quadrature node in
   *                    the test element.
   * @param[out] y1_ref First coordinate of the resulting quadrature node in the
   *                    trial element
   * @param[out] y2_ref Second coordinate of the resulting quadrature node in
   *                    the trial element.
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
   * Maps a quadratures node (ksi, eta_1, eta_2, eta_3) from the hypercube
   * (0,1)^4 to the reference triangles (the tensor product of the test and
   * trial reference triangles) for the case when trial and test triangles share
   * a vertex.
   * @param[in] ksi Value of ksi in (0,1).
   * @param[in] eta1 Value of eta_1 in (0,1).
   * @param[in] eta2 Value of eta_2 in (0,1).
   * @param[in] eta3 Value of eta_3 in (0,1).
   * @param[in] simplex Simplex index.
   * @param[out] x1_ref First coordinate of the resulting quadrature node in the
   *                    test element.
   * @param[out] x2_ref Second coordinate of the resulting quadrature node in
   *                    the test element.
   * @param[out] y1_ref First coordinate of the resulting quadrature node in the
   *                    trial element
   * @param[out] y2_ref Second coordinate of the resulting quadrature node in
   *                    the trial element.
   * @param[out] jacobian Jacobian of the transformation.
   */
  void hypercube_to_triangles_vertex( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;

  /**
   * Maps a quadratures node (ksi, eta_1, eta_2, eta_3) from the hypercube
   * (0,1)^4 to the reference triangles (the tensor product of the test and
   * trial reference triangles) for the case when trial and test triangles share
   * an edge.
   * @param[in] ksi Value of ksi in (0,1).
   * @param[in] eta1 Value of eta_1 in (0,1).
   * @param[in] eta2 Value of eta_2 in (0,1).
   * @param[in] eta3 Value of eta_3 in (0,1).
   * @param[in] simplex Simplex index.
   * @param[out] x1_ref First coordinate of the resulting quadrature node in the
   *                    test element.
   * @param[out] x2_ref Second coordinate of the resulting quadrature node in
   *                    the test element.
   * @param[out] y1_ref First coordinate of the resulting quadrature node in the
   *                    trial element.
   * @param[out] y2_ref Second coordinate of the resulting quadrature node in
   *                    the test element.
   * @param[out] jacobian Jacobian of the transformation.
   */
  void hypercube_to_triangles_edge( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;

  /**
   * Maps a quadratures node (ksi, eta_1, eta_2, eta_3) from the hypercube
   * (0,1)^4 to the reference triangles (the tensor product of the test and
   * trial reference triangles) for the case when trial and test triangles are
   * identical.
   * @param[in] ksi Value of ksi in (0,1).
   * @param[in] eta1 Value of eta_1 in (0,1).
   * @param[in] eta2 Value of eta_2 in (0,1).
   * @param[in] eta3 Value of eta_3 in (0,1).
   * @param[in] simplex Simplex index.
   * @param[out] x1_ref First coordinate of the resulting quadrature node in the
   *                    test element.
   * @param[out] x2_ref Second coordinate of the resulting quadrature node in
   *                    the test element.
   * @param[out] y1_ref First coordinate of the resulting quadrature node in the
   *                    trial element.
   * @param[out] y2_ref Second coordinate of the resulting quadrature node in
   *                    the test element.
   * @param[out] jacobian Jacobian of the transformation.
   */
  void hypercube_to_triangles_identical( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;

  /**
   * Maps all quadrature nodes (nearfield quadrature) corresponding to a given
   * triangle configuration( @p n_shared_vertices) from the reference triangles
   * to the actual geometry.
   *
   * The quadrature nodes on the reference triangles have to be given in
   * @p my_quadrature. The results are stored in this structure too.
   *
   * @param[in] x1 Coordinates of the first node of the test element.
   * @param[in] x2 Coordinates of the second node of the test element.
   * @param[in] x3 Coordinates of the third node of the test element.
   * @param[in] y1 Coordinates of the first node of the trial element.
   * @param[in] y2 Coordinates of the second node of the trial element.
   * @param[in] y3 Coordinates of the third node of the trial element.
   * @param[in] n_shared_vertices Number of vertices which the test and trial
   *                              triangle share. This determines which
   *                              quadrature points are mapped to the actual
   *                              triangles.
   * @param[in] rot_test Virtual rotation of the test element.
   * @param[in] rot_trial Virtual rotation of the trial element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void triangles_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3, int n_shared_vertices,
    int rot_test, int rot_trial, quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps all quadrature nodes (integration of Chebyshev polynomials) from the
   * reference triangle to the actual geometry.
   *
   * The quadrature nodes on the reference triangles have to be given in
   * @p my_quadrature. The results are stored in this structure too.
   *
   * @param[in] y1 Coordinates of the first node of the triangle.
   * @param[in] y2 Coordinates of the second node of the triangle.
   * @param[in] y3 Coordinates of the third node of the triangle.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   * @todo unused in this class.
   */
  void triangle_to_geometry( const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3,
    quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps points from a given axis-parallel spatial cluster to the cube [-1,1]^3
   * using the standard linear transformation.
   *
   * The points are taken from @p my_quadrature and the results are stored there
   * too.
   * @param[in,out] my_quadrature Structure holding the points to be mapped and
   *                              the results.
   * @param[in] x_start Lower border of the space cluster along x dimension.
   * @param[in] x_end Upper border of the space cluster along x dimension.
   * @param[in] y_start Lower border of the space cluster along y dimension.
   * @param[in] y_end Upper border of the space cluster along y dimension.
   * @param[in] z_start Lower border of the space cluster along z dimension.
   * @param[in] z_end Upper border of the space cluster along z dimension.
   * @todo rename the routine to better describe its action?
   * @todo unused in this class.
   */
  void cluster_to_polynomials( quadrature_wrapper & my_quadrature, sc x_start,
    sc x_end, sc y_start, sc y_end, sc z_start, sc z_end ) const;

  /**
   * Initializes data storage for moment and local contributions for all
   * clusters in the test and trial tree structure, which are owned by the
   * executing MPI process.
   */
  void initialize_moment_and_local_contributions( ) const;

  kernel_type * _kernel;  //!< Kernel used for the evaluation of the temporal
                          //!< antiderivatives.
  test_space_type * _test_space;  //!< Boundary element test space. Has to be a
                                  //!< @ref distributed_fast_spacetime_be_space.
  trial_space_type *
    _trial_space;       //!< Boundary element trial space. Has to be a
                        //!< @ref distributed_fast_spacetime_be_space.
  int _order_singular;  //!< Number of quadrature points on the line used for
                        //!< evaluation of singular integrals. By (Duffy)
                        //!< transformations singular integrals are transformed
                        //!< to integrals over the 4d cube [0,1]^4. Here,
                        //!< @p _order_singular quadrature points per dimension
                        //!< are used for integration.
  int _order_regular;  //!< Triangle quadrature order for the regular integrals.
                       //!< Polynomials on the triangle up to this order are
                       //!< integrated exactly.
  int _temp_order;     //!< Degree of Lagrange interpolation polynomials in time
                       //!< for pFMM.
  int _spat_order;  //!< Largest degree of Chebyshev polynomials for expansion
                    //!< in space in pFMM.
  int _m2l_integration_order;  //!< @p _m2l_integration_order + 1 quadrature
                               //!< points are used for the approximation of
                               //!< the m2l coefficients.
  sc _alpha;                   //!< Heat conductivity parameter.
  static constexpr std::array< int, 5 > map{ 0, 1, 2, 0,
    1 };  //!< Auxiliary array for mapping DOFs under rotation
          //!< (regularized quadrature). Realizes a fast modulo 3
          //!< operation.
  static constexpr std::array< int, 4 > n_simplices{ 1, 2, 5,
    6 };  //!< Number of simplices which have to be considered for quadrature of
          //!< integrals with singular kernels, sorted by configuration
          //!< (disjoint, shared vertex, shared edge, identical).
  int _my_rank;  //!< MPI rank of the executing process.
  const MPI_Comm *
    _comm;  //!< MPI communicator associated with the pFMM matrix.
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_ASSEMBLER_H_ */
