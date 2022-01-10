/*
Copyright (c) 2021, VSB - Technical University of Ostrava and Graz University of
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

/** @file distributed_fast_spacetime_initial_be_assembler.h
 * @brief Contains a class for assembling distributed initial pFMM matrices.
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_INITIAL_BE_ASSEMBLER_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_INITIAL_BE_ASSEMBLER_H_

#include "besthea/chebyshev_evaluator.h"
#include "besthea/distributed_initial_pFMM_matrix.h"
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/quadrature.h"
#include "besthea/settings.h"
#include "besthea/tree_structure.h"
#include "besthea/vector.h"
#include "besthea/volume_space_cluster.h"
#include "besthea/volume_space_cluster_tree.h"

#include <mpi.h>
#include <omp.h>
#include <set>
#include <vector>

namespace besthea {
  namespace bem {
    template< class kernel_type, class test_space_type, class trial_space_type >
    class distributed_fast_spacetime_initial_be_assembler;
  }
}

/**
 * Class representing an assembler for fast boundary element matrices.
 * @tparam kernel_type @ref spacetime_kernel used for the evaluation of
 *                     antiderivatives in the integration routines. E.g., when
 *                     assembling the single layer matrix the kernel is
 *                     @ref spacetime_heat_sl_kernel_antiderivative.
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
class besthea::bem::distributed_fast_spacetime_initial_be_assembler {
 private:
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< Shortcut for the full matrix
                                             //!< type.
  using pfmm_matrix_type
    = besthea::linear_algebra::distributed_initial_pFMM_matrix< kernel_type,
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
   * @param[in] trial_space Finite element trial space, see @ref _trial_space.
   * @param[in] comm  MPI communicator associated with the assembler.
   * @param[in] space_source_tree Space tree corresponding to the volume mesh,
   * see @ref _space_source_tree.
   * @param[in] order_regular_tri Triangle quadrature order for regular
   * quadrature.
   * @param[in] order_regular_tetra Tetrahedron quadrature order for regular
   * quadrature.
   * @param[in] order_regular_line  Line quadrature order for regular
   * quadrature.
   * @param[in] temp_order Degree of Lagrange interpolation polynomials in time
   *                       for pFMM matrix.
   * @param[in] spat_order Largest degree of Chebyshev polynomials for expansion
   *                       in pFMM matrix.
   * @param[in] n_recursions_singular_integrals Maximal number of recursive
   * refinements for the computation of singular matrix entries.
   * @param[in] alpha Heat conductivity parameter.
   */
  distributed_fast_spacetime_initial_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    MPI_Comm * comm, mesh::volume_space_cluster_tree & space_source_tree,
    int order_regular_tri = 4, int order_regular_tetra = 4,
    int order_regular_line = 4, int temp_order = 5, int spat_order = 5,
    lo n_recursions_singular_integrals = 2, sc alpha = 1.0 );

  distributed_fast_spacetime_initial_be_assembler(
    const distributed_fast_spacetime_initial_be_assembler & that )
    = delete;

  /**
   * Destructor.
   */
  ~distributed_fast_spacetime_initial_be_assembler( );

  /**
   * Assembles the fast spacetime matrix.
   *
   * The nearfield matrices of @p global_matrix are assembled. Furthermore all
   * structures needed for the application of the distributed initial pFMM
   * matrix vector multiplication are initialized.
   * @param[out] global_matrix Assembled pFMM matrix.
   * @param[in] info_mode If true, the nearfield matrices are not assembled.
   */
  void assemble(
    pfmm_matrix_type & global_matrix, bool info_mode = false ) const;

 private:
  /**
   * Assembles the nearfield matrices of the pFMM matrix clusterwise, i.e. by
   * considering the cluster tree associated with the test and trial space. Only
   * the nearfield matrices corresponding to leaf clusters owned by the
   * executing MPI process are assembled.
   *
   * @note The nearfield of a cluster is determined in @ref
   * besthea::mesh::distributed_initial_pFMM_matrix::initialize_nearfield_and_interaction_lists
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
    const mesh::general_spacetime_cluster * target_cluster,
    const mesh::volume_space_cluster * source_cluster,
    full_matrix_type & nearfield_matrix ) const;

  /**
   * Evaluates the definite integral of the heat kernel over a regular time
   * interval for given quadrature points.
   *
   * @param[in,out] my_quadrature Container storing the quadrature points and
   * the resulting kernel values.
   * @param[in] t0  Starting time of the considered time interval.
   * @param[in] t1  Ending time of the considered time interval.
   * @param[in] nx_data Normal vector on the surface triangle. Not used.
   */
  void evaluate_kernel_regular_interval( quadrature_wrapper & my_quadrature,
    const sc t0, const sc t1, const sc * nx_data ) const;

  /**
   * Evaluates the definite integral of the heat kernel over a time interval
   * starting at 0 for given quadrature points.
   *
   * @param[in,out] my_quadrature Container storing the quadrature points and
   * the resulting kernel values.
   * @param[in] t1  Ending time of the considered time interval.
   * @param[in] nx_data Normal vector on the surface triangle. Not used.
   */
  void evaluate_kernel_first_interval(
    quadrature_wrapper & my_quadrature, const sc t1, const sc * nx_data ) const;

  /**
   * Computes local nearfield matrix entries for all dofs in a given pair of
   * triangle and tetrahedron. (keyword local stiffness matrix)
   *
   * Regular case: The triangle and tetrahedron should be separated in space, or
   * the time interval should start at a time t0 reasonably greater than 0.
   *
   * @param[in,out] local_entries The results are written to this vector.
   * @param[in] my_quadrature Wrapper holding quadrature data.
   * @param[in] is_first_timestep  Indicates if the time interval starts at 0.
   * @param[in] t0  Starting time of the considered time interval.
   * @param[in] t1  Ending time of the considered time interval.
   * @param[in] test_index  Global spatial index of the test element (triange).
   * Not used.
   * @param[in] trial_index Index of the trial element (tetrahedron). Not used.
   * @param[in] test_area Area of the test element (triangle).
   * @param[in] trial_volume  Volume of the trial element (tetrahedron).
   * @param[in] nx_data Normal vector on the surface triangle. Not used.
   */
  void compute_local_matrix_entries_regular_case( vector_type & local_entries,
    quadrature_wrapper & my_quadrature, const bool is_first_timestep,
    const sc t0, const sc t1, const lo test_index, const lo trial_index,
    const sc test_area, const sc trial_volume, const sc * nx_data ) const;

  /**
   * Computes local nearfield matrix entries for all dofs in a given pair of
   * triangle and tetrahedron. (keyword local stiffness matrix)
   *
   * Singular case: The considered time interval starts at 0, and the triangle
   * and tetrahedron are not reasonably separated in space (e.g. they share a
   * vertex, edge, ...). \n
   * The integral is approximated by refining the tetrahedron and triangle
   * recursively until a resulting pair is either separated or the recursion
   * depth is reached. \n
   * The routine calls itself recursively.
   *
   * @param[in,out] local_entries The results are written to this vector.
   * @param[in] recursion_depth Determines the maximal number of refinements
   * for the recursive computation of the integral.
   * @param[in] my_quadrature Wrapper holding quadrature data.
   * @param[in] x1  Node 1 of the triangle.
   * @param[in] x2  Node 2 of the triangle.
   * @param[in] x3  Node 3 of the triangle.
   * @param[in] y1  Node 1 of the tetrahedron.
   * @param[in] y2  Node 2 of the tetrahedron.
   * @param[in] y3  Node 3 of the tetrahedron.
   * @param[in] y4  Node 4 of the tetrahedron.
   * @param[in] t1  Ending time of the considered time interval.
   * @param[in] test_index  Global spatial index of the test element (triange).
   * Not used.
   * @param[in] trial_index Index of the trial element (tetrahedron). Not used.
   * @param[in] test_area Area of the test element (triangle).
   * @param[in] trial_volume  Volume of the trial element (tetrahedron).
   * @param[in] nx_data Normal vector on the surface triangle. Not used.
   */
  void compute_local_matrix_entries_singular_case_recursively(
    vector_type & local_entries, const lo recursion_depth,
    quadrature_wrapper & my_quadrature,
    const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3,
    const linear_algebra::coordinates< 3 > & y4, const sc t1,
    const lo test_index, const lo trial_index, const sc test_area,
    const sc trial_volume, const sc * nx_data ) const;

  /**
   * Auxiliary function to update the local nearfield matrix corresponding to a
   * given triangle and tetrahedron (parents) by using the local nearfield
   * matrices of a pair of their children.
   *
   * Used in the routine
   * @ref compute_local_matrix_entries_singular_case_recursively.
   * @param[in,out] parent_local_entries  Vector of the local nearfield matrix
   * of the given parents, which is updated.
   * @param[in] child_pair_local_entries  Vector of the local nearfield matrix
   * of the given children.
   * @param[in] tri_child_idx Index of the child of the triangle (range 0-3).
   * @param[in] tet_child_idx Index of the child of the tetrahedron (range 0-4).
   */
  void process_childrens_local_matrix_entries(
    vector_type & parent_local_entries,
    const vector_type & child_pair_local_entries, const lo tri_child_idx,
    const lo tet_child_idx ) const;

  /**
   * Initializes quadrature structures for evaluation of nearfield integrals
   * of pFMM matrix.
   *
   * The quadrature points on the test and trial reference triangle and the
   * quadrature weights for each spatial configuration (disjoint triangles,
   * shared vertex, shared edge, identical triangles) are initialized. The
   * other structures used for nearfield quadrature are resized appropriately.
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
   * Maps the quadrature nodes from reference triangles to the actual geometry.
   * @param[in] x1 Coordinates of the first node of the test element.
   * @param[in] x2 Coordinates of the second node of the test element.
   * @param[in] x3 Coordinates of the third node of the test element.
   * @param[in] y1 Coordinates of the first node of the trial element.
   * @param[in] y2 Coordinates of the second node of the trial element.
   * @param[in] y3 Coordinates of the third node of the trial element.
   * @param[in] y4 Coordinates of the fourth node of the trial element.
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

  /**
   * Checks if a triangle and a tetrahedron are sufficiently separated to apply
   * a standard quadrature scheme for the computation of the nearfield entries
   * of the initial operator.
   *
   * The routine computes the centers of the triangle and tetrahedron, the
   * distance of these centers and the radii of the triangle and tetrahedron
   * (largest distance between the center and a vertex). If the distance is
   * greater than the sum of the radii, the triangle and tetrahedron are
   * considered to be separated, otherwise not.
   * @param[in] x1  Node 1 of the triangle.
   * @param[in] x2  Node 2 of the triangle.
   * @param[in] x3  Node 3 of the triangle.
   * @param[in] y1  Node 1 of the tetrahedron.
   * @param[in] y2  Node 2 of the tetrahedron.
   * @param[in] y3  Node 3 of the tetrahedron.
   * @param[in] y4  Node 4 of the tetrahedron.
   */
  bool triangle_and_tetrahedron_are_separated(
    const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3,
    const linear_algebra::coordinates< 3 > & y4 ) const;

  kernel_type * _kernel;  //!< Kernel used for the evaluation of the temporal
                          //!< antiderivatives.
  test_space_type * _test_space;  //!< Boundary element test space. Has to be a
                                  //!< @ref distributed_fast_spacetime_be_space.
  trial_space_type * _trial_space;  //!< Finite element trial space. Has to be a
                                    //!< @ref fe_space.
  mesh::volume_space_cluster_tree *
    _space_source_tree;  //!< Space cluster tree corresponding to the volume
  //!< mesh associated with the trial fe space.
  int _order_regular_tri;    //!< Triangle quadrature order for the regular
                             //!< integrals. Polynomials on the triangle up to
                             //!< this order are integrated exactly.
  int _order_regular_tetra;  //!< Tetrahedron quadrature order for the regular
                             //!< integrals. Polynomials on the tetrahedron up
                             //!< to this order are integrated exactly.
  int _order_regular_line;   //!< Line quadrature order for the regular
                             //!< integrals. Polynomials on a 1D line up to
                             //!< this order are integrated exactly.
  lo _n_recursions_singular_integrals;  //!< Maximal number of recursive
                                        //!< refinements for the computation of
                                        //!< singular matrix entries.
  int _temp_order;  //!< Degree of Lagrange interpolation polynomials in time
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
    6 };  //!< Number of simplices which have to be considered for quadrature
          //!< of integrals with singular kernels, sorted by configuration
          //!< (disjoint, shared vertex, shared edge, identical).
  int _my_rank;  //!< MPI rank of the executing process.
  const MPI_Comm *
    _comm;  //!< MPI communicator associated with the pFMM matrix.
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_INITIAL_BE_ASSEMBLER_H_ \
        */
