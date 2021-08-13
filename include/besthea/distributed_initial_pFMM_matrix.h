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

/** @file distributed_initial_pFMM_matrix.h
 * @brief Represents a matrix corresponding to the initial potential trace
 * operators approximated by the pFMM, distributed among a set of processes.
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_INITIAL_PFMM_MATRIX_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_INITIAL_PFMM_MATRIX_H_

#include "besthea/basis_tetra_p1.h"
#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/block_linear_operator.h"
#include "besthea/chebyshev_evaluator.h"
#include "besthea/distributed_block_vector.h"
#include "besthea/distributed_fast_spacetime_be_space.h"
#include "besthea/distributed_spacetime_cluster_tree.h"
#include "besthea/fe_space.h"
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/scheduling_time_cluster.h"
#include "besthea/settings.h"
#include "besthea/spacetime_heat_initial_m0_kernel_antiderivative.h"
#include "besthea/spacetime_heat_initial_m1_kernel_antiderivative.h"
#include "besthea/timer.h"
#include "besthea/tree_structure.h"
#include "besthea/vector.h"
#include "besthea/volume_space_cluster.h"
#include "besthea/volume_space_cluster_tree.h"

#include <mpi.h>

namespace besthea {
  namespace linear_algebra {
    template< class kernel_type, class target_space, class source_space >
    class distributed_initial_pFMM_matrix;
  }
}

/**
 * Class representing a matrix corresponding to a trace operator of initial
 * potential trace operators approximated by the pFMM method.
 */
template< class kernel_type, class target_space, class source_space >
class besthea::linear_algebra::distributed_initial_pFMM_matrix
  : public besthea::linear_algebra::linear_operator {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Wraps the mapped quadrature point so that they can be private for OpenMP
   * threads
   * @note This is needed to compute the quadratures of Chebyshev polynomials
   * in space (since these are no longer stored and hence not assembled by
   * the corresponding assembler)
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

  /**
   * Default constructor.
   */
  distributed_initial_pFMM_matrix( )
    : _my_rank( -1 ),
      _distributed_spacetime_target_tree( nullptr ),
      _space_source_tree( nullptr ),
      _temp_order( 5 ),
      _spat_order( 5 ),
      _m2l_integration_order( _spat_order ),
      _spat_contribution_size(
        ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) )
        / 6 ),
      _contribution_size( ( _temp_order + 1 ) * _spat_contribution_size ),
      _chebyshev( _spat_order ),
      _lagrange( _temp_order ),
      _alpha( 1.0 ),
      _cheb_nodes_sum_coll(
        ( _m2l_integration_order + 1 ) * ( _m2l_integration_order + 1 ) ),
      _all_poly_vals_mult_coll( ( _spat_order + 1 ) * ( _spat_order + 1 )
        * ( _m2l_integration_order + 1 ) * ( _m2l_integration_order + 1 ) ) {
  }

  distributed_initial_pFMM_matrix(
    const distributed_initial_pFMM_matrix & that )
    = delete;

  /**
   * Destructor
   */
  virtual ~distributed_initial_pFMM_matrix( ) {
    // delete all allocated nearfield matrices
    for ( auto matrix_vector : _clusterwise_nearfield_matrices ) {
      for ( auto matrix : matrix_vector ) {
        delete matrix;
      }
    }
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x using block vectors.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   * @note This routine is just a dummy here. Please use the corresponding
   * version with distributed block vectors.
   */
  virtual void apply( const vector_type & x, vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x using block vectors.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transposition.
   * @param[in] alpha
   * @param[in] beta
   * @warning Transposition is not supported.
   */
  void apply( const vector_type & x, distributed_block_vector & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /**
   * Sets the MPI communicator associated with the distributed pFMM matrix and
   * the rank of the executing process.
   * @param[in] comm  MPI communicator to be set.
   */
  void set_MPI_communicator( const MPI_Comm * comm ) {
    _comm = comm;
    MPI_Comm_rank( *_comm, &_my_rank );
  }

  /**
   * This routine initializes several data required for the execution of the
   * FMM.
   *
   * In particular, nearfield and interaction lists are initialized (first on a
   * temporal level and then for all clusters in the space-time tree) and the
   * container to store the nearfield matrices is prepared.
   * @param[in] spacetime_target_tree  The distributed spacetime tree used as
   * target tree.
   * @param[in] space_source_tree The space cluster tree used as source tree.
   */
  void initialize_fmm_data(
    mesh::distributed_spacetime_cluster_tree * spacetime_target_tree,
    mesh::volume_space_cluster_tree * space_source_tree );

  /**
   * Sets the heat conductivity parameter.
   * @param[in] alpha Heat conductivity.
   */
  void set_alpha( sc alpha ) {
    _alpha = alpha;
  }

  /**
   * Sets the dimension of the matrix.
   * @param[in] dim_domain Number of columns in the matrix.
   * @param[in] dim_range Number of rows in the matrix.
   * @note the member variables which are set are inherited from
   * @ref linear_operator.
   * @note The block structure in time in the rows is ignored. In particular,
   * @p _dim_range is the total dimension of the range.
   */
  void resize( lo dim_domain, lo dim_range ) {
    _dim_domain = dim_domain;
    _dim_range = dim_range;
  }

  /**
   * Sets the order of the Lagrange and Chebyshev polynomials and the quadrature
   * orders for numerical integration.
   * @param[in] spat_order Order of the Chebyshev polynomials.
   * @param[in] temp_order Order of the Lagrange polynomials.
   * @param[in] order_regular Quadrature order.
   */
  void set_orders( int spat_order, int temp_order, int order_regular_tri,
    int order_regular_tetra, int order_regular_line ) {
    _spat_order = spat_order;
    _temp_order = temp_order;
    _order_regular_tri = order_regular_tri;
    _order_regular_tetra = order_regular_tetra;
    _order_regular_line = order_regular_line;
    _spat_contribution_size
      = ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
    _contribution_size = ( _temp_order + 1 ) * _spat_contribution_size;
    _chebyshev.set_order( spat_order );
    _lagrange.set_order( temp_order );
  }

  /**
   * Sets the integration order for the m2l coefficients.
   * @param[in] m2l_integration_order M2L integration order.
   */
  void set_m2l_integration_order( int m2l_integration_order ) {
    _m2l_integration_order = m2l_integration_order;
  }

  /**
   * Resizes @ref _clusterwise_nearfield_matrices appropriately before the
   * initialization of the matrices.
   * @note This routine has to be called after
   * @ref initialize_nearfield_and_interaction_lists
   */
  void resize_nearfield_matrix_container( );

  /**
   * Creates a nearfield matrix for two clusters
   * @param[in] leaf_index  Index of the local leaf cluster in the target tree,
   * which acts as the target.
   * @param[in] source_index  Index of the source cluster in the nearfield list
   *                          of the target cluster.
   */
  full_matrix * create_nearfield_matrix( lou leaf_index, lou source_index );

  /**
   * Compute the spatial m2m coefficients for all local spatial levels.
   * @todo Check this!
   */
  void initialize_spatial_m2m_coeffs( );

  /**
   * Compute Chebyshev nodes and evaluate them.
   */
  void compute_chebyshev( );

  /*
   * Prints information about the underlying distributed spacetime cluster tree
   * and the operations which have to be applied.
   * @param[in] root_process  Process responsible for printing the information.
   * @param[in] print_tree_information  If true, information is printed for the
   *                                    distributed spacetime cluster tree
   *                                    corresponding to the matrix.
   */
  // void print_information(
  //   const int root_process, const bool print_tree_information = false );

 private:
  void determine_interacting_time_clusters(
    mesh::scheduling_time_cluster & current_cluster );

  void initialize_nearfield_and_interaction_lists( );

  void compute_moments_upward_path( const vector & sources,
    mesh::volume_space_cluster * current_cluster ) const;

  void apply_all_m2l_operations( ) const;

  void evaluate_local_contributions_downward_path(
    mesh::scheduling_time_cluster * current_cluster,
    distributed_block_vector & y_pFMM ) const;

  void apply_s2m_operation(
    const vector & sources, mesh::volume_space_cluster * leaf ) const;

  void apply_grouped_m2m_operation(
    mesh::volume_space_cluster * parent_cluster ) const;

  void apply_m2l_operation( const mesh::volume_space_cluster * s_src_cluster,
    mesh::general_spacetime_cluster * st_tar_cluster ) const;

  void call_l2l_operations(
    mesh::scheduling_time_cluster * child_cluster ) const;

  /**
   * Applies the L2L operations for the given parent cluster and all its
   * children for a given temporal configuration.
   * @param[in] parent_cluster  Considered spacetime parent cluster.
   * @param[in] child_configuration Indicates for which children the l2l
   *                                operations are executed:
   *                                - 0: left children w.r.t. to time.
   *                                - 1: right chilren w.r.t. to time.
   */
  void apply_grouped_l2l_operation(
    mesh::general_spacetime_cluster * parent_cluster,
    slou child_configuration ) const;

  /**
   * Applies the temporal l2l operation to a parent's local contribution and
   * adds the result to a given array.
   * @param[in] parent_local_contribution Array containing the moments of the
   *                                      child cluster.
   * @param[in] temporal_l2l_matrix Matrix used for the temporal l2l operation.
   * @param[in,out] child_local_contribution  Array to which the result is
   *                                          added.
   */
  void apply_temporal_l2l_operation( const sc * parent_local_contribution,
    const full_matrix & temporal_l2l_matrix,
    sc * child_local_contribution ) const;

  /**
   * Applies the spatial l2l operation to a parent's local contribution and adds
   * the result to a given array.
   * @param[in] parent_local Array containing the local
   *                                      contributions of the parent cluster.
   * @param[in] n_space_div_parent  Number of refinements in space executed for
   *                                the parent cluster.
   * @param[in] octant  Configuration of the child cluster with respect to its
   *                    parent in space.
   * @param[in,out] child_local  Array to which the result is
   *                                          added.
   * @note  @p n_space_div_parent and @p octant are used to determine the
   *        appropriate l2l coefficients for the operation.
   */
  void apply_spatial_l2l_operation( const sc * parent_local,
    const lo n_space_div_parent, const slou octant, sc * child_local ) const;

  /**
   * Calls all L2T operations associated with a given scheduling time cluster.
   * @param[in] t_cluster  Considered scheduling time cluster.
   * @param[in,out] output_vector Block vector to which the results are added.
   */
  void call_l2t_operations( mesh::scheduling_time_cluster * t_cluster,
    distributed_block_vector & output_vector ) const;

  /**
   * Applies the appropriate L2T operation for the given target cluster
   * depending on the boundary integral operator and writes the result to the
   * appropriate part of the output vector.
   * @param[in] cluster  Considered spacetime cluster.
   * @param[in,out] output_vector Global result vector to which the result of
   *                              the operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   */
  void apply_l2t_operation( const mesh::general_spacetime_cluster * cluster,
    distributed_block_vector & output_vector ) const;

  /**
   * Applies the L2T operation for the given target cluster for p0 basis
   * functions and writes the result to the appropriate part of the output
   * vector.
   * @param[in] cluster  Considered spacetime cluster.
   * @param[in,out] output_vector Global result vector to which the result of
   *                              the operation is added.
   * @todo This is a duplicate of
   * @ref distributed_pFMM_matrix::apply_l2t_operations_p0. Can we restructure
   * the code to get rid of duplication.
   */
  void apply_l2t_operation_p0( const mesh::general_spacetime_cluster * cluster,
    distributed_block_vector & output_vector ) const;

  /**
   * Applies the L2T operation for the given target cluster for p1 basis
   * functions and normal derivatives of spatial polynomials (for operator M1
   * (normal trace of initial potential)) and writes the result to the
   * appropriate part of the output vector.
   * @param[in] cluster  Considered spacetime cluster.
   * @param[in,out] output_vector Global result vector to which the result of
   *                              the operation is added.
   * @todo This is a duplicate of
   * @ref distributed_pFMM_matrix::apply_l2t_operations_p1_normal_drv. Can we
   * restructure the code to get rid of duplication.
   */
  void apply_l2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * cluster,
    distributed_block_vector & output_vector ) const;

  void compute_coupling_coeffs_initial_op( const vector_type & tar_time_nodes,
    const sc spat_half_size, const sc spat_center_diff,
    vector_type & buffer_for_gaussians, vector_type & coupling_coeffs ) const;

  void compute_chebyshev_quadrature_p1_volume(
    const mesh::volume_space_cluster * source_cluster,
    full_matrix & T_vol ) const;

  /**
   * Compute quadrature of the Lagrange polynomials and p0 basis functions for
   * the temporal part of a spacetime cluster
   * @param[in] source_cluster  Cluster for whose temporal component the
   *                            quadratures are computed.
   * @param[out] L  Full matrix where the quadratures are stored. The temporal
   *                elements of the cluster vary along the columns, the order
   *                of the polynomial along the rows of the matrix.
   * @todo This is a duplicate of
   * @ref distributed_pFMM_matrix::compute_lagrange_quadrature. Can we
   * restructure the code to get rid of duplication.
   */
  void compute_lagrange_quadrature(
    const mesh::general_spacetime_cluster * source_cluster,
    full_matrix & L ) const;

  /**
   * Compute quadrature of the Chebyshev polynomials and p0 basis functions for
   * the spatial part of a spacetime cluster
   * @param[in] source_cluster  Cluster for whose spatial component the
   *                            quadratures are computed.
   * @param[out] T  Full matrix where the quadratures are stored. The elements
   *                of the cluster vary along the rows, the order of the
   *                polynomial along the columns of the matrix.
   * @todo This is almost a duplicate of
   * @ref distributed_pFMM_matrix::compute_chebyshev_quadrature_p0. (different
   * spatial padding is used) Can we restructure the code to get rid of
   * duplication.
   */
  void compute_chebyshev_quadrature_p0(
    const mesh::general_spacetime_cluster * source_cluster,
    full_matrix & T ) const;

  /**
   * Computes quadrature of the normal derivatives of the Chebyshev polynomials
   * times p1 basis functions for the spatial part of a spacetime cluster.
   * @param[in] source_cluster  Cluster for whose spatial component the
   *                            quadratures are computed.
   * @param[out] T_drv  Full matrix where the quadratures are stored. The
   * nodes of the cluster vary along the rows, the order of the polynomial
   * along the columns of the matrix.
   * @todo This is almost a duplicate of
   * @ref distributed_pFMM_matrix::compute_normal_drv_chebyshev_quadrature_p1.
   * (different spatial padding is used) Can we restructure the code to get rid
   * of duplication.
   */
  void compute_normal_drv_chebyshev_quadrature_p1(
    const mesh::general_spacetime_cluster * source_cluster,
    full_matrix & T_drv ) const;

  /**
   * Initializes quadrature structures used to integrate Chebyshev polynomials
   * on triangles.
   *
   * The quadrature points and weights on the reference triangle are
   * initialized. The other structures used for integration of Chebyshev
   * polynomials are resized appropriately.
   *
   * @param[out] my_quadrature Wrapper holding quadrature data.
   * @todo This is redundant. Can we restructure the code?
   */
  void init_quadrature_polynomials( quadrature_wrapper & my_quadrature ) const;

  void init_quadrature_polynomials_tetrahedron(
    quadrature_wrapper & my_quadrature ) const;

  void tetrahedron_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & x4,
    quadrature_wrapper & my_quadrature ) const;

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
   * @todo Check if documentation makes sense in this context.
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
   * @todo This is redundant! Can we restructure the code?
   * @todo rename the routine to better describe its action?
   */
  void cluster_to_polynomials( quadrature_wrapper & my_quadrature, sc x_start,
    sc x_end, sc y_start, sc y_end, sc z_start, sc z_end ) const;

  /*
   * Counts the number of all FMM operations levelwise
   * @note m2m and l2l operations are counted for the levels of the children
   */
  // void count_fmm_operations_levelwise( std::vector< lou > & n_s2m_operations,
  //   std::vector< lou > & n_m2m_operations,
  //   std::vector< lou > & n_m2l_operations,
  //   std::vector< lou > & n_l2l_operations,
  //   std::vector< lou > & n_l2t_operations ) const;

  const MPI_Comm *
    _comm;       //!< MPI communicator associated with the pFMM matrix.
  int _my_rank;  //!< MPI rank of current process.
  mesh::distributed_spacetime_cluster_tree *
    _distributed_spacetime_target_tree;  //!< part of a distributed tree
                                         //!< hierarchically decomposing the
                                         //!< target space-time domain.
  mesh::volume_space_cluster_tree *
    _space_source_tree;  //!< Spatial cluster tree decomposing the source volume
                         //!< mesh.

  std::vector< sc >
    _maximal_spatial_paddings;  //!< Vector of maximal paddings
                                //!< at each spatial level. (levelwise maximum
                                //!< of the spatial paddings in the target and
                                //!< source tree)

  std::vector< mesh::scheduling_time_cluster * >
    _time_clusters_for_m2l;  //!< Time clusters in the scheduling tree structure
                             //!< corresponding to the target tree for which M2L
                             //!< have to be executed.
  std::vector< mesh::scheduling_time_cluster * >
    _time_clusters_for_nf;  //!< Time clusters in the scheduling tree structure
                            //!< corresponding to the target tree for which
                            //!< nearfield operations have to be
                            //!< executed.

  std::vector< vector_type >
    _m2m_coeffs_s_left;  //!< left spatial m2m matrices stored levelwise.
  std::vector< vector_type >
    _m2m_coeffs_s_right;  //!< right spatial m2m matrices stored levelwise.

  int _temp_order;  //!< degree of interpolation polynomials in time for pFMM.
  int _spat_order;  //!< degree of Chebyshev polynomials for expansion in
                    //!< space in pFMM.
  int _order_regular_tri;  //!< Triangle quadrature order for the regular
                           //!< integrals. Polynomials on the triangle up to
                           //!< this order are integrated exactly. (used e.g. in
                           //!< computation of l2t matrices)
  int _order_regular_tetra;    //!< Tetrahedron quadrature order for the regular
                               //!< integrals. Polynomials on the tetrahedron up
                               //!< to this order are integrated exactly. (used
                               //!< e.g. in computation of s2m matrices)
  int _order_regular_line;     //!< Line quadrature order for the regular
                               //!< integrals. Polynomials on a 1D line up to
                               //!< this order are integrated exactly. (used
                               //!< e.g. in computation of l2t matrices)
  int _m2l_integration_order;  //!< _m2l_integration_order + 1 quadrature
                               //!< points are used for the approximation of
                               //!< the m2l coefficients.
  int _spat_contribution_size;  //!< Spatial size of a contribution. It is
                                //!< _spat_order + 3 choose 3
  int _contribution_size;       //!< Size of a contribution (moment or local
                           //!< contribution) of a single spacetime cluster.
  mutable bem::chebyshev_evaluator
    _chebyshev;  //!< Evaluator of the Chebyshev polynomials.

  mutable bem::lagrange_interpolant
    _lagrange;  //!< Evaluator of the Lagrange polynomials.

  sc _alpha;  //!< Heat conductivity.

  std::vector< sc, besthea::allocator_type< sc > >
    _cheb_nodes_sum_coll;  //!< summed Chebyshev nodes for collapsed loop,
                           //!< aligned

  std::vector< sc, besthea::allocator_type< sc > >
    _all_poly_vals_mult_coll;  //!< summed Chebyshev nodes for collapsed loop,
                               //!< aligned

  mutable std::vector< full_matrix >
    _aux_buffer_0;  //!< Auxiliary vector used to store intermediate results in
                    //!< M2L operations.
  mutable std::vector< full_matrix >
    _aux_buffer_1;  //!< Auxiliary vector used to store intermediate results in
                    //!< M2L operations.

  std::vector< std::pair< mesh::general_spacetime_cluster *,
    std::vector< mesh::volume_space_cluster * > > >
    _interaction_list_vector;  //!< A vector of pairs. Each pair consists of a
                               //!< target space-time cluster and an associated
                               //!< interaction list of volume space clusters.

  std::vector< std::pair< mesh::general_spacetime_cluster *,
    std::vector< mesh::volume_space_cluster * > > >
    _nearfield_list_vector;  //!< A vector of pairs. Each pair consists of a
                             //!< target space-time cluster and an associated
                             //!< nearfield list of volume space clusters.

  std::vector< std::vector< full_matrix * > >
    _clusterwise_nearfield_matrices;  //!< nearfield
                                      //!< matrices for all
                                      //!< the pairs of
                                      //!< leaf clusters and
                                      //!< nearfield
                                      //!< clusters in the
                                      //!< nearfield list
                                      //!< vector.
};

/** Typedef for the distributed initial potential M0 p0-p1 PFMM matrix */
typedef besthea::linear_algebra::distributed_initial_pFMM_matrix<
  besthea::bem::spacetime_heat_initial_m0_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >
  distributed_initial_pFMM_matrix_heat_m0_p0p1;

/** Typedef for the distributed initial potential M1 p1-p1 PFMM matrix */
typedef besthea::linear_algebra::distributed_initial_pFMM_matrix<
  besthea::bem::spacetime_heat_initial_m1_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::fe_space< besthea::bem::basis_tetra_p1 > >
  distributed_initial_pFMM_matrix_heat_m1_p1p1;

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_INITIAL_PFMM_MATRIX_H_ */
