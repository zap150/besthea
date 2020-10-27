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

/** @file distributed_pFMM_matrix.h
 * @brief Represents matrix approximated by the pFMM, distributed among a set
 * of processes
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_PFMM_MATRIX_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_PFMM_MATRIX_H_

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/block_linear_operator.h"
#include "besthea/block_vector.h"
#include "besthea/chebyshev_evaluator.h"
#include "besthea/distributed_fast_spacetime_be_space.h"
#include "besthea/distributed_spacetime_cluster_tree.h"
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/local_vector_routines.h"
#include "besthea/settings.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/tree_structure.h"
#include "besthea/vector.h"

#include <list>
#include <mpi.h>
#include <unordered_map>

namespace besthea {
  namespace linear_algebra {
    template< class kernel_type, class target_space, class source_space >
    class distributed_pFMM_matrix;
  }
}

/**
 * Class representing a matrix approximated by the pFMM method.
 */
template< class kernel_type, class target_space, class source_space >
class besthea::linear_algebra::distributed_pFMM_matrix
  : public besthea::linear_algebra::block_linear_operator {
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

  /**
   * Default constructor.
   */
  distributed_pFMM_matrix( )
    : _my_rank( -1 ),
      _distributed_spacetime_tree( nullptr ),
      _scheduling_tree_structure( nullptr ),
      _temp_order( 5 ),
      _spat_order( 5 ),
      _m2l_integration_order( _spat_order ),
      _spat_contribution_size(
        ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) )
        / 6 ),
      _contribution_size( ( _temp_order + 1 ) * _spat_contribution_size ),
      _chebyshev( _spat_order ),
      _lagrange( _temp_order ),
      _alpha( 1.0 ) {
  }

  distributed_pFMM_matrix( const distributed_pFMM_matrix & that ) = delete;

  /**
   * Destructor
   */
  virtual ~distributed_pFMM_matrix( ) {
    for ( auto it = _clusterwise_nearfield_matrices.begin( );
          it != _clusterwise_nearfield_matrices.end( ); ++it ) {
      // loop over all nearfield matrices associated with a given spacetime
      // cluster and delete them.
      for ( auto matrix : it->second ) {
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
   */
  virtual void apply( const block_vector & x, block_vector & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x using block vectors for
   * single, double and adjoint double layer operators.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   */
  void apply_sl_dl( const block_vector & x, block_vector & y, bool trans,
    sc alpha, sc beta ) const;

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
   * Sets the underlying distributed spacetime tree and tree structure for
   * scheduling the operations. The size of the clusterwise nearfield
   * matrix container is set appropriately.
   * @param[in] distributed_spacetime_tree  The distributed spacetime tree. Its
   *                                        distribution tree is used as the
   *                                        scheduling tree structure.
   */
  void set_trees(
    mesh::distributed_spacetime_cluster_tree * distributed_spacetime_tree );

  /**
   * Sets the heat conductivity parameter.
   * @param[in] alpha Heat conductivity.
   */
  void set_alpha( sc alpha ) {
    _alpha = alpha;
  }

  /**
   * Sets the dimension of the matrix.
   * @param[in] block_dim Block dimension.
   * @param[in] dim_domain Number of columns in a block.
   * @param[in] dim_range Number of rows in a block.
   * @note the member variables which are set are inherited from
   *       @ref block_linear_operator.
   */
  void resize( lo block_dim, lo dim_domain, lo dim_range ) {
    _block_dim = block_dim;
    _dim_domain = dim_domain;
    _dim_range = dim_range;
  }

  /**
   * Sets the order of the Lagrange and Chebyshev polynomials and the quadrature
   * order for numerical integration.
   * @param[in] spat_order Order of the Chebyshev polynomials.
   * @param[in] temp_order Order of the Lagrange polynomials.
   * @param[in] order_regular Quadrature order.
   */
  void set_order( int spat_order, int temp_order, int order_regular ) {
    _spat_order = spat_order;
    _temp_order = temp_order;
    _order_regular = order_regular;
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
   * Fills the 4 lists used for scheduling the FMM operations by adding pointers
   * to clusters assigned to the process with id @p _my_process_id . In addition
   * it determines all pairs of clusters and process ids from which data is
   * received, and initializes the data in the scheduling time clusters which is
   * used to check the dependencies.
   * @note All lists are constructed anew, existing values are overwritten.
   * @note The clusters in the m_list are sorted using the comparison operator
   *       @ref mesh::tree_structure::compare_clusters_bottom_up_right_2_left,
   *       the others using
   *       @ref mesh::tree_structure::compare_clusters_top_down_right_2_left.
   */
  void prepare_fmm( );

  /**
   * Creates a nearfield matrix for two clusters
   * @param[in] leaf_index  Index of the local leaf cluster, which acts as the
   *                        target.
   * @param[in] source_index  Index of the source cluster in the nearfield list
   *                          of the target cluster.
   */
  full_matrix * create_nearfield_matrix( lou leaf_index, lou source_index );

  /**
   * Compute the spatial m2m coefficients for all local spatial levels.
   */
  void compute_spatial_m2m_coeffs( );

  /**
   * Pseudo-parallel FGMRES based on the implementation in MKL.
   * @param[in] rhs Right-hand side vector (cannot be const due to MKL).
   * @param[out] solution Solution vector.
   * @param[in,out] relative_residual_error Stopping criterion measuring
   * decrease of |Ax-b|/|b|, actual value on exit.
   * @param[in,out] n_iterations Maximal number of iterations, actual value on
   * exit.
   * @param[in] n_iterations_until_restart Maximal number of iterations before
   * restart.
   * @param[in] trans Use transpose of this.
   * @param[in] root_id Id of the process which executes the sequential steps
   *                    of the FGMRES implementation in MKL.
   * @todo implement a true parallel version.
   */
  bool mkl_fgmres_solve_parallel( const block_vector & rhs,
    block_vector & solution, sc & relative_residual_error, lo & n_iterations,
    lo n_iterations_until_restart = 0, bool trans = false,
    int root_id = 0 ) const;

  /**
   * Prints information about the underlying distributed spacetime cluster tree
   * and the operations which have to be applied.
   * @param[in] root_process  Process responsible for printing the information.
   */
  void print_information( const int root_process );

 private:
  /**
   * Calls all S2M operations associated with a given scheduling time cluster.
   * @param[in] sources Global sources containing the once used for the S2M
   *                    operation.
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void call_s2m_operations( const block_vector & sources,
    mesh::scheduling_time_cluster * time_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Applies the appropriate S2M operation for the given source cluster and
   * sources depending on the boundary integral operator.
   * @param[in] source_vector Global sources containing the once used for the
   *                          S2M operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   */
  void apply_s2m_operation( const block_vector & source_vector,
    mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Applies the S2M operation for the given source cluster and sources for
   * p0 basis functions (for single layer and adjoint double layer operators)
   * @param[in] source_vector Global sources containing the once used for the
   *                          S2M operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   * @todo Use buffers instead of reallocating sources and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_s2m_operation_p0( const block_vector & source_vector,
    mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Applies the S2M operation for the given source cluster and sources for
   * p1 basis functions and normal derivatives of spatial polynomials (for
   * double layer operator and hypersingular operator)
   * @param[in] source_vector Global sources containing the once used for the
   *                          S2M operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   * @todo Use buffers instead of reallocating sources and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_s2m_operations_p1_normal_drv( const block_vector & source_vector,
    mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Calls all M2M operations associated with a given scheduling time cluster.
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void call_m2m_operations( mesh::scheduling_time_cluster * time_cluster,
    bool verbose, const std::string & verbose_file ) const;

  /**
   * Applies the M2M operations for the given parent cluster and all its
   * children for a given temporal configuration.
   * @param[in] parent_cluster  Considered spacetime parent cluster.
   * @param[in] child_configuration Indicates for which children the m2m
   *                                operations are executed:
   *                                - 0: left children w.r.t. to time.
   *                                - 1: right chilren w.r.t. to time.
   */
  void apply_grouped_m2m_operation(
    mesh::general_spacetime_cluster * parent_cluster,
    slou child_configuration ) const;

  /**
   * Applies the temporal m2m operation to a child_moment and adds the result
   * to the parent moment.
   * @param[in] child_moment  Array containing the moments of the child cluster.
   * @param[in] temporal_m2m_matrix Matrix used for the temporal m2m operation.
   * @param[in,out] parent_moment Array to which the result is added.
   */
  void apply_temporal_m2m_operation( const sc * child_moment,
    const full_matrix & temporal_m2m_matrix, sc * parent_moment ) const;

  /**
   * Applies the spatial m2m operation to a child_moment and adds the result
   * to a given array.
   * @param[in] child_moment  Array containing the moments of the child cluster.
   * @param[in] n_space_div_parent  Number of refinements in space executed for
   *                                the parent cluster.
   * @param[in] octant  Configuration of the child cluster with respect to its
   *                    parent in space.
   * @param[in,out] output_array  Array to which the result is added.
   * @note  @p n_space_div_parent and @p octant are used to determine the
   *        appropriate m2m coefficients for the operation.
   */
  void apply_spatial_m2m_operation( const sc * child_moment,
    const lo n_space_div_parent, const slou octant, sc * output_array ) const;

  /**
   * Calls all M2L operations associated with a given pair of scheduling time
   * clusters.
   * @param[in] src_cluster Scheduling time cluster which acts as source in M2L.
   * @param[in] tar_cluster Scheduling time cluster which acts as target in M2L.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void call_m2l_operations( mesh::scheduling_time_cluster * src_cluster,
    mesh::scheduling_time_cluster * tar_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Applies the M2L operation for given source and target clusters.
   * @param[in] src_cluster Spacetime source cluster for the M2L operation.
   * @param[in,out] tar_cluster Spacetime target cluster for the M2L operation.
   * @todo add buffers instead of reallocation?
   */
  void apply_m2l_operation( const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const;

  /**
   * Calls all L2L operations associated with a given scheduling time cluster.
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void call_l2l_operations( mesh::scheduling_time_cluster * time_cluster,
    bool verbose, const std::string & verbose_file ) const;

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
   * Applies the temporal l2l operation to a child_moment and adds the result
   * to the parent moment.
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
   * Applies the spatial l2l operation to a child_moment and adds the result
   * to a given array.
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
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in,out] output_vector Block vector to which the results are added.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void call_l2t_operations( mesh::scheduling_time_cluster * time_cluster,
    block_vector & output_vector, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Applies the appropriate L2T operation for the given source cluster
   * depending on the boundary integral operator and writes the result to the
   * appropriate part of the output vector.
   * @param[in] cluster  Considered spacetime cluster.
   * @param[in] output_vector Global result vector to which the result of the
   *                          operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_l2t_operation( const mesh::general_spacetime_cluster * cluster,
    block_vector & output_vector ) const;

  /**
   * Applies the L2T operation for the given source cluster for p0 basis
   * functions and writes the result to the appropriate part of the output
   * vector.
   * @param[in] cluster  Considered spacetime cluster.
   * @param[in] output_vector Global result vector to which the result of the
   *                          operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_l2t_operation_p0( const mesh::general_spacetime_cluster * cluster,
    block_vector & output_vector ) const;

  /**
   * Applies the L2T operation for the given source cluster for p1 basis
   * functions and normal derivatives of spatial polynomials (for adjoint double
   * layer operator and hypersingular operator) functions and writes the result
   * to the appropriate part of the output vector.
   * @param[in] cluster  Considered spacetime cluster.
   * @param[in] output_vector Global result vector to which the result of the
   *                          operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_l2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * cluster,
    block_vector & output_vector ) const;

  /**
   * Executes all nearfield operations associated with a given scheduling time
   * cluster.
   * @param[in] cluster Time cluster whose associated nearfield operations
   *                    are executed.
   * @param[in] sources Global sources containing the once used for the
   *                    nearfield operation.
   * @param[in] trans If true, the transposed nearfield matrices are applied
   *                  otherwise the standard nearfield matrices.
   * @param[in,out] output_vector Vector to which the results are added.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void apply_nearfield_operations(
    const mesh::scheduling_time_cluster * cluster, const block_vector & sources,
    bool trans, block_vector & output_vector, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Calls MPI_Testsome for an array of Requests to check for received data.
   * @param[in,out] array_of_requests Array containing the MPI requests which
   *                                  are checked.
   * @param[in,out] array_of_indices  Array in which the indices of the
   *                                  completed requests are stored. This is
   *                                  used as an input variable to avoid
   *                                  reallocation in each function call.
   * @param[in,out] outcount  Stores the number of Requests which are completed.
   * @param[in] verbose If true, the process lists all the received data, and
   *                    reports about the time needed to process it (in case
   *                    moments in the upward path were received)
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void check_for_received_data( MPI_Request * array_of_requests,
    int array_of_indices[], int & outcount, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Returns an iterator pointing to the next cluster in the l-list whose
   * dependencies are satisfied. In case a cluster is found the status is
   * updated. If no cluster is found the iterator points to the end of the list
   * and the status is not modified.
   * @param[in] l_list  A list containing the clusters of @ref _l_list whose
   *                    operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in the list this
   *                              iterator points to it. Else it points to the
   *                              end of the list.
   * @param[out] status Set to 2 if a cluster is found.
   */
  void find_cluster_in_l_list(
    std::list< mesh::scheduling_time_cluster * > & l_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const;

  /**
   * Returns an iterator pointing to the next cluster in the m-list whose
   * dependencies are satisfied. In case a cluster is found the status is
   * updated. If no cluster is found the iterator points to the end of the list
   * and the status is not modified.
   * @param[in] m_list  A list containing the clusters of @ref _m_list whose
   *                    operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in the list this
   *                              iterator points to it. Else it points to the
   *                              end of the list.
   * @param[out] status Set to 1 if a cluster is found.
   */
  void find_cluster_in_m_list(
    std::list< mesh::scheduling_time_cluster * > & m_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const;

  /**
   * Returns an iterator pointing to the next cluster in the m2l-list whose
   * dependencies are satisfied. In case a cluster is found the status is
   * updated. If no cluster is found the iterator points to the end of the list
   * and the status is not modified.
   * @param[in] m2l_list  A list containing the clusters of @ref _m2l_list whose
   *                      operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in the list this
   *                              iterator points to it. Else it points to the
   *                              end of the list.
   * @param[out] status Set to 3 if a cluster is found.
   */
  void find_cluster_in_m2l_list(
    std::list< mesh::scheduling_time_cluster * > & m2l_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const;

  /**
   * Updates dependency flags or sends moments for M2L operations.
   * @param[in] src_cluster Considered scheduling time cluster. If a cluster in
   *                        its send list is handled by a different process, the
   *                        moments are send to this process.
   * @param[in] verbose If true, the process reports about all initiated send
   *                    operations. (Updates of dependency flags are not
   *                    reported)
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void provide_moments_for_m2l( mesh::scheduling_time_cluster * src_cluster,
    bool verbose, const std::string & verbose_file ) const;

  /**
   * Updates dependency flags or sends moments for upward path.
   * @param[in] child_cluster Considered scheduling time cluster. If its parent
   *                          is handled by a different process, the processed
   *                          moments are send from the local copy of the parent
   *                          cluster to this process.
   * @param[in] verbose If true, the process reports about all initiated send
   *                    operations. (Updates of dependency flags are not
   *                    reported)
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void provide_moments_to_parents(
    mesh::scheduling_time_cluster * child_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Sends local contributions for downward path if necessary.
   * @param[in] parent_cluster Considered scheduling time cluster. If a child
   *                           of it is handled by a different process, the
   *                           local contributions are send to this process.
   * @param[in] verbose If true, the process reports about all initiated send
   *                    operations. (Updates of dependency flags are not
   *                    reported)
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void provide_local_contributions_to_children(
    mesh::scheduling_time_cluster * parent_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Starts all receive operations given by a vector of pairs of clusters and
   * process ids.
   * @param[in,out] array_of_requests The MPI_Requests of the non-blocking
   *                                  receive operations are stored in this
   *                                  array. It is expected to have at least
   *                                  the size of @p receive_vector .
   */
  void start_receive_operations( MPI_Request array_of_requests[] ) const;

  /**
   * Compute quadrature of the Chebyshev polynomials and p0 basis functions for
   * the spatial part of a spacetime cluster
   * @param[in] source_cluster  Cluster for whose spatial component the
   *                            quadratures are computed.
   * @param[out] T  Full matrix where the quadratures are stored. The elements
   *                of the cluster vary along the rows, the order of the
   *                polynomial along the columns of the matrix.
   */
  void compute_chebyshev_quadrature_p0(
    const mesh::general_spacetime_cluster * source_cluster,
    full_matrix & T ) const;

  /**
   * Compute quadrature of the normal derivatives of the Chebyshev polynomials
   * times p1 basis functions for the spatial part of a spacetime cluster.
   * @param[in] source_cluster  Cluster for whose spatial component the
   *                            quadratures are computed.
   * @param[out] T_drv  Full matrix where the quadratures are stored. The nodes
   *                    of the cluster vary along the rows, the order of the
   *                    polynomial along the columns of the matrix.
   */
  void compute_normal_drv_chebyshev_quadrature_p1(
    const mesh::general_spacetime_cluster * source_cluster,
    full_matrix & T_drv ) const;

  /**
   * Compute quadrature of the Lagrange polynomials and p0 basis functions for
   * the temporal part of a spacetime cluster
   * @param[in] source_cluster  Cluster for whose temporal component the
   *                            quadratures are computed.
   * @param[out] L  Full matrix where the quadratures are stored. The temporal
   *                elements of the cluster vary along the columns, the order
   *                of the polynomial along the rows of the matrix.
   */
  void compute_lagrange_quadrature(
    const mesh::general_spacetime_cluster * source_cluster,
    full_matrix & L ) const;

  /*!
   * Computes coupling coefficients for the spacetime m2l operation for one of
   * the three space dimensions implicitly given.
   * @param[in] src_time_nodes  Interpolation nodes in time for the source
   *                            cluster.
   * @param[in] tar_time_nodes  Interpolation nodes in time for the target
   *                            cluster.
   * @param[in] cheb_nodes  Chebyshev nodes of degree ( _spat_order + 1 )
   * @param[in] evaluated_chebyshev Vector of evaluated Chebyshev polynomials
   *                                with degree <= _spat_order at \p cheb_nodes
   *                                as given by
   *                           @ref besthea::bem::chebyshev_evaluator::evaluate.
   * @param[in] half_size Half size in space of the current clusters along the
   *                      dimension for which the coefficients are computed.
   * @param[in] center_diff The appropriate component of the difference vector
   *                        (target_center - source_center).
   * @param[in] buffer_for_gaussians  Vector with size >= ( _spat_order + 1 )^2
   *                                  * ( _temp_order + 1 )^2 to store
   *                                  intermediate results in the computation
   *                                  of the m2l coefficients.
   * @param[in,out] coupling_coeffs Vector with size >= ( _spat_order + 1 )^2
   *                                * ( _temp_order + 1 )^2 to store m2l
   *                                coefficients.
   */
  void compute_coupling_coeffs( const vector_type & src_time_nodes,
    const vector_type & tar_time_nodes, const vector_type & cheb_nodes,
    const vector_type & evaluated_chebyshev, const sc half_size,
    const sc center_diff, vector_type & buffer_for_gaussians,
    vector_type & coupling_coeffs ) const;

  /**
   * Traverses the m_list, l_list and m2l_list of the pFMM matrix and resets the
   * dependency data (i.e. the data used to determine if the operations of a
   * cluster are ready for execution).
   */
  void reset_scheduling_clusters_dependency_data( ) const;

  /**
   * Traverses the distribution tree recursively and resets the downward path
   * status of the clusters appropriately.
   * @param[in] root  Current cluster in the tree traversal.
   */
  void reset_downward_path_status_recursively(
    mesh::scheduling_time_cluster * root ) const;

  /**
   * Initializes quadrature structures.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   * @todo This is redundant! Can we restructure the code?
   */
  void init_quadrature_polynomials( quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps the quadrature nodes from the reference triangle to the actual
   * geometry.
   * @param[in] y1 Coordinates of the first node of the test element.
   * @param[in] y2 Coordinates of the second node of the test element.
   * @param[in] y3 Coordinates of the third node of the test element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   * @todo This is redundant! Can we restructure the code?
   */
  void triangle_to_geometry( const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3,
    quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps from the spatial cluster to the interval [-1, 1] where the Chebyshev
   * polynomials are defined.
   * @param[out] my_quadrature  Structure holding mapping from the cluster
   *                            to the interval [-1,1].
   * @param[in] x_start Border of the space cluster for which the Chebyshev
   *                    polynomials are evaluated.
   * @param[in] x_end Border of the space cluster for which the Chebyshev
   *                  polynomials are evaluated.
   * @param[in] y_start Border of the space cluster for which the Chebyshev
   *                    polynomials are evaluated.
   * @param[in] y_end Border of the space cluster for which the Chebyshev
   *                  polynomials are evaluated.
   * @param[in] z_start Border of the space cluster for which the Chebyshev
   *                    polynomials are evaluated.
   * @param[in] z_end Border of the space cluster for which the Chebyshev
   *                  polynomials are evaluated.
   * @todo This is redundant! Can we restructure the code?
   */
  void cluster_to_polynomials( quadrature_wrapper & my_quadrature, sc x_start,
    sc x_end, sc y_start, sc y_end, sc z_start, sc z_end ) const;

  /**
   * Returns the ratio of entries of the nearfield blocks of the pFMM matrix
   * handled by this process and entries of the global, non-approximated matrix.
   * @note Zeros in nearfield blocks and the full matrix are counted.
   * @warning If executed in parallel, the results should be added up to get
   * a meaningful result (due to the comparison with the global number of
   * entries).
   */
  sc compute_nearfield_ratio( );

  /**
   * Returns the ratio of non-zero entries of the nearfield blocks of the
   * pFMM matrix handled by this process and non-zero entries of the global,
   * non-approximated matrix.
   * @warning If executed in parallel, the results should be added up to get
   * a meaningful result (due to the comparison with the global number of
   * entries).
   */
  sc compute_nonzero_nearfield_ratio( );

  /**
   * Counts the number of all FMM operations levelwise
   * @note m2m and l2l operations are counted for the levels of the children
   */
  void count_fmm_operations_levelwise( std::vector< lou > & n_s2m_operations,
    std::vector< lou > & n_m2m_operations,
    std::vector< lou > & n_m2l_operations,
    std::vector< lou > & n_l2l_operations,
    std::vector< lou > & n_l2t_operations );

  /**
   * Task in the M-list
   * @param[in] x Input vector
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void m_list_task( const block_vector & x,
    mesh::scheduling_time_cluster * time_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Task in the L-list
   * @param[in] y_pFMM Output vector
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void l_list_task( block_vector & y_pFMM,
    mesh::scheduling_time_cluster * time_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Task in the M2L-list
   * @param[in] y_pFMM Output vector
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void m2l_list_task( block_vector & y_pFMM,
    mesh::scheduling_time_cluster * time_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * @param[in] current_index Index of the received data.
   * @param[in] current_cluster Processed scheduling_time_cluster.
   */
  void upward_path_task(
    lou current_index, mesh::scheduling_time_cluster * current_cluster ) const;

  const MPI_Comm *
    _comm;       //!< MPI communicator associated with the pFMM matrix.
  int _my_rank;  //!< MPI rank of current process.
  mesh::distributed_spacetime_cluster_tree *
    _distributed_spacetime_tree;  //!< part of a distributed tree hierarchically
                                  //!< decomposing the space-time domain.
  mesh::tree_structure *
    _scheduling_tree_structure;  //!< Temporal tree structure used for
                                 //!< scheduling the FMM operations

  std::unordered_map< mesh::general_spacetime_cluster *,
    std::vector< full_matrix * > >
    _clusterwise_nearfield_matrices;  //!< nearfield matrices for all the space-
                                      //!< time leaf clusters and their
                                      //!< nearfield clusters.

  std::list< mesh::scheduling_time_cluster * >
    _m_list;  //!< M-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster * >
    _m2l_list;  //!< M2L-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster * >
    _l_list;  //!< L2L-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster * >
    _n_list;  //!< N-list for the execution of the FMM.
  std::vector< std::pair< mesh::scheduling_time_cluster *, lo > >
    _receive_data_information;  //!< Contains for each data which has to be
                                //!< received the corresponding scheduling time
                                //!< cluster to which the data belongs and the
                                //!< id of the process which sends it. The data
                                //!< is either the moments or the local
                                //!< contributions of the associated cluster.
                                //!< The first @p _n_moments_to_receive_upward
                                //!< entries belong to moments which have to be
                                //!< received in the upward path of the FMM, the
                                //!< next @p _n_moments_to_receive_m2l entries
                                //!< to moments which have to be received for
                                //!< M2L operations and the remaining entries to
                                //!< local contributions which have to be
                                //!< received in the downward path.
  lou _n_moments_to_receive_upward;  //!< Number of grouped moments which have
                                     //!< to be received in the upward path of
                                     //!< the FMM.
  lou _n_moments_to_receive_m2l;  //!< Number of grouped moments which have to
                                  //!< be received for M2L operations.

  std::vector< vector_type >
    _m2m_coeffs_s_dim_0_left;  //!< left spatial m2m matrices along dimension 0
                               //!< stored levelwise.
  std::vector< vector_type >
    _m2m_coeffs_s_dim_0_right;  //!< right spatial m2m matrices along
                                //!< dimension 0 stored levelwise.
  std::vector< vector_type >
    _m2m_coeffs_s_dim_1_left;  //!< left spatial m2m matrices along dimension 1
                               //!< stored levelwise.
  std::vector< vector_type >
    _m2m_coeffs_s_dim_1_right;  //!< right spatial m2m matrices along
                                //!< dimension 1 stored levelwise.
  std::vector< vector_type >
    _m2m_coeffs_s_dim_2_left;  //!< left spatial m2m matrices along dimension 2
                               //!<  stored levelwise.
  std::vector< vector_type >
    _m2m_coeffs_s_dim_2_right;  //!< right spatial m2m matrices along
                                //!< dimension 2 stored levelwise.

  int _temp_order;  //!< degree of interpolation polynomials in time for pFMM.
  int _spat_order;  //!< degree of Chebyshev polynomials for expansion in
                    //!< space in pFMM.
  int _order_regular;  //!< Triangle quadrature order for the regular integrals.
                       //!< Used for computation of quadratures in S2M steps.
  int _m2l_integration_order;   //!< _m2l_integration_order + 1 quadrature
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
};

/** Typedef for the distributed single layer p0-p0 PFMM matrix */
typedef besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >
  distributed_pFMM_matrix_heat_sl_p0p0;

/** Typedef for the distributed double layer p0-p1 PFMM matrix */
typedef besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >
  distributed_pFMM_matrix_heat_dl_p0p1;

/** Typedef for the spatially adjoint double layer p1-p0 PFMM matrix */
typedef besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >
  distributed_pFMM_matrix_heat_adl_p1p0;

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_PFMM_MATRIX_H_ */
