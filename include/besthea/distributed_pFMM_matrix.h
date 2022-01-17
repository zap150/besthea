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

/** @file distributed_pFMM_matrix.h
 * @brief Represents matrix approximated by the pFMM, distributed among a set
 * of processes
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_PFMM_MATRIX_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_PFMM_MATRIX_H_

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/block_linear_operator.h"
#include "besthea/chebyshev_evaluator.h"
#include "besthea/distributed_block_vector.h"
#include "besthea/distributed_fast_spacetime_be_space.h"
#include "besthea/distributed_spacetime_cluster_tree.h"
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/matrix.h"
#include "besthea/settings.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/timer.h"
#include "besthea/tree_structure.h"
#include "besthea/vector.h"

#include <chrono>
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
  using timer_type = besthea::tools::timer;             //!< Timer type

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
      _alpha( 1.0 ),
      _aca_eps( 1e-5 ),
      _aca_max_rank( 500 ),
      _local_full_size( 0 ),
      _local_approximated_size( 0 ),
      _cheb_nodes_integrate( _m2l_integration_order + 1 ),
      _all_poly_vals_integrate(
        ( _spat_order + 1 ) * ( _m2l_integration_order + 1 ) ),
      _cheb_nodes_sum_coll(
        ( _m2l_integration_order + 1 ) * ( _m2l_integration_order + 1 ) ),
      _all_poly_vals_mult_coll( ( _spat_order + 1 ) * ( _spat_order + 1 )
        * ( _m2l_integration_order + 1 ) * ( _m2l_integration_order + 1 ) ),
      _verbose( false ),
      _measure_tasks( false ),
      _non_nf_op_count( 0 ),
      _m_task_times( omp_get_max_threads( ) ),
      _m2l_task_times( omp_get_max_threads( ) ),
      _l_task_times( omp_get_max_threads( ) ),
      _m2t_task_times( omp_get_max_threads( ) ),
      _s2l_task_times( omp_get_max_threads( ) ),
      _n_task_times( omp_get_max_threads( ) ),
      _m_subtask_times( omp_get_max_threads( ) ),
      _m2l_subtask_times( omp_get_max_threads( ) ),
      _l_subtask_times( omp_get_max_threads( ) ),
      _m2t_subtask_times( omp_get_max_threads( ) ),
      _s2l_subtask_times( omp_get_max_threads( ) ),
      _n_subtask_times( omp_get_max_threads( ) ),
      _mpi_send_m2l_m2t_or_s2l( omp_get_max_threads( ) ),
      _mpi_send_m_parent( omp_get_max_threads( ) ),
      _mpi_send_l_children( omp_get_max_threads( ) ),
      _mpi_recv_m2l_m2t_or_s2l( omp_get_max_threads( ) ),
      _mpi_recv_m_parent( omp_get_max_threads( ) ),
      _mpi_recv_l_children( omp_get_max_threads( ) ) {
  }

  distributed_pFMM_matrix( const distributed_pFMM_matrix & that ) = delete;

  /**
   * Destructor
   */
  virtual ~distributed_pFMM_matrix( ) {
    for ( auto it = _clusterwise_nf_matrices.begin( );
          it != _clusterwise_nf_matrices.end( ); ++it ) {
      // loop over all nearfield matrices associated with a given spacetime
      // cluster and delete them.
      for ( auto matrix : it->second ) {
        delete matrix;
      }
    }
    for ( auto it = _clusterwise_spat_adm_nf_matrices.begin( );
          it != _clusterwise_spat_adm_nf_matrices.end( ); ++it ) {
      // loop over all nearfield aca matrices associated with a given spacetime
      // cluster and delete them.
      for ( auto matrix : it->second ) {
        delete matrix;
      }
    }
  }

  /**
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
  virtual void apply( const block_vector & x, block_vector & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /**
   * @brief y = beta * y + alpha * (this)^trans * x using block vectors.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const distributed_block_vector & x,
    distributed_block_vector & y, bool trans = false, sc alpha = 1.0,
    sc beta = 0.0 ) const;

  /**
   * @brief y = beta * y + alpha * (this)^trans * x using distributed block
   * vectors for single, double and adjoint double layer operators.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   * @todo we should disable trans somehow, since it is not implemented
   * correctly.
   */
  void apply_sl_dl( const distributed_block_vector & x,
    distributed_block_vector & y, bool trans, sc alpha, sc beta ) const;

  /**
   * @brief y = beta * y + alpha * (this)^trans * x using distributed block
   * vectors for the hypersingular operator.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   *                  block matrix!).
   * @param[in] alpha
   * @param[in] beta
   * @todo we should disable trans somehow, since it is not implemented
   * correctly.
   */
  void apply_hs( const distributed_block_vector & x,
    distributed_block_vector & y, bool trans, sc alpha, sc beta ) const;

  /**
   * @brief Realizes one run of the distributed pFMM algorithm. (executing all
   * farfield operations and, if @p run_count = 0 also all nearfield
   * operations).
   * @param[in] x Distributed vector which contains the sources.
   * @param[in] y_pFMM  Distributed vector to which the result is added.
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   *                  block matrix!).
   * @tparam run_count  This parameter keeps track how often the pFMM procedure
   *                    has been executed. It is used to select the appropriate
   *                    s2m and l2t operations for each run of the pFMM
   *                    algorithm for the hypersingular operator. For all other
   *                    operators it has no effect.
   * @todo we should disable trans somehow, since it is not implemented
   * correctly.
   * @note This routine is called in the routines @ref apply_sl_dl and
   *       @ref apply_hs.
   * */
  template< slou run_count >
  void apply_pFMM_procedure( const distributed_block_vector & x,
    distributed_block_vector & y_pFMM, bool trans ) const;

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
  void set_orders( int spat_order, int temp_order, int order_regular ) {
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
   * Sets the parameters of the internal ACA method used for
   * additional nearfield compression.
   * @param aca_eps Accuracy of the adaptive cross approximation.
   * @param aca_max_rank Maximum rank of the approximated matrices.
   */
  void set_aca_parameters( sc aca_eps, lo aca_max_rank ) {
    _aca_eps = aca_eps;
    _aca_max_rank = aca_max_rank;
  }

  /**
   * Returns a pointer to (const) @ref _clusters_with_nearfield_operations.
   */
  const std::vector< mesh::general_spacetime_cluster * > *
  get_pointer_to_clusters_with_nearfield_operations( ) const {
    return &_clusters_with_nearfield_operations;
  }

  /**
   * Initializes @ref _clusters_with_nearfield_operations and prepares
   * @ref _clusterwise_nf_matrices and @ref _clusterwise_spat_adm_nf_matrices by
   * creating a map for each space-time target cluster and associated list of
   * nearfield matrices.
   */
  void initialize_nearfield_containers( );

  /**
   * Fills the 4 lists used for scheduling the FMM operations by adding pointers
   * to clusters assigned to the process with id @p _my_process_id. In addition
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
   * @param[in] nf_cluster_index  Index of the nearfield cluster in
   * @ref _clusters_with_nearfield_operations, which acts as the target.
   * @param[in] source_index  Index of the source cluster in the nearfield list
   * of the target cluster.
   */
  full_matrix * create_nearfield_matrix(
    lou nf_cluster_index, lou source_index );
  /**
   * Inserts a matrix into the container of spatially admissible nearfield
   * matrices.
   * @param[in] nf_cluster_index  Index of the nearfield cluster in
   * @ref _clusters_with_nearfield_operations, which acts as the target.
   * @param[in] source_index  Index of the source cluster in the spatially
   * separated nearfield list of the target cluster.
   * @param[in] matrix Nearfield matrix (either full or low rank) inserted into
   */
  void insert_spatially_admissible_nearfield_matrix(
    lou nf_cluster_index, lou source_index, matrix * nf_matrix );

  /**
   * Compute the spatial m2m coefficients for all local spatial levels.
   */
  void initialize_spatial_m2m_coeffs( );

  /**
   * Compute Chebyshev nodes and evaluate them.
   */
  void compute_chebyshev( );

  /**
   * Prints information about the underlying distributed spacetime cluster tree
   * and the operations which have to be applied.
   * @param[in] root_process  Process responsible for printing the information.
   * @param[in] print_tree_information  If true, information is printed for the
   *                                    distributed spacetime cluster tree
   *                                    corresponding to the matrix.
   */
  void print_information(
    const int root_process, const bool print_tree_information = false );

  /**
   * Setter for verbosity during matrix-vector multiplication
   * @param[in] verbose When true, prints information to file.
   */
  void set_verbose( bool verbose ) {
    _verbose = verbose;
  }

  /**
   * Setter for task timer during matrix-vector multiplication
   * @param[in] measure_tasks When true, measures and prints timing of
   *                          individual tasks.
   */
  void set_task_timer( bool measure_tasks ) {
    _measure_tasks = measure_tasks;
  }

  /**
   * Auxiliary method that sorts clusters within the _n_list to improve shared
   * memory scalability during matrix vector multiplication.
   */
  void sort_clusters_in_n_list( );

  /**
   * Extracts the diagonal from the pFMM matrix, inverts each entry, and writes
   * the result to an output vector.
   * @param[out] inverse_diagonal Result vector. It is resized appropriately
   * during the function call.
   */
  void get_inverse_diagonal(
    distributed_block_vector & inverse_diagonal ) const;

  /**
   * Updates the theoretical total size of the local (nonapproximated)
   * nearfield blocks.
   * @param[in] size Size of the block.
   */
  void add_to_local_full_size( long size ) {
#pragma omp atomic update
    _local_full_size += size;
  }

  /**
   * Updates the total size of the local ACA-apprixmated nearfield blocks.
   * @param[in] size Size of the block.
   */
  void add_to_local_approximated_size( long size ) {
#pragma omp atomic update
    _local_approximated_size += size;
  }

  /**
   * Returns the compress ratio of the nearfield blocks approximated by ACA.
   *
   */
  sc get_local_compress_ratio( ) const {
    return (sc) _local_approximated_size / (sc) _local_full_size;
  }

 private:
  /**
   * Traverses the associated distributed space-time tree recursively and adds
   * all space-time clusters for which nearfield operations have to be executed
   * to @ref _clusters_with_nearfield_operations.
   * @param[in] current_cluster Current cluster in the tree traversal.
   */
  void determine_clusters_with_nearfield_operations_recursively(
    mesh::general_spacetime_cluster * current_cluster );

  /**
   * Calls all S2M operations associated with a given scheduling time cluster.
   * @param[in] sources Global sources containing the ones used for the S2M
   *                    operation.
   * @param[in] t_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate s2m operation for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void call_s2m_operations( const distributed_block_vector & sources,
    mesh::scheduling_time_cluster * t_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Applies the appropriate S2M operation for the given source cluster and
   * sources depending on the boundary integral operator.
   * @param[in] source_vector Global sources containing the ones used for the
   *                          S2M operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate s2m operation for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void apply_s2m_operation( const distributed_block_vector & source_vector,
    mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Applies the S2M operation for the given source cluster and sources for
   * p0 basis functions (for single layer and adjoint double layer operators)
   * @param[in] source_vector Global sources containing the ones used for the
   *                          S2M operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   * @todo Use buffers instead of reallocating sources and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_s2m_operation_p0( const distributed_block_vector & source_vector,
    mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Applies the S2M operation for the given source cluster and sources for
   * p1 basis functions and normal derivatives of spatial polynomials (for
   * double layer operator)
   * @param[in] source_vector Global sources containing the ones used for the
   *                          S2M operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   * @todo Use buffers instead of reallocating sources and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_s2m_operations_p1_normal_drv(
    const distributed_block_vector & source_vector,
    mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Applies the S2M operation for the given source cluster and sources for
   * a selected component of the surface curls of p1 basis functions (for
   * hypersingular operator)
   * @param[in] source_vector Global sources containing the ones used for the
   *                          S2M operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   * @tparam dim  Used to select the component of the surface curls (0,1 or 2).
   * @todo Use buffers instead of reallocating sources and aux buffer in every
   * function call?
   */
  template< slou dim >
  void apply_s2m_operation_curl_p1_hs(
    const distributed_block_vector & source_vector,
    mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Applies the S2M operation for the given source cluster and sources for p1
   * basis functions and a selected component of the normal derivative of the
   * Chebyshev polynomials, which are used for the expansion (for hypersingular
   * operator)
   * @param[in] source_vector Global sources containing the ones used for the
   *                          S2M operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   * @param[in] dimension Used to select the component of the normal derivatives
   *                      of the Chebyshev polynomials (0,1 or 2).
   * @todo Use buffers instead of reallocating sources and aux buffer in every
   * function call?
   */
  void apply_s2m_operation_p1_normal_hs(
    const distributed_block_vector & source_vector,
    mesh::general_spacetime_cluster * source_cluster,
    const slou dimension ) const;

  /**
   * Calls all M2M operations associated with a given scheduling time cluster.
   * @param[in] t_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void call_m2m_operations( mesh::scheduling_time_cluster * t_cluster,
    bool verbose, const std::string & verbose_file ) const;

  /**
   * Applies the M2M operations for the given parent cluster and all its
   * children for a given temporal configuration.
   * @param[in] parent_cluster  Considered spacetime parent cluster.
   * @param[in] child_configuration Indicates for which children the m2m
   *                                operations are executed:
   *                                - 0: left children w.r.t. to time.
   *                                - 1: right children w.r.t. to time.
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
    const lo n_space_div_parent, const slou octant,
    std::vector< sc > & output_array ) const;

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
   * @param[in] t_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void call_l2l_operations( mesh::scheduling_time_cluster * t_cluster,
    bool verbose, const std::string & verbose_file ) const;

  /**
   * Applies the L2L operations for the given parent cluster and all its
   * children for a given temporal configuration.
   * @param[in] parent_cluster  Considered spacetime parent cluster.
   * @param[in] child_configuration Indicates for which children the l2l
   *                                operations are executed:
   *                                - 0: left children w.r.t. to time.
   *                                - 1: right children w.r.t. to time.
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
   * Calls all M2T operations associated with a given scheduling time cluster.
   * @param[in] t_cluster  Considered scheduling time cluster.
   * @param[in,out] output_vector Block vector to which the results are added.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate m2t operation for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void call_m2t_operations( mesh::scheduling_time_cluster * t_cluster,
    distributed_block_vector & output_vector, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Applies the appropriate M2T operation for the given space-time source and
   * target clusters depending on the boundary integral operator and writes the
   * result to a local output vector.
   * @param[in] src_cluster  Considered spacetime source cluster.
   * @param[in] tar_cluster  Considered spacetime target cluster.
   * @param[in,out] local_output_vector Local result vector to which the result
   * of the operation is added.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate m2t operation for this
   *                    run in case of the hypersingular operator.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   */
  template< slou run_count >
  void apply_m2t_operation( const mesh::general_spacetime_cluster * src_cluster,
    const mesh::general_spacetime_cluster * tar_cluster,
    vector_type & local_output_vector ) const;

  /**
   * Applies an M2T operation for the given space-time source and target
   * clusters for p0 basis functions in the target cluster and writes the result
   * to the appropriate part of the local output vector.
   * @param[in] src_cluster  Considered spacetime source cluster.
   * @param[in] tar_cluster  Considered spacetime target cluster.
   * @param[in,out] local_output_vector Local result vector to which the result
   * of the operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   */
  void apply_m2t_operation_p0(
    const mesh::general_spacetime_cluster * src_cluster,
    const mesh::general_spacetime_cluster * tar_cluster,
    vector_type & local_output_vector, const lo quad_order_space = 2 ) const;

  /**
   * Applies an M2T operation for the given space-time source and target
   * clusters for p1 basis functions and normal derivatives of the kernel in the
   * target cluster and writes the result to the appropriate part of the local
   * output vector.
   * @param[in] src_cluster  Considered spacetime source cluster.
   * @param[in] tar_cluster  Considered spacetime target cluster.
   * @param[in,out] local_output_vector Local result vector to which the result
   * of the operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   */
  void apply_m2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * src_cluster,
    const mesh::general_spacetime_cluster * tar_cluster,
    vector_type & local_output_vector ) const;

  /**
   * Applies the appropriate M2Ls operation for the given space-time source and
   * target clusters, depending on the boundary integral operator.
   * @param[in] src_cluster  Considered spacetime source cluster.
   * @param[in] tar_cluster  Considered spacetime target cluster.
   */
  template< slou run_count >
  void apply_m2ls_operation(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const;

  /**
   * Applies an M2Ls operation with for the given space-time source and target
   * clusters and p0 basis functions in time.
   * @param[in] src_cluster  Considered spacetime source cluster.
   * @param[in] tar_cluster  Considered spacetime target cluster.
   * @note In the target interval we apply a Gauss quadrature rule with
   * ( _temp_order + 1 ) points, which is exact for polynomials up to order
   *  2 * _temp_order + 1. This are roughly twice as many quadrature points
   * as we use for the quadrature in time in s2m and l2t operations, so they
   * should be sufficient.
   */
  void apply_m2ls_operation_p0_time(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const;

  /**
   * Applies the appropriate L2Ls operation for the given space-time target
   * cluster, depending on the boundary integral operator.
   * @param[in] current_cluster Space-time target cluster for which the L2Ls
   * operation is executed.
   */
  template< slou run_count >
  void apply_l2ls_operation(
    mesh::general_spacetime_cluster * current_cluster ) const;

  /**
   * Applies an L2Ls operation for the given space-time target cluster and
   * piecewise constant basis functions in time.
   * @param[in] current_cluster Space-time target cluster for which the L2Ls
   * operation is executed.
   */
  void apply_l2ls_operation_p0_time(
    mesh::general_spacetime_cluster * current_cluster ) const;

  /**
   * Applies an Ls2Ls operation for the given parent and child cluster.
   * @param[in] parent_cluster  Current parent cluster.
   * @param[in] child_cluster Current child cluster.
   */
  void apply_ls2ls_operation( mesh::general_spacetime_cluster * parent_cluster,
    mesh::general_spacetime_cluster * child_cluster ) const;

  /**
   * Applies the appropriate Ls2T operation for the given space-time cluster,
   * depending on the boundary integral operator. The result is written to the
   * given global result vector.
   * @param[in,out] output_vector The results of the Ls2T operation are added to
   * the appropriate positions of this global result vector.
   * @param[in] tar_cluster  Cluster providing the local contributions and the
   * geometry for the Ls2T operation.
   * @param[in] tar_element_cluster  Cluster for whose elements Ls2T
   * operations are executed.
   * @note If @p tar_element_cluster is not provided, @p tar_cluster takes its
   * role.
   * @note @p tar_element_cluster needs to be provided only in case of
   * auxiliary Ls2T operations, in which case it should be a leaf descendant of
   * @p tar_cluster in the cluster tree.
   */
  template< slou run_count >
  void apply_ls2t_operation( distributed_block_vector & output_vector,
    const mesh::general_spacetime_cluster * tar_cluster,
    const mesh::general_spacetime_cluster * tar_element_cluster
    = nullptr ) const;

  /**
   * Applies an Ls2T operation for the given space-time cluster for p0 basis
   * functions in space and writes the result to the appropriate part of the
   * (global) output vector.
   * @param[in,out] output_vector Global result vector to which the result
   * @param[in] tar_cluster  Cluster providing the local contributions and the
   * geometry for the Ls2T operation.
   * @param[in] tar_element_cluster  Cluster for whose elements Ls2T
   * operations are executed.
   * @note See also @ref apply_ls2t_operation for more details.
   */
  void apply_ls2t_operation_p0( distributed_block_vector & output_vector,
    const mesh::general_spacetime_cluster * tar_cluster,
    const mesh::general_spacetime_cluster * tar_element_cluster ) const;

  /**
   * Applies an Ls2T operation for the given space-time cluster for p1 basis
   * functions and normal derivatives of spatial polynomials (for adjoint
   * double layer operator and hypersingular operator) and writes the result to
   * the appropriate part of the (global) output vector.
   * @param[in,out] output_vector Global result vector to which the result
   * @param[in] tar_cluster  Cluster providing the local contributions and the
   * geometry for the Ls2T operation.
   * @param[in] tar_element_cluster  Cluster for whose elements Ls2T
   * operations are executed.
   * @note See also @ref apply_ls2t_operation for more details.
   */
  void apply_ls2t_operation_p1_normal_drv(
    distributed_block_vector & output_vector,
    const mesh::general_spacetime_cluster * tar_cluster,
    const mesh::general_spacetime_cluster * tar_element_cluster ) const;

  /**
   * Calls all S2L operations associated with a given scheduling time cluster.
   * @param[in] sources Global sources containing the ones used for the S2L
   *                    operation.
   * @param[in] t_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate s2l operation for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void call_s2l_operations( const distributed_block_vector & sources,
    mesh::scheduling_time_cluster * t_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Applies the appropriate S2L operation for the given source cluster, target
   * cluster and source vector depending on the boundary integral operator.
   * @param[in] src_vector Global sources containing the ones used for the
   * S2L operation.
   * @param[in] src_cluster  Considered spacetime source cluster.
   * @param[in] tar_cluster  Considered spacetime source cluster.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate s2m operation for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void apply_s2l_operation( const distributed_block_vector & src_vector,
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const;

  /**
   * Applies an S2L operation for the given space-time source and target
   * clusters for p0 basis functions in the source cluster. The local
   * contributions of the target cluster are updated.
   * @param[in] src_vector Global sources containing the ones used for the
   * S2L operation.
   * @param[in] src_cluster  Considered spacetime source cluster.
   * @param[in] tar_cluster  Considered spacetime source cluster.
   */
  void apply_s2l_operation_p0( const distributed_block_vector & src_vector,
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster,
    const lo quad_order_space = 2 ) const;

  /**
   * Applies an S2L operation for the given space-time source and target
   * clusters for p1 basis functions and normal derivatives of the kernel in the
   * source cluster. The local contributions of the target cluster are updated.
   * @param[in] src_vector Global sources containing the ones used for the
   * S2L operation.
   * @param[in] src_cluster  Considered spacetime source cluster.
   * @param[in] tar_cluster  Considered spacetime source cluster.
   */
  void apply_s2l_operation_p1_normal_drv(
    const distributed_block_vector & src_vector,
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const;

  /**
   * Applies the appropriate S2Ms operation for a given source cluster,
   * depending on the boundary integral operator.
   * @param[in] src_vector  Global vector containing the sources used for the
   * operation.
   * @param[in] src_cluster Cluster whose spatial elements are used to compute
   * the moments via an S2Ms operation.
   * @param[in] src_geometry_cluster  Cluster whose box boundaries are used to
   * compute the moments.
   * @note If @p src_geometry_cluster is not provided, @p src_cluster takes its
   * role.
   * @note @p src_geometry_cluster needs to be provided only in case of
   * auxiliary S2Ms operations, in which case it should be a coarse ancestor of
   * @p src_cluster in the cluster tree.
   */
  template< slou run_count >
  void apply_s2ms_operation( const distributed_block_vector & src_vector,
    mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * src_geometry_cluster = nullptr ) const;

  /**
   * Applies an S2Ms operation for a given source cluster and p0 basis functions
   * in space.
   * @param[in] src_vector  Global vector containing the sources used for the
   * operation.
   * @param[in] src_cluster Cluster whose spatial elements are used to compute
   * the moments via an S2Ms operation.
   * @param[in] src_geometry_cluster  Cluster whose box boundaries are used to
   * compute the moments.
   * @note See also @ref apply_s2ms_operation for more details.
   */
  void apply_s2ms_operation_p0( const distributed_block_vector & src_vector,
    mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * src_geometry_cluster ) const;

  /**
   * Applies an S2Ms operation for a given source cluster for p1 basis functions
   * in space and normal derivatives of spatial polynomials (for double layer
   * operator).
   * @param[in] src_vector  Global vector containing the sources used for the
   * operation.
   * @param[in] src_cluster Cluster whose spatial elements are used to compute
   * the moments via an S2Ms operation.
   * @param[in] src_geometry_cluster  Cluster whose box boundaries are used to
   * compute the moments.
   * @note See also @ref apply_s2ms_operation for more details.
   */
  void apply_s2ms_operation_p1_normal_drv(
    const distributed_block_vector & src_vector,
    mesh::general_spacetime_cluster * src_cluster,
    [[maybe_unused]] mesh::general_spacetime_cluster * src_geometry_cluster
    = nullptr ) const;

  void sum_up_auxiliary_spatial_moments(
    mesh::general_spacetime_cluster * src_cluster ) const;

  /**
   * Applies an Ms2Ms operation for a given child and parent cluster.
   * @param[in] child_cluster Current child cluster.
   * @param[in] parent_cluster  Current parent cluster.
   */
  void apply_ms2ms_operation(
    const mesh::general_spacetime_cluster * child_cluster,
    mesh::general_spacetime_cluster * parent_cluster ) const;

  /**
   * Applies the appropriate Ms2M operation for a given space-time source
   * cluster, depending on the boundary integral operator.
   * @param[in] current_cluster Current source cluster.
   */
  template< slou run_count >
  void apply_ms2m_operation(
    mesh::general_spacetime_cluster * current_cluster ) const;

  /**
   * Applies an Ms2M operation for a given space-time source
   * cluster and p0 basis functions in time.
   * @param[in] current_cluster Current source cluster.
   */
  void apply_ms2m_operation_p0_time(
    mesh::general_spacetime_cluster * current_cluster ) const;

  /**
   * Applies the appropriate Ms2L operation for a given source and target
   * cluster, depending on the boundary integral operator.
   * @param[in] src_cluster Current source cluster.
   * @param[in] tar_cluster Current target cluster.
   */
  template< slou run_count >
  void apply_ms2l_operation(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const;

  /**
   * Applies an Ms2L operation for a given source and target cluster and p0
   * basis functions in time.
   * @param[in] src_cluster Current source cluster.
   * @param[in] tar_cluster Current target cluster.
   * @note In the source interval we apply a Gauss quadrature rule with
   * ( _temp_order + 1 ) points, which is exact for polynomials up to order
   *  2 * _temp_order + 1. This are roughly twice as many quadrature points
   * as we use for the quadrature in time in s2m and l2t operations, so they
   * should be sufficient.
   */
  void apply_ms2l_operation_p0_time(
    const mesh::general_spacetime_cluster * src_cluster,
    mesh::general_spacetime_cluster * tar_cluster ) const;

  /**
   * Calls all L2T operations associated with a given scheduling time cluster.
   * @param[in] t_cluster  Considered scheduling time cluster.
   * @param[in,out] output_vector Block vector to which the results are added.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate l2t operation for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void call_l2t_operations( mesh::scheduling_time_cluster * t_cluster,
    distributed_block_vector & output_vector, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Applies the appropriate L2T operation for the given target cluster
   * depending on the boundary integral operator and writes the result to the
   * appropriate part of the output vector.
   * @param[in] st_cluster  Considered spacetime cluster.
   * @param[in,out] output_vector Global result vector to which the result of
   *                              the operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate s2m operation for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void apply_l2t_operation( const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const;

  /**
   * Applies the L2T operation for the given target cluster for p0 basis
   * functions and writes the result to the appropriate part of the output
   * vector.
   * @param[in] st_cluster  Considered spacetime cluster.
   * @param[in,out] output_vector Global result vector to which the result of
   *                              the operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_l2t_operation_p0(
    const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const;

  /**
   * Applies the L2T operation for the given target cluster for p1 basis
   * functions and normal derivatives of spatial polynomials (for adjoint double
   * layer operator and hypersingular operator) and writes the result to the
   * appropriate part of the output vector.
   * @param[in] st_cluster  Considered spacetime cluster.
   * @param[in,out] output_vector Global result vector to which the result of
   *                              the operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_l2t_operation_p1_normal_drv(
    const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const;

  /**
   * Applies the L2T operation for the given target cluster for a selected
   * component of the surface curls of p1 basis functions (for hypersingular
   * operator) and writes the result to the appropriate part of the output
   * vector.
   * @param[in] st_cluster Considered spacetime cluster.
   * @param[in,out] output_vector Global result vector to which the result of
   *                              the operation is added.
   * @tparam dim  Used to select the component of the surface curls (0,1 or 2).
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   */
  template< slou dim >
  void apply_l2t_operation_curl_p1_hs(
    const mesh::general_spacetime_cluster * st_cluster,
    distributed_block_vector & output_vector ) const;

  /**
   * Applies the L2T operation for the given target cluster for p1 basis
   * functions and a selected component of the normal derivative of the
   * Chebyshev polynomials, which are used for the expansion (for hypersingular
   * operator), and writes the result to the appropriate part of the output
   * vector.
   * @param[in] st_cluster Considered spacetime cluster.
   * @param[in] dimension Used to select the component of the normal derivatives
   *                      of the Chebyshev polynomials (0,1 or 2).
   * @param[in,out] output_vector Global result vector to which the result of
   *                              the operation is added.
   * @todo Use buffers instead of reallocating targets and aux buffer in every
   * function call?
   */
  void apply_l2t_operation_p1_normal_hs(
    const mesh::general_spacetime_cluster * st_cluster, const slou dimension,
    distributed_block_vector & output_vector ) const;

  /**
   * Executes all nearfield operations associated with a given scheduling time
   * cluster.
   * @param[in] t_cluster Time cluster whose associated nearfield operations
   *                    are executed.
   * @param[in] sources Global sources containing the ones used for the
   *                    nearfield operation.
   * @param[in] trans If true, the transposed nearfield matrices are applied
   *                  otherwise the standard nearfield matrices.
   * @param[in,out] output_vector Vector to which the results are added.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   */
  void apply_nearfield_operations(
    const mesh::scheduling_time_cluster * t_cluster,
    const distributed_block_vector & sources, bool trans,
    distributed_block_vector & output_vector, bool verbose,
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
   */
  void check_for_received_data( std::vector< MPI_Request > & array_of_requests,
    std::vector< int > & array_of_indices, int & outcount ) const;

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
   * operations should be executed and whose dependencies are satisfied. In case
   * a cluster is found the status is updated. If no cluster is found the
   * iterator points to the end of the m2l_list and the status is not modified.
   * @param[in] m2l_list  A list containing the clusters of @ref _m2l_list whose
   * operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in one of the lists this
   * iterator points to it. Else it points to the end of @p m2l_list.
   * @param[out] status If a cluster is found in the m2l-list it is
   * set to 4.
   */
  void find_cluster_in_m2l_list(
    std::list< mesh::scheduling_time_cluster * > & m2l_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const;

  /**
   * Returns an iterator pointing to the next cluster in the s2l-list whose
   * operations should be executed and whose dependencies are satisfied. In case
   * a cluster is found the status is updated. If no cluster is found the
   * iterator points to the end of the s2l_list and the status is not modified.
   * @param[in] s2l_list  A list containing the clusters of @ref _s2l_list whose
   * operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in one of the lists this
   * iterator points to it. Else it points to the end of @p s2l_list.
   * @param[out] status If a cluster is found in the s2l-list it is set to 3.
   */
  void find_cluster_in_s2l_list(
    std::list< mesh::scheduling_time_cluster * > & s2l_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const;

  /**
   * Returns an iterator pointing to the next cluster in the m2t-list whose
   * dependencies are satisfied. In case a cluster is found the status is
   * updated. If no cluster is found the iterator points to the end of the list
   * and the status is not modified.
   * @param[in] m2t_list  A list containing the clusters of @ref _m2t_list whose
   * operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in the list this
   * iterator points to it. Else it points to the end of the list.
   * @param[out] status Set to 5 if a cluster is found.
   */
  void find_cluster_in_m2t_list(
    std::list< mesh::scheduling_time_cluster * > & m2t_list,
    std::list< mesh::scheduling_time_cluster * >::iterator & it_next_cluster,
    char & status ) const;

  /**
   * Updates dependency flags or sends moments for M2L or M2T operations.
   * @param[in] src_cluster Considered scheduling time cluster. If a cluster in
   * its send list or diagonal send list is handled by a different process, the
   * moments are sent to this process.
   * @param[in] verbose If true, the process reports about all initiated send
   * operations. (Updates of dependency flags are not reported)
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   * file.
   */
  void provide_moments_for_m2l_or_m2t(
    mesh::scheduling_time_cluster * src_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Updates dependency flags or sends spatial moments for hybrid S2L
   * operations.
   * @param[in] src_cluster Considered scheduling time cluster. If a cluster in
   * its s2l send list is handled by a different process, the spatial moments
   * are sent to this process.
   * @param[in] verbose If true, the process reports about all initiated send
   * operations. (Updates of dependency flags are not reported)
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   * file.
   */
  void provide_spatial_moments_for_hybrid_s2l(
    mesh::scheduling_time_cluster * src_cluster, bool verbose,
    const std::string & verbose_file ) const;

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
   *                                  the size of @p receive_vector.
   */
  void start_receive_operations(
    std::vector< MPI_Request > & array_of_requests ) const;

  /**
   * Compute quadrature of the Chebyshev polynomials and p0 basis functions for
   * the spatial part of a spacetime cluster
   * @param[out] T  Full matrix where the quadratures are stored. The elements
   * of the cluster vary along the rows, the order of the polynomial along the
   * columns of the matrix.
   * @param[in] source_elem_cluster  Cluster for whose spatial elements the
   * quadratures are computed.
   * @param[in] source_geom_cluster Cluster whose geometric box is used to
   * compute the quadratures.
   */
  void compute_chebyshev_quadrature_p0( full_matrix & T,
    const mesh::general_spacetime_cluster * source_elem_cluster,
    const mesh::general_spacetime_cluster * source_geom_cluster
    = nullptr ) const;

  /**
   * Computes quadrature of the normal derivatives of the Chebyshev polynomials
   * times p1 basis functions for the spatial part of a spacetime cluster.
   * @param[out] T_drv  Full matrix where the quadratures are stored. The
   * nodes of the cluster vary along the rows, the order of the polynomial
   * along the columns of the matrix.
   * @param[in] source_elem_cluster  Cluster for whose spatial elements the
   * quadratures are computed.
   * @param[in] source_geom_cluster Cluster whose geometric box is used to
   * compute the quadratures.
   */
  void compute_normal_drv_chebyshev_quadrature_p1( full_matrix & T_drv,
    const mesh::general_spacetime_cluster * source_elem_cluster,
    const mesh::general_spacetime_cluster * source_geom_cluster
    = nullptr ) const;

  /**
   * Computes quadrature of the Chebyshev polynomials times a selected component
   * of the surface curls of p1 basis functions for the spatial part of a
   * spacetime cluster.
   * @param[out] T_curl_along_dim Full matrix where the quadratures are stored.
   *                              The nodes of the cluster vary along the rows,
   *                              the order of the polynomial along the columns
   *                              of the matrix.
   * @param[in] source_elem_cluster  Cluster for whose spatial component the
   *                            quadratures are computed.
   * @tparam dim  Used to select the component of the surface curls (0,1 or 2).
   */
  template< slou dim >
  void compute_chebyshev_times_p1_surface_curls_along_dimension(
    full_matrix & T_curl_along_dim,
    const mesh::general_spacetime_cluster * source_elem_cluster ) const;

  /**
   * Computes quadrature of a selected component of the normal derivatives of
   * the Chebyshev polynomials times p1 basis functions for the spatial part of
   * a spacetime cluster.
   * @param[out] T_normal_along_dim Full matrix where the quadratures are
   *                                stored. The nodes of the cluster vary along
   *                                the rows, the order of the polynomial along
   *                                the columns of the matrix.
   * @param[in] dim Used to select the component of the normal derivatives of
   *                the Chebyshev polynomials (0,1 or 2).
   * @param[in] source_cluster  Cluster for whose spatial component the
   *                            quadratures are computed.
   */
  void compute_chebyshev_times_normal_quadrature_p1_along_dimension(
    full_matrix & T_normal_along_dim, const slou dim,
    const mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Compute quadrature of the Lagrange polynomials and p0 basis functions for
   * the temporal part of a spacetime cluster
   * @param[in] source_cluster  Cluster for whose temporal component the
   *                            quadratures are computed.
   * @param[out] L  Full matrix where the quadratures are stored. The temporal
   *                elements of the cluster vary along the columns, the order
   *                of the polynomial along the rows of the matrix.
   */
  void compute_lagrange_quadrature( full_matrix & L,
    const mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Compute quadrature of the derivative of Lagrange polynomials and p0 basis
   * functions for the temporal part of a spacetime cluster#
   * @param[out] L_drv  Full matrix where the quadratures are stored. The
   *                    temporal elements of the cluster vary along the columns,
   *                    the order of the polynomial along the rows of the
   *                    matrix.
   * @param[in] source_cluster  Cluster for whose temporal component the
   *                            quadratures are computed.
   */
  void compute_lagrange_drv_quadrature( full_matrix & L_drv,
    const mesh::general_spacetime_cluster * source_cluster ) const;

  /**
   * Computes coupling coefficients for the spacetime m2l operation for one of
   * the three space dimensions implicitly given.
   * @param[in] src_time_nodes  Interpolation nodes in time for the source
   *                            cluster.
   * @param[in] tar_time_nodes  Interpolation nodes in time for the target
   *                            cluster.
   * @param[in] half_size Half size in space of the current clusters along the
   *                      dimension for which the coefficients are computed.
   * @param[in] center_diff The appropriate component of the difference vector
   *                        (target_center - source_center).
   * @param[in] buffer_for_gaussians  Vector of size >= ( _spat_order + 1 )^2
   *                                  * ( _temp_order + 1 )^2 to store
   *                                  intermediate results in the computation
   *                                  of the m2l coefficients.
   * @param[in,out] coupling_coeffs Vector of size >= ( _spat_order + 1 )^2
   *                                * ( _temp_order + 1 )^2 to store m2l
   *                                coefficients.
   */
  void compute_m2l_coupling_coeffs( const vector_type & src_time_nodes,
    const vector_type & tar_time_nodes, const sc half_size,
    const sc center_diff, vector_type & buffer_for_gaussians,
    vector_type & coupling_coeffs ) const;

  /**
   * Computes the coupling coefficients for single sided FMM operations for one
   * of the three space dimensions implicitly given.
   *
   * In case of M2T operations, an expansion in the source cluster takes place,
   * in case of S2L operations an expansion in the target cluster. The
   * corresponding cluster is called expansion cluster below. By using
   * @p time_node_differences as input value, the routine can be used to compute
   * the coefficients of M2T and S2L operations at the same time.
   * @param[in] time_node_differences Difference of the target time point and
   * source time points for M2T operations, or target time points and source
   * time point for S2L operations.
   * @param[in] half_size Half size in space of the expansion cluster along the
   * dimension for which the coefficients are computed.
   * @param[in] center The appropriate component of the expansion cluster's
   * center.
   * @param[in] eval_point_space  Appropriate component of the point in
   * space, for which the coupling coefficient is evaluated.
   * @param[in] buffer_for_gaussians  Vector of size >= (_spat_order + 1)
   * * ( _temp_order + 1 ) to store intermediate results in the computation
   * of the m2l coefficients.
   * @param[in,out] coupling_coeffs Vector of size >= ( _spat_order + 1 )
   * * ( _temp_order + 1 ) to store the coupling coefficients.
   */
  void compute_single_sided_coupling_coeffs(
    const vector_type & time_node_differences, const sc half_size,
    const sc center, const sc eval_point_space,
    vector_type & buffer_for_gaussians, vector_type & coupling_coeffs ) const;

  /**
   * Computes the tensor product coupling coefficients for single sided FMM
   * operations.
   *
   * In case of M2T operations, an expansion in the source cluster takes place,
   * in case of S2L operations an expansion in the target cluster. The
   * corresponding cluster is called expansion cluster below. By using
   * @p time_node_differences as input value, the routine can be used to compute
   * the coefficients of M2T and S2L operations at the same time.
   *
   * @param[in] time_node_differences Difference of the target time point and
   * source time points for M2T operations, or target time points and source
   * time point for S2L operations.
   * @param[in] half_size_space Vector of spatial half sizes in the expansion
   * cluster.
   * @param[in] center_space  Center of the expansion cluster.
   * @param[in] eval_point_space  Spatial evaluation point (e.g. a quadrature
   * point in a source (S2L) or target (M2T) triangle).
   * @param[in] vector_of_buffers Vector containing auxiliary buffers used by
   * the routine. 4 buffers are needed with sizes greater than or equal to
   * ( _spat_order + 1 ) * ( _temp_order + 1 ).
   * @param[in,out] coupling_coeffs_tensor  The resulting coefficients are
   * written to this vector. Its size has to be >= @ref _contribution_size.
   *
   * @note The routine calls the routine
   * @ref compute_single_sided_coupling_coeffs to compute the individual
   * components of the coupling coefficients along each dimension and combines
   * them to get the tensor-product coefficients.
   */
  void compute_single_sided_coupling_coeffs_tensor(
    const vector_type & time_node_differences,
    const vector_type & half_size_space, const vector_type & center_space,
    const vector_type eval_point_space,
    const std::vector< vector_type * > & vector_of_buffers,
    vector_type & coupling_coeffs_tensor ) const;

  /**
   * Computes the coupling coefficients for single sided FMM operations
   * involving the spatial derivative of the heat kernel for one of the three
   * space dimensions implicitly given.
   *
   * In case of M2T operations, an expansion in the source cluster takes place,
   * in case of S2L operations an expansion in the target cluster. The
   * corresponding cluster is called expansion cluster below. By using
   * @p time_node_differences as input value, the routine can be used to compute
   * the coefficients of M2T and S2L operations at the same time.
   * @param[in] time_node_differences Difference of the target time point and
   * source time points for M2T operations, or target time points and source
   * time point for S2L operations.
   * @param[in] half_size Half size in space of the expansion cluster along the
   * dimension for which the coefficients are computed.
   * @param[in] center The appropriate component of the expansion cluster's
   * center.
   * @param[in] eval_point_space  Appropriate component of the point in
   * space, for which the coupling coefficient is evaluated.
   * @param[in] buffer_for_drv_gaussians  Vector of size >= (_spat_order + 1)
   * * ( _temp_order + 1 ) to store intermediate results in the computation
   * of the m2l coefficients.
   * @param[in,out] coupling_coeffs Vector of size >= ( _spat_order + 1 )
   * * ( _temp_order + 1 ) to store the coupling coefficients.
   */
  void compute_single_sided_coupling_coeffs_drv_space(
    const vector_type & time_node_differences, const sc half_size,
    const sc center, const sc eval_point_space,
    vector_type & buffer_for_drv_gaussians,
    vector_type & coupling_coeffs ) const;

  /**
   * Computes the tensor product coupling coefficients for single sided FMM
   * operations involving the normal derivative of the heat kernel (double layer
   * and adjoint double layer operator)
   *
   * In case of M2T operations, an expansion in the source cluster takes place,
   * in case of S2L operations an expansion in the target cluster. The
   * corresponding cluster is called expansion cluster below. By using
   * @p time_node_differences as input value, the routine can be used to compute
   * the coefficients of M2T and S2L operations at the same time.
   *
   * @param[in] time_node_differences Difference of the target time point and
   * source time points for M2T operations, or target time points and source
   * time point for S2L operations.
   * @param[in] half_size_space Vector of spatial half sizes in the expansion
   * cluster.
   * @param[in] center_space  Center of the expansion cluster.
   * @param[in] eval_point_space  Spatial evaluation point (e.g. a quadrature
   * point in a source (S2L) or target (M2T) triangle).
   * @param[in] normal  Normal vector of the spatial triangle (in the source
   * (S2L) or target (M2T) triangle).
   * @param[in] vector_of_buffers Vector containing auxiliary buffers used by
   * the routine. 8 buffers are needed, the first 7 with sizes greater than or
   * equal to ( _spat_order + 1 ) * ( _temp_order + 1 ), the last with size
   * greater than or equal to ( _temp_order + 1 ).
   * @param[in,out] coupling_coeffs_tensor  The resulting coefficients are
   * written to this vector. Its size has to be >= @ref _contribution_size.
   * @note The routine calls the routines
   * @ref compute_single_sided_coupling_coeffs and
   * @ref compute_single_sided_coupling_coeffs_drv_space to compute the
   * individual components of the coupling coefficients along each dimension and
   * combines them to get the tensor-product coefficients.
   */
  void compute_single_sided_coupling_coeffs_normal_drv_tensor(
    const vector_type & time_node_differences,
    const vector_type & half_size_space, const vector_type & center_space,
    const vector_type eval_point_space,
    linear_algebra::coordinates< 3 > & normal,
    const std::vector< vector_type * > & vector_of_buffers,
    vector_type & coupling_coeffs_tensor ) const;

  /**
   * Traverses the m_list, l_list and m2l_list of the pFMM matrix and resets
   * the dependency data (i.e. the data used to determine if the operations of
   * a cluster are ready for execution).
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
  void init_quadrature_polynomials(
    quadrature_wrapper & my_quadrature, const lo quadrature_order = 5 ) const;

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

  /**
   * In this routine each process counts the number of entries in its nearfield
   * matrices levelwise. The level of the target cluster determines where the
   * entries are added up to.
   *
   * @param[out] levelwise_nearfield_entries  Contains the numbers of counted
   * nearfield matrix entries for all levels in the cluster tree.
   * @note No communication among processes occurs in this routine. Each process
   * returns an individual vector.
   */
  void compute_nearfield_entries_levelwise(
    std::vector< lo > & levelwise_nearfield_entries ) const;

  /**
   * In this routine each process counts the number of entries in its nearfield
   * matrices levelwise for all those nearfield matrices which correspond to
   * clusters that are separated in time. The level of the target cluster
   * determines where the entries are added up to.
   *
   * @param[out] levelwise_time_separated_nearfield_entries  Contains the
   * numbers of counted nearfield matrix entries for time separated clusters for
   * all levels in the cluster tree.
   * @note No communication among processes occurs in this routine. Each process
   * returns an individual vector.
   */
  void compute_time_separated_nearfield_entries_levelwise(
    std::vector< lo > & levelwise_time_separated_nearfield_entries ) const;

  /**
   * Counts the number of all FMM operations levelwise
   * @param[in,out] n_s2m_operations  Container to store the numbers of
   * levelwise S2M operations.
   * @param[in,out] n_m2m_operations  Container to store the numbers of
   * levelwise M2M operations.
   * @param[in,out] n_m2l_operations  Container to store the numbers of
   * levelwise M2L operations.
   * @param[in,out] n_l2l_operations  Container to store the numbers of
   * levelwise L2L operations.
   * @param[in,out] n_l2t_operations  Container to store the numbers of
   * levelwise L2T operations.
   * @param[in,out] n_s2l_operations  Container to store the numbers of
   * levelwise S2L operations.
   * @param[in,out] n_m2t_operations  Container to store the numbers of
   * levelwise M2T operations.
   * @note m2m and l2l operations are counted for the levels of the children
   */
  void count_fmm_operations_levelwise( std::vector< lou > & n_s2m_operations,
    std::vector< lou > & n_m2m_operations,
    std::vector< lou > & n_m2l_operations,
    std::vector< lou > & n_l2l_operations,
    std::vector< lou > & n_l2t_operations,
    std::vector< lou > & n_s2l_operations,
    std::vector< lou > & n_m2t_operations ) const;

  /**
   * Primary task for a time cluster in the M-list.
   *
   * Depending on the given time cluster it executes (or schedules secondary
   * tasks for) the following operations:
   * - S2M operations for space-time leaf clusters
   * - M2M operations from the current cluster to its parent
   * - providing moments for M2L or M2T operations (sending via MPI if
   *   necessary)
   * - providing moments to the parent (sending via MPI if necessary)
   * @param[in] x Input vector; might be used to compute moments in S2M.
   * @param[in] current_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate s2m operations for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void m_list_task( const distributed_block_vector & x,
    mesh::scheduling_time_cluster * current_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Primary task for a time cluster in the L-list.
   *
   * Depending on the given time cluster it executes (or schedules secondary
   * tasks for) the following operations:
   * - L2L operations from the parent of the current cluster to itself
   * - L2T or downward send operations if the local contributions are ready.
   * @note Local contributions are ready, when all required L2L, M2L and S2L
   * operations have been executed.
   * @param[in] y_pFMM Output vector; might be updated with results from L2T
   * operations.
   * @param[in] current_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate l2t operations for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void l_list_task( distributed_block_vector & y_pFMM,
    mesh::scheduling_time_cluster * current_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Primary task for a time cluster in the M2L-list.
   *
   * Depending on the given time cluster it executes (or schedules secondary
   * tasks for) the following operations:
   * - M2L operations, where the space-time clusters associated with the given
   *   time cluster act as targets.
   * - L2T or downward send operations if the local contributions are ready.
   * @note Local contributions are ready, when all required L2L, M2L and S2L
   * operations have been executed.
   * @param[in] y_pFMM Output vector; might be updated with results from L2T
   * operations.
   * @param[in] current_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate l2t operations for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void m2l_list_task( distributed_block_vector & y_pFMM,
    mesh::scheduling_time_cluster * current_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Primary task for a time cluster in the M2L-list.
   *
   * Depending on the given time cluster it executes (or schedules secondary
   * tasks for) the following operations:
   * - M2T operations, where the space-time clusters associated with the given
   *   time cluster act as targets.
   * @param[in] y_pFMM Output vector; updated with results from M2T operations.
   * @param[in] current_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate m2t operations for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void m2t_list_task( distributed_block_vector & y_pFMM,
    mesh::scheduling_time_cluster * current_cluster, bool verbose,
    const std::string & verbose_file ) const;

  /**
   * Primary task for a time cluster in the S2L-list.
   *
   * Depending on the given time cluster it executes (or schedules secondary
   * tasks for) the following operations:
   * - S2L operations, where the space-time clusters associated with the given
   *   time cluster act as targets.
   * - L2T or downward send operations if the local contributions are ready.
   * @note Local contributions are ready, when all required L2L, M2L and S2L
   * operations have been executed.
   *
   * @param[in] x Input vector; used for S2L operations.
   * @param[in] y_pFMM Output vector; might be updated with results from L2T
   * operations.
   * @param[in] current_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output
   *                          file.
   * @tparam run_count  Run count of the corresponding pFMM procedure. It is
   *                    used to choose the appropriate s2l operations for this
   *                    run in case of the hypersingular operator.
   */
  template< slou run_count >
  void s2l_list_task( const distributed_block_vector & x,
    distributed_block_vector & y_pFMM,
    mesh::scheduling_time_cluster * current_cluster, bool verbose,
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

  std::vector< mesh::general_spacetime_cluster * >
    _clusters_with_nearfield_operations;

  std::unordered_map< mesh::general_spacetime_cluster *,
    std::vector< full_matrix * > >
    _clusterwise_nf_matrices;  //!< nearfield matrices for all the space-
                               //!< time leaf clusters and their
                               //!< nearfield clusters.
  std::unordered_map< mesh::general_spacetime_cluster *,
    std::vector< matrix * > >
    _clusterwise_spat_adm_nf_matrices;  //!< nearfield matrices for all the
                                        //!< space- time leaf clusters and
                                        //!< their nearfield clusters.

  std::list< mesh::scheduling_time_cluster * >
    _m_list;  //!< M-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster * >
    _m2l_list;  //!< M2L-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster * >
    _l_list;  //!< L2L-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster * >
    _m2t_list;  //!< M2T-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster * >
    _s2l_list;  //!< S2L-list for the execution of the FMM.
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
                                //!< next @p _n_moments_to_receive_m2l_or_m2t
                                //!< entries to moments which have to be
                                //!< received for M2L operations and the
                                //!< remaining entries to local contributions
                                //!< which have to be received in the downward
                                //!< path.
  lou _n_moments_to_receive_upward;  //!< Number of grouped moments which have
                                     //!< to be received in the upward path of
                                     //!< the FMM.
  lou _n_moments_to_receive_m2l_or_m2t;  //!< Number of grouped moments which
                                         //!< have to be received for M2L or M2T
                                         //!< operations.
  lou _n_spatial_moments_to_receive;     //!< Number of grouped spatial moments
                                      //!< which have to be received for hybrid
                                      //!< S2L operations.

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

  sc _alpha;    //!< Heat conductivity.
  sc _aca_eps;  //!< accuracy of the internal ACA used for temporal nearfield
                //!< compression
  lo _aca_max_rank;  //!< maximum allowed rank of the internal ACA used for
                     //!< temporal nearfield compression
  long long _local_full_size;  //!< total size of the (uncompressed) nearfield
                               //!< blocks
  long long _local_approximated_size;  //!< total size of the ACA-approximated
                                       //!< nearfield blocks

  vector_type
    _cheb_nodes_integrate;  //!< Chebyshev nodes used for numerical quadrature
                            //!< in computation of coupling coefficients for m2l
                            //!< or single sided expansions, aligned
  vector_type _all_poly_vals_integrate;  //!< Chebyshev nodes used for numerical
                                         //!< quadrature in computation of
                                         //!< coupling coefficients for m2l or
                                         //!< single sided expansions, aligned
  std::vector< sc, besthea::allocator_type< sc > >
    _cheb_nodes_sum_coll;  //!< summed Chebyshev nodes for collapsed loop in
                           //!< computation of coupling coefficients for m2l,
                           //!< aligned
  std::vector< sc, besthea::allocator_type< sc > >
    _all_poly_vals_mult_coll;  //!< product of Chebyshev polynomials evaluated
                               //!< in Chebyshev nodes for collapsed loop in
                               //!< computation of coupling coefficients for
                               //!< m2l, aligned

  mutable std::vector< full_matrix >
    _aux_buffer_0;  //!< Auxiliary vector used to store intermediate results in
                    //!< M2L operations.
  mutable std::vector< full_matrix >
    _aux_buffer_1;  //!< Auxiliary vector used to store intermediate results in
                    //!< M2L operations.

  bool _verbose;  //!< print info to files during matrix-vector multiplication

  bool _measure_tasks;  //!< print task time info to files during
                        //!< matrix-vector multiplications

  mutable lo _non_nf_op_count;  //!< counter to keep track of the number of
                                //!< scheduled non-nearfield operations

  /**
   * Increases @ref _non_nf_op_count.
   */
  void add_nn_operations( ) const {
#pragma omp atomic update
    _non_nf_op_count++;
  }

  /**
   * Decreases @ref _non_nf_op_count.
   */
  void reduce_nn_operations( ) const {
#pragma omp atomic update
    _non_nf_op_count--;
  }

  /**
   * @returns the value of @ref _non_nf_op_count
   */
  lo get_nn_operations( ) const {
    lo ret_val;
#pragma omp atomic read
    ret_val = _non_nf_op_count;
    return ret_val;
  }

  mutable timer_type _global_timer;  //!< structure for time measurements.

  // using clock_type = std::chrono::high_resolution_clock;
  using time_type = std::chrono::microseconds;  //!< Unit type.

  mutable std::vector< std::vector< time_type::rep > >
    _m_task_times;  //!< Contains a vector for each thread in which the
                    //!< beginning and end times of primary m-list tasks which
                    //!< this thread executed are stored.
  mutable std::vector< std::vector< time_type::rep > >
    _m2l_task_times;  //!< Same as @ref _m_task_times for primary m2l-list
                      //!< tasks.
  mutable std::vector< std::vector< time_type::rep > >
    _l_task_times;  //!< Same as @ref _m_task_times for primary l-list
                    //!< tasks.
  mutable std::vector< std::vector< time_type::rep > >
    _m2t_task_times;  //!< Same as @ref _m_task_times for primary m2t-list
                      //!< tasks.
  mutable std::vector< std::vector< time_type::rep > >
    _s2l_task_times;  //!< Same as @ref _m_task_times for primary s2l-list
                      //!< tasks.
  mutable std::vector< std::vector< time_type::rep > >
    _n_task_times;  //!< Same as @ref _m_task_times for primary n-list
                    //!< tasks.

  mutable std::vector< std::vector< time_type::rep > >
    _m_subtask_times;  //!< Contains a vector for each thread in which the
                       //!< beginning and end times of the subtasks in the
                       //!< m-list which this thread executed are stored.
  mutable std::vector< std::vector< time_type::rep > >
    _m2l_subtask_times;  //!< Same as @ref _m_subtask_times for m2l-list
                         //!< subtasks.
  mutable std::vector< std::vector< time_type::rep > >
    _l_subtask_times;  //!< Same as @ref _m_subtask_times for l-list
                       //!< subtasks.
  mutable std::vector< std::vector< time_type::rep > >
    _m2t_subtask_times;  //!< Same as @ref _m_subtask_times for m2t-list
                         //!< subtasks.
  mutable std::vector< std::vector< time_type::rep > >
    _s2l_subtask_times;  //!< Same as @ref _m_subtask_times for s2l-list
                         //!< subtasks.
  mutable std::vector< std::vector< time_type::rep > >
    _n_subtask_times;  //!< Same as @ref _m_subtask_times for n-list
                       //!< subtasks.

  mutable std::vector< std::vector< time_type::rep > >
    _mpi_send_m2l_m2t_or_s2l;  //!< Contains a vector for each thread. The
                               //!< entries in these vectors are the times when
                               //!< the sending of a group of moments to another
                               //!< process for m2l-list operations has started.
  mutable std::vector< std::vector< time_type::rep > >
    _mpi_send_m_parent;  //!< Same as @ref _mpi_send_m2l_m2t_or_s2l for sending
                         //!< moments for m-list operations.
  mutable std::vector< std::vector< time_type::rep > >
    _mpi_send_l_children;  //!< Same as @ref _mpi_send_m2l_m2t_or_s2l for
                           //!< sending local contributions for l-list
                           //!< operations.

  mutable std::vector< std::vector< time_type::rep > >
    _mpi_recv_m2l_m2t_or_s2l;  //!< Contains a vector for each thread. The
                               //!< entries in these vectors are the times when
                               //!< the thread has detected the reception of a
                               //!< group of moments needed for m2l-list
                               //!< operations.
  mutable std::vector< std::vector< time_type::rep > >
    _mpi_recv_m_parent;  //!< Same as @ref _mpi_recv_m2l_m2t_or_s2l for
                         //!< receiving moments needed for m-list operations.
  mutable std::vector< std::vector< time_type::rep > >
    _mpi_recv_l_children;  //!< Same as @ref _mpi_recv_m2l_m2t_or_s2l for
                           //!< receiving local contributions needed for l-list
                           //!< operations.

  /**
   * Saves task duration measurement per thread in files (1 per MPI rank).
   */
  void save_times( time_type::rep total_loop_duration,
    time_type::rep total_apply_duration ) const;
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

/** Typedef for the distributed spatially adjoint double layer p1-p0 PFMM matrix
 */
typedef besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_adl_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p0 > >
  distributed_pFMM_matrix_heat_adl_p1p0;

/** Typedef for the distributed hypersingular p1-p1 PFMM matrix */
typedef besthea::linear_algebra::distributed_pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 >,
  besthea::bem::distributed_fast_spacetime_be_space<
    besthea::bem::basis_tri_p1 > >
  distributed_pFMM_matrix_heat_hs_p1p1;

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_PFMM_MATRIX_H_ */
