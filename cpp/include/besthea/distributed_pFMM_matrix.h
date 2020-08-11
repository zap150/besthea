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
#include "besthea/distributed_spacetime_cluster_tree.h"
#include "besthea/distributed_fast_spacetime_be_space.h"
#include "besthea/full_matrix.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/local_vector_routines.h"
// #include "besthea/matrix.h"
#include "besthea/settings.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/tree_structure.h"
#include "besthea/vector.h"

#include <list>
#include <mpi.h>

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
      _chebyshev( _spat_order ),
      _lagrange( _temp_order ),
      _alpha( 1.0 ) {
  }

  distributed_pFMM_matrix( const distributed_pFMM_matrix & that ) = delete;

  /**
   * Destructor
   */
  virtual ~distributed_pFMM_matrix( ) { 
    for ( auto nf_matrix_vector : _clusterwise_nearfield_matrices ) {
      for ( auto matrix : nf_matrix_vector ) {
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
   * single and double layer operators.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   */
  void apply_sl_dl( const block_vector & x,
  block_vector & y, bool trans, sc alpha, sc beta ) const;

  /**
   * Sets the MPI communicator associated with the distributed pFMM matrix and
   * the rank of the executing process.
   * @param[in] comm  MPI communicator to be set.
   */
  void set_MPI_communicator( const MPI_Comm* comm ) {
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
   *                                       
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
   * @todo Are these member variables ever used?
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
    _contribution_size = ( _temp_order + 1 ) 
      * ( ( _spat_order + 3 ) * ( _spat_order + 2 ) * ( _spat_order + 1 ) ) / 6;
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
   *       @ref compare_clusters_bottom_up_right_2_left, the others using
   *       @ref compare_clusters_top_down_right_2_left .
   * @todo Determine n_list differently, when coupling with space-time cluster
   *       tree is done.
   */
  void prepare_fmm( );

  /**
   * Creates a nearfield matrix for two clusters
   * @param[in] local_leaf_index  Index of the local leaf cluster, which acts
   *                              as the target.
   * @param[in] source_index  Index of the source cluster in the nearfield list 
   *                          of the target cluster.  
   */
  full_matrix * create_nearfield_matrix( lou leaf_index, lou source_index );

 private:
  const MPI_Comm * _comm; //!< MPI communicator associated with the pFMM matrix.
  int _my_rank; //!< MPI rank of current process.
  mesh::distributed_spacetime_cluster_tree * 
    _distributed_spacetime_tree;  //!< part of a distributed tree hierarchically 
                                  //!< decomposing the space-time domain. 
  mesh::tree_structure *
    _scheduling_tree_structure; //!< Temporal tree structure used for scheduling
                                //!< the FMM operations

  std::vector< std::vector< full_matrix * > > 
    _clusterwise_nearfield_matrices;  //!< nearfield matrices for all the space-
                                      //!< time leaf clusters and their
                                      //!< nearfield clusters.
  
  std::list< mesh::scheduling_time_cluster* > 
    _m_list;  //!< M-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster* >
    _m2l_list; //!< M2L-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster* >
    _l_list;  //!< L2L-list for the execution of the FMM.
  std::list< mesh::scheduling_time_cluster* >
    _n_list;  //!< N-list for the execution of the FMM.
  std::vector< std::pair< mesh::scheduling_time_cluster*, lo > >  
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
  lou _n_moments_to_receive_upward; //!< Number of grouped moments which have to
                                    //!< be received in the upward path of the
                                    //!< FMM.
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
  int _order_regular; //!< Triangle quadrature order for the regular integrals.
                      //!< Used for computation of quadratures in S2M steps.
  int _m2l_integration_order;  //!< _m2l_integration_order + 1 quadrature
                               //!< points are used for the approximation of
                               //!< the m2l coefficients.
  int _contribution_size; //!< Size of a contribution (moment or local 
                          //!< contribution) of a single spacetime cluster.
  mutable bem::chebyshev_evaluator
    _chebyshev;  //!< Evaluator of the Chebyshev polynomials.
                 //!< @todo check if necessary in the final code

  mutable bem::lagrange_interpolant
    _lagrange;  //!< Evaluator of the Lagrange polynomials.

  sc _alpha;  //!< Heat conductivity.

  /**
   * Calls all S2M operations associated with a given scheduling time cluster.
   * @param[in] sources Global sources containing the once used for the S2M
   *                    operation.
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output file.
   */
  void call_s2m_operations( const block_vector & sources,
    mesh::scheduling_time_cluster* time_cluster, bool verbose, 
    std::string verbose_file ) const;

  /**
   * Applies the S2M operation for the given source cluster and sources.
   * @param[in] sources Global sources containing the once used for the S2M
   *                    operation.
   * @param[in] source_cluster  Considered spacetime cluster.
   * @todo Use buffers instead of reallocating sources and aux buffer in every
   * function call?
   * @todo Store the quadratures of Chebyshev polynomials in space and Lagrange
   * polynomials in time again?
   */
  void apply_s2m_operation( const block_vector & source_vector, 
    mesh::general_spacetime_cluster* source_cluster ) const;

  /**
   * Calls all M2M operations associated with a given scheduling time cluster.
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output file.
   * @todo Currently dummy routine. 
   */
  void call_m2m_operations( 
    mesh::scheduling_time_cluster* time_cluster, bool verbose, 
    std::string verbose_file );

  /**
   * Calls all M2L operations associated with a given pair of scheduling time 
   * clusters.
   * @param[in] src_cluster Scheduling time cluster which acts as source in M2L.
   * @param[in] tar_cluster Scheduling time cluster which acts as target in M2L.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output file.
   * @todo Currently dummy routine. 
   */
  void call_m2l_operations( mesh::scheduling_time_cluster* src_cluster,
    mesh::scheduling_time_cluster* tar_cluster, bool verbose, 
    std::string verbose_file );
  /**
   * Calls all L2L operations associated with a given scheduling time cluster.
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output file.
   * @todo Currently dummy routine. 
   */
  void call_l2l_operations( 
    mesh::scheduling_time_cluster* time_cluster, bool verbose, 
    std::string verbose_file );
  /**
   * Calls all L2T operations associated with a given scheduling time cluster.
   * @param[in] time_cluster  Considered scheduling time cluster.
   * @param[in,out] output_vector Vector to which the results are added.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output file.
   * @todo Currently dummy routine. 
   */
  void call_l2t_operations( mesh::scheduling_time_cluster* time_cluster, 
    std::vector< sc > & output_vector, bool verbose, std::string verbose_file );
  /**
   * Calls all nearfield operations associated with a given pair of scheduling 
   * time clusters.
   * @param[in] sources Global sources containing the once used for the nearfield 
   *                    operation.
   * @param[in] src_cluster Scheduling time cluster which acts as source for the
   *                        nearfield operations.
   * @param[in] tar_cluster Scheduling time cluster which acts as target for the
   *                        nearfield operations.
   * @param[in,out] output_vector Vector to which the results are added.
   * @param[in] verbose If true, the required time is written to file.
   * @param[in] verbose_file  If @p verbose is true, this is used as output file.
   * @todo Currently dummy routine. 
   */
  void call_nearfield_operations( const std::vector< sc > & sources,
    mesh::scheduling_time_cluster* src_cluster, 
    mesh::scheduling_time_cluster* tar_cluster, 
    std::vector< sc > & output_vector, bool verbose, std::string verbose_file );

  /**
   * Calls MPI_Testsome for an array of Requests to check for received data.
   * @param[in,out] array_of_requests Array containing the MPI requests which are 
   *                                  checked.
   * @param[in,out] array_of_indices  Array in which the indices of the completed
   *                                  requests are stored. This is used as an 
   *                                  input variable to avoid reallocation in each
   *                                  function call.
   * @param[in,out] outcount  Stores the number of Requests which are completed.
   * @param[in] verbose If true, the process lists all the received data, and 
   *                    reports about the time needed to process it (in case 
   *                    moments in the upward path were received)
   * @param[in] verbose_file  If @p verbose is true, this is used as output file.
   */
  void check_for_received_data( MPI_Request * array_of_requests, 
    int array_of_indices[ ], int & outcount, bool verbose, 
    std::string verbose_file ) const;

  /**
   * Returns an iterator pointing to the next cluster in the l-list whose 
   * dependencies are satisfied. In case a cluster is found the status is updated.
   * If no cluster is found the iterator points to the end of the list and the
   * status is not modified.
   * @param[in] l_list  A list containing the clusters of @ref _l_list whose 
   *                    operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in the list this iterator 
   *                              points to it. Else it points to the end of the
   *                              list. 
   * @param[out] status Set to 2 if a cluster is found.
   */
  void find_cluster_in_l_list( 
    std::list< mesh::scheduling_time_cluster* > & l_list,
    std::list< mesh::scheduling_time_cluster* >::iterator & 
      it_next_cluster, char & status ) const;

  /**
   * Returns an iterator pointing to the next cluster in the m-list whose 
   * dependencies are satisfied. In case a cluster is found the status is updated.
   * If no cluster is found the iterator points to the end of the list and the
   * status is not modified.
   * @param[in] m_list  A list containing the clusters of @ref _m_list whose 
   *                    operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in the list this iterator 
   *                              points to it. Else it points to the end of the
   *                              list. 
   * @param[out] status Set to 1 if a cluster is found.
   */
  void find_cluster_in_m_list( 
    std::list< mesh::scheduling_time_cluster* > & m_list,
    std::list< mesh::scheduling_time_cluster* >::iterator & 
      it_next_cluster, char & status ) const;

  /**
   * Returns an iterator pointing to the next cluster in the m2l-list whose 
   * dependencies are satisfied. In case a cluster is found the status is updated.
   * If no cluster is found the iterator points to the end of the list and the
   * status is not modified.
   * @param[in] m2l_list  A list containing the clusters of @ref _m2l_list whose 
   *                      operations have not been executed yet.
   * @param[out] it_next_cluster  If a cluster is found in the list this iterator 
   *                              points to it. Else it points to the end of the
   *                              list. 
   * @param[out] status Set to 3 if a cluster is found.
   */
  void find_cluster_in_m2l_list( 
    std::list< mesh::scheduling_time_cluster* > & m2l_list,
    std::list< mesh::scheduling_time_cluster* >::iterator & 
      it_next_cluster, char & status ) const;

  /**
   * Updates dependency flags or sends moments for M2L operations.
   * @param[in] communicator  Communicator used for sending.
   * @param[in] src_cluster Considered scheduling time cluster. If a cluster in 
   *                        its send list is handled by a different process, the
   *                        moments are send to this process.
   * @todo Currently dummy routine. Sended data needs to be exchanged.
   */
  void provide_moments_for_m2l( const MPI_Comm communicator, 
    mesh::scheduling_time_cluster* src_cluster );

  /**
   * Updates dependency flags or sends moments for upward path.
   * @param[in] communicator  Communicator used for sending.
   * @param[in] child_cluster Considered scheduling time cluster. If its parent
   *                          is handled by a different process, the processed
   *                          moments are send from the local copy of the parent
   *                          cluster to this process. 
   * @todo Currently dummy routine. Sended data needs to be exchanged.
   */
  void provide_moments_to_parents( const MPI_Comm communicator, 
    mesh::scheduling_time_cluster* child_cluster );

  /**
   * Sends local contributions for downward path if necessary.
   * @param[in] communicator  Communicator used for sending.
   * @param[in] parent_cluster Considered scheduling time cluster. If a child
   *                           of it is handled by a different process, the local
   *                           contributions are send to this process.
   * @todo Currently dummy routine. Sended data needs to be exchanged.
   */
  void provide_local_contributions_to_children( const MPI_Comm communicator, 
    mesh::scheduling_time_cluster* parent_cluster );

  /**
   * Starts all receive operations given by a vector of pairs of clusters and 
   * process ids.
   * @param[in,out] array_of_requests The MPI_Requests of the non-blocking receive
   *                                  operations are stored in this array. It is 
   *                                  expected to have at least the size of 
   *                                  @p receive_vector .
   */
  void start_receive_operations( MPI_Request array_of_requests[ ] ) const ;

  /**
   * Compute quadrature of the Chebyshev polynomials and p0 basis functions for
   * the spatial part of a spacetime cluster
   * @param[in] source_cluster  Cluster for whose spatial component the 
   *                            quadratures are computed
   * @param[out]  T Full matrix where the quadratures are stored. The elements of 
   *                the cluster vary along the rows, the order of the polynomial
   *                along the columns of the matrix.
   * @todo Include padding appropriately.
   */
  void compute_chebyshev_quadrature_p0( 
    mesh::general_spacetime_cluster* source_cluster,
    full_matrix & T ) const;

  void compute_lagrange_quadrature(
    mesh::general_spacetime_cluster* source_cluster, 
    full_matrix & L ) const;

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

};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_PFMM_MATRIX_H_ */