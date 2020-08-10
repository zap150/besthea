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
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< Full matrix type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
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
  virtual void apply( const block_vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

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
   * Sets the order of the Lagrange and Chebyshev polynomials.
   * @param[in] spat_order Order of the Chebyshev polynomials.
   * @param[in] temp_order Order of the Lagrange polynomials.
   */
  void set_order( int spat_order, int temp_order ) {
    _spat_order = spat_order;
    _temp_order = temp_order;
    _chebyshev.set_order( spat_order );
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
  full_matrix_type * create_nearfield_matrix( 
    lou leaf_index, lou source_index );

 private:
  const MPI_Comm * _comm; //!< MPI communicator associated with the pFMM matrix.
  int _my_rank; //!< MPI rank of current process.
  mesh::distributed_spacetime_cluster_tree * 
    _distributed_spacetime_tree;  //!< part of a distributed tree hierarchically 
                                  //!< decomposing the space-time domain. 
  mesh::tree_structure *
    _scheduling_tree_structure; //!< Temporal tree structure used for scheduling
                                //!< the FMM operations

  std::vector< std::vector< full_matrix_type * > > 
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
  int _m2l_integration_order;  //!< _m2l_integration_order + 1 quadrature
                               //!< points are used for the approximation of
                               //!< the m2l coefficients.
  mutable bem::chebyshev_evaluator
    _chebyshev;  //!< Evaluator of the Chebyshev polynomials.
                 //!< @todo check if necessary in the final code

  sc _alpha;  //!< Heat conductivity.                             
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_PFMM_MATRIX_H_ */