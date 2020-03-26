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

/** @file pFMM_matrix.h
 * @brief Represents matrix approximated by the pFMM method
 */

#ifndef INCLUDE_BESTHEA_PFMM_MATRIX_H_
#define INCLUDE_BESTHEA_PFMM_MATRIX_H_

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/block_linear_operator.h"
#include "besthea/chebyshev_evaluator.h"
#include "besthea/fast_spacetime_be_space.h"
#include "besthea/full_matrix.h"
#include "besthea/lagrange_interpolant.h"
#include "besthea/local_vector_routines.h"
#include "besthea/matrix.h"
#include "besthea/settings.h"
#include "besthea/space_cluster_tree.h"
#include "besthea/spacetime_cluster.h"
#include "besthea/spacetime_cluster_tree.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/sparse_matrix.h"
#include "besthea/time_cluster.h"
#include "besthea/time_cluster_tree.h"
#include "besthea/vector.h"

#include <utility>

namespace besthea {
  namespace linear_algebra {
    template< class kernel_type, class target_space, class source_space >
    class pFMM_matrix;
  }
}

/**
 * Class representing a matrix approximated by the pFMM method.
 */
template< class kernel_type, class target_space, class source_space >
class besthea::linear_algebra::pFMM_matrix
  : public besthea::linear_algebra::block_linear_operator {
 public:
  using sparse_matrix_type
    = besthea::linear_algebra::sparse_matrix;  //!< Sparse matrix type.
  using full_matrix_type
    = besthea::linear_algebra::full_matrix;  //!< Sparse matrix type.
  using space_cluster
    = besthea::mesh::space_cluster;  //!< Spacetime cluster type.
  using spacetime_cluster
    = besthea::mesh::spacetime_cluster;  //!< Spacetime cluster type.
  using spacetime_tree_type
    = besthea::mesh::spacetime_cluster_tree;  //!< Spacetime tree type.
  using time_cluster_type
    = besthea::mesh::time_cluster;  //!< Time cluster type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Default constructor.
   */
  pFMM_matrix( )
    : _spacetime_tree( nullptr ),
      _uniform( false ),
      _temp_order( 5 ),
      _spat_order( 5 ),
      _m2l_integration_order( _spat_order ),
      _chebyshev( _spat_order ),
      _alpha( 1.0 ),
      _source_space_is_p0( true ),
      _target_space_is_p0( true ) {
  }

  pFMM_matrix( const pFMM_matrix & that ) = delete;

  /**
   * Destructor
   */
  virtual ~pFMM_matrix( ) {
    #ifdef NEARFIELD_CLUSTERWISE
    for ( auto it = _clusterwise_nearfield_matrices.begin( );
          it != _clusterwise_nearfield_matrices.end( ); ++it ) {
      for ( auto it_in = ( *it ).begin( ); it_in != ( *it ).end( ); ++it_in ) {
        delete *it_in;
      }
    }
    #else
    lo matrix_idx = 0;
    for ( auto it = _nearfield_matrices.begin( );
          it != _nearfield_matrices.end( ); ++it ) {
      const std::pair< lo, lo > & indices
        = _nearfield_block_map.at( matrix_idx );

      if ( *it != nullptr ) {
        if ( _uniform && indices.second == 0 ) {
          delete *it;
          *it = nullptr;
        } else if ( !_uniform ) {
          delete *it;
          *it = nullptr;
        }
      }
      matrix_idx++;
    }
    #endif

    for ( auto it = _farfield_matrices.begin( );
          it != _farfield_matrices.end( ); ++it ) {
      if ( *it != nullptr ) {
        delete *it;
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

  //  virtual void apply2( const block_vector_type & x, block_vector_type & y,
  //    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * Sets the underlying spacetime tree. The size of the clusterwise nearfield
   * matrix container is set appropriately.
   * @param[in] spacetime_tree The tree.
   */
  void set_tree( spacetime_tree_type * spacetime_tree );

  /*!
   * Setter for the heat conductivity parameter.
   * @param[in] alpha Heat conductivity.
   */
  void set_alpha( sc alpha ) {
    this->_alpha = alpha;
  }

  /*!
   * Setter for the uniform parameter.
   * @param[in] uniform Value of the uniform parameter.
   */
  void set_uniform( bool uniform ) {
    _uniform = uniform;
  }
  /*!
   * Sets the dimension of the matrix.
   * @param[in] block_dim Block dimension.
   * @param[in] dim_domain Number of columns.
   * @param[in] dim_range Number of rows.
   */
  void resize( lo block_dim, lo dim_domain, lo dim_range ) {
    this->_block_dim = block_dim;
    this->_dim_domain = dim_domain;
    this->_dim_range = dim_range;
  }

  /*!
   * Creates a nearfield matrix.
   * @param[in] test_idx Index of test interval.
   * @param[in] trial_idx Index of trial interval.
   * @param[in] n_duplications Number of matrix duplications for uniform-time
   * assembly.
   */
  full_matrix_type * create_nearfield_matrix(
    lo test_idx, lo trial_idx, lo n_duplications = 1 );

  /*!
   * Creates a nearfield matrix for two clusters
   * @param[in] leaf_index  Index of the target leaf cluster.
   * @param[in] source_index  Index of the source cluster in the nearfield of 
   *                          the target cluster.  
   */
  full_matrix_type * create_clusterwise_nearfield_matrix( 
    lo leaf_index, lo source_index );

  /*!
   * Allocates sparse matrix of given farfield nonapproximated block and
   * returns a pointer.
   * @param[in] test_idx Index of the testing function.
   * @param[in] trial_idx Index of the trial function
   */
  full_matrix_type * create_farfield_matrix( lo test_idx, lo trial_idx );

  /*!
   * Compute the temporal m2m matrices for all levels.
   */
  void compute_temporal_m2m_matrices( );

  /*!
   * Compute the spatial m2m coefficients for all levels.
   */
  void compute_spatial_m2m_coeffs( );

  /*!
   * Applies the temporal m2m operation for a child moment and adds the result
   * to the parent moment.
   * @param[in] child_moment  Moment of the child which is passed up.
   * @param[in] level   Temporal level of the parent cluster.
   * @param[in] is_left_child True if the temporal cluster of the child is a
   *                          left child.
   * @param[in,out] parent_moment  Result of m2m operation is added to it.
   */
  void apply_temporal_m2m( full_matrix_type const & child_moment,
    const lo level, const bool is_left_child,
    full_matrix_type & parent_moment ) const;

  /*!
   * Applies the spatial m2m operation for a child moment and adds the result
   * to the parent moment.
   * @param[in] child_moment  Moment of the child which is passed up.
   * @param[in] level   Spatial level of the parent cluster.
   * @param[in] octant  Configuration of the space cluster of the child
   *                    relative to the parent ( from 8 configurations )
   * @param[in,out] parent_moment  Result of m2m operation is added to it.
   */
  void apply_spatial_m2m( full_matrix_type const & child_moment, const lo level,
    const slou octant, full_matrix_type & parent_moment ) const;

  /*!
   * Applies the temporal l2l operation for a given local contribution of a
   * parent and adds the result to a childs local contribution.
   * @param[in] parent_local  Local contribution of the parent which is passed
   *                          down.
   * @param[in] level   Temporal level of the parent cluster.
   * @param[in] is_left_child True if the temporal cluster of the child is a
   *                          left child.
   * @param[in,out] child_local  Result of l2l operation is added to it.
   */
  void apply_temporal_l2l( full_matrix_type const & parent_local,
    const lo level, const bool is_left_child,
    full_matrix_type & child_local ) const;

  /*!
   * Applies the spatial l2l operation for a given local contribution of a
   * parent and adds the result to a childs local contribution.
   * @param[in] parent_local  Local contribution of the parent which is passed
   *                          down.
   * @param[in] level   Spatial level of the parent cluster.
   * @param[in] octant  Configuration of the space cluster of the child
   *                    relative to the parent ( from 8 configurations )
   * @param[in,out] child_local  Result of l2l operation is added to it.
   */
  void apply_spatial_l2l( full_matrix_type const & parent_local, const lo level,
    const slou octant, full_matrix_type & child_local ) const;

  /*!
   * Sets order of the Lagrange and Chebyshev polynomials.
   * @param[in] spat_order Order of the Chebyshev polynomials
   * @param[in] temp_order Order of the Lagrange polynomials
   */
  void set_order( int spat_order, int temp_order ) {
    _spat_order = spat_order;
    _temp_order = temp_order;
    _chebyshev.set_order( spat_order );
  }

  /*!
   * Sets the integration order for the m2l coefficients
   * @param[in] m2l_integration_order M2L integration order.
   */
  void set_m2l_integration_order( int m2l_integration_order ) {
    this->_m2l_integration_order = m2l_integration_order;
  }

 private:
  /*!
   * @brief Executes s2m operations for all leaves of the spacetime cluster
   * and a given vector x for spatial basis functions in p0
   * @param[in] x Vector for multiplication.
   * @param[in] trans Whether to transpose the matrix.
   * @note The results are stored in the matrices \p _moment_contribution in the
   * respective spacetime clusters.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_s2m_operations(
    block_vector_type const & x, bool trans = false ) const;

  /*!
   * @brief Executes s2m operations for all leaves of the spacetime cluster
   * and a given vector x for spatial basis functions in p0
   * @param[in] x Vector for multiplication.
   * @note The results are stored in the matrices \p _moment_contribution in the
   * respective spacetime clusters.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_s2m_operations_p0( block_vector_type const & x ) const;

  /*!
   * @brief Executes s2m operations for all leaves of the spacetime cluster
   * and a given vector x for spatial basis functions in p1.
   * @param[in] x Vector for multiplication.
   * @note The results are stored in the matrices \p _moment_contribution in the
   * respective spacetime clusters.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_s2m_operations_p1( block_vector_type const & x ) const;
  
    /*!
   * @brief Executes s2m operations for all leaves of the spacetime cluster
   * and a given vector x for spatial basis functions in p1.
   * The integrals in space used for the s2m operations include the normal 
   * derivatives of the Chebyshev polynomials. 
   * @param[in] x Vector for multiplication.
   * @note The results are stored in the matrices \p _moment_contribution in the
   * respective spacetime clusters.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_s2m_operations_p1_normal_drv( block_vector_type const & x ) const;

  /*!
   * @brief Executes s2m operations for all leaves of the spacetime cluster
   * and a given vector \p x for dimension \p dim of the surface curls of p1
   * basis functions (first term of the hypersingular operator)
   * @param[in] x Vector for multiplication.
   * @param[in] dim Dimension of the surface curls (0,1,2) which is considered.
   * @note The results are stored in the matrices \p _moment_contribution in the
   * respective spacetime clusters.
   * @note The method uses matrices of integrals over polynomials and a surface
   * curl mapping, which are computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_s2m_operations_curl_p1_hs(
    block_vector_type const & x, const lo dim ) const;

  /*!
   * @brief Executes s2m operations for all leaves of the spacetime cluster
   * and a given vector \p x for dimension \p dim of the normal vector (second
   * term of the hypersingular operator)
   * @param[in] x Vector for multiplication.
   * @param[in] dim Dimension of the surface curls (0,1,2) which is considered.
   * @note The results are stored in the matrices \p _moment_contribution in the
   * respective spacetime clusters.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_s2m_operations_p1_normal_hs(
    block_vector_type const & x, const lo dim ) const;

  /*!
   * @brief Executes all m2m operations recursively.
   * @param[in] root  M2m operations are applied for this spacetime cluster and
   *                  all its descendants.
   * @param[in] buffer_matrices Vector of 8 matrices of size >=
   *    ( _temp_order + 1 ) * ( ( _spat_order + 3 ) choose 3 ).
   * \todo Currently some unnecessary m2m operations are executed (for clusters
   * without direct interactions or parental interactions).
   */
  void call_m2m_operations( spacetime_cluster * root,
    std::vector< full_matrix_type > & buffer_matrices ) const;

  /*!
   * @brief Applies the spacetime M2L operation to the moment contribution of
   * a source cluster and adds the result to the local contribution of the
   * target.
   * @param[in,out] target_cluster  Target cluster for the m2l operation.
   * @param[in] source_cluster  Source cluster for the m2l operation.
   * @param[in] buffer_for_gaussians  Vector with size >= ( _spat_order + 1 )^2
   *    * ( _temp_order + 1 )^2 to store intermediate results in the computation
   *    of the m2l coefficients.
   * @param[in] buffer_for_coeffs  Vector with size >= ( _spat_order + 1 )^2
   *    * ( _temp_order + 1 )^2 to store m2l coefficients.
   * @param[in] aux_buffer_0  Matrix with ( _temp_order + 1 )^2 times
   *  ( ( _spat_order + 3 ) choose 3 ) entries for intermediate m2l results.
   * @param[in] aux_buffer_1  Matrix with ( _temp_order + 1 )^2 times
   *  ( ( _spat_order + 3 ) choose 3 ) entries for intermediate m2l results.
   * \todo Routine requires changes if source and target cluster are from
   *       different cluster trees.
   */
  void apply_m2l_operation( spacetime_cluster * target_cluster,
    spacetime_cluster * source_cluster, vector_type & buffer_for_gaussians,
    vector_type & buffer_for_coeffs, full_matrix_type & aux_buffer_0,
    full_matrix_type & aux_buffer_1 ) const;

  /*!
   * @brief Executes all m2l operations recursively starting at the root in a
   * cluster tree.
   * @param[in] root  M2l operations are applied for this spacetime cluster and
   *                  all its descendants.
   * @param[in] buffer_for_gaussians  Vector with size >= ( _spat_order + 1 )^2
   *    * ( _temp_order + 1 )^2 to store intermediate results in the computation
   *    of the m2l coefficients.
   * @param[in] buffer_for_coeffs  Vector with size >= ( _spat_order + 1 )^2
   *    * ( _temp_order + 1 )^2 to store m2l coefficients.
   * @param[in] aux_buffer_0  Matrix with ( _temp_order + 1 )^2 times
   *  ( ( _spat_order + 3 ) choose 3 ) entries for intermediate m2l results.
   * @param[in] aux_buffer_1  Matrix with ( _temp_order + 1 )^2 times
   *  ( ( _spat_order + 3 ) choose 3 ) entries for intermediate m2l results.
   * @note The function \ref apply_m2l_operation is used to execute the m2l
   *        operations.
   */
  void call_m2l_operations( spacetime_cluster * root,
    vector_type & buffer_for_gaussians, vector_type & buffer_for_coeffs,
    full_matrix_type & aux_buffer_0, full_matrix_type & aux_buffer_1 ) const;

  /*!
   * @brief Executes all l2l operations recursively.
   * @param[in] root  L2l operations are applied for this spacetime cluster and
   *                  all its descendants.
   * @param[in] buffer_matrices Vector of 8 matrices of size >=
   *    ( _temp_order + 1 ) * ( ( _spat_order + 3 ) choose 3 ).
   * \todo Currently some unnecessary l2l operations are executed (for clusters
   * without direct interactions or parental interactions).
   */
  void call_l2l_operations( spacetime_cluster * root,
    std::vector< full_matrix_type > & buffer_matrices ) const;

  /*!
   * @brief Executes l2t operations for all leaves of the spacetime cluster
   * and adds the results to a vector y.
   * @param[in,out] y Vector to which the results are added.
   * @param[in] trans Whether to transpose the matrix.
   * @note For the computations the matrices \p _local_contribution in the
   * respective spacetime clusters are used.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_l2t_operations( block_vector_type & y, bool trans = false ) const;

  /*!
   * @brief Executes l2t operations for all leaves of the spacetime cluster
   * and adds the results to a vector y for spatial basis functions in p0.
   * @param[in,out] y Vector to which the results are added.
   * @note For the computations the matrices \p _local_contribution in the
   * respective spacetime clusters are used.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_l2t_operations_p0( block_vector_type & y ) const;

   /*!
   * @brief Executes l2t operations for all leaves of the spacetime cluster
   * and adds the results to a vector y for spatial basis functions in p1. 
   * @param[in,out] y Vector to which the results are added.
   * @note For the computations the matrices \p _local_contribution in the
   * respective spacetime clusters are used.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_l2t_operations_p1( block_vector_type & y ) const; 

  /*!
   * @brief Executes l2t operations for all leaves of the spacetime cluster
   * and adds the results to a vector y for spatial basis functions in p1.
   * The integrals in space used for the l2t operations include the normal 
   * derivatives of the Chebyshev polynomials. 
   * @param[in,out] y Vector to which the results are added.
   * @note For the computations the matrices \p _local_contribution in the
   * respective spacetime clusters are used.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_l2t_operations_p1_normal_drv( block_vector_type & y ) const;

  /*!
   * @brief Executes l2t operations for all leaves of the spacetime cluster
   * for dimension \p dim of the surface curls of p1 basis functions and adds
   * the results to a vector \p y (first term of the hypersingular operator)
   * @param[in,out] y Vector to which the results are added.
   * @param[in] dim Dimension of the surface curls (0,1,2) which is considered.
   * @note For the computations the matrices \p _local_contribution in the
   * respective spacetime clusters are used.
   * @note The method uses matrices of integrals over polynomials and a surface
   * curl mapping, which are computed in
   * \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_l2t_operations_curl_p1_hs(
    block_vector_type & y, const lo dim ) const;

  /*!
   * @brief Executes l2t operations for all leaves of the spacetime cluster
   * for dimension \p dim of the normal vector and adds the results to a
   * vector \p y (second term of the hypersingular operator)
   * @param[in,out] y Vector to which the results are added.
   * @param[in] dim Dimension of the surface curls (0,1,2) which is considered.
   * @note For the computations the matrices \p _local_contribution in the
   * respective spacetime clusters are used.
   * @note The method uses matrices of integrals over polynomials which are
   * computed in \ref besthea::bem::fast_spacetime_be_assembler.
   */
  void apply_l2t_operations_p1_normal_hs(
    block_vector_type & y, const lo dim ) const;

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
  virtual void apply_sl_dl( const block_vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x using block vectors for
   * hypersingular operator.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose of individual blocks (not the whole
   * block matrix!).
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply_hs( const block_vector_type & x, block_vector_type & y,
    bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

  /*!
   * @brief Computes coupling coefficients for the spacetime m2l operation
   *        for one of the three space dimensions implicitly given.
   * @param[in] src_time_nodes  Interpolation nodes in time for the source
   *                            cluster.
   * @param[in] tar_time_nodes  Interpolation nodes in time for the target
   *                            cluster.
   * @param[in] cheb_nodes  Chebyshev nodes of degree ( _spat_order + 1 )
   * @param[in] evaluated_chebyshev  Vector of evaluated Chebyshev polynomials
   *      with degree <= _spat_order at \p cheb_nodes as given by
   *       besthea::bem::chebyshev_evaluator::evaluate.
   * @param[in] half_size   Half size in space of the current clusters along the
   *                        dimension for which the coefficients are computed.
   * @param[in] center_diff   The appropriate component of the difference vector
   *                          (target_center - source_center).
   * @param[in] buffer_for_gaussians  Vector with size >= ( _spat_order + 1 )^2
   *    * ( _temp_order + 1 )^2 to store intermediate results in the computation
   *    of the m2l coefficients.
   * @param[in,out] coupling_coeffs Vector with size >= ( _spat_order + 1 )^2
   *    * ( _temp_order + 1 )^2 to store m2l coefficients.
   */
  void compute_coupling_coeffs( const vector_type & src_time_nodes,
    const vector_type & tar_time_nodes, const vector_type & cheb_nodes,
    const vector_type & evaluated_chebyshev, const sc half_size,
    const sc center_diff, vector_type & buffer_for_gaussians,
    vector_type & coupling_coeffs ) const;

  friend class spacetime_cluster_tree;  //!< enable the tree to access private
                                        //!< variables

  spacetime_tree_type * _spacetime_tree;  //!< tree hierarchically decomposing
                                          //!< spatial and temporal domains
  std::vector< full_matrix_type * >
    _nearfield_matrices;  //!< temporal nearfield blocks
  std::vector< std::pair< lo, lo > >
    _nearfield_block_map;  //!< mapping from block index to pair of matching
                           //!< temporal clusters

  std::vector< std::vector< full_matrix_type * > >
    _clusterwise_nearfield_matrices;  //! nearfield matrices for all the space-
                                      //! time leaf clusters and their
                                      //! nearfield clusters

  std::vector< full_matrix_type * >
    _farfield_matrices;  //!< nonapproximated temporal farfield blocks
  std::vector< std::pair< lo, lo > >
    _farfield_block_map;  //!< mapping from block index to pair of matching
                          //!< temporal clusters
  bool _uniform;          //!< specifies whether time-discretization is uniform
                          //!< (duplicates blocks)
  std::vector< full_matrix_type >
    _m2m_matrices_t_left;  //!< left temporal
                           //!< m2m matrices stored levelwise
  std::vector< full_matrix_type >
    _m2m_matrices_t_right;  //!< right temporal
                            //!< m2m matrices stored levelwise

  std::vector< vector_type >
    _m2m_coeffs_s_dim_0_left;  //!< left spatial
                               //!< m2m matrices along dimension 0 stored
                               //!< levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_0_right;  //!< right spatial
                                //!< m2m matrices along dimension 0 stored
                                //!< levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_1_left;  //!< left spatial
                               //!< m2m matrices along dimension 1 stored
                               //!< levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_1_right;  //!< right spatial
                                //!< m2m matrices along dimension 1 stored
                                //!< levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_2_left;  //!< left spatial
                               //!< m2m matrices along dimension 2 stored
                               //!< levelwise
  std::vector< vector_type >
    _m2m_coeffs_s_dim_2_right;  //!< right spatial
                                //!< m2m matrices along dimension 2 stored
                                //!< levelwise
  int _temp_order;  //!< degree of interpolation polynomials in time for pFMM

  int _spat_order;  //!< degree of Chebyshev polynomials for expansion in
                    //!< space in pFMM

  int _m2l_integration_order;  //!< _m2l_integration_order + 1 quadrature
                               //!< points are used for the approximation of
                               //!< the m2l coefficients.

  mutable bem::chebyshev_evaluator
    _chebyshev;  //!< Evaluator of the Chebyshev polynomials.
                 //!< \todo TODO check if necessary in the final code

  sc _alpha;  //!< Heat conductivity.

  bool _source_space_is_p0;  //!< True if spatial source space is p0.
                             //!< \todo TODO: control more elegantly.
  bool _target_space_is_p0;  //!< True if spatial target space is p0.
                             //!< \todo TODO: control more elegantly.

  //   besthea::bem::spacetime_heat_kernel
  //     _heat_kernel; //!< Evaluator of the Heat Kernel
};

/** Typedef for the single layer p0-p0 PFMM matrix */
typedef besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >
  pFMM_matrix_heat_sl_p0p0;

/** Typedef for the single layer p1-p1 PFMM matrix */
typedef besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >
  pFMM_matrix_heat_sl_p1p1;

/** Typedef for the double layer p0-p1 PFMM matrix */
typedef besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >
  pFMM_matrix_heat_dl_p0p1;

/** Typedef for the spatially adjoint double layer p1-p0 PFMM matrix */
typedef besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_dl_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p0 > >
  pFMM_matrix_heat_adjdl_p1p0;

/** Typedef for the hypersingular p1-p1 PFMM matrix */
typedef besthea::linear_algebra::pFMM_matrix<
  besthea::bem::spacetime_heat_hs_kernel_antiderivative,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::fast_spacetime_be_space< besthea::bem::basis_tri_p1 > >
  pFMM_matrix_heat_hs_p1p1;

#endif /* INCLUDE_BESTHEA_PFMM_MATRIX_H_ */
