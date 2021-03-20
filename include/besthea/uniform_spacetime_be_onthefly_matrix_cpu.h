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

/** @file uniform_spacetime_be_onthefly_matrix_cpu.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_CPU_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_CPU_H_

#include "besthea/block_matrix.h"

#include <array>


namespace besthea::onthefly {
  template< class kernel_type, class test_space_type, class trial_space_type >
  class uniform_spacetime_be_onthefly_matrix_cpu;

  constexpr bool quick_matrix_vals = false; // for performance testing purposes. if true, then matrix values are not calculated the correct time-consuming way, but quickly (and wrongly)
}



/**
 *  Class representing a boundary element matrix, whose elements are computed
 *  during multiplication on the fly, on the CPU.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu
  : public besthea::linear_algebra::block_matrix
{
protected:
  /**
   *  Stores quadrature nodes for reference element
   */
  struct quadrature_reference {
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

    std::array< lo, 4>
      _sizes; //!< Sizes
    
    lo _max_size; //!< Maximum size
  };

  /**
   *  Stores quadrature nodes mapped to specific element
   */
  struct quadrature_nodes {
    std::vector< sc, besthea::allocator_type< sc > >
      _xs;  //!< First coordinates of quadrature nodes
    std::vector< sc, besthea::allocator_type< sc > >
      _ys;  //!< Second coordinates of quadrature nodes
    std::vector< sc, besthea::allocator_type< sc > >
      _zs;  //!< Third coordinates of quadrature nodes

    quadrature_nodes(lo size) {
      _xs.resize( size );
      _ys.resize( size );
      _zs.resize( size );
    }
  };

public:
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using distributed_block_vector_type
    = besthea::linear_algebra::distributed_block_vector;  //!< Block vector type.

  /**
   * Constructor.
   * @param[in] kernel Spacetime kernel antiderivative object.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   * @param[in] order_singular Line quadrature order for regularized quadrature.
   * @param[in] order_regular Triangle quadrature order for regular quadrature.
   */
  uniform_spacetime_be_onthefly_matrix_cpu( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular = 4, int order_regular = 4 );

  uniform_spacetime_be_onthefly_matrix_cpu(
    const uniform_spacetime_be_onthefly_matrix_cpu & that )
    = delete;
  
  /**
   * Destructor.
   */
  virtual ~uniform_spacetime_be_onthefly_matrix_cpu( );

  /**
   * Prints info on the object.
   */
  void print_info( ) const {
    std::cout
      << "besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu"
      << std::endl;
    std::cout << "  number of blocks: " << _block_dim << std::endl;
    std::cout << "  dimension of each block: " << _dim_domain
              << " x " << _dim_range << std::endl;
  }

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans Flag for transpose (must be false, transposed
   * onthefly matrices are not yet implemented)
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( const block_vector_type & x, block_vector_type & y,
   bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const override;

  /*!
   * @brief y = beta * y + alpha * (this)^trans * x. Not implemented.
   * @param[in] x
   * @param[in,out] y
   * @param[in] trans
   * @param[in] alpha
   * @param[in] beta
   */
  virtual void apply( [[maybe_unused]] const distributed_block_vector_type & x,
    [[maybe_unused]] distributed_block_vector_type & y,
    [[maybe_unused]] bool trans = false, [[maybe_unused]] sc alpha = 1.0,
    [[maybe_unused]] sc beta = 0.0 ) const override {};

protected:

  void get_values_regular      (sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst, const quadrature_nodes & quadr_nodes_trl) const;
  void get_values_singular     (sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst, const quadrature_nodes & quadr_nodes_trl) const;
  void get_values_delta0       (sc * values_out,           lo i_test, lo i_trial, int n_shared_vertices, int rot_test, int rot_trial,
    const quadrature_nodes & quadr_nodes_tst, const quadrature_nodes & quadr_nodes_trl) const;
  void get_values_delta0special(sc * values_out,           lo i_test, lo i_trial, int n_shared_vertices, int rot_test, int rot_trial,
    const quadrature_nodes & quadr_nodes_tst, const quadrature_nodes & quadr_nodes_trl) const;

  void apply_regular_cpu(  const block_vector_type & x_perm, block_vector_type & y_perm, sc alpha = 1.0 ) const;
  void apply_regular_cpu(  const block_vector_type & x_perm, block_vector_type & y_perm, sc alpha, lo tst_elem_start, lo tst_elem_end ) const;
  void apply_singular_cpu( const block_vector_type & x_perm, block_vector_type & y_perm, sc alpha = 1.0 ) const;
  void apply_delta0_cpu(   const block_vector_type & x_perm, block_vector_type & y_perm, sc alpha = 1.0 ) const;

  /**
   * Initializes the quadrature reference structure.
   */
  void init_quadrature();

  /**
   * Determines the configuration of two triangular elements.
   * @param[in] i_test Index of the test element.
   * @param[in] i_trial Index of the trial element.
   * @param[out] type_int Type of the configuration (number of vertices shared).
   * @param[out] rot_test Virtual rotation of the test element.
   * @param[out] rot_trial Virtual rotation of the trial element.
   */
  void get_type( lo i_test, lo i_trial, int & n_shared_vertices, int & rot_test,
    int & rot_trial ) const;

  /**
   * Maps the quadrature nodes from reference triangle to specified
   * test element.
   * @param[in] i_tst Index of the test element triangle to map nodes to.
   * @param[in] n_shared_vertices Number of shared vertices between the test
   * and trial elements
   * @param[in] rot_test Virtual rotation of the test element.
   * @param[out] quadr_nodes_tst Structure to write mapped nodes to.
   */
  void triangles_to_geometry_tst( lo i_tst, int n_shared_vertices, int rot_test,
    quadrature_nodes & quadr_nodes_tst) const ;
    
  /**
   * Maps the quadrature nodes from reference triangle to specified
   * trial element.
   * @param[in] i_trl Index of the trial element triangle to map nodes to.
   * @param[in] n_shared_vertices Number of shared vertices between the test
   * and trial elements
   * @param[in] rot_trial Virtual rotation of the trial element.
   * @param[out] quadr_nodes_trl Structure to write mapped nodes to.
   */
  void triangles_to_geometry_trl( lo i_trl, int n_shared_vertices, int rot_trial,
    quadrature_nodes & quadr_nodes_trl) const ;
  
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

protected:

  quadrature_reference quadr_reference; //!< Reference quadrature nodes
  kernel_type * _kernel; //!< Kernel temporal antiderivative.
  test_space_type * _test_space; //!< Boundary element test space.
  trial_space_type * _trial_space; //!< Boundary element trial space.
  int _order_singular; //!< Line quadrature order for the singular integrals.
  int _order_regular; //!< Triangle quadrature order for the regular integrals.
  

  static constexpr std::array< int, 5 > map{ 0, 1, 2, 0,
    1 };  //!< Auxiliary array for mapping DOFs under
          // rotation (regularized quadrature). Performs fast modulo 3.

  static constexpr std::array< int, 4 > n_simplices{ 1, 2, 5,
    6 };  //!< Number of simplices for all configurations (disjoint, shared
          // vertex, shared edge, identical).

};


#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_CPU_H_ */
