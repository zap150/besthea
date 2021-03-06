
/** @file uniform_spacetime_be_onthefly_matrix_cpu.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_CPU_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_CPU_H_

#include "besthea/block_matrix.h"
#include "besthea/full_matrix.h"
#include "besthea/block_lower_triangular_toeplitz_matrix.h"
#include "besthea/uniform_spacetime_be_space.h"

#include <array>


namespace besthea::onthefly {
  template< class kernel_type, class test_space_type, class trial_space_type >
  class uniform_spacetime_be_onthefly_matrix_cpu;

  constexpr bool quick_matrix_vals = false; // for performance testing purposes. if true, then matrix values are not calculated the correct time-consuming way, but quickly (and wrongly)
}



template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu
  : public besthea::linear_algebra::block_matrix
{
protected:
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
  using matrix_type = besthea::linear_algebra::full_matrix;  //!< Matrix type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using distributed_block_vector_type
    = besthea::linear_algebra::distributed_block_vector;  //!< Block vector type.



  uniform_spacetime_be_onthefly_matrix_cpu( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular = 4, int order_regular = 4 );

  uniform_spacetime_be_onthefly_matrix_cpu(
    const uniform_spacetime_be_onthefly_matrix_cpu & that )
    = delete;
  
  virtual ~uniform_spacetime_be_onthefly_matrix_cpu( );

  void print_info( ) const {
    std::cout
      << "besthea::linear_algebra::uniform_spacetime_be_onthefly_matrix_cpu"
      << std::endl;
    std::cout << "  number of blocks: " << _block_dim << std::endl;
    std::cout << "  dimension of each block: " << _dim_domain
              << " x " << _dim_range << std::endl;
  }

  virtual void apply( const block_vector_type & x, block_vector_type & y,
   bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const override;

  virtual void apply( [[maybe_unused]] const distributed_block_vector_type & x,
    [[maybe_unused]] distributed_block_vector_type & y,
    [[maybe_unused]] bool trans = false, [[maybe_unused]] sc alpha = 1.0,
    [[maybe_unused]] sc beta = 0.0 ) const override {};
    
  void apply_cpu( const block_vector_type & x, block_vector_type & y,
   bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const;

protected:

  void get_values_regular      (sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst, const quadrature_nodes & quadr_nodes_trl) const;
  void get_values_singular     (sc * values_out, lo delta, lo i_test, lo i_trial,
    const quadrature_nodes & quadr_nodes_tst, const quadrature_nodes & quadr_nodes_trl) const;
  void get_values_delta0       (sc * values_out,           lo i_test, lo i_trial, int n_shared_vertices, int rot_test, int rot_trial,
    const quadrature_nodes & quadr_nodes_tst, const quadrature_nodes & quadr_nodes_trl) const;
  void get_values_delta0special(sc * values_out,           lo i_test, lo i_trial, int n_shared_vertices, int rot_test, int rot_trial,
    const quadrature_nodes & quadr_nodes_tst, const quadrature_nodes & quadr_nodes_trl) const;

  void apply_regular(  const block_vector_type & x_perm, block_vector_type & y_perm, sc alpha = 1.0 ) const;
  void apply_singular( const block_vector_type & x_perm, block_vector_type & y_perm, sc alpha = 1.0 ) const;
  void apply_delta0(   const block_vector_type & x_perm, block_vector_type & y_perm, sc alpha = 1.0 ) const;

  void init_quadrature();

  void get_type( lo i_test, lo i_trial, int & n_shared_vertices, int & rot_test,
    int & rot_trial ) const;

  void triangles_to_geometry_tst( lo i_tst, int n_shared_vertices, int rot_test,
    quadrature_nodes & quadr_nodes_tst) const ;
  void triangles_to_geometry_trl( lo i_trl, int n_shared_vertices, int rot_trial,
    quadrature_nodes & quadr_nodes_trl) const ;
  
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
  
  void hypercube_to_triangles_vertex( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;
    
  void hypercube_to_triangles_edge( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;

  void hypercube_to_triangles_identical( sc ksi, sc eta1, sc eta2, sc eta3,
    int simplex, sc & x1_ref, sc & x2_ref, sc & y1_ref, sc & y2_ref,
    sc & jacobian ) const;

protected:

  quadrature_reference quadr_reference;
  kernel_type * _kernel;
  test_space_type * _test_space;
  trial_space_type * _trial_space;
  int _order_singular;
  int _order_regular;
  

  static constexpr std::array< int, 5 > map{ 0, 1, 2, 0,
    1 };  //!< Auxiliary array for mapping DOFs under
          // rotation (regularized quadrature). Performs fast modulo 3.

  static constexpr std::array< int, 4 > n_simplices{ 1, 2, 5,
    6 };  //!< Number of simplices for all configurations (disjoint, shared
          // vertex, shared edge, identical).

};


#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_CPU_H_ */
