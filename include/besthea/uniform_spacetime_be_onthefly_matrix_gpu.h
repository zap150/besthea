
/** @file uniform_spacetime_be_onthefly_matrix_gpu.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_

#include "besthea/block_matrix.h"
#include "besthea/full_matrix.h"
#include "besthea/block_lower_triangular_toeplitz_matrix.h"
#include "besthea/uniform_spacetime_be_space.h"

#include <array>


namespace besthea {
  template< class kernel_type, class test_space_type, class trial_space_type >
  class uniform_spacetime_be_onthefly_matrix_gpu;
}



template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::uniform_spacetime_be_onthefly_matrix_gpu
  : public besthea::linear_algebra::block_matrix
{
private:
  struct quadrature_wrapper_readonly {
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
  };

  struct quadrature_wrapper_changing {
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

    quadrature_wrapper_changing(lo size) {
      _x1.resize( size );
      _x2.resize( size );
      _x3.resize( size );
      _y1.resize( size );
      _y2.resize( size );
      _y3.resize( size );
      _kernel_values.resize( size );
      _kernel_values_2.resize( size );
    }
  };
  lo quadr_size;

public:
  using matrix_type = besthea::linear_algebra::full_matrix;  //!< Matrix type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.



  void hello_gpu_world(int number) const;

  uniform_spacetime_be_onthefly_matrix_gpu( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular = 4, int order_regular = 4 );

  uniform_spacetime_be_onthefly_matrix_gpu(
    const uniform_spacetime_be_onthefly_matrix_gpu & that )
    = delete;
  
  virtual ~uniform_spacetime_be_onthefly_matrix_gpu( );



  sc get( lo d, lo i, lo j, quadrature_wrapper_changing & quadr_changing ) const ;

  virtual void apply( const block_vector_type & x, block_vector_type & y,
   bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const override;

  void print( std::ostream & stream = std::cout ) const;

  void print_info( ) const {
    std::cout
      << "besthea::linear_algebra::uniform_spacetime_be_onthefly_matrix_gpu"
      << std::endl;
    std::cout << "  number of blocks: " << _block_dim << std::endl;
    std::cout << "  dimension of each block: " << _dim_domain
              << " x " << _dim_range << std::endl;
  }

  bool check_equal(besthea::linear_algebra::block_lower_triangular_toeplitz_matrix & assembled, sc epsilon) const ;

private:

  void init_quadrature();

  void get_type( lo i_test, lo i_trial, int & type_int, int & rot_test,
    int & rot_trial ) const;

  void triangles_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & y1,
    const linear_algebra::coordinates< 3 > & y2,
    const linear_algebra::coordinates< 3 > & y3, int type_int, int rot_test,
    int rot_trial, quadrature_wrapper_changing & quadr_changing) const ;

  sc get_value(lo delta, lo i_test, lo i_trial, quadrature_wrapper_changing & quadr_changing, bool special = false) const ;
  
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

private:

  quadrature_wrapper_readonly my_quadrature;
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


#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_ */
