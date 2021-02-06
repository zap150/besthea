
/** @file uniform_spacetime_be_onthefly_matrix_gpu.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_

#include "besthea/block_matrix.h"
#include "besthea/full_matrix.h"
#include "besthea/block_lower_triangular_toeplitz_matrix.h"
#include "besthea/uniform_spacetime_be_space.h"
#include "besthea/uniform_spacetime_be_onthefly_matrix_cpu.h"

#include <array>


namespace besthea::onthefly {
  template< class kernel_type, class test_space_type, class trial_space_type >
  class uniform_spacetime_be_onthefly_matrix_gpu;

  // TODO: think and move these to a more sensible location
  struct quadrature_wrapper_readonly_regular_raw;
  struct quadrature_wrapper_changing_regular_raw;
  struct triangluar_surface_mesh_raw;
  struct uniform_spacetime_tensor_mesh_raw;
  struct kernel_parameters;
}

struct besthea::onthefly::quadrature_wrapper_readonly_regular_raw {
  sc _x1_ref[64];
  sc _x2_ref[64];
  sc _y1_ref[64];
  sc _y2_ref[64];
  sc _w[64];
  lo _size; // actual size
};

struct besthea::onthefly::quadrature_wrapper_changing_regular_raw {
  sc _x1[64];
  sc _x2[64];
  sc _x3[64];
  sc _y1[64];
  sc _y2[64];
  sc _y3[64];
  sc _kernel_values[64];
  sc _kernel_values_2[64];
};

struct besthea::onthefly::triangluar_surface_mesh_raw {
  sc * d_element_areas;
  sc * d_node_coords; // XYZXYZXYZXYZ...
  lo * d_element_nodes; // 123123123123...
  sc * d_element_normals; // XYZXYZXYZXYZ
  lo n_elems;
  lo n_nodes;
};

struct besthea::onthefly::uniform_spacetime_tensor_mesh_raw {
  triangluar_surface_mesh_raw surf_mesh;
  lo n_temporal_elements; // outer block dimension of the block matrix
  sc timestep;
};

struct besthea::onthefly::kernel_parameters {
  sc alpha;
  sc sqrt_alpha;
  sc alpha_2;
  sc pi;
  sc sqrt_pi;
};


template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu
  : public besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu< kernel_type, test_space_type, trial_space_type >
{

public:
  using matrix_type = besthea::linear_algebra::full_matrix;  //!< Matrix type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using distributed_block_vector_type
    = besthea::linear_algebra::distributed_block_vector;  //!< Block vector type.

public:

  uniform_spacetime_be_onthefly_matrix_gpu( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular = 4, int order_regular = 4 );

  uniform_spacetime_be_onthefly_matrix_gpu(
    const uniform_spacetime_be_onthefly_matrix_gpu & that )
    = delete;
  
  virtual ~uniform_spacetime_be_onthefly_matrix_gpu( );

  void print_info( ) const {
    std::cout
      << "besthea::linear_algebra::uniform_spacetime_be_onthefly_matrix_gpu"
      << std::endl;
    std::cout << "  number of blocks: " << this->_block_dim << std::endl;
    std::cout << "  dimension of each block: " << this->_dim_domain
              << " x " << this->_dim_range << std::endl;
  }

  virtual void apply( const block_vector_type & x, block_vector_type & y,
   bool trans = false, sc alpha = 1.0, sc beta = 0.0 ) const override;

  virtual void apply( [[maybe_unused]] const distributed_block_vector_type & x,
    [[maybe_unused]] distributed_block_vector_type & y,
    [[maybe_unused]] bool trans = false, [[maybe_unused]] sc alpha = 1.0,
    [[maybe_unused]] sc beta = 0.0 ) const override {};

protected:

  virtual void apply_regular(  const block_vector_type & x, block_vector_type & y_perm, sc alpha = 1.0 ) const override;

private:

  void copy_mesh_data_to_gpu();

  void init_gpu_constant_memory() const;

private:
  uniform_spacetime_tensor_mesh_raw mesh_raw;
  

};


#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_ */
