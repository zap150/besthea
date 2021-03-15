
/** @file uniform_spacetime_be_onthefly_matrix_gpu.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_

#include "besthea/uniform_spacetime_be_onthefly_matrix_cpu.h"
#include "besthea/gpu_onthefly_helpers.h"

#include <array>


namespace besthea::onthefly {
  template< class kernel_type, class test_space_type, class trial_space_type >
  class uniform_spacetime_be_onthefly_matrix_gpu;
}





template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::onthefly::uniform_spacetime_be_onthefly_matrix_gpu
  : public besthea::onthefly::uniform_spacetime_be_onthefly_matrix_cpu< kernel_type, test_space_type, trial_space_type >
{

public:
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using distributed_block_vector_type
    = besthea::linear_algebra::distributed_block_vector;  //!< Block vector type.

public:

  uniform_spacetime_be_onthefly_matrix_gpu( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular, int order_regular,
    const besthea::onthefly::gpu_uniform_spacetime_tensor_mesh & gpu_mesh,
    int gpu_kernel_version = 2 );

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

private:

  void init_gpu_data();

  template<int quadr_order>
  void init_gpu_quadrature_memory() const;

  void apply_regular_gpu_begin( const block_vector_type & x, const block_vector_type & y, sc alpha, timer_collection & timers ) const;
  void apply_regular_gpu_finalize( block_vector_type & y ) const;

private:
  const gpu_uniform_spacetime_tensor_mesh * gpu_mesh;
  int n_gpus;
  int gpu_kernel_version;
  
  gpu_apply_vectors_data vectors_data;
  apply_load_distribution * load_distr;

};


#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ONTHEFLY_MATRIX_GPU_H_ */
