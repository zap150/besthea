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

/** @file uniform_spacetime_be_matrix_onthefly_gpu.h
 * @brief GPU version of onthefly boudary element matrix.
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_MATRIX_ONTHEFLY_GPU_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_MATRIX_ONTHEFLY_GPU_H_

#include "besthea/gpu_onthefly_helpers.h"
#include "besthea/uniform_spacetime_be_matrix_onthefly_cpu.h"
#include "besthea/uniform_spacetime_tensor_mesh_gpu.h"

#include <array>

namespace besthea::linear_algebra::onthefly {
  template< class kernel_type, class test_space_type, class trial_space_type >
  class uniform_spacetime_be_matrix_onthefly_gpu;
}

/*!
 *  Class representing a boundary element matrix, whose elements are computed
 *  during multiplication on the fly, on the GPU.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::linear_algebra::onthefly::
  uniform_spacetime_be_matrix_onthefly_gpu
  : public besthea::linear_algebra::onthefly::
      uniform_spacetime_be_matrix_onthefly_cpu< kernel_type, test_space_type,
        trial_space_type > {
 public:
  using block_vector_type
    = besthea::linear_algebra::block_vector;            //!< Block vector type.
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.
  using distributed_block_vector_type
    = besthea::linear_algebra::distributed_block_vector;  //!< Block vector
                                                          //!< type.

 public:
  /*!
   * Constructor.
   * @param[in] kernel Spacetime kernel antiderivative object.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   * @param[in] order_singular Line quadrature order for regularized quadrature.
   * @param[in] order_regular Triangle quadrature order for regular quadrature.
   * @param[in] gpu_mesh Spacetime mesh resident on the GPU
   * @param[in] gpu_kernel_version Version of GPU multiplication algorithm
   */
  uniform_spacetime_be_matrix_onthefly_gpu( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int order_singular, int order_regular,
    const besthea::mesh::uniform_spacetime_tensor_mesh_gpu & gpu_mesh,
    int gpu_kernel_version = 1 );

  uniform_spacetime_be_matrix_onthefly_gpu(
    const uniform_spacetime_be_matrix_onthefly_gpu & that )
    = delete;

  /*!
   * Destructor.
   */
  virtual ~uniform_spacetime_be_matrix_onthefly_gpu( );

  /*!
   * Prints info on the object.
   */
  void print_info( ) const {
    std::cout << "besthea::onthefly::uniform_spacetime_be_matrix_onthefly_gpu"
              << std::endl;
    std::cout << "  number of blocks: " << this->_block_dim << std::endl;
    std::cout << "  dimension of each block: " << this->_dim_domain << " x "
              << this->_dim_range << std::endl;
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
    [[maybe_unused]] sc beta = 0.0 ) const override{ };

 private:
  /*!
   * Allocates GPU vectors, load distrubution and initializes GPU qudrature.
   */
  void init_gpu_data( );

  /*!
   * Copies regular quadrature reference nodes to GPU constant memory/
   */
  template< int quadr_order >
  void init_gpu_quadrature_memory( ) const;

  /*!
   * Asynchronously launches memory transfers and matrix multiplication kernel.
   * @param[in] x
   * @param[in] y
   * @param[in] alpha Scaling factor of vector x
   * @param[in] timers Timers for measuring elapsed time
   */
  void apply_regular_gpu_begin( const block_vector_type & x,
    const block_vector_type & y, sc alpha,
    besthea::linear_algebra::onthefly::helpers::timer_collection & timers )
    const;

  /*!
   * Waits for all GPU tasks to finish and copies vector y back to CPU memory.
   * @param[out] y
   */
  void apply_regular_gpu_finalize( block_vector_type & y ) const;

 private:
  const besthea::mesh::uniform_spacetime_tensor_mesh_gpu *
    gpu_mesh;              //!< GPU-resident mesh
  int n_gpus;              //!< Number of GPUs to use
  int gpu_kernel_version;  //!< Version of GPU multiplication algorithm

  besthea::linear_algebra::onthefly::helpers::gpu_apply_vectors_data
    vectors_data;  //!< GPU-resident vectors
  besthea::linear_algebra::onthefly::helpers::apply_load_distribution *
    load_distr;  //!< Object handling CPU-GPU load distribution
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_MATRIX_ONTHEFLY_GPU_H_ */
