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

/** @file uniform_spacetime_be_assembler.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ASSEMBLER_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ASSEMBLER_H_

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/block_lower_triangular_toeplitz_matrix.h"
#include "besthea/uniform_spacetime_be_space.h"
#include "besthea/uniform_spacetime_heat_sl_kernel_antiderivative.h"

#include <array>

namespace besthea {
  namespace bem {
    template< class kernel_type, class test_space_type, class trial_space_type >
    class uniform_spacetime_be_assembler;
  }
}

/**
 *  Class representing a boundary element matrix assembler.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::bem::uniform_spacetime_be_assembler {
 public:
  uniform_spacetime_be_assembler( ) = delete;

  /**
   * Constructor.
   * @param[in] kernel Spcetime kernel antiderivative object.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   */
  uniform_spacetime_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space );

  /**
   * Destructor.
   */
  ~uniform_spacetime_be_assembler( );

  /**
   * Assembles the spacetime matrix.
   * @param[out] global_matrix Block lower triangular Toeplitz matrix.
   */
  void assemble(
    besthea::linear_algebra::block_lower_triangular_toeplitz_matrix &
      global_matrix );

 private:
  kernel_type * _kernel;  //!< Kernel temporal antiderivative

  test_space_type * _test_space;  //!< Boundary element test space.

  trial_space_type * _trial_space;  //!< Boundary element trial space.

  const std::array< int, 5 > _map{ 0, 1, 2, 0,
    1 };  //!< Auxiliary array for mapping DOFs under
          // rotation (regularized quadrature).
};

template class besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;
template class besthea::bem::uniform_spacetime_be_assembler<
  besthea::bem::uniform_spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_ASSEMBLER_H_ */
