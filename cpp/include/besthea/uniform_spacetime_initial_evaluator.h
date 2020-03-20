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

/** @file uniform_spacetime_initial_evaluator.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_INITIAL_EVALUATOR_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_INITIAL_EVALUATOR_H_

#include "besthea/block_vector.h"
#include "besthea/fe_space.h"
#include "besthea/vector.h"

namespace besthea {
  namespace bem {
    template< class kernel_type, class space_type >
    class uniform_spacetime_initial_evaluator;
  }
}

/**
 *  Class representing a potential evaluator.
 */
template< class kernel_type, class space_type >
class besthea::bem::uniform_spacetime_initial_evaluator {
  using vector_type = linear_algebra::vector;  //!< Vector type.
  using block_vector_type
    = linear_algebra::block_vector;  //!< Block vector type.

 private:
  /**
   * Wraps the mapped quadrature point so that they can be private for OpenMP
   * threads
   */
  struct quadrature_wrapper {
    std::vector< sc, besthea::allocator_type< sc > >
      _wy;  //!< Spatial quadrature weights
    std::vector< sc, besthea::allocator_type< sc > >
      _y1_ref;  //!< First coordinates of quadrature nodes in the reference
                //!< spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y2_ref;  //!< Second coordinates of quadrature nodes in the reference
                //!< spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y3_ref;  //!< Third coordinates of quadrature nodes in the reference
                //!< spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y1;  //!< First coordinates of quadrature nodes in the spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y2;  //!< Second coordinates of quadrature nodes in the spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y3;  //!< Third coordinates of quadrature nodes in the spatial element

    std::vector< sc, besthea::allocator_type< sc > >
      _x1;  //!< First coordinates of evaluation points
    std::vector< sc, besthea::allocator_type< sc > >
      _x2;  //!< Second coordinates of evaluation points
    std::vector< sc, besthea::allocator_type< sc > >
      _x3;  //!< Third coordinates of evaluation points.

    std::vector< sc, besthea::allocator_type< sc > >
      _kernel_values;  //!< Buffer for storing kernel values.
  };

 public:
  /**
   * Constructor.
   * @param[in] kernel Spacetime kernel antiderivative object.
   * @param[in] space Boundary element space.
   * @param[in] n_timesteps Number of timesteps.
   * @param[in] timestep Timestep.
   * @param[in] order_spatial Triangle quadrature order for regular quadrature.
   */
  uniform_spacetime_initial_evaluator( kernel_type & kernel, space_type & space,
    lo n_timesteps, sc timestep, int order_spatial = 4 );

  uniform_spacetime_initial_evaluator(
    const uniform_spacetime_initial_evaluator & that )
    = delete;

  /**
   * Destructor.
   */
  ~uniform_spacetime_initial_evaluator( );

  /**
   * Evaluates the potential.
   * @param[in] x Point coordinates.
   * @param[in] density Density of the potential.
   * @param[out] result Result in the given points.
   */
  void evaluate( const std::vector< sc > & x, const vector_type & density,
    block_vector_type & result ) const;

 private:
  /**
   * Initializes quadrature structures.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( quadrature_wrapper & my_quadrature ) const;

  /**
   * Maps the quadrature nodes from the reference triangle to the actual
   * geometry.
   * @param[in] x1 Coordinates of the first node of the trial element.
   * @param[in] x2 Coordinates of the second node of the trial element.
   * @param[in] x3 Coordinates of the third node of the trial element.
   * @param[in] x4 Coordinates of the fourth node of the trial element.
   * @param[in,out] my_quadrature Structure holding the quadrature nodes.
   */
  void tetrahedron_to_geometry( const linear_algebra::coordinates< 3 > & x1,
    const linear_algebra::coordinates< 3 > & x2,
    const linear_algebra::coordinates< 3 > & x3,
    const linear_algebra::coordinates< 3 > & x4,
    quadrature_wrapper & my_quadrature ) const;

  kernel_type * _kernel;  //!< Kernel temporal antiderivative.

  space_type * _space;  //!< Boundary element space.

  lo _n_timesteps;  //!< Number of timesteps.

  sc _timestep;  //!< Timestep.

  int _order_spatial;  //!< Spatial tetrahedron quadrature order for the regular
                       //!< integrals.
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_INITIAL_EVALUATOR_H_ */
