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

/** @file fast_spacetime_be_space.h
 * @brief Boundary element space for fast BEM.
 */

#ifndef INCLUDE_BESTHEA_FAST_SPACETIME_BE_SPACE_H_
#define INCLUDE_BESTHEA_FAST_SPACETIME_BE_SPACE_H_

#include "besthea/block_vector.h"
#include "besthea/settings.h"
#include "besthea/spacetime_be_space.h"
#include "besthea/spacetime_cluster_tree.h"
#include "besthea/spacetime_tensor_mesh.h"

namespace besthea {
  namespace bem {
    template< class basis_type >
    class fast_spacetime_be_space;
  }
}

template< class basis_type >
class besthea::bem::fast_spacetime_be_space
  : public besthea::bem::spacetime_be_space< basis_type > {
  /**
   * Wraps the mapped quadrature point so that they can be private for OpenMP
   * threads
   */
  struct quadrature_wrapper {
    std::vector< sc, besthea::allocator_type< sc > >
      _wx;  //!< Spatial quadrature weights
    std::vector< sc, besthea::allocator_type< sc > >
      _x1_ref;  //!< First coordinates of quadrature nodes in the reference
                //!< spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2_ref;  //!< Second coordinates of quadrature nodes in the reference
                //!< spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x1;  //!< First coordinates of quadrature nodes in the spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2;  //!< Second coordinates of quadrature nodes in the spatial element
    std::vector< sc, besthea::allocator_type< sc > >
      _x3;  //!< Third coordinates of quadrature nodes in the spatial element

    std::vector< sc, besthea::allocator_type< sc > >
      _wt;  //!< Temporal quadrature weights
    std::vector< sc, besthea::allocator_type< sc > >
      _t_ref;  //!< Coordinates of quadrature nodes in the reference temporal
               //!< element
    std::vector< sc, besthea::allocator_type< sc > >
      _t;  //!< Coordinates of quadrature nodes in the temporal element
  };

 public:
  using st_mesh_type
    = besthea::mesh::spacetime_tensor_mesh;  //!< Spacetime mesh type.
  using block_vector_type
    = besthea::linear_algebra::block_vector;  //!< Block vector type.

  fast_spacetime_be_space( const fast_spacetime_be_space & that ) = delete;

  /**
   * Destructor.
   */
  virtual ~fast_spacetime_be_space( );

  /**
   * Constructing space from a spacetime tree.
   */
  fast_spacetime_be_space( spacetime_cluster_tree & tree ) : _tree( &tree ) {
  }

  /**
   * Returns pointer to the tree.
   */
  st_mesh_type * get_tree( ) {
    return _tree;
  }

  /**
   * Returns pointer to the tree.
   */
  const st_mesh_type * get_tree( ) const {
    return _tree;
  }

 private:
  spacetime_cluster_tree * _tree;  //!< tree storing a hierarchical subdivision
                                   //!< of the space-time domain
};

#endif /* INCLUDE_BESTHEA_FAST_SPACETIME_BE_SPACE_H_ */
