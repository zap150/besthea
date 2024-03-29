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

/** @file distributed_fast_spacetime_be_space.h
 * @brief Boundary element space for fast BEM.
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_SPACE_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_SPACE_H_

#include "besthea/distributed_block_vector.h"
#include "besthea/distributed_spacetime_cluster_tree.h"
#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/settings.h"
#include "besthea/spacetime_be_space.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/temporal_mesh.h"
#include "besthea/triangular_surface_mesh.h"

namespace besthea {
  namespace bem {
    template< class basis_type >
    class distributed_fast_spacetime_be_space;
  }
}

/**
 *  Class representing a fast boundary element space.
 */
template< class basis_type >
class besthea::bem::distributed_fast_spacetime_be_space
  : public besthea::bem::spacetime_be_space< basis_type,
      besthea::linear_algebra::distributed_block_vector > {
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

  using s_mesh_type
    = besthea::mesh::triangular_surface_mesh;        //!< Spatial mesh type.
  using t_mesh_type = besthea::mesh::temporal_mesh;  //!< Temporal mesh type.

 public:
  /**
   * Type of the basis function
   */
  typedef basis_type type;

  distributed_fast_spacetime_be_space(
    const distributed_fast_spacetime_be_space & that )
    = delete;

  /**
   * Destructor.
   */
  virtual ~distributed_fast_spacetime_be_space( );

  /**
   * Constructing space from a spacetime tree
   * @param[in] tree Distributed spacetime cluster tree.
   */
  distributed_fast_spacetime_be_space(
    mesh::distributed_spacetime_cluster_tree & tree );

  /**
   * Returns pointer to the tree.
   */
  mesh::distributed_spacetime_cluster_tree * get_tree( ) {
    return _tree;
  }

  /**
   * Returns pointer to the (unmodifiable) tree.
   */
  const mesh::distributed_spacetime_cluster_tree * get_tree( ) const {
    return _tree;
  }

  /**
   * Returns the spacetime mesh.
   */
  const mesh::distributed_spacetime_tensor_mesh & get_mesh( ) const {
    return _tree->get_mesh( );
  }

  /**
   * Projects a function to the boundary element space.
   * @param[in] f Function to be projected.
   * @param[out] projection Projection vector.
   * @param[in] order_matrix Spatial quadrature order to assemble the mass
   * matrix.
   * @param[in] order_rhs_spatial Spatial triangular quadrature order to
   * assemble the right-hand side.
   * @param[in] order_rhs_temporal Temporal line quadrature order to assemble
   * the right-hand side.
   */
  virtual void L2_projection(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    linear_algebra::distributed_block_vector & projection, int order_matrix = 2,
    int order_rhs_spatial = 5, int order_rhs_temporal = 4 ) const override;

  /**
   * Projects a function to the boundary elements space, using a weighted L2
   * scalar product, which uses the elementwise weight (h_x^{1/2} + h_t^{1/4}).
   * @param[in] f Function to be projected.
   * @param[out] projection Projection vector.
   * @param[in] order_matrix Spatial quadrature order to assemble the mass
   * matrix.
   * @param[in] order_rhs_spatial Spatial triangular quadrature order to
   * assemble the right-hand side.
   * @param[in] order_rhs_temporal Temporal line quadrature order to assemble
   * the right-hand side.
   */
  void weighted_L2_projection(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    linear_algebra::distributed_block_vector & projection, int order_matrix = 2,
    int order_rhs_spatial = 5, int order_rhs_temporal = 4 ) const;

  /**
   * Returns the L2 relative error |f-approximation|/|f|.
   * @param[in] f Function in infinite dimensional space.
   * @param[in] approximation Approximation in finite dimensional space.
   * @param[in] order_rhs_spatial Spatial triangular quadrature order to
   * assemble the right-hand side.
   * @param[in] order_rhs_temporal Temporal line quadrature order to assemble
   * the right-hand side.
   */
  virtual sc L2_relative_error(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    const linear_algebra::distributed_block_vector & approximation,
    int order_rhs_spatial = 5, int order_rhs_temporal = 4 ) const override;

  /**
   * Returns the absolute error |f-approximation| in a weighted L2 norm, where
   * the elementwise weight (h_x^{1/2} + h_t^{1/4}) is used.
   * @param[in] f Function in infinite dimensional space.
   * @param[in] approximation Approximation in finite dimensional space.
   * @param[in] order_rhs_spatial Spatial triangular quadrature order to
   * assemble the right-hand side.
   * @param[in] order_rhs_temporal Temporal line quadrature order to assemble
   * the right-hand side.
   */
  sc weighted_L2_absolute_error(
    sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
    const linear_algebra::distributed_block_vector & approximation,
    int order_rhs_spatial = 5, int order_rhs_temporal = 4 ) const;

  /*
   * Projects a function to the boundary element space. ONLY USE SPECIALIZED
   * FUNCTIONS!
   * @param[in] f Function to be projected.
   * @param[out] interpolation Interpolation vector.
   * @todo adapt this
   */
  // virtual void interpolation(
  //   sc ( *f )( sc, sc, sc, const linear_algebra::coordinates< 3 > &, sc ),
  //   block_vector_type & interpolation ) const override;

 private:
  mesh::distributed_spacetime_cluster_tree *
    _tree;  //!< distributed tree storing a hierarchical subdivision
            //!< of the local space-time domain
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_FAST_SPACETIME_BE_SPACE_H_ */
