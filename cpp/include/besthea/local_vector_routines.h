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

/** @file local_vector_routines.h
 * @brief Routines to access local parts of a block vector corresponding to a
 * clustering scheme.
 */

#ifndef INCLUDE_BESTHEA_LOCAL_VECTOR_ROUTINES_H_
#define INCLUDE_BESTHEA_LOCAL_VECTOR_ROUTINES_H_

#include "besthea/block_vector.h"
#include "besthea/fast_spacetime_be_space.h"
#include "besthea/settings.h"
#include "besthea/spacetime_cluster.h"
#include "besthea/vector.h"

// forward declaration of fast_spacetime_be_space and basis functions
namespace besthea {
  namespace bem {
    template< class basis_type >
    class fast_spacetime_be_space;
    class basis_tri_p1;
    class basis_tri_p0;
  }
}

/*!
  * Gets local part of a block vector corresponding to dofs in a spacetime 
  * cluster.
  * @param[in] cluster  Cluster determining the local dofs.
  * @param[in] block_vector Input vector in block format.
  * @param[in,out] local_vector Local part of block vector.
  * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
  *                     basis functions. It determines the dofs.
  * @warning The local vector must have the correct size.
  * @note The local vector is not a block vector anymore, but a contiguous
  *       vector.
  */
template< class space_type >
void get_local_part_of_block_vector( 
  besthea::mesh::spacetime_cluster* cluster,
  const besthea::linear_algebra::block_vector & block_vector,
  besthea::linear_algebra::vector & local_vector );

/*!
  * Adds local vector to appropriate positions of a block vector. The positions
  * are determined by the dofs in a spacetime cluster.
  * @param[in] cluster  Cluster determining the positions in the block_vector to
  *                     which the local vector is added.
  * @param[in] local_vector Local part of block vector to be added.
  * @param[in,out] block_vector Block vector to which the local vector is added.
  * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
  *                     basis functions. It determines the dofs.
  * @note The entries in the local vector are ordered according to the ordering
  *       of the time elements and spatial dofs in the spacetime cluster (time
  *       step after time step).
  */
template< class space_type >
void add_local_part_to_block_vector( 
  besthea::mesh::spacetime_cluster* cluster,
  const besthea::linear_algebra::vector & local_vector, 
  besthea::linear_algebra::block_vector & block_vector);

#endif /* INCLUDE_BESTHEA_LOCAL_VECTOR_ROUTINES_H_ */

