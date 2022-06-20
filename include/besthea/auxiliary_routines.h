/*
Copyright (c) 2022, VSB - Technical University of Ostrava and Graz University of
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

/** @file auxiliary_routines.h
 * @brief Provides some auxiliary routines which might be used in some
 * applications.
 */

#ifndef INCLUDE_AUXILIARY_ROUTINES_BESTHEA_H_
#define INCLUDE_AUXILIARY_ROUTINES_BESTHEA_H_

#include "besthea/distributed_block_vector.h"
#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/settings.h"

#include <vector>

/**
 * This routine is used to transfer the p0 projection coefficients of a function
 * on a fine mesh to a coarser mesh.
 *
 * @param[in] fine_vector p0 projection coefficients on the fine mesh.
 * @param[in] n_total_coarse_time_elems Number of time elements in the coarse
 * mesh.
 * @param[in] local_time_ref_map  A map from the indices of the time elements in
 * the coarse mesh to the indices of the respective child elements in the fine
 * mesh.
 * @param[in] local_start_idx_time  The start index of the local mesh of the
 * current process in the distributed space-time tensor mesh.
 * @param[in] space_ref_map A map from the indices of the space elements in
 * the coarse mesh to the indices of the respective child elements in the fine
 * mesh.
 * @param[in,out] coarse_output_vector  The result is stored in this vector.
 */
void sum_up_refined_mesh_vector_entries( const std::vector< sc > & fine_vector,
  const lo n_total_coarse_time_elems,
  const std::vector< std::vector< lo > > & local_time_ref_map,
  const lo local_start_idx_time,
  const std::vector< std::vector< lo > > & space_ref_map,
  std::vector< sc > & coarse_output_vector );

/**
 * Scales each entry of a distributed block vector by the inverse of the volume
 * of the corresponding space-time element. The size of the block vector has to
 * correspond to the numbers of elements in the mesh.
 *
 * This can be used to compute the p0 projection coefficients from a given
 * vector of L2 inner products of projected functions and the p0 basis
 * functions.
 *
 * @param[in] st_mesh Space-time mesh containing the elements.
 * @param[in,out] vec_to_scale  Vector which is scaled elementwise.
 */
void scale_vector_by_inv_elem_size(
  const besthea::mesh::distributed_spacetime_tensor_mesh & st_mesh,
  besthea::linear_algebra::distributed_block_vector & vec_to_scale );

/**
 * Prints integers in a cubic grid to files.
 * @param[in] values  Values that are printed. They have to be ordered first in
 * x-direction, then in y-direction and then in z-direction. E.g., the value for
 * the index combination (i_x,i_y,i_z) is given by values[i_x * edge_length^2 +
 * i_y * edge_length + i_z]
 * @param[in] edge_length Number of indices per direction, i.e. per edge.
 * @param[in] output_file_base  Basic file name, which determines the names of
 * the output files. They are of the form "BASE_z_idx_%.txt", where % is
 * replaced by the corresponding z index.
 */
bool print_integers_in_cubic_grid( std::vector< long long > values,
  lo edge_length, std::string & output_file_base );

#endif  // define INCLUDE_AUXILIARY_ROUTINES_BESTHEA_H_
