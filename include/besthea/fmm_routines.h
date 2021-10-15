/*
Copyright (c) 2021, VSB - Technical University of Ostrava and Graz University of
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

/** @file fmm_routines.h
 * @brief Provides fmm routines used for pFMM matrices and initial pFMM
 * matrices.
 */

#ifndef INCLUDE_FMM_ROUTINES_H_
#define INCLUDE_FMM_ROUTINES_H_

#include "besthea/settings.h"

namespace besthea {
  namespace linear_algebra {
    class vector;
  }
}

#include <vector>

/**
 * Computes spatial M2M coefficients (corresponding to a Chebyshev expansion of
 * some kernel) for a series of spatial clusters. The coefficients can be used
 * for various M2M and L2L operations (e.g. for a
 * @ref besthea::linear_algebra::distributed_pFMM_matrix or
 * @ref besthea::linear_algebra::distributed_initial_pFMM_matrix )
 *
 * The coefficients are computed for 1D intervals. In case of cubic 3D
 * spatial clusters the M2M coefficients coincide for all spatial dimensions.
 * @param[in] n_space_levels Total number of spatial clusters. @p n_space_levels
 * -1 M2M matrices are computed (for each parent-child combination).
 * @param[in] spat_order  Spatial expansion order used in Chebyshev expansion.
 * It determines the size of the moments.
 * @param[in] spat_half_size_bounding_box_unpadded  Spatial half size of the
 * largest spatial cluster, not including any padding. Typically it corresponds
 * to the half size of the bounding box of the spatial computation domain.
 * @param[in] spatial_paddings_per_space_level  Vector containing the spatial
 * paddings for all @p n_space_levels spatial clusters.
 * @param[in,out] m2m_coeffs_s_left Vector of vectors where the m2m coefficients
 * of parents and their left children are stored for all @p n_space_levels - 1
 * levels.
 * @param[in,out] m2m_coeffs_s_right Vector of vectors where the m2m
 * coefficients of parents and their right children are stored for all @p
 * n_space_levels - 1 levels.
 */
void compute_spatial_m2m_coeffs( const lo n_space_levels, const lo spat_order,
  const sc spat_half_size_bounding_box_unpadded,
  const std::vector< sc > & spatial_paddings_per_space_level,
  std::vector< besthea::linear_algebra::vector > & m2m_coeffs_s_left,
  std::vector< besthea::linear_algebra::vector > & m2m_coeffs_s_right );

#endif /* INCLUDE_FMM_ROUTINES_H_ */
