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

/** @file linear_algebra.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_LINEAR_ALGEBRA_H_
#define INCLUDE_BESTHEA_LINEAR_ALGEBRA_H_

#include "besthea/block_lower_triangular_toeplitz_matrix.h"
#include "besthea/block_mkl_cg_inverse.h"
#include "besthea/block_mkl_fgmres_inverse.h"
#include "besthea/block_row_matrix.h"
#include "besthea/block_vector.h"
#include "besthea/compound_block_linear_operator.h"
#include "besthea/compound_linear_operator.h"
#include "besthea/coordinates.h"
#include "besthea/distributed_block_vector.h"
#include "besthea/distributed_diagonal_matrix.h"
#include "besthea/distributed_initial_pFMM_matrix.h"
#include "besthea/distributed_pFMM_matrix.h"
#include "besthea/full_matrix.h"
#include "besthea/indices.h"
#include "besthea/low_rank_kernel.h"
#include "besthea/low_rank_matrix.h"
#include "besthea/mkl_cg_inverse.h"
#include "besthea/mkl_fgmres_inverse.h"
#include "besthea/sparse_matrix.h"
#include "besthea/vector.h"

#endif /* INCLUDE_BESTHEA_LINEAR_ALGEBRA_H_ */
