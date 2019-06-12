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

/** @file basis_tri_p1.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_BASIS_TRI_P1_H_
#define INCLUDE_BESTHEA_BASIS_TRI_P1_H_

#include "besthea/basis_function.h"
#include "besthea/triangular_surface_mesh.h"

namespace besthea {
  namespace bem {
    class basis_tri_p1;
  }
}

class besthea::bem::basis_tri_p1 : public besthea::bem::basis_function {
 public:
  basis_tri_p1( ) = delete;
  basis_tri_p1( mesh_type & mesh );
  virtual ~basis_tri_p1( );

  virtual lo dimension_local( lo i_elem );

  virtual lo dimension_global( );

  virtual void local_to_global( lo i_elem, adjacency type,
    int rotation, bool swap, std::vector< lo > indices );

  virtual void evaluate( lo i_elem, const std::vector< sc > & x1_ref,
    const std::vector< sc > & x2_ref, const sc * n, adjacency type,
    int rotation, bool swap, std::vector< matrix_type > & values );
};

#endif /* INCLUDE_BESTHEA_BASIS_TRI_P1_H_ */
