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

/** @file basis_function.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_BASIS_FUNCTION_H_
#define INCLUDE_BESTHEA_BASIS_FUNCTION_H_

#include "besthea/be_assembler.h"
#include "besthea/full_matrix.h"
#include "besthea/mesh.h"
#include "besthea/settings.h"

#include <vector>

namespace besthea {
  namespace bem {
    class basis_function;
  }
}

/**
 *  Class representing a basis function.
 */
class besthea::bem::basis_function {
 protected:
  using mesh_type = besthea::mesh::mesh;
  using adjacency = besthea::bem::adjacency;
  using matrix_type = besthea::linear_algebra::full_matrix;

 public:
  basis_function( ) : _mesh( nullptr ) {
  }

  basis_function( const basis_function & that ) = delete;

  virtual ~basis_function( ) {
  }

  virtual lo dimension_local( lo i_elem ) = 0;

  virtual lo dimension_global( ) = 0;

  virtual void local_to_global( lo i_elem, adjacency type, int rotation,
    bool swap, std::vector< lo > indices )
    = 0;

  virtual void evaluate( lo i_elem, const std::vector< sc > & x1_ref,
    const std::vector< sc > & x2_ref, const sc * n, adjacency type,
    int rotation, bool swap, std::vector< matrix_type > & values )
    = 0;

 protected:
  mesh_type * _mesh;
};

#endif /* INCLUDE_BESTHEA_BASIS_FUNCTION_H_ */
