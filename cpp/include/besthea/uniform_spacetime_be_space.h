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

/** @file be_space.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_

#include "besthea/basis_function.h"
#include "besthea/settings.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"

namespace besthea {
  namespace bem {
    class uniform_spacetime_be_space;
  }
}

/**
 *  Class representing a boundary element space.
 */
class besthea::bem::uniform_spacetime_be_space {
  using st_mesh = besthea::mesh::uniform_spacetime_tensor_mesh;
  using basis = besthea::bem::basis_function;

 public:
  uniform_spacetime_be_space( ) = delete;

  uniform_spacetime_be_space( const uniform_spacetime_be_space & that )
    = delete;

  ~uniform_spacetime_be_space( );

  /**
   * Constructing mesh from a file.
   * @param[in] space_mesh Reference to a triangular_surface_mesh.h.
   * @param[in] end_time Temporal interval set to (0,end_time).
   * @param[in] n_timesteps Number of timesteps.
   */
  uniform_spacetime_be_space(
    st_mesh & spacetime_mesh, basis & test, basis & trial );

 protected:
  st_mesh * _spacetime_mesh;  //!< uniform spacetime tensor mesh
  basis * _trial;             //!< spatial trial function (temporal is constant)
  basis * _test;              //!< spatial test function (temporal is constant)
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_BE_SPACE_H_ */
