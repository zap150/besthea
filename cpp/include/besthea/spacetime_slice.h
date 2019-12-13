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

/** @file spacetime_slice.h
 * @brief Represents a slice of a space-time mesh decomposition in a temporal
 * dimension.
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_SLICE_H_
#define INCLUDE_BESTHEA_SPACETIME_SLICE_H_

#include "besthea/spacetime_tensor_mesh.h"

namespace besthea {
  namespace mesh {
    class spacetime_slice;
  }
}

/**
 * Class representing a slice of a space-time mesh decomposition in a temporal
 * dimension.
 */
class besthea::mesh::spacetime_slice {
 public:
	/**
	 * Constructor.
	 * @param[in] space_file File with the spatial mesh.
	 * @param[in] time_file File with the temporal mesh.
	 */
  spacetime_slice(
    const std::string & space_file, const std::string & time_file )
    : _space_mesh( space_file ),
      _time_mesh( time_file ),
      _spacetime_mesh( _space_mesh, _time_mesh ) {
  }

 protected:
  triangular_surface_mesh _space_mesh; //!< Spatial mesh.
  temporal_mesh _time_mesh; //!< Temporal mesh.
  spacetime_tensor_mesh _spacetime_mesh; //!< Spacetime mesh.
};

#endif /* INCLUDE_BESTHEA_SPACETIME_SLICE_H_ */
