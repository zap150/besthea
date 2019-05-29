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

#ifndef INCLUDE_BESTHEA_TRIANGULAR_SURFACE_MESH_H_
#define INCLUDE_BESTHEA_TRIANGULAR_SURFACE_MESH_H_

#include "besthea/settings.h"

#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class triangular_surface_mesh;
  }
}  // namespace besthea

using sc = besthea::scalar;
using lo = besthea::index;

/**
 *  Class representing a triangular mesh of a 3D surface
 */
class besthea::mesh::triangular_surface_mesh {
 public:
  triangular_surface_mesh( );

  /**
   * Constructing mesh from a file.
   * @param[in] file path to the file.
   */
  triangular_surface_mesh( const std::string & file );

  triangular_surface_mesh( const triangular_surface_mesh & ) = delete;

  ~triangular_surface_mesh( );

  /**
   * Prints info on the object.
   */
  void print_info( );

  bool load( const std::string & file );

 protected:
  lo _n_nodes;     //!< number of nodes
  lo _n_elements;  //!< number of elements

  std::vector< sc > _nodes1;  //!< first coordinates of all nodes
  std::vector< sc > _nodes2;  //!< second coordinates of all nodes
  std::vector< sc > _nodes3;  //!< third coordinates of all nodes

  std::vector< lo > _elements;  //!< indices into _nodesX

  std::pair< std::vector< lo >, std::vector< lo > >
    _orientation;  //!< orientation of n := (x2-x1)x(x3-x1) and -n

  /**
   * Precomputes areas of elements.
   */
  void init_area( );

  /**
   * Precomputes exterior normal vectors of elements.
   */
  void init_normals( );

  /**
   * Initializes edges.
   */
  void init_edges( );
};

#endif /* INCLUDE_BESTHEA_TRIANGULAR_SURFACE_MESH_H_ */
