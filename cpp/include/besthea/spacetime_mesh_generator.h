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

/** @file spacetime_mesh_generator.h
 * @brief Class generating spacetime mesh decomposed into time slices.
 */

#ifndef INCLUDE_BESTHEA_SPACETIME_MESH_GENERATOR_H_
#define INCLUDE_BESTHEA_SPACETIME_MESH_GENERATOR_H_

#include "besthea/temporal_mesh.h"
#include "besthea/triangular_surface_mesh.h"

#include <sstream>
#include <string>

namespace besthea {
  namespace mesh {
    class spacetime_mesh_generator;
  }
}

/**
 * Class serving for generating of tensor-product spacetime mesh split into
 * slices for use by the distributed FMM.
 */
class besthea::mesh::spacetime_mesh_generator {
 public:
  /**
   * Constructor taking spatial and temporal mesh
   * @param[in] space_mesh Initial spatial mesh
   * @param[in] time_mesh Initial temporal mesh with nodes and elements in
   * time-ascending order. Each temporal element represents a time-slice.
   * @param[in] refinement How much to refine individual slices before saving
   * (number of bisections).
   */
  spacetime_mesh_generator( triangular_surface_mesh & space_mesh,
    temporal_mesh & time_mesh, lo refinement = 0 );

  /**
   * Constructor taking spatial and temporal mesh
   * @param[in] space_mesh Initial spatial mesh
   * @param[in] end_time End of the interval [ 0, end_time ]
   * @param[in] n_timeslices Number of timeslices to equally divide the interval
   * into. Each temporal element represents a time-slice.
   * @param[in] refinement How much to refine individual slices before saving
   * (number of bisections).
   */
  spacetime_mesh_generator( triangular_surface_mesh & space_mesh, sc end_time,
    lo n_timeslices, lo refinement = 0 );

  /**
   * Constructs generator from spatial and temporal mesh files
   * @param[in] file_space File storing the spatial mesh
   * @param[in] file_time File storing the temporal mesh with nodes and elements
   * in time-ascending order. Each temporal element represents a time-slice.
   * @param[in] refinement How much to refine individual slices before saving
   * (number of bisections).
   */
  spacetime_mesh_generator( const std::string & file_space,
    const std::string & file_time, lo refinement = 0 );

  /** Copy constructor */
  spacetime_mesh_generator( spacetime_mesh_generator & that ) = delete;

  /** Destrutor */
  ~spacetime_mesh_generator( );

  /**
   * Generates number of spacetime meshes and saves them into files.
   * @param[in] directory Directory where to save the output data
   * @param[in] file Name (prefix) of the output file
   * @param[in] suffix Suffix of the file
   */
  bool generate( const std::string & directory, const std::string & file,
    const std::string & suffix );

 protected:
  triangular_surface_mesh *
    _space_mesh;               //!< pointer to a triangular_surface_mesh.h
  temporal_mesh * _time_mesh;  //!< pointer to a temporal_mesh.h
  lo _refinement;  //!< number of bisection refinements of the initial temporal
                   //!< slices
  bool _delete_time_mesh;   //!< time mesh was created by the class, delete it
  bool _delete_space_mesh;  //!< space mesh was created by the class, delete it
};

#endif /* INCLUDE_BESTHEA_SPACETIME_MESH_GENERATOR_H_ */
