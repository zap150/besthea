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

/** @file distributed_spacetime_tensor_mesh.h
 * @brief Container holding sequence of space-time meshes distributed among MPI
 * processes.
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_TENSOR_MESH_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_TENSOR_MESH_H_

#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/temporal_mesh.h"
#include "besthea/triangular_surface_mesh.h"

#include <mpi.h>
#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class distributed_spacetime_tensor_mesh;
  }
}

/**
 * Class serving as a container holding spacetime tensor product meshes
 * distributed among MPI processes.
 */
class besthea::mesh::distributed_spacetime_tensor_mesh {
 public:
  /**
   * Constructor taking decomposition file and MPI communicator. Based on the
   * communicator, the class automatically evenly distributes meshes among MPI
   * processes.
   * @param[in] decomposition_file Path to the file describing the input mesh.
   * @param[in] comm Pointer to the MPI communicator associated with
   * decomposition.
   */
  distributed_spacetime_tensor_mesh(
    const std::string & decomposition_file, MPI_Comm * comm );

  /**
   * Destructor
   */
  ~distributed_spacetime_tensor_mesh( );

 protected:
  /**
   * Loads submeshes assigned to the current rank and merges them into one mesh.
   * @param[in] decomposition_file Path to the file describing the input mesh.
   */
  bool load( const std::string & decomposition_file );

  MPI_Comm * _comm;  //!< MPI communicator associated with the distributed mesh
  int _my_rank;      //!< MPI rank of current processor
  int _n_processes;  //!< total number of MPI processes in the communicator
  lo _n_meshes;      //!< total number of input meshes
  lo _n_meshes_per_rank;  //!< number of meshes owned by this process
  spacetime_tensor_mesh *
    _my_mesh;  //!< part of the global mesh owned by the rank
};

#endif /* INCLUDE_BESTHEA_DISTRIBUTED_SPACETIME_TENSOR_MESH_H_ */
