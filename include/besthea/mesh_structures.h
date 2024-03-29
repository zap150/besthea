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

/** @file mesh_structures.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_MESH_STRUCTURES_H_
#define INCLUDE_BESTHEA_MESH_STRUCTURES_H_

#include "besthea/distributed_spacetime_cluster_tree.h"
#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/general_spacetime_cluster.h"
#include "besthea/mesh.h"
#include "besthea/scheduling_time_cluster.h"
#include "besthea/spacetime_mesh_generator.h"
#include "besthea/spacetime_slice.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/temporal_mesh.h"
#include "besthea/tetrahedral_spacetime_mesh.h"
#include "besthea/tetrahedral_volume_mesh.h"
#include "besthea/time_cluster.h"
#include "besthea/time_cluster_tree.h"
#include "besthea/tree_structure.h"
#include "besthea/triangular_surface_mesh.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"
#include "besthea/volume_space_cluster.h"
#include "besthea/volume_space_cluster_tree.h"

#ifdef BESTHEA_USE_CUDA
#include "besthea/uniform_spacetime_tensor_mesh_gpu.h"
#endif

#endif /* INCLUDE_BESTHEA_MESH_STRUCTURES_H_ */
