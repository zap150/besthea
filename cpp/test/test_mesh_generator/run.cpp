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
#include "besthea/distributed_spacetime_tensor_mesh.h"
#include "besthea/settings.h"
#include "besthea/spacetime_mesh_generator.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

using besthea::mesh::distributed_spacetime_tensor_mesh;
using besthea::mesh::spacetime_mesh_generator;

int main( int argc, char * argv[] ) {
  int provided;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );

  int myRank;
  MPI_Comm_rank( MPI_COMM_WORLD, &myRank );

  if ( myRank == 0 ) {
    std::string space_file = "./mesh_files/cube_12.txt";
    std::string time_file = "./testfile.txt";

    spacetime_mesh_generator generator( space_file, time_file, 4 );

    generator.generate( "", "test_mesh", "txt" );
  }
  MPI_Barrier( MPI_COMM_WORLD );
  MPI_Comm comm = MPI_COMM_WORLD;
  distributed_spacetime_tensor_mesh( "test_mesh_d.txt", &comm );

  MPI_Finalize( );
}
