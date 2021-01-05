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

#include "besthea/distributed_block_vector.h"
#include "besthea/settings.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

using besthea::linear_algebra::distributed_block_vector;

int main( int argc, char * argv[] ) {
  int provided;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );

  int myRank, n_processes;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( MPI_COMM_WORLD, &myRank );
  MPI_Comm_size( comm, &n_processes );

  distributed_block_vector vector( 5, { 1, 2 } );

  distributed_block_vector vector2( 5, 4, true, MPI_COMM_WORLD );

  std::vector< lo > my_blocks;
  if ( myRank == 0 ) {
    my_blocks.push_back( 0 );
    my_blocks.push_back( 1 );
  } else if ( myRank == 1 ) {
    my_blocks.push_back( 2 );
    my_blocks.push_back( 3 );
    my_blocks.push_back( 4 );
    my_blocks.push_back( 1 );
  } else {
    my_blocks.push_back( 3 );
  }
  distributed_block_vector vector3( my_blocks, 5, 6, true, MPI_COMM_WORLD );

  vector3.set( 0, 0, 1.0 );
  vector3.set( 1, 0, 2.0 );
  vector3.set( 2, 0, 3.0 );
  vector3.set( 3, 0, 4.0 );
  vector3.set( 4, 0, 5.0 );

  std::vector< lo > my_blocks2;
  if ( myRank == 0 ) {
    my_blocks2.push_back( 0 );
    my_blocks2.push_back( 1 );
  } else if ( myRank == 1 ) {
    my_blocks2.push_back( 2 );
    my_blocks2.push_back( 3 );
    my_blocks2.push_back( 4 );
    my_blocks2.push_back( 1 );
  } else {
    my_blocks2.push_back( 3 );
  }
  distributed_block_vector vector4( my_blocks2, 5, 6, true, MPI_COMM_WORLD );

  vector4.set( 0, 0, 5.0 );
  vector4.set( 1, 0, 4.0 );
  vector4.set( 2, 0, 3.0 );
  vector4.set( 3, 0, 2.0 );
  vector4.set( 4, 0, 1.0 );
  // vector4.fill( 5.0 );

  vector3.add( vector4 );

  sc * data = new sc[ 5 * 6 ];
  vector3.copy_to_raw( data );

  if ( myRank == 0 )
    for ( lo i = 0; i < 5 * 6; ++i ) {
      std::cout << data[ i ] << " ";
    }

  sc dot = vector3.dot( vector4 );
  std::cout << "DP = " << dot << std::endl;

  distributed_block_vector vector5;
  vector5.copy( vector4 );
  vector5.print( );
  delete[] data;

  MPI_Finalize( );
}
