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

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <mpi.h>
#include <sstream>

int main( int argc, char * argv[] ) {
  int provided, rank, size;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  std::stringstream ss;

  // save old buffer and redirect output to string stream
  auto cout_buf = std::cout.rdbuf( ss.rdbuf( ) );

  int result = Catch::Session( ).run( argc, argv );

  // reset buffer
  std::cout.rdbuf( cout_buf );

  for ( int i = 1; i < size; ++i ) {
    MPI_Barrier( MPI_COMM_WORLD );
    if ( i == rank ) {
      // if all tests are passed, it's enough if we hear that from
      // the master.
      if ( ss.str( ).rfind( "All tests passed" ) == std::string::npos )
        std::cout << "MPI rank: " << rank << ", number of processes " << size
                  << std::endl
                  << ss.str( );
    }
  }
  // master prints last
  MPI_Barrier( MPI_COMM_WORLD );
  if ( rank == 0 )
    std::cout << "MPI rank: " << rank << ", number of processes " << size
              << std::endl
              << ss.str( );
  MPI_Finalize( );
  return result;
}
