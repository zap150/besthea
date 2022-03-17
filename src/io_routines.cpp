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

#include "besthea/io_routines.h"

#include <fstream>  //for ofstream and ifstream
#include <iostream>

template< class T >
void write_vector_to_bin_file(
  const std::vector< T > & print_vector, const std::string & filename ) {
  std::ofstream file_out( filename.c_str( ), std::ios::binary );
  if ( file_out.is_open( ) ) {
    lou n_chars = print_vector.size( ) * sizeof( T );
    const T * print_vector_data = print_vector.data( );
    file_out.write(
      reinterpret_cast< const char * >( print_vector_data ), n_chars );
    file_out.close( );
  } else {
    std::cout << "Error. Could not open the output file ('" << filename
              << "') for printing the the vector" << std::endl;
  }
}

template< class T >
void write_raw_data_to_bin_file(
  const T * print_data, const lou n_entries, const std::string & filename ) {
  std::ofstream file_out( filename.c_str( ), std::ios::binary );
  if ( file_out.is_open( ) ) {
    lou n_chars = n_entries * sizeof( T );
    file_out.write( reinterpret_cast< const char * >( print_data ), n_chars );
    file_out.close( );
  } else {
    std::cout << "Error. Could not open the output file ('" << filename
              << "') for printing the data: " << std::endl;
  }
}

template< class T >
std::vector< T > read_vector_from_bin_file( const std::string & filename ) {
  std::vector< T > out_vector;
  std::ifstream read_file;
  read_file.open( filename.c_str( ) );
  if ( read_file.is_open( ) ) {
    // determine the number of chars to be received
    read_file.seekg( 0, read_file.end );
    lo n_chars = read_file.tellg( );
    read_file.seekg( 0 );
    out_vector.resize( n_chars / sizeof( T ) );
    // load all chars from file
    read_file.read( reinterpret_cast< char * >( out_vector.data( ) ), n_chars );
    read_file.close( );
  } else {
    std::cout << "Error. Could not open the input file ('" << filename
              << "') for reading the data: " << std::endl;
  }
  return out_vector;
}

// explicitly instantiate required template functions
template void write_vector_to_bin_file< char >(
  const std::vector< char > & print_vector, const std::string & filename );

template void write_vector_to_bin_file< lo >(
  const std::vector< lo > & print_vector, const std::string & filename );

template void write_vector_to_bin_file< sc >(
  const std::vector< sc > & print_vector, const std::string & filename );

template void write_raw_data_to_bin_file< sc >(
  const sc * print_data, const lou n_entries, const std::string & filename );

template std::vector< char > read_vector_from_bin_file< char >(
  const std::string & filename );

template std::vector< lo > read_vector_from_bin_file< lo >(
  const std::string & filename );

template std::vector< sc > read_vector_from_bin_file< sc >(
  const std::string & filename );
