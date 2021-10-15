/*
Copyright (c) 2021, VSB - Technical University of Ostrava and Graz University of
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
  endorse or promote scaled_products derived from this software without specific
prior written permission.

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

#include "besthea/distributed_diagonal_matrix.h"

#include "besthea/distributed_block_vector.h"

void besthea::linear_algebra::distributed_diagonal_matrix::apply(
  const block_vector & /*x*/, block_vector & /*y*/, bool /*trans*/,
  sc /*alpha*/, sc /*beta*/ ) const {
  // generic method not implemented
  std::cout << "apply: NOT IMPLEMENTED for standard block vectors. Please use "
               "distributed block vectors!"
            << std::endl;
}

void besthea::linear_algebra::distributed_diagonal_matrix::apply(
  const distributed_block_vector & x, distributed_block_vector & y,
  bool /*trans*/, sc alpha, sc beta ) const {
  lo n_blocks = _diagonal.get_n_blocks( );
  std::vector< lo > my_blocks = _diagonal.get_my_blocks( );
  lo block_size = _diagonal.get_size_of_block( );
  distributed_block_vector scaled_product(
    my_blocks, n_blocks, block_size, false, _diagonal.get_comm( ) );
  for ( lou i = 0; i < my_blocks.size( ); ++i ) {
    for ( lo j = 0; j < block_size; ++j ) {
      scaled_product.set( my_blocks[ i ], j,
        alpha * _diagonal.get( my_blocks[ i ], j )
          * x.get( my_blocks[ i ], j ) );
    }
  }
  y.scale( beta );
  y.add( scaled_product );
  y.synchronize_shared_parts( );
}
