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

#include "besthea/align.h"

#include "besthea/settings.h"

#include <cstdlib>

void * besthea::memory::align::aligned_alloc(
  std::size_t alignment, std::size_t size, bool zero ) {
  std::size_t request_size = size + alignment;
  char * buf
    = (char *) ( zero ? calloc( 1, request_size ) : malloc( request_size ) );

  std::size_t remainder = ( (std::size_t) buf ) % alignment;
  std::size_t offset = alignment - remainder;
  char * ret = buf + (unsigned char) offset;

  // store how many extra bytes we allocated in the byte just before the
  // pointer we return
  *(unsigned char *) ( ret - 1 ) = offset;

  return (void *) ret;
}

void besthea::memory::align::aligned_free( void * aligned_ptr ) {
  int offset = *( ( (char *) aligned_ptr ) - 1 );
  free( ( (char *) aligned_ptr ) - offset );
}

void * besthea::memory::align::aligned_alloc( std::size_t size, bool zero ) {
  return (void *) aligned_alloc( DATA_ALIGN, size, zero );
}
