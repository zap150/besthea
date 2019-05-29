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

/** @file align.h
 * @brief Aligned memory allocation.
 */

#ifndef INCLUDE_BESTHEA_ALIGN_H_
#define INCLUDE_BESTHEA_ALIGN_H_

#include <cstddef>
#include <cstdlib>

namespace besthea {
  namespace memory {
    class align;
  }
}

class besthea::memory::align {
 public:
  align( ) = delete;

  align( const align & ) = delete;

  ~align( );

  /** Allocate aligned memory in a portable way.
   *
   * Memory allocated with aligned alloc *MUST* be freed using aligned_free.
   *
   * @param[in] alignment The number of bytes to which memory must be aligned.
   * This value *must* be <= 255.
   * @param[in] bytes The number of bytes to allocate.
   * @param[in] zero If true, the returned memory will be zeroed. If false, the
   *  contents of the returned memory are undefined.
   * @returns A pointer to `size` bytes of memory, aligned to an
   * `alignment`-byte boundary.
   */
  static void * aligned_alloc(
    std::size_t alignment, std::size_t size, bool zero = false );

  /** Free memory allocated with aligned_alloc */
  static void aligned_free( void * aligned_ptr );

  /** Allocate memory aligned to DATA_WIDTH in a portable way.
   *
   * Memory allocated with aligned alloc *MUST* be freed using aligned_free.
   *
   * @param[in] bytes The number of bytes to allocate.
   * @param[in] zero If true, the returned memory will be zeroed. If false, the
   *  contents of the returned memory are undefined.
   * @returns A pointer to `size` bytes of memory, aligned to a
   * `DATA_WIDTH`-byte boundary.
   */
  static void * aligned_alloc( std::size_t size, bool zero = false );
};

#endif
