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

/** @file settings.h
 * @brief Besthea settings.
 */

#ifndef INCLUDE_BESTHEA_SETTINGS_H_
#define INCLUDE_BESTHEA_SETTINGS_H_

#include "boost/align.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef DATA_ALIGN
#define DATA_ALIGN 64  //!< Cache-line size in bytes.
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846  //!< pi
#endif

// pragma to switch between cluster- and timestep-wise nearfield computation
// #define NEARFIELD_CLUSTERWISE

namespace besthea {
  using scalar = double;  //!< Floating point type.
  // using index = std::size_t; //!< Indexing type.
  using index = long;  //!< Indexing type.
  using index_signed
    = std::make_signed< index >::type;  //!< Signed indexing type.
  using index_unsigned
    = std::make_unsigned< index >::type;  //!< Unsigned indexing type.
  using short_index = int16_t;            //!< Signed short integer.
  using short_index_unsigned
    = std::make_unsigned< short_index >::type;  //!< Unsigned short integer.

  template< class T >
  using allocator_type = boost::alignment::aligned_allocator< T,
    DATA_ALIGN >;  //!< Aligned allocator.
};                 // namespace besthea

using sc = besthea::scalar;                  //!< Floating point type.
using lo = besthea::index;                   //!< Indexing type.
using los = besthea::index_signed;           //!< Signed indexing type.
using lou = besthea::index_unsigned;         //!< Unsigned indexing type.
using slos = besthea::short_index;           //!< Short signed indexing type.
using slou = besthea::short_index_unsigned;  //!< Short unsigned indexing type.

#endif /* INCLUDE_BESTHEA_SETTINGS_H_ */
