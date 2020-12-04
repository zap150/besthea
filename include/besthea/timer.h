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

/** @file timer.h
 * @brief Measuring elapsed time.
 */

#ifndef INCLUDE_BESTHEA_TIMER_H_
#define INCLUDE_BESTHEA_TIMER_H_

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

namespace besthea {
  namespace tools {
    class timer;
  }
}

/**
 *  Class measuring elapsed time
 */
class besthea::tools::timer {
 public:
  using clock_type = std::chrono::steady_clock;  //!< Clock type.
  // using clock_type = std::chrono::high_resolution_clock;
  using unit_type = std::chrono::milliseconds;  //!< Unit type.

  /**
   * Constructor taking a message to display.
   * @param[in] msg Message to be displayed.
   */
  timer( const std::string & msg = "" ) {
    reset( msg );
  }

  timer( const timer & that ) = delete;

  /**
   * Destructor.
   */
  ~timer( ) {
  }

  /**
   * Resets the timer.
   * @param[in] msg Message to be displayed.
   */
  void reset( const std::string & msg = "" ) {
    if ( !msg.empty( ) ) {
      std::cout << msg << std::endl;
    }
    _start = clock_type::now( );
  }

  /**
   * Returns elapsed time since the last reset (or construction).
   * @param[in] print Prints to stdout if 'true'.
   */
  std::string measure( bool print = true ) const {
    clock_type::time_point now = clock_type::now( );
    unit_type ms = std::chrono::duration_cast< unit_type >( now - _start );
    std::stringstream str;
    str << std::fixed;
    str << std::setprecision( 2 );
    str << ms.count( ) / 1000.0;
    if ( print ) {
      std::cout << "  Done in " << str.str( ) << " seconds." << std::endl;
    }
    return str.str( );
  }

 private:
  clock_type::time_point _start;  //!< Starting time point.
};

#endif /* INCLUDE_BESTHEA_TIMER_H_ */
