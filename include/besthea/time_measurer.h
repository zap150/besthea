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

/** @file time_measurer.h
 * @brief Measuring elapsed time.
 */

#ifndef INCLUDE_BESTHEA_TIME_MEASURER_H_
#define INCLUDE_BESTHEA_TIME_MEASURER_H_

#include <chrono>



namespace besthea {
  namespace tools {
    class time_measurer;
  }
}



/**
 *  Class measuring elapsed time
 */
class besthea::tools::time_measurer {

public:

  /**
   * Constructor.
   */
  time_measurer() {
    reset();
  }

  /**
   * Start the timer.
   */
  void start() {
    this->start_time = std::chrono::steady_clock::now();
  }

  /**
   * Stop the timer.
   */
  void stop() {
    std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration dur = stop_time - start_time;
    this->elapsed_time += dur;
  }

  /**
   * Reset the timer.
   */
  void reset() {
    this->elapsed_time = std::chrono::steady_clock::duration::zero();
  }

  /**
   * Returns elapsed time in seconds.
   */
  double get_time() {
    return this->elapsed_time.count() * tick_time;
  }

  /**
   * Returns elapsed time as a duration type
   */
  std::chrono::steady_clock::duration get_duration() {
    return this->elapsed_time;
  }

private:

  static constexpr double tick_time = ((double)std::chrono::steady_clock::period::num) / std::chrono::steady_clock::period::den; //!< Duration of a clock tick in seconds
  std::chrono::steady_clock::duration elapsed_time; //!< Elapsed time
  std::chrono::steady_clock::time_point start_time; //!< Time when the timer was started

};



#endif /* INCLUDE_BESTHEA_TIME_MEASURER_H_ */
