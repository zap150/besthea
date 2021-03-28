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

/** @file time_measurer_cuda.h
 * @brief Measuring elapsed time between events on a cuda device.
 */

#ifndef INCLUDE_BESTHEA_TIME_MEASURER_CUDA_H_
#define INCLUDE_BESTHEA_TIME_MEASURER_CUDA_H_

#include <iostream>
#include <cuda_runtime.h>
#include <besthea/settings.h>



namespace besthea {
  namespace tools {
    class time_measurer_cuda;
  }
}



/*!
 * Class measuring elapsed time between events on a cuda device
 */
class besthea::tools::time_measurer_cuda {

public:

  /*!
   * Default constructor. If this is used, the init method must then be called.
   */
  time_measurer_cuda() {
    this->was_inited = false;

    int curr_gpu_idx;
    cudaGetDevice(&curr_gpu_idx);
    this->init(curr_gpu_idx, 0);
  }

  /*!
   * Constructor.
   * @param[in] gpu_idx_ Index of the watched cuda device.
   * @param[in] stream_ Cuda stream to place the events in.
   */
  time_measurer_cuda(int gpu_idx_, cudaStream_t stream_) {
    this->was_inited = false;

    this->init(gpu_idx_, stream_);
  }

  /*!
   * Destructor.
   */
  ~time_measurer_cuda() {
    destroy();
  }
  
  /*!
   * Initialization method.
   * @param[in] gpu_idx_ Index of the used cuda device.
   * @param[in] stream_ Cuda stream to place the events in.
   */
  void init(int gpu_idx_, cudaStream_t stream_) {
    if(this->was_inited) {
      destroy();
    }

    int curr_gpu_idx;
    cudaGetDevice(&curr_gpu_idx);
    if(curr_gpu_idx != gpu_idx_) {
      if(besthea::settings::output_verbosity.warnings >= 1) {
        std::cerr << "BESTHEA Warning: time measurer cuda, current device is " << curr_gpu_idx << " but " << gpu_idx_ << " was provided\n";
      }
      this->gpu_idx = -1;
      return;
    }

    this->gpu_idx = gpu_idx_;
    this->stream = stream_;
    cudaSetDevice(gpu_idx);
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    reset();
    
    this->was_inited = true;
  }

  /*!
   * Destroys the cuda events.
   */
  void destroy() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  /*!
   * Submits a cuda event marking the start of the measured timespan.
   */
  void start_submit() {
    int curr_gpu_idx;
    cudaGetDevice(&curr_gpu_idx);
    if(curr_gpu_idx != gpu_idx) {
      if(besthea::settings::output_verbosity.warnings >= 1) {
        std::cerr << "BESTHEA Warning: time measurer cuda, was initialized with gpu_idx=" << gpu_idx << " but current is gpu_idx=" << curr_gpu_idx << "\n";
      }
    }

    cudaEventRecord(start_event, stream);
  }

  /*!
   * Submits a cuda event marking the end of the measured timespan.
   */
  void stop_submit() {
    int curr_gpu_idx;
    cudaGetDevice(&curr_gpu_idx);
    if(curr_gpu_idx != gpu_idx) {
      if(besthea::settings::output_verbosity.warnings >= 1) {
        std::cerr << "BESTHEA Warning: time measurer cuda, was initialized with gpu_idx=" << gpu_idx << " but current is gpu_idx=" << curr_gpu_idx << "\n";
      }
    }

    cudaEventRecord(stop_event, stream);
    this->was_time_collected = false;
  }

  /*!
   * Resets the timer.
   */
  void reset() {
    this->elapsed_time = 0.0;
    this->was_time_collected = true;
  }

  /*!
   * Synchronizes with stop event and returns elapsed time
   */
  double get_time() {
    if(!was_time_collected) {
      collect_time();
      this->was_time_collected = true;
    }
    
    return this->elapsed_time;
  }

private:

  /*!
   * Synchronizes with end event and computes elapsed time.
   */
  void collect_time() {
    int orig_gpu_idx;
    cudaGetDevice(&orig_gpu_idx);
    cudaSetDevice(gpu_idx);

    float dur;
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&dur, start_event, stop_event);
    this->elapsed_time += dur / 1000.0;
    
    cudaSetDevice(orig_gpu_idx);
  }

private:

  int gpu_idx; //!< Index of the used GPU
  cudaStream_t stream; //!< Used cuda stream
	cudaEvent_t start_event; //!< Event marking the start of measured period
	cudaEvent_t stop_event; //!< Event marking the end of the measured period
  double elapsed_time; //!< Elapsed time in seconds
  bool was_time_collected; //!< True if the stop event has been synchronized with.
  bool was_inited; //!< True if init method has been called on this instance
  
};



#endif /* INCLUDE_BESTHEA_TIME_MEASURER_CUDA_H_ */
