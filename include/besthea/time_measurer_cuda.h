
/** @file time_measurer_cuda.h
 * @brief Measuring elapsed time between events on cuda device.
 */

#ifndef INCLUDE_BESTHEA_TIME_MEASURER_CUDA_H_
#define INCLUDE_BESTHEA_TIME_MEASURER_CUDA_H_

#include <cuda_runtime.h>



namespace besthea {
  namespace tools {
    class time_measurer_cuda;
  }
}



class besthea::tools::time_measurer_cuda {
private:
  int gpu_idx;
  cudaStream_t stream;
	cudaEvent_t start_event;
	cudaEvent_t stop_event;
  double elapsed_time;
  bool was_time_collected;

public:
  time_measurer_cuda() {
  }

  time_measurer_cuda(int gpu_idx_, cudaStream_t stream_) {
    this->init(gpu_idx_, stream_);
  }

  ~time_measurer_cuda() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }
  
  void init(int gpu_idx_, cudaStream_t stream_) {
    int curr_gpu_idx;
    cudaGetDevice(&curr_gpu_idx);
    if(curr_gpu_idx != gpu_idx_) {
      fprintf(stderr, "besthea::tools::time_measurer_cuda current device is %d but %d was provided\n", curr_gpu_idx, gpu_idx_);
      this->gpu_idx = -1;
      return;
    }

    this->gpu_idx = gpu_idx_;
    this->stream = stream_;
    cudaSetDevice(gpu_idx);
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    reset();
  }

  void start_submit() {
    int curr_gpu_idx;
    cudaGetDevice(&curr_gpu_idx);
    if(curr_gpu_idx != gpu_idx) {
      fprintf(stderr, "besthea::tools::time_measurer_cuda was initialized with gpu_idx=%d but current is gpu_idx=%d\n", gpu_idx, curr_gpu_idx);
    }

    cudaEventRecord(start_event, stream);
  }

  void stop_submit() {
    int curr_gpu_idx;
    cudaGetDevice(&curr_gpu_idx);
    if(curr_gpu_idx != gpu_idx) {
      fprintf(stderr, "besthea::tools::time_measurer_cuda was initialized with gpu_idx=%d but current is gpu_idx=%d\n", gpu_idx, curr_gpu_idx);
    }

    cudaEventRecord(stop_event, stream);
    this->was_time_collected = false;
  }

  void reset() {
    this->elapsed_time = 0.0;
    this->was_time_collected = true;
  }

  double get_time() {
    if(!was_time_collected) {
      collect_time();
      this->was_time_collected = true;
    }
    
    return this->elapsed_time;
  }

private:
  void collect_time() {
    int orig_gpu_idx;
    cudaGetDevice(&orig_gpu_idx);
    cudaSetDevice(gpu_idx);

    float dur;
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&dur, start_event, stop_event);
    this->elapsed_time = dur / 1000.0;
    
    cudaSetDevice(orig_gpu_idx);
  }
};



#endif /* INCLUDE_BESTHEA_TIME_MEASURER_CUDA_H_ */
