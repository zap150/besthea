
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
  cudaStream_t stream;
	cudaEvent_t start_event;
	cudaEvent_t stop_event;
  double elapsed_time;
  bool was_time_collected;

public:
  time_measurer_cuda() {
  }

  explicit time_measurer_cuda(cudaStream_t stream) {
    this->init(stream);
  }

  ~time_measurer_cuda() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);        
  }
  
  void init(cudaStream_t stream) {
    this->stream = stream;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    reset();
  }

  void start_submit() {
		cudaEventRecord(start_event, stream);
  }

  void stop_submit() {
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
    float dur;
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&dur, start_event, stop_event);
    this->elapsed_time = dur / 1000.0;
  }
};



#endif /* INCLUDE_BESTHEA_TIME_MEASURER_CUDA_H_ */
