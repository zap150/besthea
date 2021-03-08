
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



class besthea::tools::time_measurer {
private:
  static constexpr double tick_time = ((double)std::chrono::steady_clock::period::num) / std::chrono::steady_clock::period::den;
  std::chrono::steady_clock::duration elapsed_time;
  std::chrono::steady_clock::time_point start_time;

public:
  time_measurer() {
    reset();
  }

  void start() {
    this->start_time = std::chrono::steady_clock::now();
  }

  void stop() {
    std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration dur = stop_time - start_time;
    this->elapsed_time = dur;
  }

  void reset() {
    this->elapsed_time = std::chrono::steady_clock::duration::zero();
  }

  double get_time() {
    return this->elapsed_time.count() * tick_time;
  }

  std::chrono::steady_clock::duration get_duration() {
    return this->elapsed_time;
  }
};



#endif /* INCLUDE_BESTHEA_TIME_MEASURER_H_ */
