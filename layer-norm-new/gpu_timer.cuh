#pragma once
#include <hip/hip_runtime.h>

/*
Simple timer class to time GPU functions. Remember kernels return immediately and run
asynchronously. This class will ensure proper sync/timing.
*/

struct timer{

  hipEvent_t _start, _stop;

  timer(){
    hipEventCreate(&_start);
    hipEventCreate(&_stop);
  }

  void start(){
    hipEventRecord(_start);
  }

  float stop(){
    hipEventRecord(_stop);
    hipEventSynchronize(_stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, _start, _stop);
    return milliseconds;
  }

};
