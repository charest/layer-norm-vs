/*
This file holds the instance of host_tensor. This tensor can be constructed from host_tensors
or device tensors. These objects can not be directly sent to the GPU, but have to be converted
to a device_tensor and that can be sent as a kernel argument to GPU kernels.

This object is useful for manipulating data on the CPU and transfering data between the CPU and
GPU.

DO NOT modify code in this file.
*/

#pragma once
#include "tensor.cuh"
#include "device_tensor.cuh"

#include <iostream>
#include <iomanip>

//Deleter for shared_ptr, when shared_ptr ref_count=0 memory will be freed
struct host_deleter {
    void operator()(float* p) const {
      delete[] p;
    }
};

template<int N_DIMS>
class host_tensor : public tensor<N_DIMS>{

  friend device_tensor<N_DIMS>;

  //Allocate CPU data
  virtual void alloc_data(){
    this->allocation = new float[this->get_n_elems()];
    this->data = std::shared_ptr<float>(this->allocation, host_deleter());
  }

public:

  //Copy data from other buffer to this buffer
  host_tensor<N_DIMS>& operator=(const host_tensor<N_DIMS>& other){
    this->copy(other);
    return *this;
  }

  //Construct new host_tensor based on sizes, rand=True will fill with random
  //values between -1.0 and 1.0
  host_tensor(const std::array<size_t, N_DIMS> size, bool rand=false) : tensor<N_DIMS>(size, false){
    alloc_data();
    if(rand)
      fill_random();
  }

  //Construct new tensor based on a device tensor, if copy_data=true will copy the
  //data from device to CPU.
  host_tensor(const device_tensor<N_DIMS>& other, bool copy_data=true):tensor<N_DIMS>(other.size){
    alloc_data();
    if(copy_data)
      this->copy(other);
  }


  //Construct new tensor based on a host tensor, if copy_data=true will copy the
  //data from device to CPU.
  host_tensor(const host_tensor<N_DIMS>& other, bool copy_data=true):tensor<N_DIMS>(other.size){
    alloc_data();
    if(copy_data)
      this->copy(other);
  }

  virtual void copy(const device_tensor<N_DIMS>&);
  virtual void copy(const host_tensor<N_DIMS>&);
  virtual void fill_random();
  void fill(float val);

};

//Utility functions
void print(host_tensor<2> in);
void print(host_tensor<1> in);
