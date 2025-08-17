/*
This file holds the base class for the tensor library.
Tensor is inherited by 2 classes, a host_tensor
and a device_tensor. These classes are useful for transfering
data back and forth from device and can be sent to a kernel
and used in GPU code (those functions marked as __device__

It is unlikely you need to modify this file, but for the brave...
 */

#pragma once

#include <array>
#include <memory>
#include <cassert>

#include "utils.cuh"

//Need to declare device/host tensor classes
//We want to enforce that they can copy to eachother
template<int>
class device_tensor;

template<int>
class host_tensor;

//Base class for device/host tensor classes
//N_DIMS is the dimensionality of the class
template<int N_DIMS>
class tensor{

protected:

  //Total number of elements the tensor holds
  size_t n_elems;

  //Set the above value based on size array
  void set_n_elems(){
    n_elems = 1;
    for(int i=0; i<N_DIMS; i++)
      n_elems *= size[i];
  }

  //Shared pointer for the allocation. only used to keep track of references to the data allocation.
  //Once its refcount == 0 data refered to by allocation will be deleted.
  std::shared_ptr<float> data;

  //Allocation of the data for instances of this class, could point to GPU or CPU data.
  float *allocation;

  //Derived classes must define how to allocate their own data.
  virtual void alloc_data() = 0;

  //Get the allocation pointer.
  __host__ __device__ float* get() const {return allocation;};

public:

  //Return the number of elements this tensor holds.
  __host__ __device__ size_t get_n_elems() const {return n_elems;}

  //An array of the size of each dimension in this tensor.
  const std::array<size_t, N_DIMS> size;

  //Fill this tensor with random data between -1 and 1
  virtual void fill_random() = 0;

  //Copy data from other device tensor onto this allocation.
  virtual void copy(const device_tensor<N_DIMS>&) = 0;

  //Copy data from other host tensor onto this allocation.
  virtual void copy(const host_tensor<N_DIMS>&) = 0;

  //Access the element at [x][y][z] for 3D tensors. Allocations in row major.
  __host__ __device__ __inline__ float& at(size_t x, size_t y, size_t z){
    static_assert( N_DIMS == 3, "Trying to use 3D accessor on non-3D tensor.\n");
    return *(this->get() + x * size[1] * size[2] + y * size[2] + z);
  }

  //Access the element at [x][y] for 2D tensors. Allocations in row major.
  __host__ __device__ __inline__ float& at(size_t  x, size_t y){
    static_assert( N_DIMS == 2, "Trying to use 2D accessor on non-2D tensor.\n");
    return *(this->get() + x * size[1] + y);
  }

  //Access the element at [x] for 1D tensors.
  __host__ __device__ __inline__ float& at(size_t x){
    static_assert( N_DIMS == 1, "Trying to use 1D accessor on non-1D tensor.\n");
    return *(this->get() + x);
  }

  //If tensor is 2 or 3 dimensional, will treat it as a row major 1D tensor of size n_elements.
  __host__ __device__ __inline__ float& at_linear(size_t x){
    return this->get()[x];
  }

  //Const versions of at above
  __host__ __device__ __inline__ const float& at(size_t x, size_t y, size_t z) const{
    static_assert( N_DIMS == 3, "Trying to use 3D accessor on non-3D tensor.\n");
    return *(this->get() + x * size[1] * size[2] + y * size[2] + z);
  }

  //Const versions of at above
  __host__ __device__ __inline__ const float& at(size_t x, size_t y) const{
    static_assert( N_DIMS == 2, "Trying to use 2D accessor on non-2D tensor.\n");
    return *(this->get() + x * size[1] + y);
  }

  //Const versions of at above
  __host__ __device__ __inline__ const float& at(size_t x) const{
    static_assert( N_DIMS == 1, "Trying to use 1D accessor on non-1D tensor.\n");
    return *(this->get() + x);
  }

  //Const versions of at_linear above
  __host__ __device__ __inline__ const float& at_linear(size_t x) const{
    return this->get()[x];
  }

  //Construct tensor based on size, maybe fill with random data.
  tensor(const std::array<size_t, N_DIMS> size, bool rand=false):size{size}
  {
    static_assert( N_DIMS <= 3, "Tensor class only supports upto 3 dimensions.\n");
    static_assert( N_DIMS >= 1, "Tensor class must be at least 1 dimensional.\n");
    set_n_elems();
  }

};
