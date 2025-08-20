#include "device_tensor.cuh"

template<int N_DIMS>
void device_tensor<N_DIMS>::fill_random(){
  host_tensor<N_DIMS> rand_vals(this->size, true);
  this->copy(rand_vals);
}

template<int N_DIMS>
void device_tensor<N_DIMS>::copy(const device_tensor<N_DIMS>& other){
  assert(this->get_n_elems() == other.get_n_elems());
  CHECK(hipMemcpy(this->get(), other.get(), this->get_n_elems()*sizeof(float), hipMemcpyDeviceToDevice));
};


template<int N_DIMS>
void device_tensor<N_DIMS>::copy(const host_tensor<N_DIMS>& other){
  assert(this->get_n_elems() == other.get_n_elems());
  CHECK(hipMemcpy(this->get(), other.get(), this->get_n_elems()*sizeof(float), hipMemcpyHostToDevice));
};

//Instantiate
template class device_tensor<1>;
template class device_tensor<2>;
template class device_tensor<3>;