#include "host_tensor.cuh"

//DO NOT modify any code in this file
template<int N_DIMS>
void host_tensor<N_DIMS>::fill_random(){
    for(size_t i=0; i<this->get_n_elems(); i++){
      this->get()[i] = float(rand()) / float(RAND_MAX) * 2.0 - 1.0;
    }
}

template<int N_DIMS>
void host_tensor<N_DIMS>::fill(float val){
    for(size_t i=0; i<this->get_n_elems(); i++){
      this->get()[i] = val;
    }
}

template<int N_DIMS>
void host_tensor<N_DIMS>::copy(const device_tensor<N_DIMS>& other){
  assert(this->get_n_elems() == other.get_n_elems());
  CHECK(hipMemcpy(this->get(), other.get(), this->get_n_elems()*sizeof(float), hipMemcpyDeviceToHost));
};

template<int N_DIMS>
void host_tensor<N_DIMS>::copy(const host_tensor<N_DIMS>& other){
  assert(this->get_n_elems() == other.get_n_elems());
  for(size_t i=0; i<this->get_n_elems(); i++)
    this->get()[i] = other.get()[i];
};



//Instantiate
template class host_tensor<1>;
template class host_tensor<2>;
template class host_tensor<3>;

void print(host_tensor<2> in)
{
    for (size_t i = 0; i < in.size[0]; ++i) {
        for (size_t j = 0; j < in.size[1]; ++j) {
            std::cout << std::setw(14) << std::scientific << std::setprecision(6) << in.at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
void print(host_tensor<1> in) {
    for (size_t i = 0; i < in.size[0]; ++i) {
        std::cout << std::setw(14) << std::scientific << std::setprecision(6) << in.at(i) << " ";
    }
    std::cout << std::endl;
}
