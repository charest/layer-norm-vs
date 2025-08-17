#include <hip/hip_runtime.h>

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "host_tensor.cuh"
#include "device_tensor.cuh"
#include "utils.cuh"
#include "ops.cuh"
#include "device_patterns.cuh"
#include "gpu_timer.cuh"

//#define DEBUG
#define EPS 1e-14
#define SCALE 1.9

//Size to run
#define M 304*32*4
#define N 1024*4
#define ITERATIONS 7


//Reference CPU implementation
//Do not change this function, It is the reference implementation for you to match!
host_tensor<2> op_and_normalize(host_tensor<2> &input){


  for(size_t i=0; i<input.get_n_elems(); i++){
    float val = input.at_linear(i);
    input.at_linear(i) = sinh((double) val/SCALE);
  }

  host_tensor<1> ave({input.size[0]});
  for(size_t i=0; i<input.size[0]; i++){
    float summ = 0.0;
    for(size_t j=0; j<input.size[1]; j++){
      summ += input.at(i, j);
    }
    ave.at(i) = summ / float(input.size[1]);
  }

  host_tensor<1> std_dev_sq({input.size[0]});
  for(size_t i=0; i<input.size[0]; i++){
    float summ = 0.0;
    for(size_t j=0; j<input.size[1]; j++){
      float diff = input.at(i, j) - ave.at(i);
      summ += diff * diff;
    }
    std_dev_sq.at(i) = summ / float(input.size[1]);
  }

  host_tensor<2> out(input.size);
  for(size_t i=0; i<input.size[0]; i++){
    for(size_t j=0; j<input.size[1]; j++){
        out.at(i, j) = (input.at(i, j) - ave.at(i))/sqrtf(std_dev_sq.at(i) + EPS);
    }
  }

  return out;
}

//GPU implementation
//This is a sample GPU implementation, anything and nothing can be kept from it
void op_and_normalize(
  device_tensor<2> &input,
  std::vector<hipStream_t> & streams)
{
  auto nchunk = streams.size();
  auto chunk_size = input.size[0] / nchunk;

  for (size_t i=0; i<nchunk; ++i ) {
    auto istart = i*chunk_size;
    auto iend = istart + chunk_size;
    welford(input, SCALE, EPS, istart, iend, streams[i]); // writes to input 
  }
}

//Compares a host tensor and device tensor and returns mas abs difference between them
template<int N_DIMS>
float check_result(const host_tensor<N_DIMS>& A, const device_tensor<N_DIMS>& C){
  host_tensor<N_DIMS> B(C, true);
  assert(A.get_n_elems() == B.get_n_elems());

  //std::cout << "CPU result:" << std::endl;
  //print(A);
  //std::cout << "GPU result:" << std::endl;
  //print(C);

  float max_diff = 0.0;
  for(size_t i=0; i<A.get_n_elems(); i++){
    max_diff = std::max(max_diff, std::abs(A.at_linear(i) - B.at_linear(i)));
  }
  return max_diff;
}



int main(int argc, char * argv[]) {

  srand(0);
  //srand(time(NULL));

  int device = 0;
  hipSetDevice(device);

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, device);

  std::cout << "GFX version: " << prop.gcnArchName << std::endl;
  std::cout << "HIP warp size: " << prop.warpSize << std::endl;


  bool do_profile = (argc==2 && strcmp(argv[1], "-p")==0) ? true : false;

  /*
     Do not change this section of code, this is how the user expects to interact with your
     implementation. hA and hOut is the reference implementation. hA data will be copied to
     dA so the input to the GPU function will match that of the reference. This is the tensor
     the user is expecting to give to your implementation and dOut is the tensor the user is
     expecting back from your implementation.
  */
  //Input tensor
  host_tensor<2> hA({M, N}, true);
  //hA.fill(1);
  host_tensor<2> hOut(hA, true);

  //Make copy for device ops, need to grab random numbers in hA.
  device_tensor<2> dA(hA);
  device_tensor<2> dOut(dA, true);

  //Run the CPU ops ITERATION times sequentially.
  for(int i=0; i<ITERATIONS; i++){
    hOut = op_and_normalize(hOut);
  }

  //Run the GPU ops ITERATIONS times sequentially
  //As long as dOut matches hOut you can modify anything
  //that is executed in between t.start() and t.stop().
  timer t;

  // run a warmup kernel to remove overhead
  //hipDeviceSynchronize();
  //warmup();
  //hipDeviceSynchronize();

 
  //-------------------------------------------------------
  
  // create streams
  std::vector<hipStream_t> streams(NSTREAMS);
    
  if (NSTREAMS>1) {
    for (int i=0; i<NSTREAMS; ++i )
      hipStreamCreate(&streams[i]); 
  }
  
  t.start();

  // execute kernels
  for(int i=0; i<ITERATIONS; i++){
    op_and_normalize(dOut, streams);
  }
 
 float ms = t.stop();
  
  // destroy streams
  if (NSTREAMS>1) {
    for (int i=0; i<NSTREAMS; ++i )
      hipStreamDestroy(streams[i]); 
  }

  //-------------------------------------------------------
   
  //Print the amount of time required by the gpu implementation.
  std::cout<<"Finished in "<< ms << " ms.";
  
  //Make sure the result of your implementation is correct.
  double err = check_result(hOut, dOut);
  
  std::cout 
      << " Error is " << std::scientific << std::setprecision(4) 
      << err << std::endl;

  if (err > 1.e-4) {
    std::cout << "*** Error exceeds tol!" << std::endl;
    if (!do_profile) abort();
  }

  return 0;

}
