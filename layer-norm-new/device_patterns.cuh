#include "hip/hip_runtime.h"
#include "device_tensor.cuh"

// Default warp size for device
#define WARPSIZE 32

// How many warps per block
#define NWARPS 2

// The thread block dimensions
#define BLOCKDIM (NWARPS*WARPSIZE)

// Set NVEC to 4 to try vectorized loads
#define NVEC 1

// The number of rows each block processes
#define NROWS 1

// The number of streams
#define NSTREAMS 1

// a fast version of sinh based on fast math version of exp
__device__ float fast_sinhf(float x) {
    float exp_x = __expf(x);
    float exp_neg_x = __expf(-x);
    return 0.5f * (exp_x - exp_neg_x);
}


#define SINH(x) sinh(x)
//#define SINH(x) fast_sinhf(x)

__device__ void welford_combine(
    int & count,
    float & mean,
    float & var,
    int count_b,
    float mean_b,
    float var_b, int /*tx*/)
{
    if (count_b == 0) return;
    auto new_count = count + count_b;
    auto nb_over_n = static_cast<float>(count_b) / new_count;
    auto delta = mean_b - mean;
    mean += delta * nb_over_n;
    var  += var_b + delta * delta * count * nb_over_n;
    count = new_count;
}

//--------------------------------------------------------------------
// Welford combinatory reduction
//--------------------------------------------------------------------
__device__ void welford_reduce(
    int & count,
    float & mean,
    float & var,
    size_t tx,
    size_t blocksz) 
{
    __shared__ float smem_mean [BLOCKDIM/WARPSIZE];
    __shared__ float smem_var  [BLOCKDIM/WARPSIZE];
    __shared__ int   smem_count[BLOCKDIM/WARPSIZE];
      
    // reduce accross warps via shuffling
    for (size_t stride = WARPSIZE/2; stride>0; stride/=2) {
      auto mean_b = __shfl_down(mean, stride);
      auto var_b  = __shfl_down( var, stride);
      auto count_b= __shfl_down(count,stride);
      welford_combine(count, mean, var, count_b, mean_b, var_b, tx);
    }
    
    // t0 of each warp now has the warp-wide reduction

    //store sum of each warp into smem
    if (tx % WARPSIZE == 0) {
      smem_mean [tx/WARPSIZE] = mean;
      smem_var  [tx/WARPSIZE] = var;
      smem_count[tx/WARPSIZE] = count;
    }
    
    __syncthreads();

    // repeat shuffle, only first warp this time
    if(tx < WARPSIZE){
      if (tx < blocksz/WARPSIZE) {
        mean = smem_mean [tx];
        var  = smem_var  [tx];
        count= smem_count[tx];
      }
      else {
        mean = 0;
        var  = 0;
        count= 0;
      }
      for(size_t stride = WARPSIZE/2; stride>0; stride/=2) {
        auto mean_b = __shfl_down(mean, stride);
        auto var_b  = __shfl_down( var, stride);
        auto count_b= __shfl_down(count,stride);
        welford_combine(count, mean, var, count_b, mean_b, var_b, tx);
      }
      if(tx==0) {
        smem_mean [0] = mean;
        smem_var  [0] = var;
        smem_count[0] = count;
      }
    }
    
    __syncthreads();
    
    mean  = smem_mean [0];
    var   = smem_var  [0];
    count = smem_count[0];
}

//--------------------------------------------------------------------
// Compute Mu for each row and Xij-mu_i for each element.
//--------------------------------------------------------------------

/// Kernel
__launch_bounds__(BLOCKDIM)
__global__ void  kernel_welford(
       device_tensor<2> x,
       float scale,
       float eps,
       size_t istart,
       size_t iend)
{

  size_t n = x.size[1];

  size_t tx = threadIdx.x;
  for (size_t i=blockIdx.x+istart; i<iend; i+=gridDim.x) {

    int count = 0;
    float mean = 0;
    float var = 0;

    for(size_t j=tx; j<n/NVEC; j+=blockDim.x){
      auto & v = x.at(i,j); 
      v = SINH( (double)v/scale );
      //float v = x.at(i,j); //SINH( (double)x.at(i,j)/scale );
      auto tmp = v - mean;
      count++;
      mean += tmp / count;
      var += tmp * (v - mean);
    }

    // reduce accross warps via shuffling
    welford_reduce(count, mean, var, tx, blockDim.x);
    var = sqrtf( var/count + eps );
    
    for (size_t j = tx; j < n; j += blockDim.x)
        x.at(i, j) = (x.at(i, j) - mean) / var;

  }

}

/// Kernel wrapper for average calc
void welford(
  const device_tensor<2>& x,
  float scale,
  float eps,
  int istart,
  size_t iend,
  hipStream_t stream)
{
  auto nblock = (iend-istart)/NROWS; // give each block a row
  auto nthread = BLOCKDIM/NVEC;

  kernel_welford <<<nblock, nthread, 0, stream>>>(x, scale, eps, istart, iend);
}

// warmup the gpu
__global__ void kernel_warmup(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
	for (size_t i=0; i<100000000000; ++i)
  	ib += ia + tid; 
}

void warmup()
{
  int device;
  hipGetDevice(&device);

  struct hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
	int grid(props.multiProcessorCount * props.maxBlocksPerMultiProcessor);
	int block(2*props.warpSize);
  kernel_warmup<<<grid, block>>>();
}
