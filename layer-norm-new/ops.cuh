//Simple math operations to be used with device_patterns.cuh

struct square_op{
  __host__ __device__ static inline float op(const float& a){
    return a*a;
  }
};

struct sinh_op{
  __host__ __device__ static inline float op(const float& a){
    return sinh( (double) a );
  }
};

struct square_root_op{
  __host__ __device__ static inline float op(const float& a){
    return sqrtf(a);
  }
};

struct add_op{
  __host__ __device__ static inline float op(const float& a, const float& b){
    return a+b;
  }

  //Init value for reduction use of this op
  __host__ __device__ static inline float init(){
    return 0.0;
  }
};

struct mul_op{
  __host__ __device__ static inline float op(const float& a, const float& b){
    return a*b;
  }
};

struct div_op{
  __host__ __device__ static inline float op(const float& a, const float& b){
    return a/b;
  }
};

struct sub_op{
  __host__ __device__ static inline float op(const float& a, const float& b){
    return a-b;
  }
};
