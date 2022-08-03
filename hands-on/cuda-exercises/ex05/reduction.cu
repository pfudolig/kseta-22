#include <iostream>
#include <numeric>
#include <random>
#include <vector>


// Here you can set the device ID that was assigned to you
#define MYDEVICE 0
const int num_blocks = 4096;
const int block_size = 64;

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int* input, int* per_block_results,
                          const size_t n)
{
//  const int block_size = blockDim.x;

  __shared__ int sdata[num_blocks];
  int thread = threadIdx.x;
  int global_thread = threadIdx.x + blockIdx.x * blockDim.x;
  //sdata[global_thread] = 0;
  if (global_thread < n) {
    input = &sdata[global_thread];
    __syncthreads();

    //sdata[global_thread] = input[global_thread];
    //__syncthreads();
    //per_block_results[global_thread] += sdata[thread]; //sum threads
  }
  printf("%d\n",input);
  
}


/*
//  const int block_size = blockDim.x;
  //__shared__ int sdata[16];
  int thread = threadIdx.x;
  int global_thread = threadIdx.x + blockIdx.x * blockDim.x;
  //const int total_thread = blockIdx.x * blockDim.x;
  //int x = per_block_results[global_thread]; //sum threads
  __shared__ int sdata[n];
  sdata[thread] = sum;
  __syncthreads();

  printf("%d",sum);
/*
//  int total_threads = blockDim.x * gridDim.x;
  sdata[thread] = input[global_thread];
  __syncthreads();
  int x = per_block_results[global_thread]; //sum threads
  printf("%d",x);
//  atomicAdd(&input,sdata[global_thread])
//  per_block_results = &input;
*/




////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{

  //this part basically picks random integers out of a random vector of values to use
  std::random_device rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(-10, 10); //produces random integer from -10 to 10

  // create array of 256ki elements
  const int num_elements = 1 << 18;
  // generate random input on the host
  std::vector<int> h_input(num_elements);
  for (auto& elt : h_input) {
    elt = distrib(gen);
  }

  const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0); //sum of elements in range, starting at 0
  std::cerr << "Host sum: " << host_result << std::endl;




  // //Part 1 of 6: move input to device memory
  int n = num_elements * sizeof(int); //sizeof int is 4bytes
  int* d_input = 0;
  cudaMalloc((void**)&d_input,n);
  cudaMemcpy(d_input,h_input.data(),n,cudaMemcpyHostToDevice);
  int* d_partial_sums_and_total;
  cudaMalloc(&d_partial_sums_and_total,block_size * sizeof(int));
  block_sum<<<num_blocks, block_size>>>(d_input, d_partial_sums_and_total,
                                        n);
  cudaDeviceSynchronize();



/* 
  std::vector<int> host(block_size);

//  int i = 0; 
//  std::fill(host.begin(),host.end(),i++);
  cudaMemcpy(d_input,host.data(),n,cudaMemcpyHostToDevice);
  std::cout << *d_input << std::endl;


  // // Part 1 of 6: allocate the partial sums: How much space does it need?
  int* d_partial_sums_and_total;
  cudaMalloc(&d_partial_sums_and_total,n);
  // // Part 1 of 6: launch one kernel to compute, per-block, a partial sum. How
  // much shared memory does it need?
  //block_sum<<<num_blocks, block_size>>>(d_input, d_partial_sums_and_total,
//                                        n);

  //std::cout << d_partial_sums_and_total << std::endl;
  int* final;
  cudaMalloc(&final,n);
  cudaMemcpy(final,d_partial_sums_and_total,n,cudaMemcpyDeviceToDevice);
  //std::cout << *final << std::endl;
*/










  /*
  //cudaDeviceSynchronize(); //Unsure yet if I need this

  // // // Part 1 of 6: compute the sum of the partial sums
  // int* d_final;
  // cudaMalloc(&d_final,n);
  // block_sum<<<dimGrid,dimBlock>>>(d_partial_sums_and_total,d_final,n);

  // // // Part 1 of 6: copy the result back to the host
  // //int* device_result = 0;
  // //cudaMemcpy(device_result,d_final,n,cudaMemcpyDeviceToHost);
  // std::cout << "Device sum: " << *d_final << std::endl;

  // // Part 1 of 6: deallocate device memory
  //cudaFree(d_input);
  //cudaFree(d_partial_sums_and_total);
  //cudaFree(d_final);
  //cudaFree(device_result);
*/ 
  return 0;
}
