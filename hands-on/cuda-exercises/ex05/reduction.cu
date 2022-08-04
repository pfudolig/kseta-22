#include <iostream>
#include <numeric>
#include <random>
#include <vector>


// Here you can set the device ID that was assigned to you
#define MYDEVICE 0
const int num_blocks = 4096;
const int block_size = 64;

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int* input, int* per_block_results, const size_t n)
{
  __shared__ int sdata[block_size];
  //int thread = threadIdx.x;
  int global_thread = threadIdx.x + blockIdx.x * blockDim.x;
  int s = blockDim.x / 2;
  if (global_thread < n) {
    sdata[threadIdx.x] = input[global_thread];
    __syncthreads();
    while (s > 0) {
      if (threadIdx.x < s) {
        sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[s + threadIdx.x];
        __syncthreads();
      }
      s = s/2;
      printf("%d\n",s);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      per_block_results[blockIdx.x] = sdata[0]; //save just the first element of our sdata array
    }
  }
  //printf("%d",per_block_results[blockIdx.x]);
}

__global__ void saxpy(unsigned int n, double a, double* x, double* y)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; //global index of each thread
  if (i < n) //if i is greater than some value
    y[i] = a * x[i] + y[i]; //to ensure no out of bounds memory access (#threads must be divisible by #blocks)
}


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





  int n = num_elements * sizeof(int); //sizeof int is 4bytes
  int* d_input = 0;
  cudaMalloc((void**)&d_input,n);
  
  int* d_partial_sums_and_total;
  cudaMalloc((void**)&d_partial_sums_and_total,num_blocks * sizeof(int)); //make num_blocks?

  int* d_final;
  cudaMalloc((void**)&d_final,sizeof(int));

  std::vector<int> host(num_blocks * sizeof(int)); 

  //cudaEvent_t start,stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);

  cudaMemcpy(d_input,h_input.data(),n,cudaMemcpyHostToDevice);

  //cudaEventRecord(start);

  block_sum<<<num_blocks, block_size>>>(d_input, d_partial_sums_and_total,n);
  cudaDeviceSynchronize();
  //cudaEventRecord(stop);
  //cudaEventSynchronize(stop);
  cudaMemcpy(host.data(),d_partial_sums_and_total,num_blocks,cudaMemcpyDeviceToHost);

 //Checking vectors
 // for (int i=0; i<h_input.size();i++) {
  //  printf("%d\n",h_input[i]);
  //}
  printf("%d\n",host.size()); //size is num elems when launch w/ n but should be num_blocks
  //printf("%d\n",h_input.begin());
  //printf("%d\n",h_input.end());
  //std::cout << *h_input.data() << std::endl;

/*
  block_sum<<<1,num_blocks>>>(d_partial_sums_and_total,d_final,block_size*sizeof(int));
  cudaDeviceSynchronize();
  h_input.resize(0);
  cudaMemcpy(h_input.data(),d_final,sineof(int),cudaMemcpyDeviceToHost);

  printf("%d\n",h_input.size());
  printf("%d\n",h_input.begin());
  printf("%d\n",h_input.end());
  //std::cout << "Device sum: " << h_input << std::endl;
*/
/*
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop); //write recorded time to milliseconds variable 

  double maxError = 0.; //just error check
  for (unsigned int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i] - 4.0));

  }
  float thruput = 3 * N * sizeof(double) / milliseconds / 10e6;
  std::cout << "throughput: " << thruput << std::endl;


  cudaFree(d_input);
  cudaFree(d_partial_sums_and_total);
  cudaFree(d_final);
*/

  return 0;
}
