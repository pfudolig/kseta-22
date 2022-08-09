// UNFINISHED

#include <iostream>
#include <numeric>
#include <random>
#include <vector>


// Here you can set the device ID that was assigned to you
#define MYDEVICE 1
const int num_blocks = 16; //4096
const int block_size = 64; //64

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int* input, int* per_block_results, int* final, const size_t n)
{
  __shared__ int sdata[block_size];
  //int thread = threadIdx.x;
  int global_thread = threadIdx.x + blockIdx.x * blockDim.x;
  int s = blockDim.x / 2;
  if (global_thread < n) {
    sdata[threadIdx.x] = input[global_thread];
  }
  __syncthreads();
  //printf("globaldone");
    //printf("start\n");
  while (s > 0) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[s + threadIdx.x];
      //printf("%d\n",sdata[threadIdx.x]);
      //printf("mid");
    }
    __syncthreads();
    //printf("%d\n",sdata[threadIdx.x]);
    //printf("mid");
    s = s/2;
  }
    //printf("whiledone");
    //printf("%d\n",sdata[threadIdx.x]);
  per_block_results[blockIdx.x] = 0;
  if (threadIdx.x == 0) {
    per_block_results[blockIdx.x] = sdata[0]; //save just the first element of our sdata array

    printf("savedone");
    printf("%d\n",per_block_results[blockIdx.x]); //here prints num_blocks we need, good
  }
  int nb = num_blocks;
  if (threadIdx.x < nb) {
    printf("ok");
    per_block_results[threadIdx.x] = per_block_results[blockIdx.x];
    printf("%d\n",per_block_results[threadIdx.x]);
  }
  /*s = blockDim.x / 2;
  while (nb > 0) {
    if (threadIdx.x < nb) {
      printf("Work");
      final[threadIdx.x] = final[threadIdx.x] + final[s + threadIdx.x];
      printf("done");
    }
    __syncthreads();
    //printf("%d\n",sdata[threadIdx.x]);
    //printf("mid");
    s = s/2;
  }
  printf("mid");
  //printf("%d\n",final[blockIdx.x]);*/
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
  const int num_elements = 1 << 10; //18
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

  std::vector<int> host(sizeof(int)); 


  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_input,h_input.data(),n,cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  block_sum<<<num_blocks, block_size>>>(d_input, d_partial_sums_and_total, d_final, n);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaMemcpy(host.data(),d_final,sizeof(int),cudaMemcpyDeviceToHost);
  std::cout << "Device sum: " << *host.data() << std::endl;


  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop); //write recorded time to milliseconds variable 

  float thruput = 3 * num_elements * sizeof(int) / milliseconds / 10e6;
  std::cout << "throughput: " << thruput << std::endl;


  cudaFree(d_input);
  cudaFree(d_partial_sums_and_total);
  cudaFree(d_final);

  return 0;
}
