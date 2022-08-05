#include <iostream>
#include <numeric>
#include <random>
#include <vector>


// Here you can set the device ID that was assigned to you
#define MYDEVICE 1
const int num_blocks = 4; //4096
const int block_size = 8; //64

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int* input, int* per_block_results, const size_t n)
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
      //printf("%d\n",per_block_results[blockIdx.x]);
      //printf("%d\n",blockIdx.x); {
    per_block_results[blockIdx.x] = sdata[0]; //save just the first element of our sdata array
    printf("savedone");
    printf("%d\n",per_block_results[blockIdx.x]); //here prints just two we need, good
  }
      //if (blockIdx.x > num_blocks) and if (per_block_results[blockIdx.x == 0]) { 
        //__syncthreads();
  //per_block_results[blockIdx.x] = per_block_results[num_blocks];
    //want to save sum to just one thread element in perblocks array, not every thread element
  //printf("final");
  //printf("%d\n",per_block_results[blockIdx.x]); //here prints as a block of 1st and a block of 2nd
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
  const int num_elements = 1 << 5; //18
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
 
  int* d_final;
  cudaMalloc((void**)&d_final,sizeof(int));

  std::vector<int> host(sizeof(int)); 

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_input,h_input.data(),n,cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  block_sum<<<num_blocks, block_size>>>(d_input, d_final,n);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaMemcpy(host.data(),d_final,sizeof(int),cudaMemcpyDeviceToHost);
  std::cout << "Device sum: " << *host.data() << std::endl;
  //printf("Device sum: ");
  //printf("%d\n",host.data());
 //Checking vectors
 // for (int i=0; i<h_input.size();i++) {
  //  printf("%d\n",h_input[i]);
  //}
  //printf("%d\n",host.size()); //size is num elems when launch w/ n but should be num_blocks
  //printf("%d\n",host.begin());
  //printf("%d\n",host.end());
  //std::cout << *h_input.data() << std::endl;


  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop); //write recorded time to milliseconds variable 

  float thruput = 3 * num_elements * sizeof(int) / milliseconds / 10e6;
  std::cout << "throughput: " << thruput << std::endl;


  cudaFree(d_input);
  cudaFree(d_final);

  return 0;
}
