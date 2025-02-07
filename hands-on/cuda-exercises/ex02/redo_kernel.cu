#include <cassert>
#include <iostream>
#include <vector>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);
 
// Part 3 of 5: implement the kernel
__global__ void myFirstKernel(int* d_a) //bc d_a is a pointer
{
  int i = threadIdx.x + blockIdx.x * blockDim.x; //global indices of threads
  d_a[i] = blockIdx.x + threadIdx.x + 42; //given
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  cudaSetDevice(MYDEVICE);

  // pointer for device memory
  int* d_a;

  // define grid and block size
  int numBlocks          = 8;
  int numThreadsPerBlock = 8;

  // host vector
  // hint: you might want to reserve some memory
  std::vector<int> h_a; //at start, array was size 0
  h_a.resize(numBlocks * numThreadsPerBlock); //host vector was empty ***set size of host vector total # of threads

  // Part 1 of 5: allocate host and device memory
  size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int); //also total number of threads
  cudaMalloc(&d_a, memSize); //assign memory to the location of pointer

  // Part 2 of 5: configure and launch kernel
  dim3 dimGrid(numBlocks);
  dim3 dimBlock(numThreadsPerBlock);
  myFirstKernel<<<dimGrid,dimBlock>>>(d_a);

  // block until the device has completed
  cudaDeviceSynchronize();

  // check if kernel execution generated an error
  checkCUDAError("kernel execution");

  // Part 4 of 5: device to host copy
  cudaMemcpy(h_a.data(),d_a,memSize,cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
  checkCUDAError("cudaMemcpy");

  // Part 5 of 5: verify the data returned to the host is correct
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
     // assert(h_a[i * numThreadsPerBlock + j] == i + j + 42);
    }
  }

  // free device memory
  cudaFree(d_a);

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  std::cout << "Correct!" << std::endl;

  return 0;
}

void checkCUDAError(const char* msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::cerr << "Cuda error: " << msg << " " << cudaGetErrorString(err)
              << std::endl;
    exit(-1);
  }
}