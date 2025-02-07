#include <cassert>
#include <iostream>
#include <vector>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);
// Part 2 of 4: implement the kernel
__global__ void kernel(int* a, int dimx, int dimy)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y; //fill with indices of threads
    int col = threadIdx.x + blockIdx.x * blockDim.x; //same
    int val = row*dimx + col; //formula given was row*N + col
    a[val] = val; //make every value in the device pt equal to the result from the formula
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
  cudaSetDevice(MYDEVICE);
  // Part 1 and 4 of 4: set the dimensions of the matrix
  int dimx = 19;
  int dimy = 67;

  std::vector<int> h_a(dimx * dimy);
  int num_bytes = dimx * dimy * sizeof(int);

  int* d_a = 0; // device and host pointers

  // allocate memory on the device
  cudaMalloc(&d_a,num_bytes);

  if (NULL == d_a) {
    std::cerr << "couldn't allocate memory" << std::endl;
    return 1;
  }

  // Part 2 of 4: define grid and block size and launch the kernel
  dim3 grid, block;
  block.x = 1;
  block.y = dimy;
  grid.x  = dimx;
  grid.y  = 1;

  kernel<<<grid, block>>>(d_a, dimx, dimy);
  // block until the device has completed
  cudaDeviceSynchronize();
  // check if kernel execution generated an error
  checkCUDAError("kernel execution");
  // device to host copy
  cudaMemcpy(h_a.data(),d_a,num_bytes,cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
  checkCUDAError("cudaMemcpy");
  // verify the data returned to the host is correct
  for (int row = 0; row < dimy; row++) {
    for (int col = 0; col < dimx; col++)
      assert(h_a[row * dimx + col] == row * dimx + col);
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