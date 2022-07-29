#include <iostream>
#include <vector>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 0 //memory clock rate 6251 Mhz, 384-bit memory bus width -> 6.251e9Hz, 48 bytes
__global__ void saxpy(unsigned int n, double a, double* x, double* y)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; //global index of each thread
  if (i < n) //if i is greater than some value
    y[i] = a * x[i] + y[i]; //to ensure no out of bounds memory access (#threads must be divisible by #blocks)
}

int main(void)
{
  cudaSetDevice(MYDEVICE);

  // 1<<N is the equivalent to 2^N
  unsigned int N = 20 * (1 << 20); //why 20?????
  double *d_x, *d_y;
  std::vector<double> x(N, 16.); //a vector of doubles that is size N and filled with 1's
  std::vector<double> y(N, 16.); //just initialization, can be changed, did we just pick for fun?

  cudaMalloc(&d_x, N * sizeof(double)); //allocate sizes of devices to the devices
  cudaMalloc(&d_y, N * sizeof(double));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); //assign start and stop values to the pointers?

  cudaMemcpy(d_x, x.data(), N * sizeof(double), cudaMemcpyHostToDevice); //host to device data transfer
  cudaMemcpy(d_y, y.data(), N * sizeof(double), cudaMemcpyHostToDevice); //also initialization

  cudaEventRecord(start); //start recording the time

  saxpy<<<(N + 511) / 512, 512>>>(N, 2.0, d_x, d_y); //N+511 blocks in the grid, 512 threads per block

  cudaEventRecord(stop);
  cudaEventSynchronize(stop); //stop recording

  cudaMemcpy(x.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost); //transfer new data over to host
  cudaMemcpy(y.data(), d_y, N * sizeof(double), cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop); //write recorded time to milliseconds variable 

  double maxError = 0.; //just error check
  for (unsigned int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i] - 4.0));

  }
  float thruput = 3 * N * sizeof(double) / milliseconds / 10e6; //thruput formula?
  std::cout << "throughput: " << thruput << std::endl;

  //float bandwidth = 6.251e9 * 48 * 2 / 10e9; //GB/s
  //std::cout << "bandwidth: " << bandwidth << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
} //final result throughput for device 0 (NVIDIA A10) is 55.1805 -> 55180.5 over seconds


//first tests:
// N = 20 * (1 << 20), x(N, 1.), y(N, 2.) :: 55.08 thruput
// N = 25 * (1 << 25), x(N, 1.), y(N, 2.) :: 55.9601 thruput, took a bit to load
// N = 30 * (1 << 30), same xy :: Error shift fcount is too large
// N = 40 * (1 << 40), same xy :: Error shift fcount is too large
// N = 20 * (1 << 20), x(N, 2.), y(N, 2.) :: 55.0337 thruput

//log tests: N = 20 * (1 << 20)
// N = 20 * (1 << 20), x(N, 2.), y(N, 4.) :: 55.1766 thruput
// x(N, 4.), y(N, 8.) :: 55.492 thruput
// x(N, 8.), y(N, 16.) :: 55.7496 thruput
// x(N, 16.), y(N, 32.) :: 55.5096 thruput
// x(N, 32.), y(N, 64.) :: 55.363 thruput
// x(N, 100.), y(N, 200.) :: 55.6116 thruput
// Conclusion: at N=20 the thruput oscillates around ~~55-56 GB/s, wide scope of variation

//log tests: N = 25 * (1 << 25)
// x(N, 2.), y(N, 4.) :: 55.947 thruput
// x(N, 4.), y(N, 8.) :: 56.0041 thruput
// x(N, 8.), y(N, 16.) :: 55.9286 thruput
// x(N, 16.), y(N, 32.) :: 55.9562 thruput
// x(N, 32.), y(N, 64.) :: 55.9298 thruput
// x(N, 100.), y(N, 200.) :: 55.9356 thruput
// Conclusion: at N=25 the thruput oscillates around ~56 GB/s, smaller scope of variation than lower N size

//log tests: N = 20 * (1 << 20)
// x = y = 2 :: 55.722
// x = y = 4 :: 55.3124
// x = y = 8 :: 55.1863
// x = y = 16 :: 55.0877
// Conclusion: thruput decreases the more elements used when vector sizes are equal

// Explanation:
// Greater N = greater size of host/device memory & greater number of blocks in the grid
// Can be a good thing sometimes to increase efficiency, but also such thing as too many blocks
// (creates too much communication overhead and resource contention)
// so makes sense throughput comes up then back down